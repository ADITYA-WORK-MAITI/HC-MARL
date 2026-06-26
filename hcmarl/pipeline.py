"""
HC-MARL End-to-End Pipeline.

Orchestrates the complete human-centric multi-agent control loop:
    1. State observation     -- read x_i(t) = [MR, MA, MF] per worker
    2. Task allocation       -- NSWF centralised planner (Eq 33)
    3. Load translation      -- map task to target load per muscle group
    4. Neural drive          -- baseline controller (Eq 35) or RL policy
    5. Safety filtering      -- ECBF dual-barrier QP (Eq 20)
    6. State update          -- integrate 3CC-r ODEs (Eqs 2--4)
    7. Repeat

Mathematical reference: HC-MARL Framework v12, Section 7.3.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Optional

import numpy as np

from hcmarl.ecbf_filter import ECBFFilter, ECBFParams
from hcmarl.nswf_allocator import AllocationResult, NSWFAllocator, NSWFParams
from hcmarl.three_cc_r import (
    MUSCLE_REGISTRY,
    MuscleParams,
    ThreeCCr,
    ThreeCCrState,
    get_muscle,
)
from hcmarl.utils import load_yaml, get_logger

logger = get_logger(__name__)


# =========================================================================
# Task demand profiles (Def 7.1)
# =========================================================================

@dataclasses.dataclass
class TaskProfile:
    """Demand profile for a single task (Def 7.1).

    Each task is characterised by target loads across relevant muscle groups:
        TL_g^{(j)} in [0, 1] for each muscle group g.

    Attributes:
        task_id: Unique task identifier (1-indexed for productive tasks).
        name: Human-readable task name.
        demands: Dict mapping muscle group name -> target load (fraction of MVC).
    """

    task_id: int
    name: str
    demands: dict[str, float]

    def __post_init__(self) -> None:
        for muscle, load in self.demands.items():
            if load < 0.0 or load > 1.0:
                raise ValueError(
                    f"Target load for {muscle} in task '{self.name}' "
                    f"must be in [0, 1], got {load}"
                )

    def get_load(self, muscle_name: str) -> float:
        """Get target load for a specific muscle group.

        Returns 0.0 if the task does not involve this muscle.
        """
        return self.demands.get(muscle_name, 0.0)


# =========================================================================
# Worker state
# =========================================================================

@dataclasses.dataclass
class WorkerState:
    """Complete state of a single human worker.

    Maintains one ThreeCCrState per muscle group being tracked,
    plus the worker's current task assignment.
    """

    worker_id: int
    muscle_states: dict[str, ThreeCCrState]
    current_task: Optional[int] = None  # None or 0 = resting

    @classmethod
    def fresh(cls, worker_id: int, muscle_names: list[str]) -> WorkerState:
        """Create a fully rested worker."""
        return cls(
            worker_id=worker_id,
            muscle_states={name: ThreeCCrState.fresh() for name in muscle_names},
        )

    def max_fatigue(self) -> float:
        """Return the maximum MF across all tracked muscle groups."""
        return max(s.MF for s in self.muscle_states.values())

    def fatigue_for_allocation(self) -> float:
        """Return the fatigue level used by the NSWF allocator.

        S-8: Uses max(MF) across all muscles as the aggregate scalar MF_i
        for worker i in the NSWF objective (Eq 32-33). The math doc writes
        D_i(MF_i) with a single scalar per worker but does not specify how
        to aggregate across muscle groups.

        Design choice: max(MF) is the conservative bottleneck measure.
        Rationale:
          1. The binary safety cost (safety_cost in reward_functions.py)
             triggers when ANY muscle exceeds theta_max. Using max(MF) for
             the disagreement utility aligns the allocation objective with
             the safety constraint -- both are bottleneck-driven.
          2. A worker with one highly fatigued muscle is operationally
             limited by that muscle regardless of how fresh the others are.
          3. Consistent with reward_functions.nswf_reward() which also uses
             max(MF) (see S-15 documentation there).
        """
        return self.max_fatigue()


# =========================================================================
# Pipeline
# =========================================================================

class HCMARLPipeline:
    """End-to-end HC-MARL control pipeline (Section 7.3).

    Coordinates the 3CC-r model, ECBF filter, and NSWF allocator into
    the seven-step loop described in Section 7.3 of the framework.

    Args:
        num_workers: Number of human workers (N).
        muscle_names: List of muscle groups to track per worker.
        task_profiles: List of available productive tasks.
        ecbf_params_per_muscle: ECBF design parameters per muscle group.
        nswf_params: NSWF allocator parameters.
        kp: Baseline neural drive controller gain (Eq 35).
        dt: Integration time step in minutes.
    """

    def __init__(
        self,
        num_workers: int,
        muscle_names: list[str],
        task_profiles: list[TaskProfile],
        ecbf_params_per_muscle: dict[str, ECBFParams],
        nswf_params: Optional[NSWFParams] = None,
        kp: float = 10.0,
        dt: float = 1.0,
    ) -> None:
        self.num_workers = num_workers
        self.muscle_names = muscle_names
        self.task_profiles = task_profiles
        self.dt = dt

        # -- Build muscle models (one per muscle group) --
        self.muscle_params: dict[str, MuscleParams] = {}
        self.models: dict[str, ThreeCCr] = {}
        for name in muscle_names:
            mp = get_muscle(name)
            self.muscle_params[name] = mp
            self.models[name] = ThreeCCr(params=mp, kp=kp)

        # -- Build ECBF filters (one per muscle group) --
        self.ecbf_filters: dict[str, ECBFFilter] = {}
        for name in muscle_names:
            ep = ecbf_params_per_muscle.get(name)
            if ep is None:
                raise ValueError(f"Missing ECBF params for muscle '{name}'")
            self.ecbf_filters[name] = ECBFFilter(
                muscle=self.muscle_params[name],
                ecbf_params=ep,
            )

        # -- Build NSWF allocator --
        self.allocator = NSWFAllocator(nswf_params or NSWFParams())

        # -- Initialise worker states --
        self.workers: list[WorkerState] = [
            WorkerState.fresh(i, muscle_names) for i in range(num_workers)
        ]

        # -- State tracking --
        self.time: float = 0.0
        self.step_count: int = 0
        self.history: list[dict[str, Any]] = []

    # -----------------------------------------------------------------
    # Step 1: State observation
    # -----------------------------------------------------------------

    def _observe_states(self) -> list[WorkerState]:
        """Read current worker states (Step 1 of Section 7.3)."""
        return self.workers

    # -----------------------------------------------------------------
    # Step 2: Task allocation (NSWF)
    # -----------------------------------------------------------------

    def _allocate_tasks(
        self, utility_matrix: np.ndarray
    ) -> AllocationResult:
        """Solve NSWF allocation (Step 2 of Section 7.3).

        Args:
            utility_matrix: Shape (N, M) productivity utilities.

        Returns:
            AllocationResult from the NSWF solver.
        """
        fatigue = np.array([w.fatigue_for_allocation() for w in self.workers])
        return self.allocator.allocate(utility_matrix, fatigue)

    # -----------------------------------------------------------------
    # Steps 3--6: Per-worker update
    # -----------------------------------------------------------------

    def _update_worker(
        self,
        worker: WorkerState,
        task_id: int,  # 0 = rest, >= 1 = productive task
    ) -> dict[str, Any]:
        """Execute steps 3--6 for a single worker.

        3. Load translation: look up TL per muscle for assigned task.
        4. Neural drive: baseline controller produces C_nom.
        5. Safety filtering: ECBF QP clips C_nom -> C*.
        6. State update: Euler step with C*.

        Returns:
            Dict with diagnostic info for logging.
        """
        worker.current_task = task_id
        diagnostics: dict[str, Any] = {"worker_id": worker.worker_id, "task": task_id}

        if task_id == 0:
            # Resting: C = 0, TL = 0 for all muscles
            task_profile = None
        else:
            # Find task profile
            task_profile = None
            for tp in self.task_profiles:
                if tp.task_id == task_id:
                    task_profile = tp
                    break
            if task_profile is None:
                raise ValueError(f"Unknown task_id {task_id}")

        for muscle_name in self.muscle_names:
            model = self.models[muscle_name]
            ecbf = self.ecbf_filters[muscle_name]
            state = worker.muscle_states[muscle_name]

            # Step 3: Load translation (Def 7.1)
            if task_profile is not None:
                target_load = task_profile.get_load(muscle_name)
            else:
                target_load = 0.0

            # Step 4: Neural drive (Eq 35)
            C_nom = model.baseline_neural_drive(target_load, state.MA)

            # Step 5: Safety filtering (Eq 20)
            C_safe, ecbf_diag = ecbf.filter(state, C_nom, target_load)

            # Step 6: State update (Eqs 2--4 via Euler)
            new_state = model.step_euler(state, C_safe, target_load, self.dt)
            worker.muscle_states[muscle_name] = new_state

            diagnostics[muscle_name] = {
                "MR": new_state.MR,
                "MA": new_state.MA,
                "MF": new_state.MF,
                "TL": target_load,
                "C_nom": C_nom,
                "C_safe": C_safe,
                "h": ecbf_diag.h,
                "h2": ecbf_diag.h2,
                "psi_1": ecbf_diag.psi_1,
                "was_clipped": ecbf_diag.was_clipped,
            }

        return diagnostics

    # -----------------------------------------------------------------
    # Main step (Section 7.3, complete loop)
    # -----------------------------------------------------------------

    def step(
        self,
        utility_matrix: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Execute one complete allocation round (Steps 1--7).

        Args:
            utility_matrix: Shape (N, M) productivity utilities.
                If None, uses a default uniform utility of 1.0 for all
                worker-task pairs.

        Returns:
            Dict containing allocation result and per-worker diagnostics.
        """
        N = self.num_workers
        M = len(self.task_profiles)

        # Default utility matrix
        if utility_matrix is None:
            utility_matrix = np.ones((N, M))

        # Step 1: Observe
        workers = self._observe_states()

        # Step 2: Allocate
        allocation = self._allocate_tasks(utility_matrix)
        logger.debug(
            "Allocation: %s",
            {i: j for i, j in allocation.assignments.items()},
        )

        # Steps 3--6: Update each worker
        worker_diagnostics = []
        for i, worker in enumerate(workers):
            task_id = allocation.assignments.get(i, 0)
            diag = self._update_worker(worker, task_id)
            worker_diagnostics.append(diag)

        # Step 7: Advance time
        self.time += self.dt
        self.step_count += 1

        step_result = {
            "step": self.step_count,
            "time": self.time,
            "allocation": allocation,
            "workers": worker_diagnostics,
        }
        self.history.append(step_result)

        return step_result

    # -----------------------------------------------------------------
    # Configuration loading
    # -----------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str) -> HCMARLPipeline:
        """Construct pipeline from a YAML configuration file.

        Args:
            config_path: Path to the YAML config file.

        Returns:
            Configured HCMARLPipeline instance.
        """
        cfg = load_yaml(config_path)

        num_workers = cfg.get("num_workers", 4)
        muscle_names = cfg.get("muscle_names", ["shoulder", "elbow", "grip"])
        dt = cfg.get("dt", 1.0)
        kp = cfg.get("kp", 1.0)

        # Task profiles
        task_profiles = []
        for i, t_cfg in enumerate(cfg.get("tasks", []), start=1):
            task_profiles.append(TaskProfile(
                task_id=i,
                name=t_cfg.get("name", f"task_{i}"),
                demands=t_cfg.get("demands", {}),
            ))

        # ECBF params per muscle
        ecbf_cfg = cfg.get("ecbf", {})
        ecbf_params_per_muscle = {}
        for m_name in muscle_names:
            m_ecbf = ecbf_cfg.get(m_name, {})
            mp = get_muscle(m_name)
            # S-7: Additive 10pp margin above theta_min_max (Eq 25-26).
            # Multiplicative (1.1x) gave razor-thin margins for high-theta_min_max
            # muscles (shoulder 6.3pp, elbow 3.9pp). Additive 10pp is meaningful
            # for all muscles regardless of theta_min_max magnitude.
            default_theta = min(mp.theta_min_max + 0.10, 0.95)
            if "theta_max" not in m_ecbf:
                import warnings
                warnings.warn(
                    f"No explicit theta_max for '{m_name}' in config. "
                    f"Using default {default_theta:.3f} "
                    f"(theta_min_max={mp.theta_min_max:.3f} + 0.10). "
                    f"Specify ecbf.{m_name}.theta_max in your config for "
                    f"reproducible experiments.",
                    UserWarning, stacklevel=2,
                )
            ecbf_params_per_muscle[m_name] = ECBFParams(
                theta_max=m_ecbf.get("theta_max", default_theta),
                alpha1=m_ecbf.get("alpha1", 0.5),
                alpha2=m_ecbf.get("alpha2", 0.5),
                alpha3=m_ecbf.get("alpha3", 0.5),
            )

        # NSWF params
        nswf_cfg = cfg.get("nswf", {})
        nswf_params = NSWFParams(
            kappa=nswf_cfg.get("kappa", 1.0),
            epsilon=nswf_cfg.get("epsilon", 1e-3),
        )

        return cls(
            num_workers=num_workers,
            muscle_names=muscle_names,
            task_profiles=task_profiles,
            ecbf_params_per_muscle=ecbf_params_per_muscle,
            nswf_params=nswf_params,
            kp=kp,
            dt=dt,
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the current pipeline state."""
        lines = [
            f"HC-MARL Pipeline | Step {self.step_count} | Time {self.time:.1f} min",
            f"Workers: {self.num_workers} | Muscles: {self.muscle_names}",
            f"Tasks: {[tp.name for tp in self.task_profiles]}",
            "",
        ]
        for w in self.workers:
            task_str = f"Task {w.current_task}" if w.current_task else "Resting"
            max_mf = w.max_fatigue()
            lines.append(
                f"  Worker {w.worker_id}: {task_str} | Max MF = {max_mf:.4f}"
            )
        return "\n".join(lines)
