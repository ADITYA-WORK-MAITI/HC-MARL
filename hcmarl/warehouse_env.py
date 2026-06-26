"""
HC-MARL Phase 2: Warehouse Multi-Agent Environment
====================================================
PettingZoo-compatible parallel environment wrapping the HC-MARL pipeline.
Each worker agent observes its own physiological state x_i(t) = [MR, MA, MF]
per muscle group and receives task assignments from the NSWF allocator.

Integrates: 3CC-r fatigue model, ECBF safety filter, NSWF task allocation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
import yaml
import copy
from hcmarl.envs.reward_functions import nswf_reward, safety_cost
from hcmarl.ecbf_filter import ECBFFilter, ECBFParams
from hcmarl.three_cc_r import ThreeCCrState, get_muscle as _get_muscle_params


# ---------------------------------------------------------------------------
# Single-agent wrapper (Gymnasium API) for debugging / single-worker tests
# ---------------------------------------------------------------------------

class SingleWorkerWarehouseEnv(gym.Env):
    """Single-worker warehouse env for unit testing and baselines."""

    metadata = {"render_modes": ["human", "ansi"], "name": "SingleWorkerWarehouse-v0"}

    def __init__(
        self,
        muscle_groups: Optional[Dict] = None,
        tasks: Optional[Dict] = None,
        theta_max: Optional[Dict] = None,
        dt: float = 1.0,
        max_steps: int = 60,
        kappa: float = 1.0,
        render_mode: Optional[str] = None,
        ecbf_mode: str = "on",
    ):
        super().__init__()
        if ecbf_mode not in ("on", "off"):
            raise ValueError(f"ecbf_mode must be 'on' or 'off', got '{ecbf_mode}'")
        self.ecbf_mode = ecbf_mode
        self.render_mode = render_mode
        self.dt = dt
        self.max_steps = max_steps
        self.kappa = kappa

        # Default muscle groups: shoulder, elbow, grip
        # These are ISOMETRIC (F, R) from Table 1 — for sustained warehouse
        # task holds. Dynamic tasks require re-calibration (see
        # real_data_calibration.py for dynamic regime with 30-180x larger F).
        self.muscle_groups = muscle_groups or {
            "shoulder": {"F": 0.01820, "R": 0.00168, "r": 15},
            "elbow":    {"F": 0.00912, "R": 0.00094, "r": 15},
            "grip":     {"F": 0.00980, "R": 0.00064, "r": 30},  # Looft et al. 2018 Table 2
        }
        self.muscle_names = list(self.muscle_groups.keys())
        self.n_muscles = len(self.muscle_names)

        # Default tasks with per-muscle target loads
        # Sources: Granata & Marras 1995 (trunk), Hoozemans et al. 2004 (shoulder/grip),
        #          de Looze et al. 2000 (shoulder pushing), Snook & Ciriello 1991 (carry)
        self.tasks = tasks or {
            "heavy_lift":   {"shoulder": 0.45, "elbow": 0.30, "grip": 0.55},  # Hoozemans 2004, Granata 1995
            "light_sort":   {"shoulder": 0.10, "elbow": 0.15, "grip": 0.20},  # Nordander et al. 2000
            "carry":        {"shoulder": 0.25, "elbow": 0.20, "grip": 0.45},  # Snook & Ciriello 1991
            "rest":         {"shoulder": 0.00, "elbow": 0.00, "grip": 0.00},
        }
        self.task_names = list(self.tasks.keys())
        self.n_tasks = len(self.task_names)

        # Safety thresholds per muscle (must satisfy Eq 26: theta_max >= F/(F+R*r)).
        # Under Frey-Law 2012 Table 1 (verified): shoulder theta_min_max=41.9%,
        # elbow 39.3%, grip 33.8% (r=30). Grip raised 0.35->0.45 for an 11.2pp
        # margin; Byström & Fransson-Hall 1994 mean-intermittent ceiling
        # of 17% MVC must be respected by duty cycle (checked in reward_functions).
        self.theta_max = theta_max or {
            "shoulder": 0.70, "elbow": 0.45, "grip": 0.45,
        }

        # C-6.A: Build per-muscle ECBFFilter instances
        self.ecbf_filters = {}
        for m in self.muscle_names:
            mp = _get_muscle_params(m)
            theta = self.theta_max.get(m, 0.7)
            theta = max(theta, mp.theta_min_max + 0.01)
            params = ECBFParams(theta_max=theta, alpha1=0.05, alpha2=0.05, alpha3=0.1)
            self.ecbf_filters[m] = ECBFFilter(muscle=mp, ecbf_params=params)

        # Observation: [MR, MA, MF] per muscle + current_step_normalised
        obs_dim = self.n_muscles * 3 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action: choose a task index (discrete)
        self.action_space = spaces.Discrete(self.n_tasks)

        # State
        self.state = None
        self.current_step = 0

    def _get_obs(self) -> np.ndarray:
        obs = []
        for m in self.muscle_names:
            obs.extend([self.state[m]["MR"], self.state[m]["MA"], self.state[m]["MF"]])
        obs.append(self.current_step / self.max_steps)
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, task_name: str) -> float:
        """NSWF reward matching Eq 33 + Eq 32 of the math doc."""
        task = self.tasks[task_name]
        productivity = sum(task[m] for m in self.muscle_names)
        fatigue = {m: self.state[m]["MF"] for m in self.muscle_names}
        return nswf_reward(productivity, fatigue, self.theta_max, kappa=self.kappa)

    def _integrate_3cc_r(self, task_name: str):
        """Euler integration of 3CC-r ODEs for one timestep.

        Returns:
            (ecbf_interventions, ecbf_clip_total): number of muscles where
            the ECBF clipped the neural drive, and the total clipped magnitude.
        """
        task = self.tasks[task_name]
        kp = 1.0  # proportional gain for baseline controller (Eq 35)
        ecbf_interventions = 0
        ecbf_clip_total = 0.0

        for m in self.muscle_names:
            params = self.muscle_groups[m]
            F = params["F"]
            R_base = params["R"]
            r = params["r"]
            TL = task[m]

            MR = self.state[m]["MR"]
            MA = self.state[m]["MA"]
            MF = self.state[m]["MF"]

            # Neural drive (proportional controller, Eq 35)
            C_nominal = kp * max(TL - MA, 0.0) if TL > 0 else 0.0

            R_eff = R_base if TL > 0 else R_base * r

            if self.ecbf_mode == "on":
                # C-6.A: Use canonical ECBFFilter instead of inlined bounds
                state = ThreeCCrState(MR=MR, MA=MA, MF=MF)
                C, _infeasible = self.ecbf_filters[m].filter_analytical(state, C_nominal, TL)
            else:
                C = max(0.0, C_nominal)

            # Track ECBF interventions
            if C_nominal > 1e-9 and (C_nominal - C) > 1e-9:
                ecbf_interventions += 1
                ecbf_clip_total += C_nominal - C

            # ODEs (Eqs 2-4)
            dMA = C - F * MA
            dMF = F * MA - R_eff * MF
            dMR = R_eff * MF - C

            # Euler step
            MA_new = MA + dMA * self.dt
            MF_new = MF + dMF * self.dt
            MR_new = MR + dMR * self.dt

            # Conservation-preserving guard (no renormalization)
            MA_new = max(0.0, MA_new)
            MF_new = max(0.0, MF_new)
            MR_new = 1.0 - MA_new - MF_new
            if MR_new < 0.0:
                s = MA_new + MF_new
                if s > 0:
                    MA_new /= s
                    MF_new /= s
                MR_new = 0.0

            self.state[m] = {"MR": MR_new, "MA": MA_new, "MF": MF_new}

        return ecbf_interventions, float(ecbf_clip_total)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.state = {}
        for m in self.muscle_names:
            self.state[m] = {"MR": 1.0, "MA": 0.0, "MF": 0.0}
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        task_name = self.task_names[action]
        ecbf_interventions, ecbf_clip_total = self._integrate_3cc_r(task_name)
        self.current_step += 1

        obs = self._get_obs()
        reward = self._compute_reward(task_name)
        terminated = self.current_step >= self.max_steps
        truncated = False

        fatigue = {m: self.state[m]["MF"] for m in self.muscle_names}
        cost = safety_cost(fatigue, self.theta_max)
        info = {
            "task": task_name,
            "fatigue": fatigue,
            "violations": sum(1 for m in self.muscle_names if fatigue[m] > self.theta_max[m]),
            "cost": cost,
            "ecbf_interventions": ecbf_interventions,
            "ecbf_clip_total": ecbf_clip_total,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            lines = [f"Step {self.current_step}/{self.max_steps}"]
            for m in self.muscle_names:
                s = self.state[m]
                lines.append(
                    f"  {m}: MR={s['MR']:.3f} MA={s['MA']:.3f} MF={s['MF']:.3f} "
                    f"(Θmax={self.theta_max[m]:.2f})"
                )
            return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-agent wrapper (PettingZoo Parallel API)
# ---------------------------------------------------------------------------

class WarehouseMultiAgentEnv:
    """
    PettingZoo-style parallel environment for N warehouse workers.
    Each worker independently observes its physiological state and
    receives task assignments. The NSWF allocator handles centralised
    coordination (used in the centralised critic for MAPPO).
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "WarehouseMA-v0"}

    def __init__(
        self,
        n_workers: int = 4,
        muscle_groups: Optional[Dict] = None,
        tasks: Optional[Dict] = None,
        theta_max: Optional[Dict] = None,
        dt: float = 1.0,
        max_steps: int = 60,
        kappa: float = 1.0,
        ecbf_mode: str = "on",
    ):
        if ecbf_mode not in ("on", "off"):
            raise ValueError(f"ecbf_mode must be 'on' or 'off', got '{ecbf_mode}'")
        self.ecbf_mode = ecbf_mode
        self.n_workers = n_workers
        self.dt = dt
        self.max_steps = max_steps
        self.kappa = kappa

        # Isometric (F, R) from Table 1 — for sustained warehouse task holds.
        # Dynamic tasks require re-calibration; see real_data_calibration.py.
        self.muscle_groups = muscle_groups or {
            "shoulder": {"F": 0.01820, "R": 0.00168, "r": 15},
            "elbow":    {"F": 0.00912, "R": 0.00094, "r": 15},
            "grip":     {"F": 0.00980, "R": 0.00064, "r": 30},  # Looft et al. 2018 Table 2
        }
        self.muscle_names = list(self.muscle_groups.keys())
        self.n_muscles = len(self.muscle_names)

        self.tasks = tasks or {
            "heavy_lift":   {"shoulder": 0.45, "elbow": 0.30, "grip": 0.55},  # Hoozemans 2004, Granata 1995
            "light_sort":   {"shoulder": 0.10, "elbow": 0.15, "grip": 0.20},  # Nordander et al. 2000
            "carry":        {"shoulder": 0.25, "elbow": 0.20, "grip": 0.45},  # Snook & Ciriello 1991
            "rest":         {"shoulder": 0.00, "elbow": 0.00, "grip": 0.00},
        }
        self.task_names = list(self.tasks.keys())
        self.n_tasks = len(self.task_names)

        self.theta_max = theta_max or {
            "shoulder": 0.70, "elbow": 0.45, "grip": 0.45,  # grip raised 0.35->0.45: theta_min_max=33.8% (Frey-Law 2012 Table 1 corrected), 11.2pp margin
        }

        # C-6.A: Build per-muscle ECBFFilter instances (shared across workers)
        self.ecbf_filters = {}
        for m in self.muscle_names:
            mp = _get_muscle_params(m)
            theta = self.theta_max.get(m, 0.7)
            theta = max(theta, mp.theta_min_max + 0.01)
            params = ECBFParams(theta_max=theta, alpha1=0.05, alpha2=0.05, alpha3=0.1)
            self.ecbf_filters[m] = ECBFFilter(muscle=mp, ecbf_params=params)

        # Agent IDs
        self.possible_agents = [f"worker_{i}" for i in range(n_workers)]
        self.agents = list(self.possible_agents)

        # Spaces
        obs_dim = self.n_muscles * 3 + 1  # [MR,MA,MF]*muscles + normalised_step
        self.observation_spaces = {
            agent: spaces.Box(0.0, 1.0, (obs_dim,), np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(self.n_tasks)
            for agent in self.possible_agents
        }

        # Global state for centralised critic: all workers' states concatenated
        global_obs_dim = n_workers * self.n_muscles * 3 + 1
        self.global_observation_space = spaces.Box(
            0.0, 1.0, (global_obs_dim,), np.float32
        )

        self.states = {}
        self.current_step = 0

    def _get_obs(self, agent: str) -> np.ndarray:
        idx = int(agent.split("_")[1])
        obs = []
        for m in self.muscle_names:
            s = self.states[idx][m]
            obs.extend([s["MR"], s["MA"], s["MF"]])
        obs.append(self.current_step / self.max_steps)
        return np.array(obs, dtype=np.float32)

    def _get_global_obs(self) -> np.ndarray:
        obs = []
        for i in range(self.n_workers):
            for m in self.muscle_names:
                s = self.states[i][m]
                obs.extend([s["MR"], s["MA"], s["MF"]])
        obs.append(self.current_step / self.max_steps)
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, worker_idx: int, task_name: str) -> float:
        """NSWF reward matching Eq 33 + Eq 32 of the math doc."""
        task = self.tasks[task_name]
        productivity = sum(task[m] for m in self.muscle_names)
        fatigue = {m: self.states[worker_idx][m]["MF"] for m in self.muscle_names}
        return nswf_reward(productivity, fatigue, self.theta_max, kappa=self.kappa)

    def _integrate_worker(self, worker_idx: int, task_name: str):
        """Integrate 3CC-r for one worker, one timestep, with ECBF filtering.

        Returns:
            (ecbf_interventions, ecbf_clip_total): number of muscles where
            the ECBF clipped the neural drive, and the total clipped magnitude.
        """
        task = self.tasks[task_name]
        kp = 1.0  # proportional gain (Eq 35)
        ecbf_interventions = 0
        ecbf_clip_total = 0.0

        for m in self.muscle_names:
            params = self.muscle_groups[m]
            F = params["F"]
            R_base = params["R"]
            r_mult = params["r"]
            TL = task[m]

            s = self.states[worker_idx][m]
            MR, MA, MF = s["MR"], s["MA"], s["MF"]

            C_nominal = kp * max(TL - MA, 0.0) if TL > 0 else 0.0
            R_eff = R_base if TL > 0 else R_base * r_mult

            if self.ecbf_mode == "on":
                # C-6.A: Use canonical ECBFFilter instead of inlined bounds
                state_obj = ThreeCCrState(MR=MR, MA=MA, MF=MF)
                C, _infeasible = self.ecbf_filters[m].filter_analytical(state_obj, C_nominal, TL)
            else:
                C = max(0.0, C_nominal)

            # Track ECBF interventions
            if C_nominal > 1e-9 and (C_nominal - C) > 1e-9:
                ecbf_interventions += 1
                ecbf_clip_total += C_nominal - C

            dMA = C - F * MA
            dMF = F * MA - R_eff * MF
            dMR = R_eff * MF - C

            MA_new = max(0.0, MA + dMA * self.dt)
            MF_new = max(0.0, MF + dMF * self.dt)
            MR_new = 1.0 - MA_new - MF_new
            if MR_new < 0.0:
                s = MA_new + MF_new
                if s > 0:
                    MA_new /= s
                    MF_new /= s
                MR_new = 0.0

            self.states[worker_idx][m] = {"MR": MR_new, "MA": MA_new, "MF": MF_new}

        return ecbf_interventions, float(ecbf_clip_total)

    def reset(self, seed=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        self.agents = list(self.possible_agents)
        self.current_step = 0
        self.states = {}
        for i in range(self.n_workers):
            self.states[i] = {}
            for m in self.muscle_names:
                self.states[i][m] = {"MR": 1.0, "MA": 0.0, "MF": 0.0}

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        rewards = {}
        infos = {}

        for agent in self.agents:
            idx = int(agent.split("_")[1])
            task_idx = actions[agent]
            task_name = self.task_names[task_idx]
            ecbf_interventions, ecbf_clip_total = self._integrate_worker(idx, task_name)
            rewards[agent] = self._compute_reward(idx, task_name)
            fatigue = {m: self.states[idx][m]["MF"] for m in self.muscle_names}
            cost = safety_cost(fatigue, self.theta_max)
            infos[agent] = {
                "task": task_name,
                "fatigue": fatigue,
                "violations": sum(1 for m in fatigue if fatigue[m] > self.theta_max[m]),
                "cost": cost,
                "ecbf_interventions": ecbf_interventions,
                "ecbf_clip_total": ecbf_clip_total,
            }

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        observations = {agent: self._get_obs(agent) for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        """Global state for centralised critic."""
        return self._get_global_obs()


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------

def make_single_env(**kwargs) -> SingleWorkerWarehouseEnv:
    return SingleWorkerWarehouseEnv(**kwargs)

def make_multi_env(**kwargs) -> WarehouseMultiAgentEnv:
    return WarehouseMultiAgentEnv(**kwargs)
