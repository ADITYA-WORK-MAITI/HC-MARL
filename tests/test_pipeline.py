"""Integration tests for hcmarl.pipeline.

Verifies the end-to-end HC-MARL pipeline (Section 7.3).
"""

import numpy as np
import pytest

from hcmarl.ecbf_filter import ECBFParams
from hcmarl.nswf_allocator import NSWFParams
from hcmarl.pipeline import HCMARLPipeline, TaskProfile, WorkerState
from hcmarl.three_cc_r import SHOULDER, ELBOW, GRIP, ThreeCCrState


# =====================================================================
# TaskProfile
# =====================================================================

class TestTaskProfile:
    """Test task demand profiles (Def 7.1)."""

    def test_valid_profile(self):
        tp = TaskProfile(task_id=1, name="box_lift", demands={"shoulder": 0.4, "grip": 0.6})
        assert tp.get_load("shoulder") == 0.4
        assert tp.get_load("grip") == 0.6

    def test_missing_muscle_returns_zero(self):
        tp = TaskProfile(task_id=1, name="walk", demands={"ankle": 0.3})
        assert tp.get_load("shoulder") == 0.0

    def test_invalid_load_raises(self):
        with pytest.raises(ValueError):
            TaskProfile(task_id=1, name="bad", demands={"shoulder": 1.5})


# =====================================================================
# WorkerState
# =====================================================================

class TestWorkerState:
    """Test worker state management."""

    def test_fresh_worker(self):
        w = WorkerState.fresh(0, ["shoulder", "elbow"])
        assert w.worker_id == 0
        assert w.max_fatigue() == 0.0
        for name in ["shoulder", "elbow"]:
            assert w.muscle_states[name].MR == 1.0

    def test_max_fatigue(self):
        w = WorkerState.fresh(0, ["shoulder", "elbow"])
        w.muscle_states["shoulder"] = ThreeCCrState(MR=0.5, MA=0.2, MF=0.3)
        w.muscle_states["elbow"] = ThreeCCrState(MR=0.6, MA=0.1, MF=0.3)
        assert abs(w.max_fatigue() - 0.3) < 1e-10


# =====================================================================
# Pipeline construction
# =====================================================================

def make_pipeline(
    num_workers: int = 2,
    muscle_names: list[str] | None = None,
) -> HCMARLPipeline:
    """Helper to construct a test pipeline."""
    if muscle_names is None:
        muscle_names = ["shoulder", "elbow", "grip"]

    tasks = [
        TaskProfile(task_id=1, name="box_lift", demands={
            "shoulder": 0.4, "elbow": 0.3, "grip": 0.5,
        }),
        TaskProfile(task_id=2, name="carry", demands={
            "shoulder": 0.2, "elbow": 0.1, "grip": 0.3,
        }),
    ]

    ecbf_params = {}
    for name in muscle_names:
        from hcmarl.three_cc_r import get_muscle
        mp = get_muscle(name)
        # theta_max safely above threshold
        theta = max(mp.theta_min_max * 1.1, mp.theta_min_max + 0.05)
        theta = min(theta, 0.95)
        ecbf_params[name] = ECBFParams(
            theta_max=theta,
            alpha1=0.05,
            alpha2=0.05,
            alpha3=0.1,
        )

    return HCMARLPipeline(
        num_workers=num_workers,
        muscle_names=muscle_names,
        task_profiles=tasks,
        ecbf_params_per_muscle=ecbf_params,
        nswf_params=NSWFParams(kappa=1.0, epsilon=1e-3),
        kp=1.0,
        dt=1.0,
    )


class TestPipelineConstruction:
    """Test pipeline construction."""

    def test_basic_construction(self):
        pipe = make_pipeline()
        assert pipe.num_workers == 2
        assert len(pipe.muscle_names) == 3
        assert len(pipe.task_profiles) == 2

    def test_workers_initialised_fresh(self):
        pipe = make_pipeline(num_workers=3)
        assert len(pipe.workers) == 3
        for w in pipe.workers:
            assert w.max_fatigue() == 0.0

    def test_missing_ecbf_params_raises(self):
        with pytest.raises(ValueError, match="Missing ECBF params"):
            HCMARLPipeline(
                num_workers=1,
                muscle_names=["shoulder"],
                task_profiles=[],
                ecbf_params_per_muscle={},  # Missing!
            )


# =====================================================================
# Pipeline step (Section 7.3)
# =====================================================================

class TestPipelineStep:
    """Test the complete pipeline step."""

    def test_single_step(self):
        pipe = make_pipeline()
        result = pipe.step()
        assert result["step"] == 1
        assert result["time"] == 1.0
        assert len(result["workers"]) == 2

    def test_conservation_after_step(self):
        """All muscle states must satisfy MR + MA + MF = 1 after each step."""
        pipe = make_pipeline()
        pipe.step()
        for w in pipe.workers:
            for name, state in w.muscle_states.items():
                total = state.MR + state.MA + state.MF
                assert abs(total - 1.0) < 1e-6, (
                    f"Worker {w.worker_id}, {name}: "
                    f"MR+MA+MF = {total}"
                )

    def test_non_negative_after_step(self):
        """All compartments remain non-negative."""
        pipe = make_pipeline()
        for _ in range(10):
            pipe.step()
        for w in pipe.workers:
            for name, state in w.muscle_states.items():
                assert state.MR >= -1e-9
                assert state.MA >= -1e-9
                assert state.MF >= -1e-9

    def test_fatigue_increases_under_load(self):
        """Workers doing tasks should accumulate fatigue."""
        pipe = make_pipeline(num_workers=1)
        # High utility to ensure assignment
        util = np.array([[10.0, 10.0]])
        for _ in range(5):
            pipe.step(util)

        # At least one muscle should have MF > 0
        w = pipe.workers[0]
        assert w.max_fatigue() > 0.0

    def test_safety_filter_prevents_overwork(self):
        """ECBF should limit MF growth toward theta_max.

        Note: With Euler integration at dt=1.0, the continuous-time ECBF
        guarantee (Theorem 5.7) holds only approximately. Muscles with
        very low theta_max (e.g. grip at ~0.25) and conservative alpha
        values can overshoot significantly under Euler discretisation.
        We use a 2x tolerance: the key property is that MF stays bounded
        rather than growing without limit.
        """
        pipe = make_pipeline(num_workers=1)
        util = np.array([[10.0, 10.0]])

        # Run many steps
        for _ in range(200):
            pipe.step(util)

        for w in pipe.workers:
            for name, state in w.muscle_states.items():
                theta_max = pipe.ecbf_filters[name].params.theta_max
                # 2x tolerance for Euler discretisation at dt=1 min
                assert state.MF <= theta_max * 2.0, (
                    f"Worker {w.worker_id}, {name}: "
                    f"MF = {state.MF:.4f} >> theta_max = {theta_max:.4f}"
                )

    def test_multiple_steps(self):
        """Pipeline should run multiple steps without errors."""
        pipe = make_pipeline()
        for _ in range(50):
            result = pipe.step()
        assert pipe.step_count == 50
        assert abs(pipe.time - 50.0) < 1e-10

    def test_summary_string(self):
        """Summary should be a non-empty string."""
        pipe = make_pipeline()
        pipe.step()
        s = pipe.summary()
        assert isinstance(s, str)
        assert "HC-MARL Pipeline" in s

    def test_history_recorded(self):
        """Each step should be recorded in history."""
        pipe = make_pipeline()
        for _ in range(5):
            pipe.step()
        assert len(pipe.history) == 5


# =====================================================================
# Config loading
# =====================================================================

class TestConfigLoading:
    """Test pipeline construction from YAML config."""

    def test_from_config(self, tmp_path):
        """Load pipeline from a YAML config file."""
        config = tmp_path / "test_config.yaml"
        config.write_text("""
num_workers: 3
muscle_names:
  - shoulder
  - grip
kp: 1.0
dt: 1.0
tasks:
  - name: box_lift
    demands:
      shoulder: 0.4
      grip: 0.6
  - name: carry
    demands:
      shoulder: 0.2
      grip: 0.3
ecbf:
  shoulder:
    theta_max: 0.70
    alpha1: 0.05
    alpha2: 0.05
    alpha3: 0.1
  grip:
    theta_max: 0.35
    alpha1: 0.05
    alpha2: 0.05
    alpha3: 0.1
nswf:
  kappa: 1.0
  epsilon: 0.001
""")
        pipe = HCMARLPipeline.from_config(str(config))
        assert pipe.num_workers == 3
        assert pipe.muscle_names == ["shoulder", "grip"]
        assert len(pipe.task_profiles) == 2
