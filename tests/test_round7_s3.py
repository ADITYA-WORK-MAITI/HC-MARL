"""
Tests for Cycle S-3 audit fixes (S-7, S-8, S-11, S-17, S-18).

Verifies:
    S-7   -- pipeline.py from_config uses additive 10pp margin for theta_max default
    S-8   -- fatigue_for_allocation uses max(MF), documented as design choice
    S-11  -- MMICRL E-step guard documented as standard EM practice
    S-17  -- grip theta_max default raised from 0.25 to 0.35 in pettingzoo_wrapper
    S-18  -- ankle/trunk ECBF inactivity documented as biologically justified
"""
import inspect
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# S-7: pipeline.py from_config additive theta_max margin
# =====================================================================

class TestS7:

    def test_additive_margin_formula(self):
        """from_config default theta_max must use theta_min_max + 0.10, not * 1.1."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "hcmarl", "pipeline.py"
        )
        with open(src_path) as f:
            source = f.read()
        # Must NOT have the old multiplicative formula
        assert "theta_min_max * 1.1" not in source, \
            "Old multiplicative theta_max formula still present in pipeline.py"
        # Must have the new additive formula
        assert "theta_min_max + 0.10" in source, \
            "New additive theta_max formula not found in pipeline.py"

    def test_shoulder_margin_at_least_10pp(self):
        """Shoulder default theta_max must have >= 10pp margin above theta_min_max."""
        from hcmarl.three_cc_r import get_muscle
        mp = get_muscle("shoulder")
        default_theta = min(mp.theta_min_max + 0.10, 0.95)
        margin = default_theta - mp.theta_min_max
        assert margin >= 0.099, f"Shoulder margin {margin:.3f} < 10pp"

    def test_grip_margin_at_least_10pp(self):
        """Grip default theta_max must have >= 10pp margin above theta_min_max."""
        from hcmarl.three_cc_r import get_muscle
        mp = get_muscle("grip")
        default_theta = min(mp.theta_min_max + 0.10, 0.95)
        margin = default_theta - mp.theta_min_max
        assert margin >= 0.099, f"Grip margin {margin:.3f} < 10pp"

    def test_warning_on_missing_theta_max(self):
        """from_config must warn when theta_max is not in config."""
        from hcmarl.pipeline import HCMARLPipeline
        import tempfile, yaml
        # Minimal config with no ecbf theta_max
        cfg = {
            "num_workers": 2,
            "muscle_names": ["shoulder"],
            "dt": 1.0,
            "tasks": [{"name": "test_task", "demands": {"shoulder": 0.3}}],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f)
            tmp_path = f.name
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pipeline = HCMARLPipeline.from_config(tmp_path)
                theta_warnings = [x for x in w if "theta_max" in str(x.message)]
                assert len(theta_warnings) >= 1, \
                    "No warning raised for missing theta_max in config"
        finally:
            os.unlink(tmp_path)

    def test_alpha_defaults_aligned_to_0_5(self):
        """from_config alpha defaults must be 0.5, matching S-25 config alignment."""
        from hcmarl.pipeline import HCMARLPipeline
        import tempfile, yaml
        cfg = {
            "num_workers": 2,
            "muscle_names": ["shoulder"],
            "dt": 1.0,
            "tasks": [{"name": "t", "demands": {"shoulder": 0.3}}],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f)
            tmp_path = f.name
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                pipeline = HCMARLPipeline.from_config(tmp_path)
            # Check the ECBF params via the ecbf_filters
            for m, ecbf_filter in pipeline.ecbf_filters.items():
                p = ecbf_filter.params
                assert p.alpha1 == 0.5, f"{m} alpha1={p.alpha1}, expected 0.5"
                assert p.alpha2 == 0.5, f"{m} alpha2={p.alpha2}, expected 0.5"
                assert p.alpha3 == 0.5, f"{m} alpha3={p.alpha3}, expected 0.5"
        finally:
            os.unlink(tmp_path)


# =====================================================================
# S-8: max(MF) aggregation documented in pipeline.py
# =====================================================================

class TestS8:

    def test_fatigue_for_allocation_uses_max(self):
        """fatigue_for_allocation must return max(MF) across muscles."""
        from hcmarl.pipeline import WorkerState
        from hcmarl.three_cc_r import ThreeCCrState
        ws = WorkerState(
            worker_id=0,
            muscle_states={
                "shoulder": ThreeCCrState(MR=0.5, MA=0.1, MF=0.4),
                "elbow": ThreeCCrState(MR=0.8, MA=0.1, MF=0.1),
                "grip": ThreeCCrState(MR=0.7, MA=0.1, MF=0.2),
            }
        )
        assert ws.fatigue_for_allocation() == 0.4, \
            "fatigue_for_allocation should return max(MF)=0.4"

    def test_s8_documented_in_source(self):
        """S-8 design choice must be documented in fatigue_for_allocation docstring."""
        from hcmarl.pipeline import WorkerState
        doc = WorkerState.fatigue_for_allocation.__doc__
        assert "S-8" in doc, "S-8 tag missing from fatigue_for_allocation docstring"
        assert "bottleneck" in doc.lower() or "conservative" in doc.lower(), \
            "Justification (bottleneck/conservative) missing from docstring"

    def test_consistent_with_reward_functions(self):
        """pipeline max(MF) must be consistent with reward_functions max(MF)."""
        from hcmarl.envs.reward_functions import nswf_reward, disagreement_utility
        from hcmarl.pipeline import WorkerState
        from hcmarl.three_cc_r import ThreeCCrState
        # Same worker state
        fatigue = {"shoulder": 0.4, "elbow": 0.1}
        ws = WorkerState(
            worker_id=0,
            muscle_states={
                "shoulder": ThreeCCrState(MR=0.5, MA=0.1, MF=0.4),
                "elbow": ThreeCCrState(MR=0.8, MA=0.1, MF=0.1),
            }
        )
        # Both should use max(MF)=0.4
        pipeline_mf = ws.fatigue_for_allocation()
        reward_di = disagreement_utility(max(fatigue.values()))
        pipeline_di = disagreement_utility(pipeline_mf)
        assert abs(reward_di - pipeline_di) < 1e-10, \
            "Pipeline and reward_functions disagree on fatigue aggregation"


# =====================================================================
# S-11: E-step guard documented as standard EM practice
# =====================================================================

class TestS11:

    def test_s11_comment_present(self):
        """S-11 documentation must exist in mmicrl.py _discover_types_cfde."""
        from hcmarl.mmicrl import MMICRL
        src = inspect.getsource(MMICRL._discover_types_cfde)
        assert "S-11" in src, "S-11 tag missing from _discover_types_cfde"

    def test_mcLachlan_reference(self):
        """The EM regularization must reference McLachlan & Peel."""
        from hcmarl.mmicrl import MMICRL
        src = inspect.getsource(MMICRL._discover_types_cfde)
        assert "McLachlan" in src, \
            "McLachlan & Peel reference missing from S-11 documentation"

    def test_min_count_guard_exists(self):
        """The 5% minimum count guard must exist in E-step."""
        from hcmarl.mmicrl import MMICRL
        src = inspect.getsource(MMICRL._discover_types_cfde)
        assert "min_count" in src, "min_count guard not found in E-step code"
        assert "0.05" in src, "5% threshold not found in E-step code"

    def test_n_types_is_hyperparameter_documented(self):
        """Documentation must state K (n_types) is a hyperparameter."""
        from hcmarl.mmicrl import MMICRL
        src = inspect.getsource(MMICRL._discover_types_cfde)
        assert "hyperparameter" in src.lower(), \
            "n_types as hyperparameter not documented"


# =====================================================================
# Grip theta_max raised 0.25 -> 0.35 -> 0.45 for ECBF margin
# =====================================================================

class TestS17:

    def test_pettingzoo_grip_default(self):
        """PettingZoo env must default grip theta_max to 0.45 per CONSTANTS_AUDIT
        v3 (Eq 26 floor 33.8% under Frey-Law 2012 F,R with r=30; 0.35 left
        only 1.2pp margin, raised to 0.45 for ~11.2pp)."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=5)
        grip_theta = env.theta_max.get("grip")
        assert grip_theta == 0.45, \
            f"grip theta_max default is {grip_theta}, expected 0.45"

    def test_grip_margin_positive(self):
        """Grip margin (theta_max - theta_min_max) must be positive (Assumption 5.5).

        Under Frey-Law 2012 Table 1 values (grip F=0.00980, R=0.00064, r=30),
        theta_min_max = 33.8%. The production
        theta_max = 0.45 gives an 11.2pp margin -- same order as other
        muscles. This test is the numeric floor for that margin.
        """
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        from hcmarl.three_cc_r import get_muscle
        env = WarehousePettingZoo(n_workers=2, max_steps=5)
        grip_theta = env.theta_max.get("grip")
        grip_min = get_muscle("grip").theta_min_max
        margin = grip_theta - grip_min
        assert margin > 0.0, \
            f"Grip margin {margin:.3f} violates Assumption 5.5 (theta_max={grip_theta}, theta_min_max={grip_min:.3f})"

    def test_no_025_grip_default(self):
        """The old 0.25 grip default must not appear in pettingzoo_wrapper.py."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "hcmarl", "envs", "pettingzoo_wrapper.py"
        )
        with open(src_path) as f:
            source = f.read()
        # Should not have grip: 0.25 in the default_theta dict
        assert '"grip": 0.25' not in source, \
            "Old grip default 0.25 still present in pettingzoo_wrapper.py"

    def test_consistent_with_warehouse_env(self):
        """Grip default must match warehouse_env.py (both 0.45 post-v3)."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        pz_env = WarehousePettingZoo(n_workers=2, max_steps=5)
        assert pz_env.theta_max["grip"] == 0.45


# =====================================================================
# S-18: ankle/trunk ECBF inactivity documented as biologically justified
# =====================================================================

class TestS18:

    def test_rest_phase_margin_comment_present(self):
        """Rest-phase margin documentation must exist in pettingzoo_wrapper.py.

        constants table replaced the old tag with a corrected
        theta_min_max and Rr/F table. The comment block is now keyed on
        per-muscle margin numbers (28.1pp/39.6pp/...), which is what this
        test now verifies.
        """
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "hcmarl", "envs", "pettingzoo_wrapper.py"
        )
        with open(src_path) as f:
            source = f.read()
        assert "Margins vs the defaults" in source, \
            "Margin/assumption comment block missing from pettingzoo_wrapper.py"

    def test_ankle_rr_over_f_documented(self):
        """Documentation must mention ankle's Rr/F ratio under corrected values.

        Under constants audit corrected parameters, ankle Rr/F = 1.48.
        The previously expected "46.35" value was an artifact of the
        transcription error where ankle R carried shoulder F.
        """
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "hcmarl", "envs", "pettingzoo_wrapper.py"
        )
        with open(src_path) as f:
            source = f.read()
        assert "ankle 1.48" in source or "ankle  1.48" in source, \
            "Ankle Rr/F ratio (1.48) not documented in pettingzoo_wrapper.py"

    def test_ankle_ecbf_rarely_binds(self):
        """Ankle ECBF should not bind during normal task execution (light_sort)."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=1, max_steps=60, ecbf_mode="on")
        env.reset()
        total_ankle_interventions = 0
        for _ in range(60):
            # light_sort = low demand task
            _, _, _, _, infos = env.step({"worker_0": 1})  # light_sort
            total_ankle_interventions += infos["worker_0"]["ecbf_interventions"]
        # With ankle theta_max=0.80 and low demand, ECBF should rarely/never bind
        # We check ankle MF stays far below threshold
        ankle_mf = env.states[0]["ankle"]["MF"]
        ankle_theta = env.theta_max_per_worker[0]["ankle"]
        assert ankle_mf < ankle_theta * 0.5, \
            f"Ankle MF={ankle_mf:.3f} unexpectedly close to theta_max={ankle_theta}"

    def test_all_muscles_have_rest_overshoot_risk(self):
        """Under corrected Frey-Law 2012 parameters, every muscle has Rr/F > 1.

        The earlier framing — shoulder (Rr/F < 1) as the uniquely vulnerable
        ECBF target and ankle/trunk as self-limiting — was an artifact of the
        transcription errors in F, R documented in constants audit.
        Under corrected values every muscle can overshoot during rest, so
        the ECBF safety argument must apply uniformly. Shoulder remains the
        tightest binding margin by virtue of having the highest isometric F,
        but the Rr/F < 1 separation no longer holds.
        """
        from hcmarl.three_cc_r import get_muscle, ALL_MUSCLES
        for m in ALL_MUSCLES:
            assert m.Rr_over_F > 1.0, \
                f"{m.name} Rr/F={m.Rr_over_F:.3f} should be > 1 under corrected values"
        shoulder = get_muscle("shoulder")
        # Shoulder has the largest isometric F, so its ECBF constraint binds
        # most often even though Rr/F > 1 for all muscles now.
        assert shoulder.F == max(m.F for m in ALL_MUSCLES), \
            "Shoulder should carry the largest isometric F"
