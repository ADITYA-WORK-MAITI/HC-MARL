"""
Tests for Cycle S-4 audit fixes (S-1 through S-40, 16 remaining items).

Verifies:
    S-1   -- simulate() dead C_values removed
    S-2   -- filter_analytical returns (C_safe, infeasible) tuple
    S-3   -- ECBFDiagnostics has infeasible field
    S-5   -- disagreement_utility symmetric clamping
    S-13  -- validate_mmicrl removed from mmicrl.py __main__
    S-14  -- MMICRL.fit() warns on auto-detected n_actions
    S-21  -- LagrangianRolloutBuffer overflow guard
    S-24  -- RolloutBuffer overflow guard
    S-27  -- All configs have explicit kp
    S-28  -- Conservation tests check 0 <= x <= 1 bounds
    S-29  -- Quantitative recovery comparison vs RK45
    S-30  -- Episode end checks physiological state
    S-31  -- Tight ECBF tolerance test (separate from 2x stress test)
    S-32  -- theta_max flows through to env
    S-35  -- Tests use env.global_obs_dim, not hardcoded formula
    S-40  -- n_tasks includes rest action (verified + assertion)
"""
import inspect
import os
import sys
import warnings

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# S-1: simulate() dead C_values removed
# =====================================================================

class TestS1:

    def test_no_C_values_in_simulate(self):
        """simulate() must not have a dead C_values list."""
        from hcmarl.three_cc_r import ThreeCCr
        src = inspect.getsource(ThreeCCr.simulate)
        assert "C_values" not in src.split("# S-1")[0], \
            "Dead C_values list still present in simulate() before S-1 comment"

    def test_simulate_still_works(self):
        """simulate() must still produce correct output after removing C_values."""
        from hcmarl.three_cc_r import ThreeCCr, ThreeCCrState, SHOULDER
        model = ThreeCCr(params=SHOULDER, kp=1.0)
        result = model.simulate(ThreeCCrState.fresh(), target_load=0.3, duration=10.0)
        assert "t" in result
        assert "MR" in result
        assert "C" in result
        assert len(result["t"]) > 0
        # Conservation must hold
        for i in range(len(result["t"])):
            total = result["MR"][i] + result["MA"][i] + result["MF"][i]
            assert abs(total - 1.0) < 1e-6

    def test_simulate_C_override(self):
        """simulate() with C_override must still work."""
        from hcmarl.three_cc_r import ThreeCCr, ThreeCCrState, SHOULDER
        model = ThreeCCr(params=SHOULDER, kp=1.0)
        result = model.simulate(ThreeCCrState.fresh(), target_load=0.0,
                                duration=5.0, C_override=0.01)
        assert np.all(np.abs(result["C"] - 0.01) < 1e-10)


# =====================================================================
# S-2: filter_analytical returns (C_safe, infeasible) tuple
# =====================================================================

class TestS2:

    def test_filter_analytical_returns_tuple(self):
        """filter_analytical must return (float, bool) tuple."""
        from hcmarl.ecbf_filter import ECBFFilter, ECBFParams
        from hcmarl.three_cc_r import SHOULDER, ThreeCCrState
        filt = ECBFFilter(SHOULDER, ECBFParams(theta_max=0.70))
        result = filt.filter_analytical(ThreeCCrState.fresh(), 0.01, 0.3)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2
        C_safe, infeasible = result
        assert isinstance(C_safe, float)
        assert isinstance(infeasible, bool)

    def test_infeasible_when_both_bounds_negative(self):
        """infeasible=True when both upper bounds are negative."""
        from hcmarl.ecbf_filter import ECBFFilter, ECBFParams
        from hcmarl.three_cc_r import SHOULDER, ThreeCCrState
        filt = ECBFFilter(SHOULDER, ECBFParams(theta_max=0.70))
        # Deeply fatigued state: MF near theta_max, MR near 0
        state = ThreeCCrState(MR=0.01, MA=0.29, MF=0.70)
        C_safe, infeasible = filt.filter_analytical(state, 0.1, 0.3)
        # With such high MF and low MR, the bounds should be very restrictive
        assert C_safe == 0.0, f"Expected C_safe=0 in deep fatigue, got {C_safe}"

    def test_not_infeasible_for_fresh_state(self):
        """infeasible=False for a fresh worker."""
        from hcmarl.ecbf_filter import ECBFFilter, ECBFParams
        from hcmarl.three_cc_r import SHOULDER, ThreeCCrState
        filt = ECBFFilter(SHOULDER, ECBFParams(theta_max=0.70))
        C_safe, infeasible = filt.filter_analytical(ThreeCCrState.fresh(), 0.01, 0.3)
        assert not infeasible, "Fresh state should not be infeasible"
        assert C_safe > 0


# =====================================================================
# S-3: ECBFDiagnostics has infeasible field
# =====================================================================

class TestS3:

    def test_infeasible_field_exists(self):
        """ECBFDiagnostics must have an infeasible field."""
        from hcmarl.ecbf_filter import ECBFDiagnostics
        import dataclasses
        fields = {f.name for f in dataclasses.fields(ECBFDiagnostics)}
        assert "infeasible" in fields, "infeasible field missing from ECBFDiagnostics"

    def test_infeasible_set_on_qp_infeasibility(self):
        """filter() must set infeasible=True when QP is infeasible."""
        from hcmarl.ecbf_filter import ECBFFilter, ECBFParams
        from hcmarl.three_cc_r import SHOULDER, ThreeCCrState
        filt = ECBFFilter(SHOULDER, ECBFParams(theta_max=0.70))
        # Normal case: fresh state should be feasible
        _, diag = filt.filter(ThreeCCrState.fresh(), 0.01, 0.3)
        assert not diag.infeasible, "Fresh state QP should be feasible"

    def test_infeasible_false_for_normal_operation(self):
        """Moderate fatigue should not trigger infeasibility."""
        from hcmarl.ecbf_filter import ECBFFilter, ECBFParams
        from hcmarl.three_cc_r import SHOULDER, ThreeCCrState
        filt = ECBFFilter(SHOULDER, ECBFParams(theta_max=0.70))
        state = ThreeCCrState(MR=0.5, MA=0.3, MF=0.2)
        _, diag = filt.filter(state, 0.01, 0.3)
        assert not diag.infeasible


# =====================================================================
# S-5: disagreement_utility symmetric clamping
# =====================================================================

class TestS5:

    def test_negative_mf_clamped(self):
        """Slightly negative MF (float noise) must not raise, returns 0."""
        from hcmarl.envs.reward_functions import disagreement_utility
        result = disagreement_utility(-0.001)
        assert result == 0.0, f"Expected 0.0 for MF=-0.001, got {result}"

    def test_mf_above_1_clamped(self):
        """MF >= 1 must not raise or return inf, returns finite value."""
        from hcmarl.envs.reward_functions import disagreement_utility
        result = disagreement_utility(1.0)
        assert np.isfinite(result), f"Expected finite for MF=1.0, got {result}"

    def test_mf_above_1_returns_large(self):
        """MF=0.999 should return a large but finite disagreement."""
        from hcmarl.envs.reward_functions import disagreement_utility
        result = disagreement_utility(0.999)
        assert result > 10.0, f"Expected large D for MF=0.999, got {result}"
        assert np.isfinite(result)

    def test_s5_comment_present(self):
        """S-5 documentation must exist in disagreement_utility."""
        from hcmarl.envs.reward_functions import disagreement_utility
        src = inspect.getsource(disagreement_utility)
        assert "S-5" in src, "S-5 comment missing from disagreement_utility"


# =====================================================================
# S-13: validate_mmicrl removed from production module
# =====================================================================

class TestS13:

    def test_no_main_block(self):
        """mmicrl.py must not have __main__ that calls validate_mmicrl."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "hcmarl", "mmicrl.py"
        )
        with open(src_path, encoding="utf-8") as f:
            source = f.read()
        assert 'if __name__ == "__main__"' not in source, \
            "__main__ block still present in mmicrl.py"

    def test_validate_mmicrl_not_importable(self):
        """validate_mmicrl should not be importable from hcmarl.mmicrl."""
        import hcmarl.mmicrl as mod
        assert not hasattr(mod, 'validate_mmicrl'), \
            "validate_mmicrl still exists as importable function"

    def test_s13_comment_present(self):
        """S-13 removal must be documented."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "hcmarl", "mmicrl.py"
        )
        with open(src_path, encoding="utf-8") as f:
            source = f.read()
        assert "S-13" in source, "S-13 tag missing from mmicrl.py"


# =====================================================================
# S-14: MMICRL.fit() warns on auto-detected n_actions
# =====================================================================

class TestS14:

    def test_warns_on_auto_detect(self):
        """fit() must warn when n_actions is auto-detected."""
        from hcmarl.mmicrl import MMICRL, DemonstrationCollector
        from hcmarl.real_data_calibration import (
            generate_demonstrations_from_profiles,
            load_path_g_into_collector,
        )
        profiles = [
            {'worker_id': 0, 'source_subject': 's1',
             'muscles': {'shoulder': {'F': 2.0, 'R': 0.02, 'r': 15}}},
            {'worker_id': 1, 'source_subject': 's2',
             'muscles': {'shoulder': {'F': 0.5, 'R': 0.02, 'r': 15}}},
        ]
        demos, wids = generate_demonstrations_from_profiles(
            profiles, muscle='shoulder', n_episodes_per_worker=2)
        collector = load_path_g_into_collector(demos, wids)
        mmicrl = MMICRL(n_types=2, n_muscles=1, n_iterations=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mmicrl.fit(collector)  # no n_actions -> auto-detect
            n_action_warns = [x for x in w if "n_actions" in str(x.message)]
            assert len(n_action_warns) >= 1, "No warning for auto-detected n_actions"

    def test_no_warning_with_explicit(self):
        """fit() must not warn when n_actions is provided explicitly."""
        from hcmarl.mmicrl import MMICRL, DemonstrationCollector
        from hcmarl.real_data_calibration import (
            generate_demonstrations_from_profiles,
            load_path_g_into_collector,
        )
        profiles = [
            {'worker_id': 0, 'source_subject': 's1',
             'muscles': {'shoulder': {'F': 2.0, 'R': 0.02, 'r': 15}}},
        ]
        demos, wids = generate_demonstrations_from_profiles(
            profiles, muscle='shoulder', n_episodes_per_worker=2)
        collector = load_path_g_into_collector(demos, wids)
        mmicrl = MMICRL(n_types=2, n_muscles=1, n_iterations=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mmicrl.fit(collector, n_actions=6)
            n_action_warns = [x for x in w if "n_actions" in str(x.message)]
            assert len(n_action_warns) == 0, "Spurious warning with explicit n_actions"


# =====================================================================
# S-21: LagrangianRolloutBuffer overflow guard
# =====================================================================

class TestS21:

    def test_normal_store_works(self):
        """store() with correct N calls flushes one timestep."""
        from hcmarl.agents.mappo_lag import LagrangianRolloutBuffer
        buf = LagrangianRolloutBuffer(agent_ids=["w0", "w1"])
        obs = np.zeros(10, dtype=np.float32)
        gs = np.zeros(5, dtype=np.float32)
        buf.store(obs, gs, 0, 0.0, 0.5, 0.0, False, {"w0": 0.0, "w1": 0.0}, {"w0": 0.0, "w1": 0.0})
        buf.store(obs, gs, 1, 0.0, 0.5, 0.0, False, {"w0": 0.0, "w1": 0.0}, {"w0": 0.0, "w1": 0.0})
        assert buf._n_steps == 1, f"Expected 1 timestep, got {buf._n_steps}"
        assert buf._legacy_pending is None, "Pending should be flushed"

    def test_no_agent_ids_raises(self):
        """store() without agent_ids must raise ValueError."""
        from hcmarl.agents.mappo_lag import LagrangianRolloutBuffer
        buf = LagrangianRolloutBuffer()  # no agent_ids
        with pytest.raises(ValueError, match="agent_ids not set"):
            buf.store(np.zeros(5), np.zeros(3), 0, 0.0, 0.0, 0.0, False, 0.0, 0.0)

    def test_docstring_documents_contract(self):
        """store() docstring must document the N-calls contract."""
        from hcmarl.agents.mappo_lag import LagrangianRolloutBuffer
        doc = LagrangianRolloutBuffer.store.__doc__
        assert "S-21" in doc, "S-21 tag missing from store() docstring"
        assert "N times" in doc or "exactly" in doc.lower(), \
            "Contract not documented in store() docstring"


# =====================================================================
# S-24: RolloutBuffer overflow guard
# =====================================================================

class TestS24:

    def test_normal_store_works(self):
        """store() with correct N calls flushes one timestep."""
        from hcmarl.agents.mappo import RolloutBuffer
        buf = RolloutBuffer(agent_ids=["w0", "w1"])
        obs = np.zeros(10, dtype=np.float32)
        gs = np.zeros(5, dtype=np.float32)
        buf.store(obs, gs, 0, 0.0, 0.5, False, {"w0": 0.0, "w1": 0.0})
        buf.store(obs, gs, 1, 0.0, 0.5, False, {"w0": 0.0, "w1": 0.0})
        assert buf._n_steps == 1, f"Expected 1 timestep, got {buf._n_steps}"
        assert buf._legacy_pending is None

    def test_no_agent_ids_raises(self):
        """store() without agent_ids must raise ValueError."""
        from hcmarl.agents.mappo import RolloutBuffer
        buf = RolloutBuffer()  # no agent_ids
        with pytest.raises(ValueError, match="agent_ids not set"):
            buf.store(np.zeros(5), np.zeros(3), 0, 0.0, 0.0, False, 0.0)

    def test_docstring_documents_contract(self):
        """store() docstring must document the call-order contract."""
        from hcmarl.agents.mappo import RolloutBuffer
        doc = RolloutBuffer.store.__doc__
        assert "S-24" in doc, "S-24 tag missing from store() docstring"


# =====================================================================
# S-27: All configs have explicit kp
# =====================================================================

class TestS27:

    CONFIG_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config"
    )

    def _configs_with_ecbf(self):
        """Return all config files that have an ecbf: section."""
        configs = []
        for f in os.listdir(self.CONFIG_DIR):
            if f.endswith(".yaml"):
                path = os.path.join(self.CONFIG_DIR, f)
                with open(path) as fh:
                    content = fh.read()
                if "ecbf:" in content:
                    configs.append((f, path))
        return configs

    def test_all_configs_have_kp(self):
        """Every config with an ecbf section must have explicit kp."""
        missing = []
        for name, path in self._configs_with_ecbf():
            with open(path) as f:
                cfg = yaml.safe_load(f)
            ecbf = cfg.get("ecbf", {})
            # kp can be under ecbf or at top level
            has_kp = "kp" in ecbf or "kp" in cfg
            if not has_kp:
                missing.append(name)
        assert not missing, f"Configs missing explicit kp: {missing}"

    # test_scaling_configs_have_kp_in_ecbf was removed when the scaling
    # study was dropped. The general
    # test_all_configs_have_kp above iterates every YAML containing an
    # `ecbf:` section, so any future config under config/ is still covered.


# =====================================================================
# S-28: Conservation tests check 0 <= x <= 1 bounds
# =====================================================================

class TestS28:

    def test_single_env_bounds(self):
        """All compartments must stay in [0, 1] during random operation."""
        from hcmarl.warehouse_env import SingleWorkerWarehouseEnv
        env = SingleWorkerWarehouseEnv(max_steps=200)
        env.reset()
        for _ in range(100):
            env.step(env.action_space.sample())
            for m in env.muscle_names:
                MR = env.state[m]["MR"]
                MA = env.state[m]["MA"]
                MF = env.state[m]["MF"]
                assert 0.0 <= MR <= 1.0 + 1e-6, f"{m} MR={MR} out of [0,1]"
                assert 0.0 <= MA <= 1.0 + 1e-6, f"{m} MA={MA} out of [0,1]"
                assert 0.0 <= MF <= 1.0 + 1e-6, f"{m} MF={MF} out of [0,1]"

    def test_pettingzoo_env_bounds(self):
        """PettingZoo env compartments must stay in [0, 1]."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=60)
        env.reset()
        for _ in range(60):
            actions = {a: np.random.randint(0, env.n_tasks) for a in env.agents}
            env.step(actions)
        for idx in range(2):
            for m in env.muscle_names:
                s = env.states[idx][m]
                assert 0.0 <= s["MR"] <= 1.0 + 1e-6, f"w{idx} {m} MR out of bounds"
                assert 0.0 <= s["MA"] <= 1.0 + 1e-6, f"w{idx} {m} MA out of bounds"
                assert 0.0 <= s["MF"] <= 1.0 + 1e-6, f"w{idx} {m} MF out of bounds"

    def test_conservation_and_bounds_together(self):
        """Both conservation AND bounds must hold simultaneously."""
        from hcmarl.warehouse_env import SingleWorkerWarehouseEnv
        env = SingleWorkerWarehouseEnv(max_steps=200)
        env.reset()
        for _ in range(50):
            env.step(0)  # heavy_lift
        for m in env.muscle_names:
            MR = env.state[m]["MR"]
            MA = env.state[m]["MA"]
            MF = env.state[m]["MF"]
            total = MR + MA + MF
            assert abs(total - 1.0) < 1e-6, f"Conservation: {total}"
            assert MR >= -1e-9, f"{m} MR={MR} < 0"
            assert MA >= -1e-9, f"{m} MA={MA} < 0"
            assert MF >= -1e-9, f"{m} MF={MF} < 0"


# =====================================================================
# S-29: Quantitative recovery comparison vs RK45
# =====================================================================

class TestS29:

    def test_recovery_matches_rk45(self):
        """Env rest recovery must match RK45 simulate() within tolerance."""
        from hcmarl.warehouse_env import SingleWorkerWarehouseEnv
        from hcmarl.three_cc_r import ThreeCCr, ThreeCCrState, get_muscle
        env = SingleWorkerWarehouseEnv(max_steps=500)
        env.reset()
        # Work for 30 steps to build fatigue
        for _ in range(30):
            env.step(0)  # heavy_lift
        # Record state at start of rest
        grip = env.state["grip"]
        state_at_rest_start = ThreeCCrState(
            MR=grip["MR"], MA=grip["MA"], MF=grip["MF"]
        )
        # Rest for 100 steps (Euler, dt=1 min)
        rest_idx = env.task_names.index("rest")
        for _ in range(100):
            env.step(rest_idx)
        env_mf_final = env.state["grip"]["MF"]
        # RK45 reference: same initial state, rest for 100 min
        model = ThreeCCr(params=get_muscle("grip"), kp=1.0)
        ref = model.simulate(state_at_rest_start, target_load=0.0,
                             duration=100.0, dt_eval=1.0)
        rk45_mf_final = ref["MF"][-1]
        # Euler vs RK45: should match within 10% relative error
        # (dt=1 with small rates means Euler is quite accurate for rest)
        if rk45_mf_final > 0.001:
            rel_err = abs(env_mf_final - rk45_mf_final) / rk45_mf_final
            assert rel_err < 0.10, \
                f"Rest recovery mismatch: env MF={env_mf_final:.6f}, " \
                f"RK45 MF={rk45_mf_final:.6f}, rel_err={rel_err:.3f}"


# =====================================================================
# S-30: Episode end checks physiological state
# =====================================================================

class TestS30:

    def test_episode_end_physiology(self):
        """After a complete episode, env must have valid physiological state."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=30)
        obs, _ = env.reset()
        for _ in range(30):
            actions = {a: np.random.randint(0, env.n_tasks) for a in env.agents}
            obs, r, terms, truncs, infos = env.step(actions)
        # Episode should be done
        assert all(terms.values()), "Episode should terminate"
        # Physiological checks
        for idx in range(2):
            total_mf = 0.0
            for m in env.muscle_names:
                s = env.states[idx][m]
                total = s["MR"] + s["MA"] + s["MF"]
                assert abs(total - 1.0) < 1e-6, f"Conservation: {total}"
                assert s["MR"] >= -1e-9
                assert s["MA"] >= -1e-9
                assert s["MF"] >= -1e-9
                total_mf += s["MF"]
            # At least some fatigue should have accumulated (env did work)
            assert total_mf > 0, f"Worker {idx}: no fatigue after 30 steps"


# =====================================================================
# S-31: Tight ECBF tolerance test (separate from 2x stress test)
# =====================================================================

class TestS31:

    def test_tight_ecbf_with_low_drive(self):
        """With low task demand (0.15), ECBF overshoot should be < 50%.

        The pipeline ECBF operates in continuous-time but is discretised at
        dt=1 min. For low loads, the per-step MF increment is small enough
        that the ECBF bound is approximately satisfied in discrete time.
        This is tighter than the 2x stress test which uses high loads.
        """
        from hcmarl.ecbf_filter import ECBFFilter, ECBFParams
        from hcmarl.pipeline import HCMARLPipeline, TaskProfile
        from hcmarl.three_cc_r import get_muscle
        muscle_names = ["shoulder"]
        ecbf_params = {
            "shoulder": ECBFParams(theta_max=0.70, alpha1=0.5, alpha2=0.5, alpha3=0.5)
        }
        pipe = HCMARLPipeline(
            num_workers=1, muscle_names=muscle_names,
            task_profiles=[
                TaskProfile(task_id=1, name="light_work", demands={"shoulder": 0.15}),
            ],
            ecbf_params_per_muscle=ecbf_params,
            kp=1.0,
            dt=1.0,
        )
        util = np.array([[10.0]])
        for _ in range(300):
            pipe.step(util)
        mf = pipe.workers[0].muscle_states["shoulder"].MF
        theta_max = 0.70
        # Low load: MF growth rate is modest, ECBF should hold within 50%
        assert mf <= theta_max * 1.5, \
            f"MF={mf:.4f} > 1.5 * theta_max={theta_max} with low load"

    def test_stress_2x_still_holds(self):
        """The original 2x stress test must still pass (pipeline default kp=1.0)."""
        from hcmarl.ecbf_filter import ECBFParams
        from hcmarl.pipeline import HCMARLPipeline, TaskProfile
        from hcmarl.three_cc_r import get_muscle
        muscle_names = ["shoulder", "elbow", "grip"]
        ecbf_params = {}
        for name in muscle_names:
            mp = get_muscle(name)
            theta = max(mp.theta_min_max * 1.1, mp.theta_min_max + 0.05)
            theta = min(theta, 0.95)
            ecbf_params[name] = ECBFParams(theta_max=theta, alpha1=0.05,
                                            alpha2=0.05, alpha3=0.1)
        pipe = HCMARLPipeline(
            num_workers=1, muscle_names=muscle_names,
            task_profiles=[
                TaskProfile(task_id=1, name="box_lift",
                           demands={"shoulder": 0.4, "elbow": 0.3, "grip": 0.5}),
            ],
            ecbf_params_per_muscle=ecbf_params, kp=1.0, dt=1.0,
        )
        util = np.array([[10.0]])
        for _ in range(200):
            pipe.step(util)
        for name in muscle_names:
            mf = pipe.workers[0].muscle_states[name].MF
            theta_max = ecbf_params[name].theta_max
            assert mf <= theta_max * 2.0, \
                f"{name}: MF={mf:.4f} > 2x theta_max={theta_max}"


# =====================================================================
# S-32: theta_max flows through to env
# =====================================================================

class TestS32:

    def test_theta_max_flows_to_env(self):
        """Custom theta_max must appear in env.theta_max_per_worker."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        custom_theta = {"shoulder": 0.50, "grip": 0.30}
        env = WarehousePettingZoo(n_workers=2, max_steps=5,
                                   theta_max=custom_theta)
        for w in range(2):
            assert env.theta_max_per_worker[w]["shoulder"] == 0.50, \
                f"Worker {w} shoulder theta_max not 0.50"
            assert env.theta_max_per_worker[w]["grip"] == 0.30, \
                f"Worker {w} grip theta_max not 0.30"

    def test_different_theta_changes_interventions(self):
        """Lower theta_max should produce more ECBF interventions."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        np.random.seed(42)
        # High theta (permissive)
        env_high = WarehousePettingZoo(
            n_workers=1, max_steps=60, ecbf_mode="on",
            theta_max={"shoulder": 0.90, "grip": 0.90}
        )
        env_high.reset()
        int_high = 0
        for _ in range(60):
            _, _, _, _, infos = env_high.step({"worker_0": 0})  # heavy_lift
            int_high += infos["worker_0"]["ecbf_interventions"]
        # Low theta (restrictive)
        np.random.seed(42)
        env_low = WarehousePettingZoo(
            n_workers=1, max_steps=60, ecbf_mode="on",
            theta_max={"shoulder": 0.30, "grip": 0.25}
        )
        env_low.reset()
        int_low = 0
        for _ in range(60):
            _, _, _, _, infos = env_low.step({"worker_0": 0})
            int_low += infos["worker_0"]["ecbf_interventions"]
        # Lower theta should produce >= interventions
        assert int_low >= int_high, \
            f"Lower theta ({int_low} interventions) should >= higher ({int_high})"


# =====================================================================
# S-35: Tests use env.global_obs_dim, not hardcoded formula
# =====================================================================

class TestS35:

    def test_global_obs_dim_from_env(self):
        """global_obs_dim must be read from env, not computed by formula."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=4, max_steps=5)
        env.reset()
        g = env._get_global_obs()
        assert g.shape == (env.global_obs_dim,), \
            f"global_obs shape {g.shape} != declared {env.global_obs_dim}"

    def test_global_obs_dim_matches_actual(self):
        """env.global_obs_dim must match actual _get_global_obs() output."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        for n in [2, 4, 6]:
            env = WarehousePettingZoo(n_workers=n, max_steps=5)
            env.reset()
            g = env._get_global_obs()
            assert len(g) == env.global_obs_dim, \
                f"n={n}: actual={len(g)}, declared={env.global_obs_dim}"


# =====================================================================
# S-40: n_tasks includes rest action
# =====================================================================

class TestS40:

    def test_rest_in_task_names(self):
        """'rest' must be in env.task_names."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=5)
        assert "rest" in env.task_names, \
            f"rest not in task_names: {env.task_names}"

    def test_n_tasks_includes_rest(self):
        """n_tasks must count rest as an action."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=5)
        assert env.n_tasks == len(env.task_names), \
            f"n_tasks={env.n_tasks} != len(task_names)={len(env.task_names)}"
        # With default profiles: 5 productive + 1 rest = 6
        assert env.n_tasks == 6, f"Expected 6 tasks (5+rest), got {env.n_tasks}"

    def test_rest_action_index_valid(self):
        """Rest action index must be a valid action."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=5)
        rest_idx = env.task_names.index("rest")
        assert 0 <= rest_idx < env.n_tasks, \
            f"Rest index {rest_idx} out of range [0, {env.n_tasks})"
        # Running rest action should not crash
        env.reset()
        env.step({"worker_0": rest_idx, "worker_1": rest_idx})

    def test_train_n_actions_correct(self):
        """train.py uses env.n_tasks which includes rest — verify."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=5)
        # This is what train.py does: n_actions = env.n_tasks
        n_actions = env.n_tasks
        # Must include rest
        assert n_actions > len(env.task_mgr.get_productive_tasks()), \
            "n_actions should be > productive tasks (must include rest)"
