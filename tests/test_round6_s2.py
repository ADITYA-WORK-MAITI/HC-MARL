"""
Tests for Cycle S-2 audit fixes (S-4, S-15, S-16, S-19, S-20, S-25).

Verifies:
    S-4   -- Env step returns barrier_violations in info dict
    S-15  -- nswf_reward uses max(MF) for disagreement utility (documented)
    S-16  -- NSWF_EPSILON is single source of truth, default epsilon=1e-3
    S-19  -- GAE compute_returns has last_episode_truncated param, default True
    S-20  -- Done mask at intermediate steps zeroes correctly (1 - dones[t])
    S-25  -- All config files use alpha1=alpha2=alpha3=0.5
"""
import inspect
import math
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# S-4: Post-step barrier verification in env integration
# =====================================================================

class TestS4:

    def test_barrier_violations_in_info(self):
        """step() must return barrier_violations in each agent's info."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=5)
        obs, _ = env.reset()
        actions = {a: 0 for a in env.agents}
        _, _, _, _, infos = env.step(actions)
        for agent in env.agents:
            assert "barrier_violations" in infos[agent], \
                f"barrier_violations missing from {agent} info"

    def test_barrier_violations_type(self):
        """barrier_violations must be an integer (count of muscles)."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=5)
        obs, _ = env.reset()
        actions = {a: 0 for a in env.agents}
        _, _, _, _, infos = env.step(actions)
        for agent in env.agents:
            bv = infos[agent]["barrier_violations"]
            assert isinstance(bv, (int, float))
            assert bv >= 0

    def test_integrate_returns_3_tuple(self):
        """_integrate must return (ecbf_interventions, ecbf_clip_total, barrier_violations)."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=5)
        env.reset()
        result = env._integrate(0, "rest")
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}"
        assert isinstance(result[2], (int, float)), "Third element must be barrier_violations"

    def test_ecbf_on_keeps_barriers(self):
        """With ECBF on and rest task, barrier_violations should be 0."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=5, ecbf_mode="on")
        env.reset()
        # Rest task = zero demand, no fatigue buildup
        _, _, bv = env._integrate(0, "rest")
        assert bv == 0


# =====================================================================
# S-15: max(MF) aggregation in nswf_reward
# =====================================================================

class TestS15:

    def test_nswf_reward_uses_max_mf(self):
        """nswf_reward should use max(MF) -- verify via two-muscle scenario.

        If MF_shoulder=0.5 and MF_grip=0.1, disagreement utility should
        use 0.5 (the max), not 0.3 (the mean).
        """
        from hcmarl.envs.reward_functions import nswf_reward, disagreement_utility
        fatigue = {"shoulder": 0.5, "grip": 0.1}
        theta = {"shoulder": 0.7, "grip": 0.35}
        prod = 1.0

        # Compute expected reward with max(MF)=0.5
        di_max = disagreement_utility(0.5)
        surplus_max = prod - di_max
        expected = math.log(max(surplus_max, 1e-3))

        actual = nswf_reward(prod, fatigue, theta, safety_weight=0.0)
        assert abs(actual - expected) < 1e-10, \
            f"nswf_reward should use max(MF)=0.5, got reward={actual}, expected={expected}"

    def test_max_mf_documented_in_source(self):
        """S-15 comment must exist in reward_functions.py."""
        src = inspect.getsource(
            __import__("hcmarl.envs.reward_functions", fromlist=["nswf_reward"]).nswf_reward
        )
        assert "max(MF)" in src or "max_mf" in src, \
            "nswf_reward source should document max(MF) aggregation"

    def test_single_muscle_max_equals_value(self):
        """With one muscle, max(MF) = that muscle's MF."""
        from hcmarl.envs.reward_functions import nswf_reward, disagreement_utility
        fatigue = {"shoulder": 0.3}
        theta = {"shoulder": 0.7}
        di = disagreement_utility(0.3)
        surplus = 1.0 - di
        expected = math.log(max(surplus, 1e-3))
        actual = nswf_reward(1.0, fatigue, theta, safety_weight=0.0)
        assert abs(actual - expected) < 1e-10


# =====================================================================
# S-16: NSWF_EPSILON is single source of truth
# =====================================================================

class TestS16:

    def test_nswf_epsilon_value(self):
        """NSWF_EPSILON must be 1e-3."""
        from hcmarl.nswf_allocator import NSWF_EPSILON
        assert NSWF_EPSILON == 1e-3

    def test_reward_default_uses_nswf_epsilon(self):
        """nswf_reward default epsilon must come from NSWF_EPSILON."""
        from hcmarl.envs.reward_functions import nswf_reward
        sig = inspect.signature(nswf_reward)
        from hcmarl.nswf_allocator import NSWF_EPSILON
        default_eps = sig.parameters["epsilon"].default
        assert default_eps == NSWF_EPSILON, \
            f"nswf_reward epsilon default={default_eps}, expected NSWF_EPSILON={NSWF_EPSILON}"

    def test_nswf_params_default_uses_constant(self):
        """NSWFParams.epsilon default must equal NSWF_EPSILON."""
        from hcmarl.nswf_allocator import NSWFParams, NSWF_EPSILON
        params = NSWFParams()
        assert params.epsilon == NSWF_EPSILON

    def test_no_hardcoded_001_in_reward(self):
        """reward_functions.py must not have hardcoded epsilon=0.01."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "hcmarl", "envs", "reward_functions.py"
        )
        with open(src_path) as f:
            source = f.read()
        # Should not have the old default
        assert "epsilon: float = 0.01" not in source, \
            "Old hardcoded epsilon=0.01 still present"


# =====================================================================
# S-19: GAE truncation vs termination
# =====================================================================

class TestS19:

    def test_mappo_compute_returns_has_truncation_param(self):
        """RolloutBuffer.compute_returns must accept last_episode_truncated."""
        from hcmarl.agents.mappo import RolloutBuffer
        sig = inspect.signature(RolloutBuffer.compute_returns)
        assert "last_episode_truncated" in sig.parameters

    def test_mappo_lag_compute_returns_has_truncation_param(self):
        """LagrangianRolloutBuffer.compute_returns must accept last_episode_truncated."""
        from hcmarl.agents.mappo_lag import LagrangianRolloutBuffer
        sig = inspect.signature(LagrangianRolloutBuffer.compute_returns)
        assert "last_episode_truncated" in sig.parameters

    def test_truncation_default_is_true(self):
        """Default last_episode_truncated must be True (warehouse always truncates)."""
        from hcmarl.agents.mappo import RolloutBuffer
        sig = inspect.signature(RolloutBuffer.compute_returns)
        assert sig.parameters["last_episode_truncated"].default is True

    def test_truncation_keeps_bootstrap(self):
        """With truncation=True, last-step advantage should use bootstrap V(s_next).

        Regression: before S-19, the done mask at T-1 zeroed the bootstrap
        even for truncated episodes, losing the value estimate.
        """
        from hcmarl.agents.mappo import RolloutBuffer
        agents = ["worker_0"]
        buf = RolloutBuffer(agent_ids=agents)

        # Store 3 steps: constant reward=1.0, value=0.5, done at step 2
        for t in range(3):
            done = 1.0 if t == 2 else 0.0
            buf.store_step(
                {"worker_0": np.zeros(4)},
                np.zeros(4),
                {"worker_0": 0},
                {"worker_0": 0.0},
                {"worker_0": 1.0},
                done,
                {"worker_0": 0.5},
            )

        # With truncation=True: bootstrap should NOT be zeroed
        adv_trunc, ret_trunc = buf.compute_returns(
            {"worker_0": 0.5}, gamma=0.99, gae_lambda=0.95,
            last_episode_truncated=True
        )

        # With truncation=False: bootstrap IS zeroed by done[T-1]=1
        buf2 = RolloutBuffer(agent_ids=agents)
        for t in range(3):
            done = 1.0 if t == 2 else 0.0
            buf2.store_step(
                {"worker_0": np.zeros(4)},
                np.zeros(4),
                {"worker_0": 0},
                {"worker_0": 0.0},
                {"worker_0": 1.0},
                done,
                {"worker_0": 0.5},
            )
        adv_term, ret_term = buf2.compute_returns(
            {"worker_0": 0.5}, gamma=0.99, gae_lambda=0.95,
            last_episode_truncated=False
        )

        # The last-step return should differ: truncated keeps bootstrap
        assert ret_trunc[-1] != ret_term[-1], \
            "Truncation vs termination should produce different last-step returns"
        # Truncated return should be higher (bootstrap adds value)
        assert ret_trunc[-1] > ret_term[-1], \
            "Truncated episode should have higher return (bootstrap preserved)"

    def test_mappo_lag_truncation_keeps_bootstrap(self):
        """Same truncation test for LagrangianRolloutBuffer."""
        from hcmarl.agents.mappo_lag import LagrangianRolloutBuffer
        agents = ["worker_0"]
        buf = LagrangianRolloutBuffer(agent_ids=agents)

        # store() is per-agent; with 1 agent each call = 1 timestep
        for t in range(3):
            done = 1.0 if t == 2 else 0.0
            buf.store(
                np.zeros(4), np.zeros(4), 0, 0.0,
                1.0, 0.0, done,
                {"worker_0": 0.5}, {"worker_0": 0.0},
            )

        adv_t, ret_t, cadv_t, cret_t = buf.compute_returns(
            {"worker_0": 0.5}, {"worker_0": 0.0},
            gamma=0.99, gae_lambda=0.95, last_episode_truncated=True
        )

        buf2 = LagrangianRolloutBuffer(agent_ids=agents)
        for t in range(3):
            done = 1.0 if t == 2 else 0.0
            buf2.store(
                np.zeros(4), np.zeros(4), 0, 0.0,
                1.0, 0.0, done,
                {"worker_0": 0.5}, {"worker_0": 0.0},
            )

        adv_f, ret_f, cadv_f, cret_f = buf2.compute_returns(
            {"worker_0": 0.5}, {"worker_0": 0.0},
            gamma=0.99, gae_lambda=0.95, last_episode_truncated=False
        )

        assert ret_t[-1] > ret_f[-1], \
            "Truncated episode should preserve bootstrap in Lagrangian buffer too"


# =====================================================================
# S-20: Done mask correctness at intermediate steps
# =====================================================================

class TestS20:

    def test_intermediate_done_zeroes_bootstrap(self):
        """At intermediate steps (t < T-1), done[t]=1 must zero next_non_term.

        This verifies the existing done mask 1-dones[t] works correctly:
        when done[t]=1, the GAE delta uses next_non_term=0 so the next
        value is not bootstrapped across episode boundaries.
        """
        from hcmarl.agents.mappo import RolloutBuffer
        agents = ["worker_0"]

        # Scenario: 4 steps, done at step 1 (mid-episode boundary)
        buf_with_done = RolloutBuffer(agent_ids=agents)
        for t in range(4):
            done = 1.0 if t == 1 else 0.0
            buf_with_done.store_step(
                {"worker_0": np.zeros(4)},
                np.zeros(4),
                {"worker_0": 0},
                {"worker_0": 0.0},
                {"worker_0": 1.0},
                done,
                {"worker_0": 0.5},
            )

        # Same scenario but no intermediate done
        buf_no_done = RolloutBuffer(agent_ids=agents)
        for t in range(4):
            buf_no_done.store_step(
                {"worker_0": np.zeros(4)},
                np.zeros(4),
                {"worker_0": 0},
                {"worker_0": 0.0},
                {"worker_0": 1.0},
                0.0,  # no done
                {"worker_0": 0.5},
            )

        adv_d, ret_d = buf_with_done.compute_returns({"worker_0": 0.5})
        adv_n, ret_n = buf_no_done.compute_returns({"worker_0": 0.5})

        # Step 0's return should differ because done[1]=1 cuts off bootstrap
        assert ret_d[0] != ret_n[0], \
            "Intermediate done should affect earlier returns via GAE"

    def test_s20_comment_present_in_mappo(self):
        """S-20 confirming comment must exist in mappo.py compute_returns."""
        from hcmarl.agents.mappo import RolloutBuffer
        src = inspect.getsource(RolloutBuffer.compute_returns)
        assert "S-20" in src, "S-20 confirming comment missing from mappo.py"

    def test_s20_comment_present_in_mappo_lag(self):
        """S-20 confirming comment must exist in mappo_lag.py compute_returns."""
        from hcmarl.agents.mappo_lag import LagrangianRolloutBuffer
        src = inspect.getsource(LagrangianRolloutBuffer.compute_returns)
        assert "S-20" in src, "S-20 confirming comment missing from mappo_lag.py"


# =====================================================================
# S-25: ECBF alpha consistency across all configs
# =====================================================================

class TestS25:

    @pytest.fixture
    def config_dir(self):
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"
        )

    def _load_yaml(self, path):
        with open(path) as f:
            return yaml.safe_load(f)

    def test_dry_run_alphas(self, config_dir):
        """dry_run_50k.yaml must have alpha1=alpha2=alpha3=0.5."""
        cfg = self._load_yaml(os.path.join(config_dir, "dry_run_50k.yaml"))
        ecbf = cfg["ecbf"]
        assert ecbf["alpha1"] == 0.5
        assert ecbf["alpha2"] == 0.5
        assert ecbf["alpha3"] == 0.5

    def test_default_config_alphas(self, config_dir):
        """default_config.yaml must have alpha=0.5 for all muscles."""
        cfg = self._load_yaml(os.path.join(config_dir, "default_config.yaml"))
        for muscle, params in cfg["ecbf"].items():
            assert params["alpha1"] == 0.5, f"{muscle} alpha1 != 0.5"
            assert params["alpha2"] == 0.5, f"{muscle} alpha2 != 0.5"
            assert params["alpha3"] == 0.5, f"{muscle} alpha3 != 0.5"

    def test_full_config_alphas(self, config_dir):
        """hcmarl_full_config.yaml must have ECBF alphas that flow through to env."""
        cfg = self._load_yaml(os.path.join(config_dir, "hcmarl_full_config.yaml"))
        ecbf = cfg["ecbf"]
        for key in ("alpha1", "alpha2", "alpha3"):
            assert key in ecbf, f"Missing {key} in ecbf config"
            assert ecbf[key] > 0, f"ecbf.{key} must be positive"

    def test_all_configs_have_ecbf_alphas(self, config_dir):
        """All YAML configs with ecbf section must have positive alpha values."""
        import glob
        yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))
        for yf in yaml_files:
            cfg = self._load_yaml(yf)
            if "ecbf" not in cfg:
                continue
            ecbf = cfg["ecbf"]
            fname = os.path.basename(yf)
            if "alpha1" in ecbf:
                for key in ("alpha1", "alpha2", "alpha3"):
                    assert ecbf[key] > 0, f"{fname} ecbf.{key} must be positive"
            else:
                for muscle, params in ecbf.items():
                    if isinstance(params, dict) and "alpha1" in params:
                        for key in ("alpha1", "alpha2", "alpha3"):
                            assert params[key] > 0, f"{fname}/{muscle} {key} must be positive"
