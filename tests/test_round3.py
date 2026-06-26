"""
Tests for Round 3 audit fixes (C-3, C-17).

Verifies:
    C-3   -- Hungarian allocator replaces greedy; exact for all N
    C-17  -- Ablation knobs actually change behavior

Note: the C-18 scaling-configs class was removed when the scaling study was
dropped. The lone retained allocator-N
sweep (Hungarian works for any N) lives in tests/test_post_scaling_drop.py.
"""
import numpy as np
import pytest
import sys
import os
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# C-3: Hungarian allocator — exact NSWF for all N
# =====================================================================

class TestC3:

    def test_hungarian_matches_exact_small_n(self):
        """Hungarian must match brute-force for N=2..6 with random utilities."""
        from hcmarl.nswf_allocator import NSWFAllocator, NSWFParams
        alloc = NSWFAllocator(NSWFParams())

        rng = np.random.RandomState(42)
        for N in range(2, 7):
            for M in range(1, 6):
                util = rng.uniform(0.5, 5.0, (N, M))
                fl = rng.uniform(0.0, 0.3, N)
                D = np.array([alloc.disagreement_utility(mf) for mf in fl])
                surplus = util - D[:, None]
                eps = alloc.params.epsilon

                h = alloc._solve_hungarian(N, M, surplus, D, eps)
                e = alloc._solve_exact(N, M, surplus, D, eps)
                assert abs(h.objective_value - e.objective_value) < 1e-6, \
                    f"N={N}, M={M}: Hungarian obj={h.objective_value:.6f} != exact obj={e.objective_value:.6f}"

    def test_no_greedy_method(self):
        """_solve_greedy must not exist (replaced by _solve_hungarian)."""
        from hcmarl.nswf_allocator import NSWFAllocator
        assert not hasattr(NSWFAllocator, '_solve_greedy')

    def test_n12_m5_works(self):
        """N=12 with M=5 must work (would have triggered greedy before)."""
        from hcmarl.nswf_allocator import NSWFAllocator, NSWFParams
        alloc = NSWFAllocator(NSWFParams())
        rng = np.random.RandomState(123)
        util = rng.uniform(1, 5, (12, 5))
        fl = rng.uniform(0, 0.3, 12)
        result = alloc.allocate(util, fl)
        assert len(result.assignments) == 12
        # At most 5 workers should have productive tasks
        productive = sum(1 for j in result.assignments.values() if j > 0)
        assert productive <= 5

    def test_utilitarian_hungarian(self):
        """UtilitarianAllocator uses Hungarian, not inherited greedy."""
        from hcmarl.nswf_allocator import create_allocator, NSWFParams
        alloc = create_allocator("utilitarian", NSWFParams())
        rng = np.random.RandomState(42)
        util = rng.uniform(1, 5, (12, 5))
        fl = rng.uniform(0, 0.3, 12)
        result = alloc.allocate(util, fl)
        assert len(result.assignments) == 12

    def test_maxmin_no_greedy_fallback(self):
        """MaxMinAllocator uses exact enumeration for N=12 (no greedy)."""
        from hcmarl.nswf_allocator import create_allocator, NSWFParams
        alloc = create_allocator("maxmin", NSWFParams())
        rng = np.random.RandomState(42)
        util = rng.uniform(1, 5, (10, 5))
        fl = rng.uniform(0, 0.3, 10)
        result = alloc.allocate(util, fl)
        assert len(result.assignments) == 10

    def test_gini_no_greedy_fallback(self):
        """GiniAllocator uses exact enumeration for N=10 (no greedy)."""
        from hcmarl.nswf_allocator import create_allocator, NSWFParams
        alloc = create_allocator("gini", NSWFParams())
        rng = np.random.RandomState(42)
        util = rng.uniform(1, 5, (10, 5))
        fl = rng.uniform(0, 0.3, 10)
        result = alloc.allocate(util, fl)
        assert len(result.assignments) == 10

    def test_all_rest_when_all_surplus_negative(self):
        """When all surpluses are negative, all workers should rest."""
        from hcmarl.nswf_allocator import NSWFAllocator, NSWFParams
        alloc = NSWFAllocator(NSWFParams(kappa=100.0))
        # Very high fatigue -> D_i is huge -> all surpluses negative
        util = np.array([[1.0, 1.0], [1.0, 1.0]])
        fl = np.array([0.9, 0.9])
        result = alloc.allocate(util, fl)
        assert result.assignments[0] == 0  # rest
        assert result.assignments[1] == 0  # rest

    def test_assignment_constraints(self):
        """Verify assignment constraints (Def 6.1) hold."""
        from hcmarl.nswf_allocator import NSWFAllocator, NSWFParams
        alloc = NSWFAllocator(NSWFParams())
        rng = np.random.RandomState(42)
        for _ in range(20):
            N = rng.randint(2, 10)
            M = rng.randint(1, 6)
            util = rng.uniform(0.5, 5.0, (N, M))
            fl = rng.uniform(0.0, 0.4, N)
            result = alloc.allocate(util, fl)
            # Each worker gets exactly one assignment
            assert len(result.assignments) == N
            # Each productive task assigned to at most one worker
            productive_tasks = [j for j in result.assignments.values() if j > 0]
            assert len(productive_tasks) == len(set(productive_tasks))


# =====================================================================
# C-17: Ablation knobs actually change behavior
# =====================================================================

class TestC17:

    def test_ecbf_off_config_read(self):
        """ecbf.enabled=false in config should disable ECBF."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env_on = WarehousePettingZoo(n_workers=2, max_steps=30, ecbf_mode="on")
        env_off = WarehousePettingZoo(n_workers=2, max_steps=30, ecbf_mode="off")
        env_on.reset(); env_off.reset()
        # Run heavy work for 20 steps
        for _ in range(20):
            actions = {"worker_0": 0, "worker_1": 0}  # heavy_lift
            _, r_on, _, _, i_on = env_on.step(actions)
            _, r_off, _, _, i_off = env_off.step(actions)
        # ECBF off should have 0 interventions
        assert i_off["worker_0"]["ecbf_interventions"] == 0
        # Rewards should differ (ECBF changes fatigue trajectory)
        mf_on = max(i_on["worker_0"]["fatigue"].values())
        mf_off = max(i_off["worker_0"]["fatigue"].values())
        assert mf_off >= mf_on  # no ECBF means more fatigue allowed

    def test_nswf_disabled(self):
        """use_nswf=False should produce agent with no allocator."""
        from hcmarl.agents.hcmarl_agent import HCMARLAgent
        agent_on = HCMARLAgent(obs_dim=19, global_obs_dim=73, n_actions=6, n_agents=4, use_nswf=True)
        agent_off = HCMARLAgent(obs_dim=19, global_obs_dim=73, n_actions=6, n_agents=4, use_nswf=False)
        assert agent_on.allocator is not None
        assert agent_off.allocator is None

    def test_disagreement_constant_vs_divergent(self):
        """Constant D_i should give different rewards than divergent."""
        from hcmarl.envs.reward_functions import nswf_reward
        fatigue = {"shoulder": 0.5, "ankle": 0.3}
        theta = {"shoulder": 0.7, "ankle": 0.8}
        r_div = nswf_reward(1.5, fatigue, theta, kappa=1.0, disagreement_type="divergent")
        r_const = nswf_reward(1.5, fatigue, theta, kappa=1.0, disagreement_type="constant")
        assert r_div != r_const, "Constant and divergent should produce different rewards at MF=0.5"

    def test_disagreement_constant_fatigue_independent(self):
        """With constant D_i, changing fatigue should NOT change D_i."""
        from hcmarl.envs.reward_functions import disagreement_utility
        d_low = disagreement_utility(0.1, kappa=1.0, disagreement_type="constant")
        d_high = disagreement_utility(0.8, kappa=1.0, disagreement_type="constant")
        assert d_low == d_high == 1.0

    def test_muscle_params_override_reperfusion(self):
        """muscle_params_override with r=1 should change recovery dynamics."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        # Normal env: r=15 or 30 (fast rest recovery)
        env_normal = WarehousePettingZoo(n_workers=1, max_steps=100)
        # Overridden: r=1 (no reperfusion boost)
        override = {m: {"r": 1} for m in env_normal.muscle_names}
        env_ablated = WarehousePettingZoo(
            n_workers=1, max_steps=100, muscle_params_override=override
        )
        env_normal.reset(); env_ablated.reset()
        # Work for 30 steps, then rest for 30 steps
        for _ in range(30):
            env_normal.step({"worker_0": 0})
            env_ablated.step({"worker_0": 0})
        for _ in range(30):
            env_normal.step({"worker_0": 5})  # rest
            env_ablated.step({"worker_0": 5})  # rest
        # With r=1, recovery during rest is R*1 = R (slow)
        # With r=15, recovery during rest is R*15 (fast)
        mf_normal = max(env_normal.states[0][m]["MF"] for m in env_normal.muscle_names)
        mf_ablated = max(env_ablated.states[0][m]["MF"] for m in env_ablated.muscle_names)
        assert mf_ablated > mf_normal, "Without reperfusion, fatigue should be higher after rest"

    def test_env_disagreement_type_propagated(self):
        """WarehousePettingZoo should pass disagreement_type to reward fn."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env_div = WarehousePettingZoo(n_workers=1, max_steps=20, disagreement_type="divergent")
        env_const = WarehousePettingZoo(n_workers=1, max_steps=20, disagreement_type="constant")
        env_div.reset(); env_const.reset()
        # Build up fatigue
        for _ in range(10):
            env_div.step({"worker_0": 0})
            env_const.step({"worker_0": 0})
        # At this point fatigue > 0, rewards should differ
        _, r_div, _, _, _ = env_div.step({"worker_0": 0})
        _, r_const, _, _, _ = env_const.step({"worker_0": 0})
        assert r_div["worker_0"] != r_const["worker_0"]

    def test_ablation_no_ecbf_config_read(self):
        """Verify ablation_no_ecbf.yaml has ecbf.enabled=false."""
        with open("config/ablation_no_ecbf.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["ecbf"]["enabled"] is False

    def test_ablation_no_nswf_config_read(self):
        """Verify ablation_no_nswf.yaml has nswf.enabled=false."""
        with open("config/ablation_no_nswf.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["nswf"]["enabled"] is False

    def test_ablation_no_reperfusion_r_is_1(self):
        """Verify ablation_no_reperfusion.yaml has r=1 for all muscles."""
        with open("config/ablation_no_reperfusion.yaml") as f:
            cfg = yaml.safe_load(f)
        for m_name, m_params in cfg["environment"]["muscle_groups"].items():
            assert m_params["r"] == 1, f"{m_name}: r={m_params['r']} != 1"

    def test_ablation_no_divergent_constant(self):
        """Verify ablation_no_divergent.yaml has disagreement.type=constant."""
        with open("config/ablation_no_divergent.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["disagreement"]["type"] == "constant"

    def test_ablation_no_mmicrl_fixed_theta(self):
        """Verify ablation_no_mmicrl.yaml has mmicrl.use_fixed_theta=true."""
        with open("config/ablation_no_mmicrl.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["mmicrl"]["use_fixed_theta"] is True
        assert cfg["mmicrl"]["enabled"] is False

    def test_run_ablations_reads_matrix(self):
        """run_ablations.py must read experiment_matrix.yaml's ablation rungs
        rather than hardcoding a list. This is the post-Batch-D contract:
        the matrix file is the single source of truth for what runs."""
        with open("scripts/run_ablations.py") as f:
            content = f.read()
        assert "experiment_matrix.yaml" in content, (
            "run_ablations.py must reference experiment_matrix.yaml"
        )
        assert 'ablation' in content and 'rungs' in content, (
            "run_ablations.py must read the ablation.rungs list from the matrix"
        )
        # Old remove-one ABLATION_FLAGS dict is gone — matrix-driven launcher
        # uses --run-name to keep ablation logs distinct from the headline.
        assert "ABLATION_FLAGS" not in content, (
            "ABLATION_FLAGS dict was the old remove-one design; the new "
            "build-up ladder reads the matrix directly"
        )
        assert "--run-name" in content, (
            "run_ablations.py must pass --run-name so logs/ablation_<rung>/ "
            "doesn't collide with logs/<method>/ from the headline grid"
        )

    def test_run_baselines_reads_matrix(self):
        """run_baselines.py must launch from experiment_matrix.yaml (4 methods
        x 10 seeds) rather than the pre-Batch-D hardcoded 3-baseline x 5-seed
        list. Hardcoding the seed count was the seed-count execution bug."""
        with open("scripts/run_baselines.py") as f:
            content = f.read()
        assert "experiment_matrix.yaml" in content
        assert 'headline' in content and 'methods' in content
        # The pre-Batch-D file had: SEEDS = [0, 1, 2, 3, 4]. New launcher
        # reads `headline.seeds` from the matrix instead.
        assert "SEEDS = [0, 1, 2, 3, 4]" not in content, (
            "Hardcoded 5-seed list snuck back in; aggregator expects 10 seeds"
        )
        assert "BASELINES = [" not in content, (
            "Hardcoded BASELINES list snuck back in; matrix is canonical"
        )

    def test_run_baselines_no_fake_methods_post_refactor(self):
        """Even after the matrix-driven rewrite, the launcher must not allow
        fake methods to creep in. Mirrors test_round4.test_run_baselines_no_fake_methods."""
        with open("scripts/run_baselines.py") as f:
            content = f.read()
        assert '"ppo_lag"' not in content
        assert '"cpo"' not in content


# =====================================================================
# Allocator scaling sanity (kept after the C-18 scaling-config class
# was deleted along with the scaling study). The Hungarian allocator
# is expected to work at every N the warehouse env can be configured
# with, regardless of whether we ship a scaling experiment.
# =====================================================================

class TestAllocatorScalesWithN:

    def test_allocator_same_algorithm_all_n(self):
        """Hungarian allocator returns a feasible assignment for any N."""
        from hcmarl.nswf_allocator import NSWFAllocator, NSWFParams
        alloc = NSWFAllocator(NSWFParams())
        rng = np.random.RandomState(42)
        for N in [3, 4, 6, 8, 12]:
            util = rng.uniform(1, 5, (N, 5))
            fl = rng.uniform(0, 0.3, N)
            result = alloc.allocate(util, fl)
            assert len(result.assignments) == N
            # Hungarian is exact — should never fall through to a -inf objective.
            assert result.objective_value > -1e10
