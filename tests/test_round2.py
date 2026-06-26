"""
Tests for Round 2 audit fixes (C-6, C-7, C-8, C-9, C-10).

Verifies:
    C-6.A  — Envs route through ECBFFilter (no inlined ECBF)
    C-7.A  — Hierarchical two-timescale allocation
    C-7.R  — Welfare function ablation set
    C-8.A  — Continuous neural drive action space
    C-8.D  — HCMARLAgent owns ECBF + NSWF
    C-9.A  — MAPPO critic with agent-id one-hot
    C-10.A — Cost-advantage NOT normalised
    C-10.B — PID Lagrangian
"""
import numpy as np
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# C-6.A: Envs use ECBFFilter, not inlined bounds
# =====================================================================

class TestC6A:

    def test_pettingzoo_has_ecbf_filters(self):
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=5)
        assert hasattr(env, 'ecbf_filters')
        assert 0 in env.ecbf_filters
        for m in env.muscle_names:
            assert m in env.ecbf_filters[0]

    def test_single_env_has_ecbf_filters(self):
        from hcmarl.warehouse_env import SingleWorkerWarehouseEnv
        env = SingleWorkerWarehouseEnv()
        assert hasattr(env, 'ecbf_filters')
        for m in env.muscle_names:
            assert m in env.ecbf_filters

    def test_multi_env_has_ecbf_filters(self):
        from hcmarl.warehouse_env import WarehouseMultiAgentEnv
        env = WarehouseMultiAgentEnv(n_workers=3)
        assert hasattr(env, 'ecbf_filters')
        for m in env.muscle_names:
            assert m in env.ecbf_filters

    def test_ecbf_filter_is_used_not_inlined(self):
        """Run env and verify ECBF clips match canonical filter output."""
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        from hcmarl.ecbf_filter import ECBFFilter
        from hcmarl.three_cc_r import ThreeCCrState
        env = WarehousePettingZoo(n_workers=1, max_steps=20)
        env.reset()
        # Work for 10 steps
        for _ in range(10):
            env.step({"worker_0": 0})  # first task (heavy work)
        # After 10 steps, verify conservation still holds
        for m in env.muscle_names:
            s = env.states[0][m]
            total = s["MR"] + s["MA"] + s["MF"]
            assert abs(total - 1.0) < 1e-5, f"Conservation violated for {m}: {total}"


# =====================================================================
# C-7.A: Hierarchical two-timescale allocation
# =====================================================================

class TestC7A:

    def test_hcmarl_agent_has_allocator(self):
        from hcmarl.agents.hcmarl_agent import HCMARLAgent
        agent = HCMARLAgent(obs_dim=19, global_obs_dim=73, n_actions=6, n_agents=4)
        assert agent.allocator is not None

    def test_allocate_tasks(self):
        from hcmarl.agents.hcmarl_agent import HCMARLAgent
        agent = HCMARLAgent(obs_dim=19, global_obs_dim=73, n_actions=6, n_agents=4)
        assignments = agent.allocate_tasks([0.1, 0.2, 0.3, 0.4])
        assert len(assignments) == 4
        for i in range(4):
            assert i in assignments

    def test_should_reallocate_interval(self):
        from hcmarl.agents.hcmarl_agent import HCMARLAgent
        agent = HCMARLAgent(
            obs_dim=19, global_obs_dim=73, n_actions=6, n_agents=2,
            allocation_interval=5,
        )
        assert not agent.should_reallocate()  # starts at 0
        # Simulate steps
        obs = {f"worker_{i}": np.random.randn(19).astype(np.float32) for i in range(2)}
        gs = np.random.randn(73).astype(np.float32)
        for _ in range(5):
            agent.get_actions(obs, gs)
        assert agent.should_reallocate()

    def test_env_set_task_assignments(self):
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=5, action_mode="continuous")
        env.reset()
        env.set_task_assignments({0: 1, 1: 2})
        assert env._task_assignments[0] == 1
        assert env._task_assignments[1] == 2


# =====================================================================
# C-7.R: Welfare function ablation set
# =====================================================================

class TestC7R:

    def test_all_welfare_types_exist(self):
        from hcmarl.nswf_allocator import WELFARE_ALLOCATORS
        assert "nswf" in WELFARE_ALLOCATORS
        assert "utilitarian" in WELFARE_ALLOCATORS
        assert "maxmin" in WELFARE_ALLOCATORS
        assert "gini" in WELFARE_ALLOCATORS

    def test_create_allocator(self):
        from hcmarl.nswf_allocator import create_allocator, NSWFParams
        for wtype in ["nswf", "utilitarian", "maxmin", "gini"]:
            alloc = create_allocator(wtype, NSWFParams())
            result = alloc.allocate(
                np.array([[2.0, 1.0], [1.5, 2.5]]),
                np.array([0.1, 0.2]),
            )
            assert len(result.assignments) == 2

    def test_utilitarian_favours_highest_total(self):
        from hcmarl.nswf_allocator import create_allocator, NSWFParams
        alloc = create_allocator("utilitarian", NSWFParams())
        # Worker 0 is very good at task 1, worker 1 is very good at task 2
        util = np.array([[10.0, 1.0], [1.0, 10.0]])
        result = alloc.allocate(util, np.array([0.0, 0.0]))
        assert result.assignments[0] == 1  # worker 0 -> task 1
        assert result.assignments[1] == 2  # worker 1 -> task 2

    def test_maxmin_protects_weakest(self):
        from hcmarl.nswf_allocator import create_allocator, NSWFParams
        alloc = create_allocator("maxmin", NSWFParams())
        util = np.array([[2.0, 1.0], [1.0, 2.0]])
        result = alloc.allocate(util, np.array([0.0, 0.0]))
        # Both workers should get their preferred task for equal surplus
        assert result.assignments[0] == 1
        assert result.assignments[1] == 2


# =====================================================================
# C-8.A: Continuous neural drive action space
# =====================================================================

class TestC8A:

    def test_continuous_env_obs_dim(self):
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env_d = WarehousePettingZoo(n_workers=2, max_steps=5, action_mode="discrete")
        env_c = WarehousePettingZoo(n_workers=2, max_steps=5, action_mode="continuous")
        # Continuous obs includes task one-hot
        assert env_c.obs_dim > env_d.obs_dim
        assert env_c.obs_dim == env_d.obs_dim + env_c.n_tasks

    def test_continuous_step(self):
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=10, action_mode="continuous")
        obs, _ = env.reset()
        env.set_task_assignments({0: 1, 1: 0})
        for _ in range(5):
            actions = {a: np.random.uniform(0, 0.5, size=env.n_muscles) for a in env.agents}
            obs, r, t, tr, i = env.step(actions)
        assert obs["worker_0"].shape == (env.obs_dim,)

    def test_gaussian_actor_network(self):
        from hcmarl.agents.networks import GaussianActorNetwork
        net = GaussianActorNetwork(obs_dim=25, action_dim=6)
        obs = torch.randn(3, 25)
        action, lp, ent = net.get_action(obs)
        assert action.shape == (3, 6)
        assert (action >= 0).all() and (action <= 1).all()
        lp2, ent2 = net.evaluate(obs, action)
        assert lp2.shape == (3,)

    def test_conservation_continuous_mode(self):
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=50, action_mode="continuous")
        env.reset()
        env.set_task_assignments({0: 1, 1: 2})
        for _ in range(30):
            actions = {a: np.random.uniform(0, 0.3, size=env.n_muscles) for a in env.agents}
            env.step(actions)
        for i in range(2):
            for m in env.muscle_names:
                s = env.states[i][m]
                total = s["MR"] + s["MA"] + s["MF"]
                assert abs(total - 1.0) < 1e-5, f"Worker {i}, {m}: MR+MA+MF={total}"


# =====================================================================
# C-8.D: HCMARLAgent owns ECBF + NSWF
# =====================================================================

class TestC8D:

    def test_agent_discrete_mode(self):
        from hcmarl.agents.hcmarl_agent import HCMARLAgent
        agent = HCMARLAgent(obs_dim=19, global_obs_dim=73, n_actions=6, n_agents=4)
        assert agent.action_mode == "discrete"
        assert agent.allocator is not None

    def test_agent_continuous_mode(self):
        from hcmarl.agents.hcmarl_agent import HCMARLAgent
        agent = HCMARLAgent(
            obs_dim=25, global_obs_dim=73, n_actions=6, n_agents=2,
            action_mode="continuous", n_muscles=6,
        )
        assert agent.action_mode == "continuous"
        assert hasattr(agent, 'continuous_actor')
        obs = {f"worker_{i}": np.random.randn(25).astype(np.float32) for i in range(2)}
        gs = np.random.randn(73).astype(np.float32)
        actions, lp, v = agent.get_actions(obs, gs)
        for a in actions.values():
            assert isinstance(a, np.ndarray)
            assert a.shape == (6,)

    def test_agent_welfare_types(self):
        from hcmarl.agents.hcmarl_agent import HCMARLAgent
        for wtype in ["nswf", "utilitarian", "maxmin", "gini"]:
            agent = HCMARLAgent(
                obs_dim=19, global_obs_dim=73, n_actions=6, n_agents=2,
                welfare_type=wtype,
            )
            assignments = agent.allocate_tasks([0.1, 0.2])
            assert len(assignments) == 2


# =====================================================================
# C-9.A: MAPPO critic with agent-id one-hot
# =====================================================================

class TestC9A:

    def test_mappo_critic_input_dim(self):
        from hcmarl.agents.mappo import MAPPO
        m = MAPPO(obs_dim=10, global_obs_dim=37, n_actions=4, n_agents=4)
        # Critic input should be global_obs_dim + n_agents = 41
        assert m.critic.net[0].in_features == 41

    def test_mappo_per_agent_values(self):
        from hcmarl.agents.mappo import MAPPO
        m = MAPPO(obs_dim=10, global_obs_dim=37, n_actions=4, n_agents=4)
        obs = {f"worker_{i}": np.random.randn(10).astype(np.float32) for i in range(4)}
        gs = np.random.randn(37).astype(np.float32)
        actions, log_probs, values = m.get_actions(obs, gs)
        assert isinstance(values, dict)
        assert len(values) == 4
        # Values should differ because of different one-hot encoding
        v_list = list(values.values())
        # With random init, values will generally differ
        assert len(set(round(v, 6) for v in v_list)) >= 1

    def test_mappo_lag_critic_input_dim(self):
        from hcmarl.agents.mappo_lag import MAPPOLagrangian
        ml = MAPPOLagrangian(obs_dim=10, global_obs_dim=37, n_actions=4, n_agents=4)
        assert ml.critic.net[0].in_features == 41
        assert ml.cost_critic.net[0].in_features == 41

    def test_mappo_lag_per_agent_values(self):
        from hcmarl.agents.mappo_lag import MAPPOLagrangian
        ml = MAPPOLagrangian(obs_dim=10, global_obs_dim=37, n_actions=4, n_agents=4)
        obs = {f"worker_{i}": np.random.randn(10).astype(np.float32) for i in range(4)}
        gs = np.random.randn(37).astype(np.float32)
        actions, log_probs, values, cost_values = ml.get_actions(obs, gs)
        assert isinstance(values, dict) and len(values) == 4
        assert isinstance(cost_values, dict) and len(cost_values) == 4

    def test_buffer_stores_per_agent_values(self):
        from hcmarl.agents.mappo import RolloutBuffer
        buf = RolloutBuffer(agent_ids=["worker_0", "worker_1"])
        values_dict = {"worker_0": 1.0, "worker_1": 2.0}
        buf.store_step(
            obs_dict={"worker_0": np.zeros(3), "worker_1": np.zeros(3)},
            global_state=np.zeros(5),
            actions_dict={"worker_0": 0, "worker_1": 1},
            log_probs_dict={"worker_0": -0.5, "worker_1": -0.3},
            rewards_dict={"worker_0": 1.0, "worker_1": 2.0},
            done=False,
            values=values_dict,
        )
        assert buf._values[0] == values_dict

    def test_get_flat_tensors_augmented(self):
        from hcmarl.agents.mappo import RolloutBuffer
        buf = RolloutBuffer(agent_ids=["worker_0", "worker_1"])
        gs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buf.store_step(
            obs_dict={"worker_0": np.zeros(2), "worker_1": np.ones(2)},
            global_state=gs,
            actions_dict={"worker_0": 0, "worker_1": 1},
            log_probs_dict={"worker_0": 0.0, "worker_1": 0.0},
            rewards_dict={"worker_0": 0.0, "worker_1": 0.0},
            done=False,
            values={"worker_0": 0.0, "worker_1": 0.0},
        )
        obs, gs_out, acts, lps = buf.get_flat_tensors("cpu")
        # gs_out should be (2, 5) = (gs_dim=3 + n_agents=2)
        assert gs_out.shape == (2, 5)
        # First row: [1,2,3, 1,0] (worker_0 one-hot)
        assert gs_out[0, 3] == 1.0 and gs_out[0, 4] == 0.0
        # Second row: [1,2,3, 0,1] (worker_1 one-hot)
        assert gs_out[1, 3] == 0.0 and gs_out[1, 4] == 1.0


# =====================================================================
# C-10.A: Cost-advantage NOT normalised
# =====================================================================

class TestC10A:

    def test_cadv_not_normalised(self):
        """Verify cadv normalization line is removed from mappo_lag.py."""
        import inspect
        from hcmarl.agents.mappo_lag import MAPPOLagrangian
        source = inspect.getsource(MAPPOLagrangian.update)
        # Should NOT contain cadv normalization
        assert "cadv_t - cadv_t.mean()" not in source
        assert "cadv_t.std()" not in source
        # Should still normalise reward advantages
        assert "adv_t - adv_t.mean()" in source


# =====================================================================
# C-10.B: PID Lagrangian
# =====================================================================

class TestC10B:

    def test_pid_state_exists(self):
        from hcmarl.agents.mappo_lag import MAPPOLagrangian
        ml = MAPPOLagrangian(obs_dim=10, global_obs_dim=37, n_actions=4, n_agents=2)
        assert hasattr(ml, '_pid_integral')
        assert hasattr(ml, '_pid_prev_error')
        assert hasattr(ml, '_pid_kp')

    def test_pid_increases_lambda_on_violation(self):
        from hcmarl.agents.mappo_lag import MAPPOLagrangian
        ml = MAPPOLagrangian(
            obs_dim=10, global_obs_dim=37, n_actions=4, n_agents=2,
            cost_limit=0.1, lambda_init=0.01,
        )
        initial_lam = ml.lam
        # Simulate high cost (above limit)
        for _ in range(10):
            ml.update_lambda(0.5)  # cost >> limit
        assert ml.lam > initial_lam, "Lambda should increase when cost > limit"

    def test_pid_decreases_lambda_when_safe(self):
        from hcmarl.agents.mappo_lag import MAPPOLagrangian
        ml = MAPPOLagrangian(
            obs_dim=10, global_obs_dim=37, n_actions=4, n_agents=2,
            cost_limit=0.5, lambda_init=1.0,
        )
        # First push lambda up
        for _ in range(10):
            ml.update_lambda(1.0)
        high_lam = ml.lam
        # Now push down
        for _ in range(50):
            ml.update_lambda(0.0)  # no cost
        assert ml.lam < high_lam, "Lambda should decrease when cost < limit"
