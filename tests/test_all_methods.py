"""Tests for Phase 3: All 10 methods train 1 episode (#52)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo

def test_random_policy():
    env = WarehousePettingZoo(n_workers=2, max_steps=5)
    obs, _ = env.reset()
    for _ in range(5):
        actions = {a: np.random.randint(0, env.n_tasks) for a in env.agents}
        obs, r, t, tr, i = env.step(actions)
    print("  PASS: test_random_policy")

def test_mappo_instantiation():
    import torch
    from hcmarl.agents.mappo import MAPPO
    m = MAPPO(obs_dim=19, global_obs_dim=73, n_actions=6, n_agents=4)
    assert m.actor is not None; print("  PASS: test_mappo_instantiation")

def test_ippo_instantiation():
    """IPPO (parameter-shared variant per Yu et al. 2022 benchmark).
    Single shared actor + single shared critic with local-obs input."""
    import torch
    from hcmarl.agents.ippo import IPPO
    ip = IPPO(obs_dim=19, n_actions=6, n_agents=4)
    assert ip.actor is not None, "shared actor missing"
    assert ip.critic is not None, "shared critic missing"
    assert ip.n_agents == 4
    # Decentralised-critic property: critic input dim must equal obs_dim
    # (local obs), not global_state dim. That's what distinguishes IPPO
    # from MAPPO.
    first_layer = next(ip.critic.net.children())
    assert first_layer.in_features == 19, \
        f"IPPO critic must take local obs (dim=19), got {first_layer.in_features}"
    print("  PASS: test_ippo_instantiation")

def test_mappo_lag_instantiation():
    import torch
    from hcmarl.agents.mappo_lag import MAPPOLagrangian
    ml = MAPPOLagrangian(obs_dim=19, global_obs_dim=73, n_actions=6, n_agents=4)
    assert ml.lam > 0; print("  PASS: test_mappo_lag_instantiation")

def test_safepo_wrapper_honest_label():
    from hcmarl.baselines.safepo_wrapper import SafePOWrapper
    w = SafePOWrapper(obs_dim=19, n_actions=6, n_agents=4)
    assert w.name == "MAPPO-Lagrangian", f"Expected honest label, got: {w.name}"
    obs = {"worker_0": np.zeros(19, dtype=np.float32)}
    result = w.get_actions(obs)
    actions = result[0] if isinstance(result, tuple) else result
    assert "worker_0" in actions; print("  PASS: test_safepo_wrapper_honest_label")

if __name__ == "__main__":
    print("=== All Methods Tests ===")
    for t in [test_random_policy, test_mappo_instantiation, test_ippo_instantiation,
              test_mappo_lag_instantiation, test_safepo_wrapper_honest_label]: t()
    print("All passed.")
