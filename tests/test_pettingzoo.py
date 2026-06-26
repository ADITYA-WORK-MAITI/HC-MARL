"""Tests for Phase 2: PettingZoo wrapper (#24)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo

def test_reset():
    env = WarehousePettingZoo(n_workers=3)
    obs, infos = env.reset()
    assert len(obs) == 3; print("  PASS: test_reset")

def test_parallel_step():
    env = WarehousePettingZoo(n_workers=4, max_steps=10)
    obs, _ = env.reset()
    actions = {a: 0 for a in env.agents}
    obs2, r, terms, truncs, infos = env.step(actions)
    assert len(obs2) == 4; assert len(r) == 4; print("  PASS: test_parallel_step")

def test_global_obs():
    env = WarehousePettingZoo(n_workers=4)
    env.reset()
    g = env._get_global_obs()
    expected = 4 * env.n_muscles * 3 + 1
    assert g.shape == (expected,); print("  PASS: test_global_obs")

def test_episode_completes():
    env = WarehousePettingZoo(n_workers=2, max_steps=5)
    obs, _ = env.reset()
    for _ in range(5):
        actions = {a: np.random.randint(0, env.n_tasks) for a in env.agents}
        obs, r, terms, truncs, infos = env.step(actions)
    assert all(terms.values()); print("  PASS: test_episode_completes")

if __name__ == "__main__":
    print("=== PettingZoo Tests ===")
    for t in [test_reset, test_parallel_step, test_global_obs, test_episode_completes]: t()
    print("All passed.")
