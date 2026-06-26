"""Tests for Phase 2: Full integration."""
import sys, os, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
from hcmarl.logger import HCMARLLogger

def test_full_episode_with_logger():
    env = WarehousePettingZoo(n_workers=3, max_steps=20)
    with tempfile.TemporaryDirectory() as _log_dir:
        logger = HCMARLLogger(log_dir=_log_dir, use_wandb=False)
        obs, _ = env.reset()
        total_reward = 0.0
        for step in range(20):
            actions = {a: np.random.randint(0, env.n_tasks) for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)
            total_reward += sum(rewards.values())
            if all(terms.values()): break
        ep_data = {"total_reward": total_reward, "n_steps": 20, "n_workers": 3,
                   "n_muscles": env.n_muscles, "total_violations": 0, "safe_steps": 20,
                   "tasks_completed": 40, "peak_fatigue": 0.3, "forced_rests": 2,
                   "tasks_per_worker": [14, 13, 13], "recovery_times": []}
        metrics = logger.compute_episode_metrics(ep_data)
        assert len(metrics) == 9; print("  PASS: test_full_episode_with_logger")

if __name__ == "__main__":
    print("=== Env Integration Tests ===")
    test_full_episode_with_logger()
    print("All passed.")
