"""Tests for Phase 3: HC-MARL agent (#40)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_agent_init():
    import torch
    from hcmarl.agents.hcmarl_agent import HCMARLAgent
    agent = HCMARLAgent(obs_dim=19, global_obs_dim=73, n_actions=6, n_agents=4)
    assert agent.n_agents == 4; print("  PASS: test_agent_init")

def test_agent_get_actions():
    import torch
    from hcmarl.agents.hcmarl_agent import HCMARLAgent
    agent = HCMARLAgent(obs_dim=19, global_obs_dim=73, n_actions=6, n_agents=4)
    obs = {f"worker_{i}": np.random.randn(19).astype(np.float32) for i in range(4)}
    gs = np.random.randn(73).astype(np.float32)
    actions, lp, v = agent.get_actions(obs, gs)
    assert len(actions) == 4
    for a in actions.values(): assert 0 <= a < 6
    print("  PASS: test_agent_get_actions")

def test_agent_save_load():
    import torch, tempfile
    from hcmarl.agents.hcmarl_agent import HCMARLAgent
    agent = HCMARLAgent(obs_dim=19, global_obs_dim=73, n_actions=6, n_agents=4)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
        tmp_path = fh.name
    try:
        agent.save(tmp_path)
        agent2 = HCMARLAgent(obs_dim=19, global_obs_dim=73, n_actions=6, n_agents=4)
        agent2.load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    print("  PASS: test_agent_save_load")

if __name__ == "__main__":
    print("=== HC-MARL Agent Tests ===")
    for t in [test_agent_init, test_agent_get_actions, test_agent_save_load]: t()
    print("All passed.")
