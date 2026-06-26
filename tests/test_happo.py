"""Smoke test for HAPPO agent.

Confirms construction, action selection, buffer round-trip, update loop,
and save/load -- the contract train.py and run_baselines.py rely on.
"""
import numpy as np
import os
import tempfile

import torch

from hcmarl.agents.happo import HAPPO


def _seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_happo_construct():
    _seed()
    agent = HAPPO(obs_dim=19, global_obs_dim=109, n_actions=6, n_agents=4,
                  device="cpu")
    assert len(agent.actors) == 4
    assert len(agent.actor_optims) == 4
    assert agent.critic is not None
    # Actors must have INDEPENDENT parameters (heterogeneous).
    p0 = next(agent.actors[0].parameters()).detach().clone()
    p1 = next(agent.actors[1].parameters()).detach().clone()
    # With a fixed seed, two freshly-initialised networks DO produce
    # different parameters (orthogonal init draws fresh randomness per call).
    assert not torch.allclose(p0, p1), "Per-agent actors must be independent"


def test_happo_get_actions_contract():
    _seed()
    n_agents = 4
    agent = HAPPO(obs_dim=19, global_obs_dim=77, n_actions=6, n_agents=n_agents,
                  device="cpu")
    obs = {f"worker_{i}": np.random.randn(19).astype(np.float32) for i in range(n_agents)}
    gs = np.random.randn(77).astype(np.float32)
    actions, log_probs, values = agent.get_actions(obs, gs)
    assert set(actions.keys()) == set(obs.keys())
    assert all(isinstance(v, int) and 0 <= v < 6 for v in actions.values())
    assert all(isinstance(v, float) for v in log_probs.values())
    assert all(isinstance(v, float) for v in values.values())


def test_happo_buffer_and_update():
    _seed()
    n_agents = 4
    obs_dim, gobs_dim, n_actions = 19, 77, 6
    agent = HAPPO(obs_dim=obs_dim, global_obs_dim=gobs_dim,
                  n_actions=n_actions, n_agents=n_agents,
                  batch_size=64, n_epochs=2, device="cpu")
    # Synthesise 256 steps of rollout (4 agents x 64 timesteps -> 256).
    T = 64
    for t in range(T):
        obs = {f"worker_{i}": np.random.randn(obs_dim).astype(np.float32)
               for i in range(n_agents)}
        gs = np.random.randn(gobs_dim).astype(np.float32)
        actions, log_probs, values = agent.get_actions(obs, gs)
        rewards = {f"worker_{i}": float(np.random.randn()) for i in range(n_agents)}
        done = (t == T - 1)
        agent.buffer.store_step(
            obs_dict=obs, global_state=gs, actions_dict=actions,
            log_probs_dict=log_probs, rewards_dict=rewards, done=done,
            values=values,
        )
    metrics = agent.update()
    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])
    # Buffer should be cleared after update.
    assert len(agent.buffer) == 0


def test_happo_save_load_roundtrip():
    _seed()
    a1 = HAPPO(obs_dim=19, global_obs_dim=77, n_actions=6, n_agents=4,
               device="cpu")
    a2 = HAPPO(obs_dim=19, global_obs_dim=77, n_actions=6, n_agents=4,
               device="cpu")
    # Sanity: pre-load they should differ (different RNG draws per construct).
    obs = {f"worker_{i}": np.random.randn(19).astype(np.float32) for i in range(4)}
    gs = np.random.randn(77).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "happo.pt")
        a1.save(path)
        a2.load(path)
    # Post-load: same params -> same deterministic logits.
    for actor1, actor2 in zip(a1.actors, a2.actors):
        for p1, p2 in zip(actor1.parameters(), actor2.parameters()):
            assert torch.allclose(p1, p2)
    for p1, p2 in zip(a1.critic.parameters(), a2.critic.parameters()):
        assert torch.allclose(p1, p2)


def test_happo_per_agent_optimiser_state_persists():
    """The save/load contract MUST round-trip per-agent optimiser state.
    Without this, --resume from checkpoint silently re-initialises Adam
    momentum/variance on every restart and bit-identical reproducibility
    is lost."""
    _seed()
    a1 = HAPPO(obs_dim=19, global_obs_dim=77, n_actions=6, n_agents=4,
               device="cpu", batch_size=8, n_epochs=1)
    # Take one optimiser step on each per-agent actor so Adam state populates.
    for j, opt in enumerate(a1.actor_optims):
        loss = a1.actors[j](torch.randn(1, 19)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    a2 = HAPPO(obs_dim=19, global_obs_dim=77, n_actions=6, n_agents=4,
               device="cpu", batch_size=8, n_epochs=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "happo_state.pt")
        a1.save(path)
        a2.load(path)
    # Every per-agent optimiser must have non-empty state and match.
    for o1, o2 in zip(a1.actor_optims, a2.actor_optims):
        s1, s2 = o1.state_dict()["state"], o2.state_dict()["state"]
        assert len(s1) > 0
        assert set(s1.keys()) == set(s2.keys())
