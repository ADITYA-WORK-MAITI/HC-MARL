"""
HC-MARL Phase 3 (#49): IPPO — Parameter-Shared Variant (PS-IPPO).

This is the "IPPO" baseline as implemented in the MAPPO benchmark of
Yu et al. 2022 (NeurIPS) — a parameter-shared actor-critic where each
agent uses the *same* network weights but acts on its own local
observation, and the critic sees only local observations (no global
state). That decentralised-critic property is what distinguishes IPPO
from MAPPO.

Why parameter-shared: the prior per-agent-network IPPO performed
6 serial forward passes per env step and 6 serial optimiser steps per
PPO epoch. On a single GPU that was CUDA-launch bound at ~50-60 SPS.
Shared parameters + batched forward collapses this to one actor + one
critic call per step, matching MAPPO throughput while preserving the
algorithmic identity (decentralised critic on local obs).

References:
    Yu, C. et al. (2022) "The Surprising Effectiveness of PPO in
        Cooperative Multi-Agent Games." NeurIPS. Section 4.3 of that
        paper describes the IPPO implementation we follow here:
        parameter-shared actor and critic, decentralised critic on
        local observations.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict

from hcmarl.agents.networks import ActorNetwork, CriticNetwork
from hcmarl.agents.mappo import RolloutBuffer


class IPPO:
    """Parameter-shared Independent PPO.

    Same algorithmic niche as the original IPPO (decentralised critic on
    local obs, no centralised info), but with one actor and one critic
    shared across all agents. Each agent acts on its own local obs and
    is evaluated by a critic that sees only that agent's own local obs.

    Key contrasts:
      - vs original per-agent IPPO: we share one actor + one critic
        across agents -> 1 CUDA launch per step (not n_agents).
      - vs MAPPO: the critic input is each agent's local obs, NOT the
        global state. So V(o_i), not V(s). This is the defining IPPO
        property.
    """

    def __init__(self, obs_dim, n_actions, n_agents,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 entropy_coeff=0.01, max_grad_norm=0.5,
                 n_epochs=10, batch_size=256,
                 hidden_dim=64, critic_hidden_dim=128,
                 device="cpu"):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Single shared actor and single shared critic (local-obs critic).
        self.actor = ActorNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim, critic_hidden_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # One multi-agent rollout buffer (same class as MAPPO) so GAE is
        # computed per-agent but the update sees a flattened batch.
        agent_ids = [f"worker_{i}" for i in range(n_agents)]
        self.buffer = RolloutBuffer(agent_ids=agent_ids)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def get_actions(self, observations: Dict[str, np.ndarray], global_state=None):
        """Act with one batched forward through the shared actor+critic.

        Contract matches MAPPO.get_actions so train.py works unchanged:
            returns (actions, log_probs, values) as dicts keyed by agent.
            values[agent_id] = V(o_i) from the shared critic on that
            agent's LOCAL observation (not the global state).
        """
        sorted_agents = sorted(observations.keys())
        N = len(sorted_agents)
        obs_batch = torch.from_numpy(
            np.stack([observations[a] for a in sorted_agents])
        ).float().to(self.device)

        with torch.no_grad():
            action_batch, lp_batch, _ = self.actor.get_action(obs_batch)
            # IPPO defining property: critic uses local obs, not global state.
            value_batch = self.critic(obs_batch)

        a_np = action_batch.cpu().numpy()
        lp_np = lp_batch.cpu().numpy()
        v_np = value_batch.cpu().numpy()
        actions, log_probs, values = {}, {}, {}
        for i, aid in enumerate(sorted_agents):
            actions[aid] = int(a_np[i])
            log_probs[aid] = float(lp_np[i])
            values[aid] = float(v_np[i])
        return actions, log_probs, values

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self):
        """PPO update. Batched across all agents thanks to param sharing."""
        if len(self.buffer) < self.batch_size:
            return {}

        obs, _, acts, old_lp = self.buffer.get_flat_tensors(self.device)
        # obs shape: (T*N, obs_dim). The second return (global states + one-hot)
        # is unused in IPPO because our critic takes local obs only.

        # Per-agent bootstrap: each agent's last observation -> its own V(o_i).
        with torch.no_grad():
            last_obs_list = []
            for a in self.buffer.agent_ids:
                last_obs_list.append(self.buffer._obs[a][-1])
            last_obs_t = torch.from_numpy(np.stack(last_obs_list)).float().to(self.device)
            last_values_t = self.critic(last_obs_t)
            last_values = {
                a: float(last_values_t[i].item())
                for i, a in enumerate(self.buffer.agent_ids)
            }

        advantages, returns = self.buffer.compute_returns(
            last_values, self.gamma, self.gae_lambda
        )
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        last_actor_loss = last_critic_loss = 0.0
        for _ in range(self.n_epochs):
            new_lp, entropy = self.actor.evaluate(obs, acts)
            ratio = (new_lp - old_lp).exp()
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy.mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optim.step()

            # Critic uses local obs (IPPO), not global state.
            values = self.critic(obs)
            critic_loss = nn.functional.mse_loss(values, ret_t)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optim.step()

            last_actor_loss = actor_loss.item()
            last_critic_loss = critic_loss.item()

        self.buffer.clear()
        return {"actor_loss": last_actor_loss, "critic_loss": last_critic_loss}

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
        }, path)

    def load(self, path):
        # weights_only=False: PyTorch 2.6+ rejects optimiser state-dicts
        # under its default weights_only=True. Without this, --resume
        # silently fails to restore optimiser state and bit-identical
        # resume breaks.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "actor_optim" in ckpt:
            self.actor_optim.load_state_dict(ckpt["actor_optim"])
        if "critic_optim" in ckpt:
            self.critic_optim.load_state_dict(ckpt["critic_optim"])

    # Legacy transition-store API kept for backward compat with any test
    # that calls store_transition(i, obs, action, log_prob, reward, done).
    # Delegates to buffer.store_step with a per-agent value computed on
    # the shared critic. Prefer get_actions() returning a values dict and
    # calling buffer.store_step directly in new code.
    def store_transition(self, agent_idx, obs, action, log_prob, reward, done):
        # PS-IPPO is called once per agent per timestep by train.py's
        # per-agent loop. RolloutBuffer.store_step() expects ALL agents'
        # data in a single dict call; routing each agent through it
        # individually broke the buffer's agent_id set on the first call
        # (KeyError 'worker_1' on the second timestep). The buffer.store()
        # legacy shim is the correct entry point: it accumulates N
        # per-agent calls in agent_id order, then flushes one timestep
        # via store_step() automatically. Set agent_ids on first call so
        # the shim has its call-order contract anchored.
        if not self.buffer.agent_ids:
            self.buffer.agent_ids = [f"worker_{i}" for i in range(self.n_agents)]
            self.buffer._obs = {a: [] for a in self.buffer.agent_ids}
            self.buffer._actions = {a: [] for a in self.buffer.agent_ids}
            self.buffer._log_probs = {a: [] for a in self.buffer.agent_ids}
            self.buffer._rewards = {a: [] for a in self.buffer.agent_ids}
        obs_t = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            value = float(self.critic(obs_t).item())
        self.buffer.store(
            obs=obs,
            global_state=obs,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            values=value,
        )
