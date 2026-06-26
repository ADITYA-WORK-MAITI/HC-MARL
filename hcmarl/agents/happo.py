"""
HC-MARL: Heterogeneous-Agent PPO (HAPPO) baseline.

Reference:
    Kuba, J. G., Chen, R., Wen, M., Wen, Y., Sun, F., Wang, J., & Yang, Y.
    (2022). Trust Region Policy Optimisation in Multi-Agent Reinforcement
    Learning. ICLR. arXiv:2109.11251.

Algorithmic identity (Algorithm 1, PPO-clip variant in Eq. 11):
  - Per-agent (heterogeneous) actors -- no parameter sharing across agents.
  - Single centralised critic V(s, agent_id) shared across agents.
  - Sequential per-agent update with random permutation each epoch.
  - Per-agent surrogate uses the joint advantage decomposition (Lemma 1):
        A^{i_{1:m}}_pi(s, a^{i_{1:m}}) = sum_j A^{i_j}_pi(s, a^{i_{1:j-1}}, a^{i_j})
    which lets each agent maximise a clipped-PPO objective scaled by the
    importance product M^{i_{1:m-1}} of agents already updated in this
    permutation.

Differences from MAPPO that matter for the contribution claim:
  - No parameter sharing -> each agent class is allowed to specialise.
  - Sequential update with monotonic-improvement guarantee per Theorem 3.
  - No safety / cost machinery (HAPPO is unconstrained).

This file deliberately mirrors mappo.py's RolloutBuffer / get_actions /
update / save / load contract so train.py and run_baselines.py work
without a special-case branch.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict

from hcmarl.agents.networks import ActorNetwork, CriticNetwork
from hcmarl.agents.mappo import RolloutBuffer


class HAPPO:
    """Heterogeneous-Agent PPO with per-agent actors + centralised critic.

    The critic input is (global_state || agent_id one-hot) so it produces
    per-agent value estimates V(s, agent_id) -- same C-9.A pattern MAPPO
    uses.
    """

    def __init__(self, obs_dim, global_obs_dim, n_actions, n_agents,
                 lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 entropy_coeff=0.01, max_grad_norm=0.5,
                 n_epochs=10, batch_size=256,
                 hidden_dim=64, critic_hidden_dim=128, device="cpu"):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Heterogeneous actors -- one ActorNetwork per agent slot.
        self.actors = nn.ModuleList([
            ActorNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
            for _ in range(n_agents)
        ])
        self.actor_optims = [
            optim.Adam(a.parameters(), lr=lr_actor) for a in self.actors
        ]

        # Centralised critic with agent-id one-hot augmentation (C-9.A).
        self.critic = CriticNetwork(global_obs_dim + n_agents, critic_hidden_dim).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)

        agent_ids = [f"worker_{i}" for i in range(n_agents)]
        self.buffer = RolloutBuffer(agent_ids=agent_ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _augment_gs(self, gs_tensor, agent_idx):
        one_hot = torch.zeros(self.n_agents, device=self.device)
        one_hot[agent_idx] = 1.0
        return torch.cat([gs_tensor, one_hot])

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def get_actions(self, observations: Dict[str, np.ndarray], global_state: np.ndarray):
        """One forward per agent (no parameter sharing). Returns dicts
        compatible with mappo.get_actions: (actions, log_probs, values).
        """
        sorted_agents = sorted(observations.keys())
        actions, log_probs, values = {}, {}, {}
        gs_np = np.asarray(global_state, dtype=np.float32)
        for i, agent_id in enumerate(sorted_agents):
            obs_t = torch.from_numpy(observations[agent_id]).float().unsqueeze(0).to(self.device)
            one_hot = np.zeros(self.n_agents, dtype=np.float32)
            one_hot[i] = 1.0
            gs_aug = np.concatenate([gs_np, one_hot]).astype(np.float32)
            gs_t = torch.from_numpy(gs_aug).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, lp, _ = self.actors[i].get_action(obs_t)
                value = self.critic(gs_t)
            actions[agent_id] = int(action.item())
            log_probs[agent_id] = float(lp.item())
            values[agent_id] = float(value.item())
        return actions, log_probs, values

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self):
        """HAPPO update: sequential per-agent PPO-clip with M-product term.

        For each PPO epoch:
            1. Draw a random permutation of agents.
            2. Initialise M_running = 1 (vector of length T).
            3. For each agent j_m in permutation order:
               a. Compute the surrogate using (M_running.detach() * A_{j_m})
                  as the effective advantage.
               b. PPO-clip update on actor[j_m] only.
               c. Update M_running by the new policy ratio of agent j_m,
                  computed under no_grad after the optimiser step.
            4. Centralised critic update on flattened (T*N) targets.
        """
        if len(self.buffer) < self.batch_size:
            return {}

        T = self.buffer._n_steps
        N = self.n_agents

        # -- Compute per-agent advantages via the existing buffer pipeline.
        with torch.no_grad():
            last_gs = torch.FloatTensor(
                self.buffer._global_states[-1]
            ).to(self.device)
            last_values = {}
            for i, a in enumerate(self.buffer.agent_ids):
                gs_aug = self._augment_gs(last_gs, i)
                last_values[a] = self.critic(gs_aug).item()

        adv_flat, ret_flat = self.buffer.compute_returns(
            last_values, self.gamma, self.gae_lambda
        )
        # Buffer interleaves (T, N) -> (T*N,) row-major. Restore the (T, N)
        # layout so each agent's trajectory is contiguous along axis=0.
        adv_TN = adv_flat.reshape(T, N)
        ret_TN = ret_flat.reshape(T, N)

        # -- Per-agent tensors.
        per_obs, per_acts, per_old_lp, per_adv, per_ret = {}, {}, {}, {}, {}
        for i, a in enumerate(self.buffer.agent_ids):
            per_obs[i] = torch.from_numpy(np.array(self.buffer._obs[a], dtype=np.float32)).to(self.device)
            per_acts[i] = torch.from_numpy(np.array(self.buffer._actions[a], dtype=np.int64)).to(self.device)
            per_old_lp[i] = torch.from_numpy(np.array(self.buffer._log_probs[a], dtype=np.float32)).to(self.device)
            adv_i = torch.from_numpy(adv_TN[:, i].astype(np.float32)).to(self.device)
            # Per-agent advantage normalisation (HAPPO common practice).
            per_adv[i] = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)
            per_ret[i] = torch.from_numpy(ret_TN[:, i].astype(np.float32)).to(self.device)

        # -- Per-agent augmented global-state tensors (T, global_dim+N).
        gs_np = np.array(self.buffer._global_states, dtype=np.float32)  # (T, global_dim)
        per_gs = []
        for i in range(N):
            one_hot = np.zeros(N, dtype=np.float32)
            one_hot[i] = 1.0
            gs_aug = np.concatenate([gs_np, np.broadcast_to(one_hot, (T, N))], axis=1)
            per_gs.append(torch.from_numpy(gs_aug).to(self.device))

        last_actor_loss = last_critic_loss = 0.0

        for _ in range(self.n_epochs):
            # Random permutation order of agents (HAPPO Algorithm 1, line 5).
            perm = np.random.permutation(N)
            # M_running: importance product over agents already updated in
            # this permutation (Eq. 11, the M^{i_{1:m-1}} term).
            M_running = torch.ones(T, device=self.device)

            for j in perm:
                # Effective advantage for this agent: M_{1:m-1} * A_j.
                m_adv = (M_running.detach() * per_adv[j])

                new_lp, entropy = self.actors[j].evaluate(per_obs[j], per_acts[j])
                ratio = (new_lp - per_old_lp[j]).exp()
                surr1 = ratio * m_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * m_adv
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy.mean()

                self.actor_optims[j].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[j].parameters(), self.max_grad_norm)
                self.actor_optims[j].step()
                last_actor_loss = actor_loss.item()

                # Update M_running with the post-update ratio of agent j.
                # This is what HAPPO requires: subsequent agents in the
                # permutation see the freshly-updated policy of agent j.
                with torch.no_grad():
                    new_lp_after, _ = self.actors[j].evaluate(per_obs[j], per_acts[j])
                    new_ratio = (new_lp_after - per_old_lp[j]).exp()
                    # Hard clamp to [0.5, 2.0]. The lower bound is critical:
                    # if any element of new_ratio collapses to 0, M_running
                    # becomes the zero vector and every subsequent agent in
                    # the permutation sees zero advantage -> zero gradient
                    # -> no update for the rest of the epoch.
                    new_ratio = new_ratio.clamp(0.5, 2.0)
                    M_running = M_running * new_ratio

            # Centralised critic update on all (T*N) targets.
            gs_all = torch.cat(per_gs, dim=0)               # (T*N, dim)
            ret_all = torch.cat([per_ret[i] for i in range(N)], dim=0)  # (T*N,)
            values = self.critic(gs_all)
            critic_loss = nn.functional.mse_loss(values, ret_all)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optim.step()
            last_critic_loss = critic_loss.item()

        self.buffer.clear()
        return {"actor_loss": last_actor_loss, "critic_loss": last_critic_loss}

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save({
            "actors": [a.state_dict() for a in self.actors],
            "critic": self.critic.state_dict(),
            "actor_optims": [o.state_dict() for o in self.actor_optims],
            "critic_optim": self.critic_optim.state_dict(),
            "n_agents": self.n_agents,
        }, path)

    def load(self, path):
        # weights_only=False mirrors mappo / mappo_lag for PyTorch 2.6+
        # compat. Bit-identical resume relies on the optimiser state-dicts
        # round-tripping; True drops them silently.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("n_agents", self.n_agents) != self.n_agents:
            raise ValueError(
                f"HAPPO checkpoint has n_agents={ckpt['n_agents']} "
                f"but agent was constructed with n_agents={self.n_agents}."
            )
        for a, sd in zip(self.actors, ckpt["actors"]):
            a.load_state_dict(sd)
        self.critic.load_state_dict(ckpt["critic"])
        if "actor_optims" in ckpt:
            for o, sd in zip(self.actor_optims, ckpt["actor_optims"]):
                o.load_state_dict(sd)
        if "critic_optim" in ckpt:
            self.critic_optim.load_state_dict(ckpt["critic_optim"])
