"""
HC-MARL Phase 3 (#35): MAPPO Algorithm
Multi-Agent PPO with centralised critic and decentralised actors.
Ref: Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (NeurIPS 2022).

Buffer structure: Per-agent trajectory storage. GAE computed per-agent.
Flattened for PPO update since the actor has shared weights.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
from hcmarl.agents.networks import ActorNetwork, CriticNetwork


class RolloutBuffer:
    """Per-agent rollout storage for correct GAE computation.

    Stores data as lists-of-dicts keyed by agent_id. Each agent's
    trajectory is kept separate so GAE is computed correctly per-agent
    (each agent has its own reward stream).

    For the PPO update, per-agent data is flattened since the actor
    shares weights across all agents (CTDE paradigm).
    """
    def __init__(self, agent_ids=None):
        self.agent_ids = agent_ids or []
        self.clear()

    def clear(self):
        # Per-agent storage: {agent_id: [list of values per timestep]}
        self._obs = {a: [] for a in self.agent_ids}
        self._global_states = []  # Shared across agents (one per timestep)
        self._actions = {a: [] for a in self.agent_ids}
        self._log_probs = {a: [] for a in self.agent_ids}
        self._rewards = {a: [] for a in self.agent_ids}
        self._dones = []  # Shared (episode-level termination)
        self._values = []  # Per-agent critic values: dict {agent_id: float} per timestep (C-9.A)
        self._n_steps = 0

    def store_step(self, obs_dict, global_state, actions_dict, log_probs_dict,
                   rewards_dict, done, values):
        """Store one timestep for ALL agents simultaneously.

        Args:
            obs_dict: {agent_id: obs_array}
            global_state: global observation for centralised critic
            actions_dict: {agent_id: action_int}
            log_probs_dict: {agent_id: log_prob_float}
            rewards_dict: {agent_id: reward_float}
            done: bool/float, shared episode termination
            values: dict {agent_id: float} — per-agent critic values (C-9.A)
        """
        # Initialise agent_ids on first call if not set
        if not self.agent_ids:
            self.agent_ids = sorted(obs_dict.keys())
            self._obs = {a: [] for a in self.agent_ids}
            self._actions = {a: [] for a in self.agent_ids}
            self._log_probs = {a: [] for a in self.agent_ids}
            self._rewards = {a: [] for a in self.agent_ids}

        for a in self.agent_ids:
            self._obs[a].append(obs_dict[a])
            self._actions[a].append(actions_dict[a])
            self._log_probs[a].append(log_probs_dict.get(a, 0.0))
            self._rewards[a].append(rewards_dict[a])

        self._global_states.append(global_state)
        self._dones.append(float(done))
        # C-9.A: store per-agent values (dict or scalar for backward compat)
        if isinstance(values, dict):
            self._values.append(values)
        else:
            # Legacy scalar: broadcast to all agents
            self._values.append({a: float(values) for a in self.agent_ids})
        self._n_steps += 1

    # Legacy compatibility: store() for per-agent calls from old train.py
    # This is a shim — new code should use store_step()
    _legacy_pending = None

    def store(self, obs, global_state, action, log_prob, reward, done, values):
        """Legacy per-agent store. Accumulates N calls then flushes one timestep.

        S-24: Call-order contract — this method must be called exactly N
        times per timestep (once per agent), in the order matching
        self.agent_ids (which is sorted(actions.keys()) from train.py).
        The i-th call stores data for agent_ids[i]. After the N-th call,
        the accumulated data is flushed. Prefer store_step() for new code.
        """
        if self._legacy_pending is None:
            self._legacy_pending = {
                'obs': {}, 'actions': {}, 'log_probs': {}, 'rewards': {},
                'global_state': global_state, 'done': done, 'value': values,
                '_call_count': 0,
            }

        # Assign to next agent in order
        if not self.agent_ids:
            raise ValueError("RolloutBuffer: agent_ids not set. "
                             "Pass agent_ids at construction time.")

        idx = self._legacy_pending['_call_count']
        # S-24: Guard against overflow — more calls than agents
        if idx >= len(self.agent_ids):
            raise RuntimeError(
                f"RolloutBuffer.store() called {idx+1} times "
                f"for a {len(self.agent_ids)}-agent buffer. Expected exactly "
                f"{len(self.agent_ids)} calls per timestep."
            )
        if idx < len(self.agent_ids):
            a = self.agent_ids[idx]
            self._legacy_pending['obs'][a] = obs
            self._legacy_pending['actions'][a] = action
            self._legacy_pending['log_probs'][a] = log_prob
            self._legacy_pending['rewards'][a] = reward
        self._legacy_pending['_call_count'] += 1

        # When all agents have stored, flush
        if self._legacy_pending['_call_count'] >= len(self.agent_ids):
            self.store_step(
                self._legacy_pending['obs'],
                self._legacy_pending['global_state'],
                self._legacy_pending['actions'],
                self._legacy_pending['log_probs'],
                self._legacy_pending['rewards'],
                self._legacy_pending['done'],
                self._legacy_pending['value'],
            )
            self._legacy_pending = None

    def compute_returns(self, last_values, gamma=0.99, gae_lambda=0.95,
                         last_episode_truncated=True):
        """Compute GAE per-agent with per-agent value baselines, then flatten.

        C-9.A fix: Each agent uses its OWN value baseline V(s, agent_id)
        rather than a shared V(s), enabling proper per-agent credit assignment.

        S-19 fix: Distinguishes truncation from termination at the last step.
        For truncated episodes (time limit), the bootstrap V(s_next) is used.
        For terminated episodes (true terminal state), the bootstrap is 0.
        Without this distinction the agent learns a value function biased
        toward 0 near the time-limit boundary, which corrupts late-episode
        credit assignment.

        Args:
            last_values: dict {agent_id: float} -- bootstrap values for each agent.
                For backward compat, also accepts a scalar (broadcast to all).
            last_episode_truncated: If True, the episode ended by time limit
                (not true termination), so the last bootstrap should NOT be
                zeroed by the done mask. Default True since the warehouse env
                always truncates at max_steps.

        Returns:
            advantages: (T * N_agents,) flattened
            returns: (T * N_agents,) flattened
        """
        T = self._n_steps
        if T == 0:
            return np.zeros(0), np.zeros(0)

        dones = np.array(self._dones)    # (T,)

        # Handle scalar last_value for backward compat
        if not isinstance(last_values, dict):
            last_values = {a: float(last_values) for a in self.agent_ids}

        all_advantages = []
        all_returns = []

        for a in self.agent_ids:
            # Per-agent value trajectory
            values_a = np.array([v[a] for v in self._values])  # (T,)
            rewards_a = np.array(self._rewards[a])  # (T,)
            adv_a = np.zeros(T)
            last_gae = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = last_values[a]
                    # S-19: for truncated episodes, don't zero the bootstrap
                    # S-20: done mask at intermediate steps correctly handles
                    # episode boundaries (next_non_term=0 when done[t]=1)
                    next_non_term = 1.0 if last_episode_truncated else (1.0 - dones[t])
                else:
                    next_val = values_a[t + 1]
                    next_non_term = 1.0 - dones[t]
                delta = rewards_a[t] + gamma * next_val * next_non_term - values_a[t]
                last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
                adv_a[t] = last_gae
            ret_a = adv_a + values_a
            all_advantages.append(adv_a)
            all_returns.append(ret_a)

        # Interleave: for each timestep t, stack all agents
        # Result shape: (T * N,)
        advantages = np.stack(all_advantages, axis=1).reshape(-1)  # (T, N) -> (T*N,)
        returns = np.stack(all_returns, axis=1).reshape(-1)

        return advantages, returns

    def get_flat_tensors(self, device):
        """Get flattened tensors for PPO update.

        Returns obs, global_states, actions, log_probs as (T*N, ...) tensors.

        C-9.A fix: Global states are augmented with agent-id one-hot so the
        critic receives (global_state, agent_id) and can produce per-agent
        value estimates instead of learning the team mean.
        """
        T = self._n_steps
        N = len(self.agent_ids)

        # Flatten obs: (T, N, obs_dim) -> (T*N, obs_dim)
        obs_list = []
        for t in range(T):
            for a in self.agent_ids:
                obs_list.append(self._obs[a][t])
        obs = torch.FloatTensor(np.array(obs_list)).to(device)

        # Global states: augment with agent-id one-hot (C-9.A)
        gs_list = []
        for t in range(T):
            for i in range(N):
                one_hot = np.zeros(N, dtype=np.float32)
                one_hot[i] = 1.0
                gs_aug = np.concatenate([self._global_states[t], one_hot])
                gs_list.append(gs_aug)
        gs = torch.FloatTensor(np.array(gs_list)).to(device)

        # Actions and log_probs: flatten same way.
        # Appendix F fix: auto-detect action dtype. Discrete-mode
        # actors store integer task indices (LongTensor); continuous-mode
        # GaussianActorNetwork stores per-muscle neural drive in [0, 1]
        # (FloatTensor of shape (n_muscles,) per agent per step). The previous
        # unconditional LongTensor cast silently truncated continuous actions
        # to int, which then crashed at Categorical.log_prob shape validation.
        acts_list, lp_list = [], []
        for t in range(T):
            for a in self.agent_ids:
                acts_list.append(self._actions[a][t])
                lp_list.append(self._log_probs[a][t])
        acts_arr = np.array(acts_list)
        if acts_arr.dtype.kind == 'f':
            # Continuous actions: Gaussian-actor neural drive vectors.
            acts = torch.FloatTensor(acts_arr).to(device)
        else:
            # Discrete actions: Categorical task indices.
            acts = torch.LongTensor(acts_arr).to(device)
        lps = torch.FloatTensor(np.array(lp_list)).to(device)

        return obs, gs, acts, lps

    def __len__(self):
        return self._n_steps * max(len(self.agent_ids), 1)


class MAPPO:
    """Multi-Agent PPO with shared actor weights and centralised critic."""

    def __init__(self, obs_dim, global_obs_dim, n_actions, n_agents,
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, entropy_coeff=0.01, max_grad_norm=0.5,
                 n_epochs=10, batch_size=256, device="cpu"):
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.actor = ActorNetwork(obs_dim, n_actions).to(self.device)
        # C-9.A: Critic input = global_obs + agent_id one-hot, so it can
        # produce per-agent value estimates (Yu et al. 2022, Section 3.2)
        self.critic = CriticNetwork(global_obs_dim + n_agents).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)

        agent_ids = [f"worker_{i}" for i in range(n_agents)]
        self.buffer = RolloutBuffer(agent_ids=agent_ids)

    def _augment_gs(self, gs_tensor, agent_idx):
        """Append agent-id one-hot to global state for per-agent critic (C-9.A)."""
        one_hot = torch.zeros(self.n_agents, device=self.device)
        one_hot[agent_idx] = 1.0
        return torch.cat([gs_tensor, one_hot])

    def get_actions(self, observations: Dict[str, np.ndarray], global_state: np.ndarray):
        # T1 throughput: stack all agent obs into a single (N, obs_dim) forward
        # instead of N per-agent (1, obs_dim) forwards. On L4 the per-call
        # CUDA-launch + Categorical.sample overhead is ~1 ms regardless of
        # batch, so collapsing 6 actor + 6 critic launches into 1 each cuts
        # ~10 ms off every env step. Per-seed
        # reproducibility is preserved (Categorical.sample is still seeded);
        # bit-identity vs. the old per-agent path is NOT, since RNG is drawn
        # in one batched call rather than N scalar calls. That breaks an
        # apples-to-apples resume from pre-T1 checkpoints.
        sorted_agents = sorted(observations.keys())
        N = len(sorted_agents)
        obs_batch = torch.from_numpy(
            np.stack([observations[a] for a in sorted_agents])
        ).float().to(self.device)
        # Critic input = (global_state || agent_id one-hot) per agent.
        # One-hot width is self.n_agents (critic was built with that dim), not
        # N — N may be smaller than self.n_agents when a caller passes partial
        # observations (e.g. SafePOWrapper.test harness).
        gs_np = np.asarray(global_state, dtype=np.float32)
        gs_tiled = np.broadcast_to(gs_np, (N, gs_np.shape[0])).copy()
        one_hot = np.eye(self.n_agents, dtype=np.float32)[:N]
        gs_aug = np.concatenate([gs_tiled, one_hot], axis=1)
        gs_aug_t = torch.from_numpy(gs_aug).to(self.device)
        with torch.no_grad():
            action_batch, lp_batch, _ = self.actor.get_action(obs_batch)
            value_batch = self.critic(gs_aug_t)
        action_np = action_batch.cpu().numpy()
        lp_np = lp_batch.cpu().numpy()
        val_np = value_batch.cpu().numpy()
        actions, log_probs, values = {}, {}, {}
        for i, agent_id in enumerate(sorted_agents):
            actions[agent_id] = int(action_np[i])
            log_probs[agent_id] = float(lp_np[i])
            values[agent_id] = float(val_np[i])
        return actions, log_probs, values

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        obs, gs, acts, old_lp = self.buffer.get_flat_tensors(self.device)

        with torch.no_grad():
            # C-9.A: per-agent bootstrap values
            last_gs = torch.FloatTensor(
                self.buffer._global_states[-1]
            ).to(self.device)
            last_values = {}
            for i, a in enumerate(self.buffer.agent_ids):
                gs_aug = self._augment_gs(last_gs, i)
                last_values[a] = self.critic(gs_aug).item()

        advantages, returns = self.buffer.compute_returns(
            last_values, self.gamma, self.gae_lambda
        )
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.n_epochs):
            new_lp, entropy = self.actor.evaluate(obs, acts)
            ratio = (new_lp - old_lp).exp()
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy.mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optim.step()

            values = self.critic(gs)
            critic_loss = nn.functional.mse_loss(values, ret_t)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optim.step()
            total_loss += (actor_loss.item() + critic_loss.item())

        self.buffer.clear()
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
        }, path)

    def load(self, path):
        # weights_only=False: PyTorch 2.6+ defaults to True, which rejects
        # the optimizer state-dicts we save here. Without this, --resume
        # silently fails to restore optimizer momentum/variance and the
        # bit-identical resume guarantee evaporates.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "actor_optim" in ckpt:
            self.actor_optim.load_state_dict(ckpt["actor_optim"])
        if "critic_optim" in ckpt:
            self.critic_optim.load_state_dict(ckpt["critic_optim"])
