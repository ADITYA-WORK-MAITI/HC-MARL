"""
HC-MARL Phase 3 (#36): MAPPO-Lagrangian
MAPPO with cost critic and dual variable lambda for constraint satisfaction.
Actor loss: L_clip + λ * cost_advantage (penalise unsafe actions).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from hcmarl.agents.networks import ActorNetwork, CriticNetwork, CostCriticNetwork


class LagrangianRolloutBuffer:
    """Per-agent rollout storage with cost tracking for Lagrangian PPO.

    Like MAPPO's RolloutBuffer, stores data per-agent so GAE is computed
    correctly per-agent (each agent has its own reward/cost stream).
    Flattened for PPO update since the actor shares weights.
    """
    def __init__(self, agent_ids=None):
        self.agent_ids = agent_ids or []
        self.clear()

    def clear(self):
        self._obs = {a: [] for a in self.agent_ids}
        self._global_states = []
        self._actions = {a: [] for a in self.agent_ids}
        self._log_probs = {a: [] for a in self.agent_ids}
        self._rewards = {a: [] for a in self.agent_ids}
        self._costs = {a: [] for a in self.agent_ids}
        self._dones = []
        self._values = []
        self._cost_values = []
        self._n_steps = 0
        self._legacy_pending = None

    def store(self, obs, global_state, action, log_prob, reward, cost, done, values, cost_values):
        """Per-agent store. Accumulates N calls then flushes one timestep.

        S-21: Call-order contract — this method must be called exactly N
        times per timestep (once per agent), in the order matching
        self.agent_ids. The i-th call stores data for agent_ids[i].
        After the N-th call, the accumulated data is flushed to the
        internal buffers as one complete timestep. Calling more than N
        times before flush raises RuntimeError.

        Args:
            values: dict {agent_id: float} -- per-agent critic values (C-9.A).
                Also accepts scalar for backward compat.
            cost_values: dict {agent_id: float} -- per-agent cost critic values.
                Also accepts scalar for backward compat.
        """
        if self._legacy_pending is None:
            self._legacy_pending = {
                'obs': {}, 'actions': {}, 'log_probs': {}, 'rewards': {},
                'costs': {},
                'global_state': global_state, 'done': done,
                'value': values, 'cost_value': cost_values,
                '_call_count': 0,
            }

        if not self.agent_ids:
            raise ValueError("LagrangianRolloutBuffer: agent_ids not set. "
                             "Pass agent_ids at construction time.")

        idx = self._legacy_pending['_call_count']
        # S-21: Guard against overflow — more calls than agents
        if idx >= len(self.agent_ids):
            raise RuntimeError(
                f"LagrangianRolloutBuffer.store() called {idx+1} times "
                f"for a {len(self.agent_ids)}-agent buffer. Expected exactly "
                f"{len(self.agent_ids)} calls per timestep."
            )
        if idx < len(self.agent_ids):
            a = self.agent_ids[idx]
            self._legacy_pending['obs'][a] = obs
            self._legacy_pending['actions'][a] = action
            self._legacy_pending['log_probs'][a] = log_prob
            self._legacy_pending['rewards'][a] = reward
            self._legacy_pending['costs'][a] = cost
        self._legacy_pending['_call_count'] += 1

        if self._legacy_pending['_call_count'] >= len(self.agent_ids):
            p = self._legacy_pending
            for a in self.agent_ids:
                self._obs[a].append(p['obs'][a])
                self._actions[a].append(p['actions'][a])
                self._log_probs[a].append(p['log_probs'][a])
                self._rewards[a].append(p['rewards'][a])
                self._costs[a].append(p['costs'][a])
            self._global_states.append(p['global_state'])
            self._dones.append(float(p['done']))
            # C-9.A: per-agent values (dict or scalar)
            v = p['value']
            cv = p['cost_value']
            if isinstance(v, dict):
                self._values.append(v)
            else:
                self._values.append({a: float(v) for a in self.agent_ids})
            if isinstance(cv, dict):
                self._cost_values.append(cv)
            else:
                self._cost_values.append({a: float(cv) for a in self.agent_ids})
            self._n_steps += 1
            self._legacy_pending = None

    def compute_returns(self, last_values, last_cost_values, gamma=0.99, gae_lambda=0.95,
                         last_episode_truncated=True):
        """Compute reward GAE and cost GAE per-agent with per-agent baselines, then flatten.

        C-9.A fix: Uses per-agent value baselines instead of shared values.
        S-19 fix: Proper truncation handling -- see RolloutBuffer.compute_returns.

        Args:
            last_values: dict {agent_id: float} or scalar -- bootstrap values.
            last_cost_values: dict {agent_id: float} or scalar -- bootstrap cost values.
            last_episode_truncated: If True, don't zero bootstrap at T-1 (time limit).
        """
        T = self._n_steps
        if T == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

        dones = np.array(self._dones)

        # Handle scalar for backward compat
        if not isinstance(last_values, dict):
            last_values = {a: float(last_values) for a in self.agent_ids}
        if not isinstance(last_cost_values, dict):
            last_cost_values = {a: float(last_cost_values) for a in self.agent_ids}

        all_advantages, all_returns = [], []
        all_cost_advantages, all_cost_returns = [], []

        for a in self.agent_ids:
            rewards_a = np.array(self._rewards[a])
            costs_a = np.array(self._costs[a])
            values_a = np.array([v[a] for v in self._values])  # (T,) per agent
            cost_values_a = np.array([v[a] for v in self._cost_values])  # (T,)

            # Reward GAE
            adv_a = np.zeros(T)
            last_gae = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = last_values[a]
                    # S-19: truncation -> keep bootstrap; termination -> zero it
                    # S-20: done mask at intermediate steps correctly handles
                    # episode boundaries (next_non_term=0 when done[t]=1)
                    next_non_term = 1.0 if last_episode_truncated else (1.0 - dones[t])
                else:
                    next_val = values_a[t + 1]
                    next_non_term = 1.0 - dones[t]
                delta = rewards_a[t] + gamma * next_val * next_non_term - values_a[t]
                last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
                adv_a[t] = last_gae
            all_advantages.append(adv_a)
            all_returns.append(adv_a + values_a)

            # Cost GAE
            cadv_a = np.zeros(T)
            last_gae = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_cv = last_cost_values[a]
                    # S-19/S-20: same truncation/done-mask logic as reward GAE
                    next_non_term = 1.0 if last_episode_truncated else (1.0 - dones[t])
                else:
                    next_cv = cost_values_a[t + 1]
                    next_non_term = 1.0 - dones[t]
                delta = costs_a[t] + gamma * next_cv * next_non_term - cost_values_a[t]
                last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
                cadv_a[t] = last_gae
            all_cost_advantages.append(cadv_a)
            all_cost_returns.append(cadv_a + cost_values_a)

        # Interleave: (T, N) -> (T*N,)
        advantages = np.stack(all_advantages, axis=1).reshape(-1)
        returns = np.stack(all_returns, axis=1).reshape(-1)
        cost_advantages = np.stack(all_cost_advantages, axis=1).reshape(-1)
        cost_returns = np.stack(all_cost_returns, axis=1).reshape(-1)

        return advantages, returns, cost_advantages, cost_returns

    def get_flat_tensors(self, device):
        """Get flattened tensors for PPO update. Shape: (T*N, ...).

        C-9.A fix: Global states augmented with agent-id one-hot.
        """
        T = self._n_steps
        N = len(self.agent_ids)
        obs_list, gs_list, acts_list, lp_list = [], [], [], []
        for t in range(T):
            for i, a in enumerate(self.agent_ids):
                obs_list.append(self._obs[a][t])
                acts_list.append(self._actions[a][t])
                lp_list.append(self._log_probs[a][t])
                # Augment global state with agent-id one-hot
                one_hot = np.zeros(N, dtype=np.float32)
                one_hot[i] = 1.0
                gs_aug = np.concatenate([self._global_states[t], one_hot])
                gs_list.append(gs_aug)

        import torch
        obs = torch.FloatTensor(np.array(obs_list)).to(device)
        gs = torch.FloatTensor(np.array(gs_list)).to(device)
        acts = torch.LongTensor(np.array(acts_list)).to(device)
        lps = torch.FloatTensor(np.array(lp_list)).to(device)
        return obs, gs, acts, lps

    def __len__(self):
        return self._n_steps * max(len(self.agent_ids), 1)


class MAPPOLagrangian:
    """MAPPO with Lagrangian constraint handling."""
    def __init__(self, obs_dim, global_obs_dim, n_actions, n_agents,
                 lr_actor=3e-4, lr_critic=1e-3, lr_lambda=5e-3,
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 entropy_coeff=0.01, max_grad_norm=0.5,
                 n_epochs=10, batch_size=256,
                 cost_limit=0.1, lambda_init=0.5, device="cpu"):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cost_limit = cost_limit

        self.log_lambda = nn.Parameter(torch.tensor(np.log(max(lambda_init, 1e-8)), device=self.device))
        self.actor = ActorNetwork(obs_dim, n_actions).to(self.device)
        # C-9.A: Critic input = global_obs + agent_id one-hot for per-agent values
        self.critic = CriticNetwork(global_obs_dim + n_agents).to(self.device)
        self.cost_critic = CostCriticNetwork(global_obs_dim + n_agents).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.cost_critic_optim = optim.Adam(self.cost_critic.parameters(), lr=lr_critic)
        self.lambda_optim = optim.Adam([self.log_lambda], lr=lr_lambda)
        # C-10.B: PID Lagrangian state (Stooke et al. 2020)
        self._pid_integral = 0.0
        self._pid_prev_error = 0.0
        self._pid_kp = 1.0
        self._pid_ki = 0.01
        self._pid_kd = 0.01
        agent_ids = [f"worker_{i}" for i in range(n_agents)]
        self.buffer = LagrangianRolloutBuffer(agent_ids=agent_ids)

    @property
    def lam(self):
        return self.log_lambda.exp().item()

    def _augment_gs(self, gs_tensor, agent_idx):
        """Append agent-id one-hot to global state for per-agent critic (C-9.A)."""
        one_hot = torch.zeros(self.n_agents, device=self.device)
        one_hot[agent_idx] = 1.0
        return torch.cat([gs_tensor, one_hot])

    def get_actions(self, observations, global_state):
        # T1 throughput: stack all agent obs into a single (N, obs_dim) actor
        # forward + single (N, global_dim+N) critic + cost_critic forward.
        # See mappo.py::get_actions for the full rationale.
        sorted_agents = sorted(observations.keys())
        N = len(sorted_agents)
        obs_batch = torch.from_numpy(
            np.stack([observations[a] for a in sorted_agents])
        ).float().to(self.device)
        gs_np = np.asarray(global_state, dtype=np.float32)
        gs_tiled = np.broadcast_to(gs_np, (N, gs_np.shape[0])).copy()
        # One-hot width = self.n_agents; see mappo.py::get_actions for why.
        one_hot = np.eye(self.n_agents, dtype=np.float32)[:N]
        gs_aug = np.concatenate([gs_tiled, one_hot], axis=1)
        gs_aug_t = torch.from_numpy(gs_aug).to(self.device)
        with torch.no_grad():
            action_batch, lp_batch, _ = self.actor.get_action(obs_batch)
            value_batch = self.critic(gs_aug_t)
            cost_value_batch = self.cost_critic(gs_aug_t)
        action_np = action_batch.cpu().numpy()
        lp_np = lp_batch.cpu().numpy()
        val_np = value_batch.cpu().numpy()
        cval_np = cost_value_batch.cpu().numpy()
        actions, log_probs, values, cost_values = {}, {}, {}, {}
        for i, agent_id in enumerate(sorted_agents):
            actions[agent_id] = int(action_np[i])
            log_probs[agent_id] = float(lp_np[i])
            values[agent_id] = float(val_np[i])
            cost_values[agent_id] = float(cval_np[i])
        return actions, log_probs, values, cost_values

    def update(self):
        """PPO update with Lagrangian cost penalty on actor loss."""
        if len(self.buffer) < self.batch_size:
            return {}

        obs, gs, acts, old_lp = self.buffer.get_flat_tensors(self.device)

        with torch.no_grad():
            # C-9.A: per-agent bootstrap values
            last_gs = torch.FloatTensor(
                self.buffer._global_states[-1]
            ).to(self.device)
            last_values, last_cost_values = {}, {}
            for i, a in enumerate(self.buffer.agent_ids):
                gs_aug = self._augment_gs(last_gs, i)
                last_values[a] = self.critic(gs_aug).item()
                last_cost_values[a] = self.cost_critic(gs_aug).item()

        advantages, returns, cost_advantages, cost_returns = self.buffer.compute_returns(
            last_values, last_cost_values, self.gamma, self.gae_lambda
        )

        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        cadv_t = torch.FloatTensor(cost_advantages).to(self.device)
        cret_t = torch.FloatTensor(cost_returns).to(self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        # C-10.A fix: Do NOT normalise cost advantages. Normalising cadv_t
        # to mean~0 decouples the primal update (lam * cadv) from the dual
        # update (raw mean_cost - cost_limit), breaking the Lagrangian.
        # See Stooke et al. 2020 Section 5 for analysis of this footgun.

        lam = self.log_lambda.exp().detach()

        for _ in range(self.n_epochs):
            new_lp, entropy = self.actor.evaluate(obs, acts)
            ratio = (new_lp - old_lp).exp()
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
            reward_loss = -torch.min(surr1, surr2).mean()

            # Lagrangian cost penalty: λ * max(surr) is conservative (safe) clipping
            cost_surr1 = ratio * cadv_t
            cost_surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * cadv_t
            cost_loss = lam * torch.max(cost_surr1, cost_surr2).mean()

            actor_loss = reward_loss + cost_loss - self.entropy_coeff * entropy.mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optim.step()

            # Reward critic
            values = self.critic(gs)
            critic_loss = nn.functional.mse_loss(values, ret_t)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optim.step()

            # Cost critic
            cost_values = self.cost_critic(gs)
            cost_critic_loss = nn.functional.mse_loss(cost_values, cret_t)
            self.cost_critic_optim.zero_grad()
            cost_critic_loss.backward()
            nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
            self.cost_critic_optim.step()

        self.buffer.clear()
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "cost_critic_loss": cost_critic_loss.item(),
            "lambda": self.lam,
        }

    def update_lambda(self, mean_cost):
        """PID Lagrangian update (Stooke et al. 2020, C-10.B).

        Replaces raw gradient ascent with a PID controller on the cost
        violation signal (mean_cost - cost_limit). This prevents the
        oscillation and overshoot typical of primal-dual methods.
        """
        error = mean_cost - self.cost_limit
        self._pid_integral += error
        derivative = error - self._pid_prev_error
        self._pid_prev_error = error

        # PID output determines lambda
        pid_output = (self._pid_kp * error
                      + self._pid_ki * self._pid_integral
                      + self._pid_kd * derivative)

        # Set lambda directly from PID (softplus to keep positive)
        new_lam = max(0.0, pid_output)
        with torch.no_grad():
            self.log_lambda.data = torch.tensor(
                np.log(max(new_lam, 1e-8)), device=self.device
            )

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "cost_critic": self.cost_critic.state_dict(),
            "log_lambda": self.log_lambda.data,
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "cost_critic_optim": self.cost_critic_optim.state_dict(),
            "lambda_optim": self.lambda_optim.state_dict(),
        }, path)

    def load(self, path):
        # weights_only=False: PyTorch 2.6+ defaults to True, which rejects
        # the optimizer state-dicts and the log_lambda tensor we save here.
        # Without this, --resume silently loses the dual variable and
        
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.cost_critic.load_state_dict(ckpt["cost_critic"])
        self.log_lambda.data = ckpt["log_lambda"]
        if "actor_optim" in ckpt:
            self.actor_optim.load_state_dict(ckpt["actor_optim"])
        if "critic_optim" in ckpt:
            self.critic_optim.load_state_dict(ckpt["critic_optim"])
        if "cost_critic_optim" in ckpt:
            self.cost_critic_optim.load_state_dict(ckpt["cost_critic_optim"])
        if "lambda_optim" in ckpt:
            self.lambda_optim.load_state_dict(ckpt["lambda_optim"])
