"""
HC-MARL: MAPPO-Lagrangian Baseline Wrapper
Wraps MAPPOLagrangian with SafePO-compatible interface.
This is an honest MAPPO-Lagrangian baseline (cost critic + dual variable).
"""
import numpy as np
import torch
from typing import Dict

from hcmarl.agents.mappo_lag import MAPPOLagrangian


class SafePOWrapper:
    """MAPPO-Lagrangian baseline with SafePO-compatible interface.
    Always uses our own MAPPOLagrangian (no external SafePO dependency)."""

    def __init__(self, obs_dim=19, n_actions=6, n_agents=4, config=None, device="cpu"):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.config = config or {}
        self.device = device
        self.name = "MAPPO-Lagrangian"
        self._safepo_available = False
        self._safepo_agent = None

        # Always use our own MAPPO-Lagrangian (no external SafePO dependency).
        if not self._safepo_available:
            algo = self.config.get("algorithm", {})
            global_obs_dim = n_agents * (obs_dim - 1) + 1  # match pettingzoo wrapper
            self._fallback = MAPPOLagrangian(
                obs_dim=obs_dim,
                global_obs_dim=global_obs_dim,
                n_actions=n_actions,
                n_agents=n_agents,
                lr_actor=algo.get("lr_actor", 3e-4),
                lr_critic=algo.get("lr_critic", 1e-3),
                lr_lambda=algo.get("lambda_lr", 5e-3),
                cost_limit=algo.get("cost_limit", 0.05),
                lambda_init=algo.get("lambda_init", 1.0),
                device=device,
            )

    @property
    def buffer(self):
        if self._safepo_available:
            return None
        return self._fallback.buffer

    @property
    def lam(self):
        if self._safepo_available:
            return 0.0
        return self._fallback.lam

    def get_actions(self, observations, global_state=None, **kwargs):
        if self._safepo_available:
            actions = {}
            log_probs = {}
            for agent_id, obs in observations.items():
                try:
                    action = self._safepo_agent.predict(obs)
                    actions[agent_id] = int(action) if np.isscalar(action) else int(action[0])
                except Exception:
                    actions[agent_id] = np.random.randint(0, self.n_actions)
                log_probs[agent_id] = 0.0
            return actions, log_probs, 0.0, 0.0
        else:
            if global_state is None:
                # Construct global state: concatenate all agent obs (minus time), then time
                all_obs = []
                for agent_id in sorted(observations.keys()):
                    all_obs.append(observations[agent_id][:-1])  # drop time step
                # Pad to expected n_agents if fewer agents present
                obs_per_agent = len(list(observations.values())[0]) - 1
                n_present = len(all_obs)
                for _ in range(self.n_agents - n_present):
                    all_obs.append(np.zeros(obs_per_agent, dtype=np.float32))
                all_obs = np.concatenate(all_obs)
                time_frac = list(observations.values())[0][-1]
                global_state = np.append(all_obs, time_frac).astype(np.float32)
            result = self._fallback.get_actions(observations, global_state)
            # Return full tuple for train.py compatibility
            return result

    def update(self):
        if self._safepo_available:
            return {}
        return self._fallback.update()

    def update_lambda(self, mean_cost):
        if not self._safepo_available:
            self._fallback.update_lambda(mean_cost)

    def save(self, path):
        if self._safepo_available and self._safepo_agent:
            self._safepo_agent.save(path)
        elif not self._safepo_available:
            self._fallback.save(path)

    def load(self, path):
        if self._safepo_available and self._safepo_agent:
            self._safepo_agent.load(path)
        elif not self._safepo_available:
            self._fallback.load(path)
