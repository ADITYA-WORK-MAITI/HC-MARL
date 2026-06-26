"""
HC-MARL Agent: Full pipeline agent per Section 7.3.

Orchestrates:
    1. NSWF allocator (outer loop) — assigns tasks to workers (Eq 33)
    2. MAPPO policy (inner loop) — outputs neural drive C per muscle (Remark 7.2)
    3. ECBF safety filter — clips C to safe set (Eq 20, handled in env)

Supports two action modes:
    - "discrete": agent selects tasks (backward compatible, pre-Round-2)
    - "continuous": agent outputs continuous neural drive per muscle (math-doc-faithful)
"""
import torch
import numpy as np
from typing import Dict, Optional
from hcmarl.agents.mappo import MAPPO
from hcmarl.agents.networks import GaussianActorNetwork
from hcmarl.nswf_allocator import NSWFAllocator, NSWFParams, create_allocator


class HCMARLAgent:
    """Complete HC-MARL agent: NSWF + MAPPO + ECBF (Section 7.3).

    In continuous mode (Remark 7.2), the agent owns:
        - An NSWF allocator for task assignment at the outer timescale
        - A MAPPO policy with GaussianActor for continuous neural drive
        - ECBF filtering is delegated to the environment (C-6.A)

    In discrete mode (backward compatible):
        - Wraps MAPPO with discrete action space
        - ECBF filtering handled in env
    """

    def __init__(self, obs_dim, global_obs_dim, n_actions, n_agents,
                 theta_max=None, ecbf_params=None, use_nswf=True,
                 action_mode="discrete", n_muscles=None,
                 welfare_type="nswf", nswf_params=None,
                 allocation_interval=30, device="cpu",
                 **mappo_kwargs):
        """
        Args:
            obs_dim: Per-agent observation dimension.
            global_obs_dim: Global state dimension for centralized critic.
            n_actions: Number of discrete actions (tasks) if discrete mode.
            n_agents: Number of workers/agents.
            theta_max: Safety thresholds per muscle (for allocator utility).
            ecbf_params: ECBF parameters (stored for reference; filtering in env).
            use_nswf: Whether to use NSWF allocation.
            action_mode: "discrete" or "continuous".
            n_muscles: Number of muscles (action dim in continuous mode).
            welfare_type: Welfare function for allocator ("nswf", "utilitarian", etc).
            nswf_params: NSWFParams for the allocator.
            allocation_interval: Steps between allocator calls (K in two-timescale).
            device: Torch device.
        """
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.action_mode = action_mode
        self.n_muscles = n_muscles or 6  # default from task_profiles
        self.allocation_interval = allocation_interval
        self.device = device

        # Store ECBF params for reference (actual filtering in env, C-6.A)
        self.theta_max = theta_max or {}
        self.ecbf_params = ecbf_params or {}

        # NSWF allocator (C-7.A)
        if use_nswf:
            self.allocator = create_allocator(
                welfare_type,
                params=nswf_params or NSWFParams(),
            )
        else:
            self.allocator = None

        # MAPPO policy — forward algorithm hyperparams from config
        self.mappo = MAPPO(
            obs_dim, global_obs_dim, n_actions, n_agents, device=device,
            **mappo_kwargs
        )

        # In continuous mode, replace the discrete actor with Gaussian
        if action_mode == "continuous":
            self.continuous_actor = GaussianActorNetwork(
                obs_dim=obs_dim, action_dim=self.n_muscles
            ).to(torch.device(device))
            self.continuous_actor_optim = torch.optim.Adam(
                self.continuous_actor.parameters(), lr=3e-4
            )
            # Appendix F fix: wire MAPPO.update()'s actor path
            # through the Gaussian actor in continuous mode. Without this swap
            # MAPPO.update() calls self.actor.evaluate(obs, acts) on the
            # discrete ActorNetwork (Categorical), which rejects continuous-
            # action tensors with a shape-validation ValueError at the first
            # PPO update boundary (Gate 0 failure mode #1).
            #
            # The swap retargets MAPPO's actor + actor_optim to the
            # GaussianActorNetwork that act() already uses. The discrete
            # ActorNetwork constructed by MAPPO.__init__ becomes unreferenced
            # and is GC'd; its optimizer is replaced. Discrete-mode runs (the
            # only mode used in EXP1/EXP2) are entirely unaffected -- this
            # branch only fires when action_mode == "continuous".
            self.mappo.actor = self.continuous_actor
            self.mappo.actor_optim = self.continuous_actor_optim

        # Allocation state
        self._current_assignments = {i: 0 for i in range(n_agents)}
        self._steps_since_allocation = 0

    def allocate_tasks(self, fatigue_levels, utility_matrix=None):
        """Run the NSWF allocator to assign tasks (outer timescale, C-7.A).

        Args:
            fatigue_levels: dict or array of per-worker max fatigue.
            utility_matrix: (N, M) utilities. If None, uses uniform.

        Returns:
            dict {worker_idx: task_idx}
        """
        if self.allocator is None:
            return self._current_assignments

        if isinstance(fatigue_levels, dict):
            fl = np.array([fatigue_levels.get(f"worker_{i}", fatigue_levels.get(i, 0.0))
                           for i in range(self.n_agents)])
        else:
            fl = np.asarray(fatigue_levels)

        if utility_matrix is None:
            n_productive = max(1, self.n_actions - 1)  # exclude rest
            utility_matrix = np.ones((self.n_agents, n_productive))

        result = self.allocator.allocate(utility_matrix, fl)
        self._current_assignments = result.assignments
        self._steps_since_allocation = 0
        return result.assignments

    def should_reallocate(self):
        """Check if it's time to run the allocator (every K steps)."""
        return self._steps_since_allocation >= self.allocation_interval

    def get_actions(self, observations, global_state=None):
        """Get agent actions.

        In discrete mode: returns task indices (backward compatible).
        In continuous mode: returns per-muscle C_nom arrays.
        """
        if global_state is None:
            global_state = np.concatenate(list(observations.values()))

        self._steps_since_allocation += 1

        if self.action_mode == "continuous":
            return self._get_continuous_actions(observations, global_state)
        else:
            actions, log_probs, values = self.mappo.get_actions(observations, global_state)
            return actions, log_probs, values

    def _get_continuous_actions(self, observations, global_state):
        """Get continuous neural drive actions (Remark 7.2).

        Appendix F probe instrumentation: captures the Gaussian actor's
        mean log_std across agents and stores it on
        ``self.last_gaussian_log_std_mean`` for diagnostic logging. This
        confirms the continuous-mode policy is exploring (log_std not
        collapsed near LOG_STD_MIN=-5.0, not diverged near LOG_STD_MAX=2.0)
        without leaking gradients.
        """
        actions, log_probs, values = {}, {}, {}
        gs = torch.FloatTensor(global_state).to(torch.device(self.device))
        sorted_agents = sorted(observations.keys())

        log_std_accum = []
        for i, agent_id in enumerate(sorted_agents):
            obs_t = torch.FloatTensor(observations[agent_id]).to(torch.device(self.device))
            with torch.no_grad():
                action, lp, _, log_std_mean = self.continuous_actor.get_action_with_log_std(obs_t)
            # action is (n_muscles,) in [0, 1]
            actions[agent_id] = action.cpu().numpy()
            log_probs[agent_id] = lp.item()
            log_std_accum.append(float(log_std_mean.item()))
            # Per-agent value from critic (C-9.A)
            gs_aug = self.mappo._augment_gs(gs, i)
            values[agent_id] = self.mappo.critic(gs_aug).item()

        # Appendix F diagnostic: mean log_std across agents this step.
        # Read by train.py loop and aggregated per episode.
        self.last_gaussian_log_std_mean = (
            float(np.mean(log_std_accum)) if log_std_accum else 0.0
        )

        return actions, log_probs, values

    def update(self):
        """PPO update. Delegates to MAPPO."""
        return self.mappo.update() or {}

    def save(self, path):
        state = {
            "mappo_actor": self.mappo.actor.state_dict(),
            "mappo_critic": self.mappo.critic.state_dict(),
            "mappo_actor_optim": self.mappo.actor_optim.state_dict(),
            "mappo_critic_optim": self.mappo.critic_optim.state_dict(),
        }
        if self.action_mode == "continuous":
            state["continuous_actor"] = self.continuous_actor.state_dict()
            state["continuous_actor_optim"] = self.continuous_actor_optim.state_dict()
        torch.save(state, path)

    def load(self, path):
        # weights_only=False: PyTorch 2.6+ defaults to True, which rejects
        # the optimizer state-dicts (and continuous-actor optim) we save
        # here. Without this, --resume silently fails to restore optimizer
        # state and the bit-identical resume guarantee evaporates.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.mappo.actor.load_state_dict(ckpt["mappo_actor"])
        self.mappo.critic.load_state_dict(ckpt["mappo_critic"])
        if "mappo_actor_optim" in ckpt:
            self.mappo.actor_optim.load_state_dict(ckpt["mappo_actor_optim"])
        if "mappo_critic_optim" in ckpt:
            self.mappo.critic_optim.load_state_dict(ckpt["mappo_critic_optim"])
        if self.action_mode == "continuous" and "continuous_actor" in ckpt:
            self.continuous_actor.load_state_dict(ckpt["continuous_actor"])
            if "continuous_actor_optim" in ckpt:
                self.continuous_actor_optim.load_state_dict(ckpt["continuous_actor_optim"])
