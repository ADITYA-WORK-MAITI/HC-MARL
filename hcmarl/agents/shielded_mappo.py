"""
HC-MARL: Shielded-MAPPO baseline (non-RL safety baseline).

Wraps the MAPPO actor with a hand-designed static-threshold shield. The
shield is the simplest possible safety mechanism: at action-selection
time, if any muscle required by the proposed task is already at or above
(theta_max - safety_margin), the action is overridden to `rest`. No
learning, no QP, no continuous adjustment -- just an if-statement.

Why this baseline matters
-------------------------
HCMARL's contribution claim is that the analytical ECBF-QP filter buys
better safety/throughput trade-off than a hand-coded threshold check.
Without this baseline, a reviewer cannot tell whether the QP machinery is
load-bearing. With it, the comparison is transparent: ECBF should keep
workers on heavy tasks at reduced neural drive (continuous adjustment)
while shielded-MAPPO bounces them to rest (discrete refusal). That's a
testable hypothesis. If shielded-MAPPO matches HCMARL, ECBF is over-
engineered for this task; if HCMARL beats it on throughput while matching
on safety, ECBF's complexity is justified.

How shielding interacts with PPO updates
----------------------------------------
The shield modifies the action AFTER the policy samples it. Two design
choices are possible:
  A. Store the policy's INTENDED action and its log-prob, but step the
     env on the shielded action. PPO ratio uses log_p(intended).
  B. Store the SHIELDED action and recompute its log-prob under the
     policy. PPO ratio uses log_p(shielded).
We pick (B). Under (A), the buffer's (s, a, log_p) trio doesn't match
the actual (s, a') the env saw, and PPO's importance ratio would compute
on a fictitious action. (B) keeps everything consistent: the policy is
trained on the shielded behaviour, with the shield treated as part of
the effective policy. Recomputing log_p(rest) costs one forward pass
through the actor, negligible at warehouse-env scale.

Observation schema (depended on, must stay in sync with pettingzoo_wrapper)
--------------------------------------------------------------------------
Per-agent observation layout for action_mode='discrete':
  obs = [MR_m, MA_m, MF_m for each m in muscle_names]
        + [step_norm]
        + [type_one_hot]   (if n_types > 0)
So obs[mi*3 + 2] is muscle mi's MF. We read MF directly; no Euler step,
no prediction. This is "static threshold" in the simplest reading.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from hcmarl.agents.mappo import MAPPO


class ShieldedMAPPO(MAPPO):
    """MAPPO + static-threshold task-refusal shield.

    Constructor takes the env's muscle ordering, theta_max table, and task
    -> demand map so it can decide which actions to refuse. These come
    from the same env config block train.py reads, so wiring is trivial.
    """

    def __init__(self, obs_dim, global_obs_dim, n_actions, n_agents,
                 muscle_names: Sequence[str],
                 theta_max: Dict[str, float],
                 task_names: Sequence[str],
                 task_demands: Dict[str, Dict[str, float]],
                 rest_task_name: str = "rest",
                 safety_margin: float = 0.05,
                 demand_threshold: float = 0.0,
                 **mappo_kwargs):
        super().__init__(obs_dim=obs_dim, global_obs_dim=global_obs_dim,
                         n_actions=n_actions, n_agents=n_agents,
                         **mappo_kwargs)
        self.muscle_names: List[str] = list(muscle_names)
        self.theta_max = dict(theta_max)
        self.task_names: List[str] = list(task_names)
        self.task_demands = {t: dict(d) for t, d in task_demands.items()}
        self.safety_margin = float(safety_margin)
        # Strictly positive demand counts as "this muscle is exercised by
        # this task". Pure-zero entries (e.g. heavy_lift's ankle=0.10 IS
        # nonzero -- ankle DOES get exercised) drop only the literal zeros
        # like 'rest'.
        self.demand_threshold = float(demand_threshold)

        if rest_task_name not in self.task_names:
            raise ValueError(
                f"rest_task_name='{rest_task_name}' not in task_names={self.task_names}; "
                f"shielded-MAPPO needs a refuse-target action"
            )
        self.rest_action_idx = self.task_names.index(rest_task_name)

        # Shield diagnostics (cumulative across an update window; cleared
        # by the caller via reset_shield_stats).
        self._n_shielded_calls = 0
        self._n_total_calls = 0

    # ------------------------------------------------------------------
    # Public diagnostics
    # ------------------------------------------------------------------

    @property
    def shield_rate(self) -> float:
        """Fraction of action-selection calls in which at least one agent
        was shielded. Reset by reset_shield_stats. Used by HCMARLLogger
        to surface a 'how often is the shield engaging' signal during
        training -- if shield_rate is near 1, the policy never learns to
        avoid violations on its own; if near 0, the shield is dormant
        and the comparison effectively reduces to MAPPO."""
        if self._n_total_calls == 0:
            return 0.0
        return self._n_shielded_calls / self._n_total_calls

    def reset_shield_stats(self) -> None:
        self._n_shielded_calls = 0
        self._n_total_calls = 0

    # ------------------------------------------------------------------
    # Shielding logic
    # ------------------------------------------------------------------

    def _mf_per_muscle(self, obs: np.ndarray) -> Dict[str, float]:
        """Read per-muscle MF from a single agent's observation.
        Schema per pettingzoo_wrapper._get_obs:
          obs[mi*3 : mi*3+3] = (MR, MA, MF) for muscle mi.
        """
        out = {}
        for mi, name in enumerate(self.muscle_names):
            out[name] = float(obs[mi * 3 + 2])
        return out

    def _should_shield(self, obs: np.ndarray, intended_action: int) -> bool:
        """Return True if the intended action would push the worker into
        the unsafe zone on any required muscle."""
        if intended_action == self.rest_action_idx:
            return False
        if not (0 <= intended_action < len(self.task_names)):
            return False
        task = self.task_names[intended_action]
        demands = self.task_demands.get(task, {})
        mf = self._mf_per_muscle(obs)
        for muscle, demand in demands.items():
            if demand <= self.demand_threshold:
                continue
            tm = self.theta_max.get(muscle)
            if tm is None:
                continue
            if mf.get(muscle, 0.0) >= max(0.0, tm - self.safety_margin):
                return True
        return False

    def _logprob_under_policy(self, obs_t: torch.Tensor, action: int) -> float:
        """Recompute log pi(action | obs) without grad. Needed when the
        shield substitutes an action -- the buffer must store log_prob
        consistent with the action that was executed (Option B)."""
        with torch.no_grad():
            logits = self.actor(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            lp = dist.log_prob(torch.tensor([action], device=self.device))
        return float(lp.item())

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def get_actions(self, observations: Dict[str, np.ndarray], global_state: np.ndarray):
        actions, log_probs, values = super().get_actions(observations, global_state)

        any_shielded = False
        sorted_agents = sorted(observations.keys())
        for agent_id in sorted_agents:
            obs = observations[agent_id]
            intended = actions[agent_id]
            if self._should_shield(obs, intended):
                any_shielded = True
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                actions[agent_id] = self.rest_action_idx
                # Option B: store log_prob of the SHIELDED action so PPO
                # ratio is internally consistent.
                log_probs[agent_id] = self._logprob_under_policy(
                    obs_t, self.rest_action_idx
                )
        self._n_total_calls += 1
        if any_shielded:
            self._n_shielded_calls += 1
        return actions, log_probs, values
