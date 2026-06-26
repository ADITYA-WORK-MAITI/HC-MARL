"""
HC-MARL Phase 2: Baseline Methods
====================================
9 baseline methods for comparison against HC-MARL (MAPPO + ECBF + NSWF).
Each baseline is a callable that takes observations and returns actions.

Baselines:
 1. Random           - Uniform random task selection
 2. Round-Robin      - Cyclic task assignment
 3. Greedy           - Always pick highest-productivity task
 4. Greedy-Safe      - Greedy but switch to rest if MF > threshold
 5. PPO-Unconstrained - Single-agent PPO without safety filter
 6. PPO-Lagrangian   - PPO with Lagrangian constraint penalty
 7. MAPPO-NoFilter   - MAPPO without ECBF safety filter
 8. MAPPO-Lagrangian - MAPPO with Lagrangian multiplier for safety
 9. Fixed-Schedule   - Fixed work/rest pattern (industry standard)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Random Baseline
# ---------------------------------------------------------------------------

class RandomBaseline:
    """Uniform random task selection."""

    def __init__(self, n_tasks: int, seed: int = 42):
        self.n_tasks = n_tasks
        self.rng = np.random.RandomState(seed)
        self.name = "Random"

    def get_actions(self, observations: Dict[str, np.ndarray], **kwargs) -> Dict[str, int]:
        return {agent: self.rng.randint(0, self.n_tasks) for agent in observations}


# ---------------------------------------------------------------------------
# 2. Round-Robin Baseline
# ---------------------------------------------------------------------------

class RoundRobinBaseline:
    """Cyclic task assignment: each worker rotates through tasks."""

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        self.step_count = 0
        self.name = "Round-Robin"

    def get_actions(self, observations: Dict[str, np.ndarray], **kwargs) -> Dict[str, int]:
        actions = {}
        for i, agent in enumerate(sorted(observations.keys())):
            actions[agent] = (i + self.step_count) % self.n_tasks
        self.step_count += 1
        return actions


# ---------------------------------------------------------------------------
# 3. Greedy Baseline
# ---------------------------------------------------------------------------

class GreedyBaseline:
    """Always pick the highest-productivity task (index 0 = heavy_lift)."""

    def __init__(self, best_task_idx: int = 0):
        self.best_task_idx = best_task_idx
        self.name = "Greedy"

    def get_actions(self, observations: Dict[str, np.ndarray], **kwargs) -> Dict[str, int]:
        return {agent: self.best_task_idx for agent in observations}


# ---------------------------------------------------------------------------
# 4. Greedy-Safe Baseline
# ---------------------------------------------------------------------------

class GreedySafeBaseline:
    """
    Greedy but switches to rest (last task) when any muscle's MF exceeds threshold.
    Observation format: [MR, MA, MF] per muscle + normalised_step.
    """

    def __init__(self, n_tasks: int, n_muscles: int = 3, mf_threshold: float = 0.5,
                 best_task_idx: int = 0):
        self.n_tasks = n_tasks
        self.n_muscles = n_muscles
        self.mf_threshold = mf_threshold
        self.best_task_idx = best_task_idx
        self.rest_idx = n_tasks - 1  # rest is last task
        self.name = "Greedy-Safe"

    def get_actions(self, observations: Dict[str, np.ndarray], **kwargs) -> Dict[str, int]:
        actions = {}
        for agent, obs in observations.items():
            # Extract MF values (index 2, 5, 8, ... in obs)
            mf_values = [obs[3 * i + 2] for i in range(self.n_muscles)]
            if max(mf_values) > self.mf_threshold:
                actions[agent] = self.rest_idx
            else:
                actions[agent] = self.best_task_idx
        return actions


# ---------------------------------------------------------------------------
# 5. PPO-Unconstrained (single-agent, no safety)
# ---------------------------------------------------------------------------

class PPOUnconstrainedBaseline:
    """
    Single-agent PPO without any safety constraints.
    Uses the same actor architecture as MAPPO but no ECBF filter.
    Actions are chosen purely from the learned policy.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(hidden_dim, obs_dim).astype(np.float32) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(n_actions, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(n_actions, dtype=np.float32)
        self.n_actions = n_actions
        self.name = "PPO-Unconstrained"

    def _forward(self, obs: np.ndarray) -> np.ndarray:
        h = np.maximum(0, self.W1 @ obs + self.b1)
        logits = self.W2 @ h + self.b2
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        return np.clip(probs, 1e-8, 1.0)

    def get_actions(self, observations: Dict[str, np.ndarray], **kwargs) -> Dict[str, int]:
        actions = {}
        for agent, obs in observations.items():
            probs = self._forward(obs)
            actions[agent] = int(np.random.choice(self.n_actions, p=probs / probs.sum()))
        return actions


# ---------------------------------------------------------------------------
# 6. PPO-Lagrangian (single-agent, Lagrangian safety penalty)
# ---------------------------------------------------------------------------

class PPOLagrangianBaseline:
    """
    PPO with Lagrangian multiplier for safety constraint.
    reward_adjusted = reward - lambda * constraint_cost
    lambda is updated via dual gradient ascent.
    """

    def __init__(self, obs_dim: int, n_actions: int, n_muscles: int = 3,
                 theta_max: float = 0.5, lambda_init: float = 1.0,
                 lambda_lr: float = 0.01, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(64, obs_dim).astype(np.float32) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = rng.randn(n_actions, 64).astype(np.float32) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(n_actions, dtype=np.float32)
        self.n_actions = n_actions
        self.n_muscles = n_muscles
        self.theta_max = theta_max
        self.lam = lambda_init
        self.lambda_lr = lambda_lr
        self.name = "PPO-Lagrangian"

    def _forward(self, obs: np.ndarray) -> np.ndarray:
        h = np.maximum(0, self.W1 @ obs + self.b1)
        logits = self.W2 @ h + self.b2
        # Penalise actions that would increase fatigue when already high
        mf_values = [obs[3 * i + 2] for i in range(self.n_muscles)]
        avg_mf = np.mean(mf_values)
        if avg_mf > self.theta_max * 0.8:
            # Boost rest action (last index)
            logits[-1] += self.lam * avg_mf
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        return np.clip(probs, 1e-8, 1.0)

    def update_lambda(self, constraint_violation: float):
        """Dual ascent: increase lambda when constraints are violated."""
        self.lam = max(0.0, self.lam + self.lambda_lr * constraint_violation)

    def get_actions(self, observations: Dict[str, np.ndarray], **kwargs) -> Dict[str, int]:
        actions = {}
        for agent, obs in observations.items():
            probs = self._forward(obs)
            actions[agent] = int(np.random.choice(self.n_actions, p=probs / probs.sum()))
        return actions


# ---------------------------------------------------------------------------
# 7. MAPPO-NoFilter (MAPPO without ECBF)
# ---------------------------------------------------------------------------

class MAPPONoFilterBaseline:
    """MAPPO actor without ECBF safety filtering. Same network, no barrier."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(hidden_dim, obs_dim).astype(np.float32) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(n_actions, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(n_actions, dtype=np.float32)
        self.n_actions = n_actions
        self.name = "MAPPO-NoFilter"

    def get_actions(self, observations: Dict[str, np.ndarray], **kwargs) -> Dict[str, int]:
        actions = {}
        for agent, obs in observations.items():
            h = np.maximum(0, self.W1 @ obs + self.b1)
            logits = self.W2 @ h + self.b2
            logits -= logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()
            probs = np.clip(probs, 1e-8, 1.0)
            actions[agent] = int(np.random.choice(self.n_actions, p=probs / probs.sum()))
        return actions


# ---------------------------------------------------------------------------
# 8. MAPPO-Lagrangian
# ---------------------------------------------------------------------------

class MAPPOLagrangianBaseline(PPOLagrangianBaseline):
    """Multi-agent version of PPO-Lagrangian. Shared weights across agents."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "MAPPO-Lagrangian"


# ---------------------------------------------------------------------------
# 9. Fixed-Schedule Baseline
# ---------------------------------------------------------------------------

class FixedScheduleBaseline:
    """
    Industry standard: work for work_duration steps, rest for rest_duration steps.
    E.g., 15 min work, 5 min rest (work_duration=15, rest_duration=5).
    """

    def __init__(self, n_tasks: int, work_duration: int = 15, rest_duration: int = 5,
                 work_task_idx: int = 0):
        self.n_tasks = n_tasks
        self.work_duration = work_duration
        self.rest_duration = rest_duration
        self.work_task_idx = work_task_idx
        self.rest_task_idx = n_tasks - 1
        self.step_count = 0
        self.name = "Fixed-Schedule"

    def get_actions(self, observations: Dict[str, np.ndarray], **kwargs) -> Dict[str, int]:
        cycle = self.work_duration + self.rest_duration
        phase = self.step_count % cycle
        self.step_count += 1

        if phase < self.work_duration:
            task = self.work_task_idx
        else:
            task = self.rest_task_idx

        return {agent: task for agent in observations}


# ---------------------------------------------------------------------------
# Helper: create all baselines
# ---------------------------------------------------------------------------

def create_all_baselines(
    obs_dim: int, n_actions: int, n_muscles: int = 3, seed: int = 42
) -> List:
    """Instantiate all 9 baseline methods."""
    return [
        RandomBaseline(n_actions, seed),
        RoundRobinBaseline(n_actions),
        GreedyBaseline(best_task_idx=0),
        GreedySafeBaseline(n_actions, n_muscles, mf_threshold=0.5),
        PPOUnconstrainedBaseline(obs_dim, n_actions, seed=seed),
        PPOLagrangianBaseline(obs_dim, n_actions, n_muscles, seed=seed),
        MAPPONoFilterBaseline(obs_dim, n_actions, seed=seed),
        MAPPOLagrangianBaseline(obs_dim, n_actions, n_muscles, seed=seed),
        FixedScheduleBaseline(n_actions, work_duration=15, rest_duration=5),
    ]
