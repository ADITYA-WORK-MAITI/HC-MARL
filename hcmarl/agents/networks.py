"""
HC-MARL Phase 3 (#34): Actor-Critic Neural Networks
MLP-based actor and critic with orthogonal initialization.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

def init_weights(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class ActorNetwork(nn.Module):
    """Decentralised actor: π(a|o_i) — maps local observation to action distribution."""
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.net.apply(lambda m: init_weights(m, np.sqrt(2)))
        init_weights(self.net[-1], gain=0.01)  # small init for output

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # logits

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()


class CriticNetwork(nn.Module):
    """Centralised critic: V(s) — maps global state to scalar value."""
    def __init__(self, global_obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.net.apply(lambda m: init_weights(m, np.sqrt(2)))
        init_weights(self.net[-1], gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class GaussianActorNetwork(nn.Module):
    """Continuous-action actor: pi(C|o_i) — maps local obs to per-muscle neural drive.

    Outputs mean and log_std of a squashed Gaussian (tanh-Normal).
    This implements Remark 7.2 of the math doc: "the RL agent replaces
    the proportional controller with a learned policy pi_theta(C | x, task)
    that outputs a continuous neural drive."

    Args:
        obs_dim: Local observation dimension per agent.
        action_dim: Number of continuous action dimensions (= n_muscles).
        hidden_dim: Hidden layer width.
    """
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.backbone.apply(lambda m: init_weights(m, np.sqrt(2)))
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        init_weights(self.mean_head, gain=0.01)
        init_weights(self.log_std_head, gain=0.01)

    def forward(self, obs: torch.Tensor):
        """Returns (mean, log_std) for the Gaussian policy."""
        h = self.backbone(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def get_action(self, obs: torch.Tensor):
        """Sample an action and return (action, log_prob, entropy).

        Action is squashed to [0, 1] via sigmoid (neural drive C in [0, 1]).
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        # Reparameterised sample
        x = dist.rsample()
        # Squash to [0, 1] via sigmoid
        action = torch.sigmoid(x)
        # Log-prob with sigmoid correction: log p(a) = log p(x) - log |da/dx|
        # da/dx = sigmoid(x) * (1 - sigmoid(x))
        log_prob = dist.log_prob(x) - torch.log(action * (1 - action) + 1e-8)
        log_prob = log_prob.sum(-1)  # sum over action dims
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """Evaluate log-prob of given actions.

        action is in [0, 1] (sigmoid-squashed); invert to get x.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        # Inverse sigmoid: x = log(a / (1 - a))
        action_clamped = action.clamp(1e-6, 1 - 1e-6)
        x = torch.log(action_clamped / (1 - action_clamped))
        log_prob = dist.log_prob(x) - torch.log(action_clamped * (1 - action_clamped) + 1e-8)
        log_prob = log_prob.sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy

    def get_action_with_log_std(self, obs: torch.Tensor):
        """Like get_action but additionally returns the mean log_std for
        diagnostic logging in the Appendix F continuous-mode probe.

        Used by HCMARLAgent._get_continuous_actions when in continuous mode
        to expose the Gaussian-actor exploration signal — verifies the
        policy is exploring (log_std not collapsed to LOG_STD_MIN, not
        diverged to LOG_STD_MAX). Returned log_std is detached (no grad).
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        action = torch.sigmoid(x)
        log_prob = dist.log_prob(x) - torch.log(action * (1 - action) + 1e-8)
        log_prob = log_prob.sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, log_std.mean().detach()


class CostCriticNetwork(nn.Module):
    """Cost critic for Lagrangian methods: V_c(s) — predicts cumulative cost."""
    def __init__(self, global_obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.net.apply(lambda m: init_weights(m, np.sqrt(2)))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
