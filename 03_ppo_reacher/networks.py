"""Actor and critic networks for PPO Reacher."""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Gaussian policy: maps observations to Normal(mean, std) over actions.

    Mean is input-dependent; std is a learned parameter (input-independent).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, act_dim)

        # log-space so std > 0 after exp(); init ~0.6 to avoid over-exploration
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor):
        """Return (mean, std) of the action distribution."""
        features = self.shared_net(obs)
        mean = self.mean_head(features)
        # clamp to [~0.14, 1.0] — prevents collapse or blow-up
        log_std_clamped = torch.clamp(self.log_std, min=-2.0, max=0.0)
        std = log_std_clamped.exp()
        return mean, std

    def get_action(self, obs: torch.Tensor):
        """Sample action + log_prob for rollout collection."""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor):
        """Re-evaluate stored action under current policy (for PPO ratio)."""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """State-value network V(s) for actor-critic."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


if __name__ == "__main__":
    OBS_DIM = 11  # Reacher-v5
    ACT_DIM = 2

    policy = PolicyNetwork(OBS_DIM, ACT_DIM)
    value_net = ValueNetwork(OBS_DIM)

    policy_params = sum(p.numel() for p in policy.parameters())
    value_params = sum(p.numel() for p in value_net.parameters())

    print("PolicyNetwork architecture:")
    print(policy)
    print(f"Trainable parameters: {policy_params:,}\n")

    print("ValueNetwork architecture:")
    print(value_net)
    print(f"Trainable parameters: {value_params:,}")

    dummy_obs = torch.randn(4, OBS_DIM)
    mean, std = policy(dummy_obs)
    values = value_net(dummy_obs)
    print(f"\nDummy forward — mean: {mean.shape}, "
          f"std: {std.shape}, values: {values.shape}")
