"""Neural network definitions for the PPO Reacher agent.

Contains the actor (PolicyNetwork) and critic (ValueNetwork) used in
the actor-critic PPO implementation.
"""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Gaussian policy network (actor) for continuous action spaces.

    Maps observations to a factored Gaussian distribution over actions.
    The mean is input-dependent; the standard deviation is a learned
    parameter independent of the input.

    Args:
        obs_dim: Dimensionality of the observation space.
        act_dim: Dimensionality of the action space.
        hidden_dim: Number of units in each hidden layer. Default: 64.
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

        # log_std is a learnable parameter, not input-dependent.
        # Parameterising in log-space ensures std > 0 after exp().
        # Initialise at -0.5 so starting std ≈ 0.6 (avoids early over-exploration).
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor):
        """Compute the distribution parameters for the given observations.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim).

        Returns:
            mean: Action means, shape (batch_size, act_dim).
            std:  Action standard deviations, shape (act_dim,).
        """
        features = self.shared_net(obs)
        mean = self.mean_head(features)
        # Clamp log_std to prevent std from collapsing or exploding.
        # Range [-2, 0] corresponds to std in [~0.14, 1.0] — tighter upper bound
        # stops the policy from staying too stochastic and destabilising training.
        log_std_clamped = torch.clamp(self.log_std, min=-2.0, max=0.0)
        std = log_std_clamped.exp()
        return mean, std

    def get_action(self, obs: torch.Tensor):
        """Sample an action and compute its log-probability.

        Used during environment interaction (rollout collection).

        Args:
            obs: Observation tensor of shape (obs_dim,) or (batch_size, obs_dim).

        Returns:
            action:   Sampled action tensor.
            log_prob: Sum of log-probabilities across action dimensions.
        """
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor):
        """Re-evaluate a stored action under the current policy.

        Used during the PPO update step to compute the probability ratio
        between the new and old policies.

        Args:
            obs:    Observation tensor of shape (batch_size, obs_dim).
            action: Action tensor of shape (batch_size, act_dim).

        Returns:
            log_prob: Log-probability of the action under the current policy.
            entropy:  Differential entropy of the current distribution.
        """
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """State-value network (critic) for actor-critic methods.

    Estimates V(s) — the expected discounted return from state s.

    Args:
        obs_dim:    Dimensionality of the observation space.
        hidden_dim: Number of units in each hidden layer. Default: 64.
    """

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
        """Predict the state value for the given observations.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim).

        Returns:
            Scalar value estimate per observation, shape (batch_size,).
        """
        return self.net(obs).squeeze(-1)


if __name__ == "__main__":
    OBS_DIM = 11  # Reacher-v5 observation dimension
    ACT_DIM = 2   # Reacher-v5 action dimension

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

    # Smoke test with a dummy batch
    dummy_obs = torch.randn(4, OBS_DIM)
    mean, std = policy(dummy_obs)
    values = value_net(dummy_obs)
    print(f"\nDummy forward pass — mean shape: {mean.shape}, "
          f"std shape: {std.shape}, values shape: {values.shape}")
