"""Actor-Critic networks for PPO (MLP and CNN variants)."""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """MLP actor-critic with separate actor/critic trunks.

    obs -> actor trunk -> action mean + learnable log_std
    obs -> critic trunk -> state value
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(obs)
        action_std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


# ─── CNN encoder for pixel observations ──────────────────────

class CNNEncoder(nn.Module):
    """Conv encoder: pixel images -> latent vector.

    Input:  (batch, channels, 84, 84)  e.g. channels=9 for 3-frame RGB stack
    Output: (batch, latent_dim)
    """

    def __init__(self, in_channels: int = 9, latent_dim: int = 256) -> None:
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, latent_dim),
            nn.ReLU(),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.convs(x)       # (batch, 64, 7, 7)
        h = self.flatten(h)     # (batch, 3136)
        h = self.fc(h)          # (batch, 256)
        return h


class PixelActorCritic(nn.Module):
    """Actor-Critic that takes pixel observations through a CNN encoder."""

    def __init__(
        self,
        in_channels: int = 9,
        act_dim: int = 8,
        latent_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.encoder = CNNEncoder(in_channels, latent_dim)

        self.actor_mean = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )

        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self._initialize_heads()

    def _initialize_heads(self) -> None:
        for module in self.actor_mean:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)
        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        return self.encoder(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        latent = self._encode(obs)

        action_mean = self.actor_mean(latent)
        action_std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(latent).squeeze(-1)

        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self._encode(obs)
        return self.critic(latent).squeeze(-1)
