"""PPO agent wrapping ActorCritic network, optimizer, and rollout buffer."""

from typing import Tuple

import numpy as np
import torch
from torch import nn

from networks import ActorCritic
from ppo_buffer import PPOBuffer


class PPOAgent:
    """PPO-Clip agent for continuous control."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
    ) -> None:
        self.network = ActorCritic(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = PPOBuffer(buffer_size, obs_dim, act_dim)

        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def select_action(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Choose action given normalized obs (no grad). Returns (action, log_prob, value)."""
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action, log_prob, _, value = self.network.get_action_and_value(
                obs_tensor
            )
            return action.numpy(), log_prob.item(), value.item()

    def update(self) -> None:
        """Run PPO-Clip updates on the current buffer, then reset it."""
        for _epoch in range(self.update_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                obs = batch["observations"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                _, new_log_probs, entropy, new_values = (
                    self.network.get_action_and_value(obs, actions)
                )

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1 - self.clip_range, 1 + self.clip_range
                    )
                    * advantages
                )

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (new_values - returns).pow(2).mean()
                entropy_loss = -entropy.mean()
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

        self.buffer.reset()
