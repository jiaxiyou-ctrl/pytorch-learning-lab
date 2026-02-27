"""Core PPO algorithm: rollout collection, GAE, PPO-Clip update."""

from typing import Dict, Tuple, List

import numpy as np
import torch

from networks import PolicyNetwork, ValueNetwork


def collect_experience(
    env,
    policy: PolicyNetwork,
    value_net: ValueNetwork,
    num_steps: int,
) -> Dict:
    """Collect a fixed-length trajectory under the current policy.

    Handles episode resets automatically. Returns dict with keys:
    obs, actions, rewards, log_probs, values, dones, last_value.
    """
    all_obs: List = []
    all_actions: List = []
    all_rewards: List = []
    all_log_probs: List = []
    all_values: List = []
    all_dones: List = []

    observation, _ = env.reset()

    for _ in range(num_steps):
        obs_tensor = torch.FloatTensor(observation)

        with torch.no_grad():
            action, log_prob = policy.get_action(obs_tensor)
            value = value_net(obs_tensor)

        action_np = np.clip(action.numpy(), env.action_space.low, env.action_space.high)

        all_obs.append(observation.copy())
        all_actions.append(action_np.copy())
        all_log_probs.append(log_prob.item())
        all_values.append(value.item())

        observation, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        all_rewards.append(reward)
        all_dones.append(done)

        if done:
            observation, _ = env.reset()

    with torch.no_grad():
        last_value = value_net(torch.FloatTensor(observation)).item()

    return {
        "obs": all_obs,
        "actions": all_actions,
        "rewards": all_rewards,
        "log_probs": all_log_probs,
        "values": all_values,
        "dones": all_dones,
        "last_value": last_value,
    }


def compute_advantages(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """GAE-lambda advantage and return computation.

    Episode boundaries (done=True) mask the bootstrap value.
    """
    advantages: List[float] = []
    gae = 0.0
    values_extended = values + [last_value]

    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values_extended[t + 1] * mask - values_extended[t]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def ppo_update(
    policy: PolicyNetwork,
    value_net: ValueNetwork,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    batch: Dict,
    clip_epsilon: float = 0.2,
    update_epochs: int = 10,
    mini_batch_size: int = 64,
) -> Tuple[float, float]:
    """PPO-Clip update over collected batch.

    Policy loss: clipped surrogate + entropy bonus (0.01).
    Value loss: clipped MSE with normalized returns.
    """
    obs_tensor = torch.FloatTensor(np.array(batch["obs"]))
    act_tensor = torch.FloatTensor(np.array(batch["actions"]))
    old_log_prob_tensor = torch.FloatTensor(np.array(batch["log_probs"]))
    old_values_tensor = torch.FloatTensor(np.array(batch["values"]))
    adv_tensor = torch.FloatTensor(np.array(batch["advantages"]))
    ret_tensor = torch.FloatTensor(np.array(batch["returns"]))

    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    # Normalize returns so value loss doesn't explode on large/sparse rewards
    returns_mean = ret_tensor.mean()
    returns_std = ret_tensor.std() + 1e-8
    ret_tensor_normalized = (ret_tensor - returns_mean) / returns_std

    total_policy_loss = 0.0
    total_value_loss = 0.0
    update_count = 0
    n = len(batch["obs"])

    for _ in range(update_epochs):
        indices = np.random.permutation(n)

        for start in range(0, n, mini_batch_size):
            batch_idx = indices[start: start + mini_batch_size]

            b_obs = obs_tensor[batch_idx]
            b_act = act_tensor[batch_idx]
            b_old_log_prob = old_log_prob_tensor[batch_idx]
            b_old_val = old_values_tensor[batch_idx]
            b_adv = adv_tensor[batch_idx]
            b_ret = ret_tensor_normalized[batch_idx]

            # Policy update
            new_log_prob, entropy = policy.evaluate_action(b_obs, b_act)
            ratio = (new_log_prob - b_old_log_prob).exp()

            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            policy_loss = policy_loss - 0.01 * entropy.mean()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            policy_optimizer.step()

            # Value update (clipped MSE)
            b_old_val_normalized = (b_old_val - returns_mean) / returns_std
            value_pred = value_net(b_obs)
            value_pred_clipped = b_old_val_normalized + torch.clamp(
                value_pred - b_old_val_normalized, -clip_epsilon, clip_epsilon
            )
            value_loss_unclipped = (value_pred - b_ret).pow(2)
            value_loss_clipped = (value_pred_clipped - b_ret).pow(2)
            # Pessimistic (max) prevents value from jumping too far per update
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
            value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            update_count += 1

    avg_policy_loss = total_policy_loss / max(update_count, 1)
    avg_value_loss = total_value_loss / max(update_count, 1)
    return avg_policy_loss, avg_value_loss
