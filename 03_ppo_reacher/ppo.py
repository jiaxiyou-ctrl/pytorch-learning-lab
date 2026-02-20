"""Core PPO algorithm components.

This module contains three pure functions that implement the three
phases of a PPO iteration:

1. collect_experience — rollout collection under the current policy
2. compute_advantages — GAE advantage and return computation
3. ppo_update         — PPO-Clip policy and value network update
"""

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

    Runs the policy in the environment for ``num_steps`` steps, handling
    episode resets automatically. All tensors are detached from the
    computation graph.

    Args:
        env:       A Gymnasium environment instance.
        policy:    The actor network used to sample actions.
        value_net: The critic network used to estimate state values.
        num_steps: Number of environment steps to collect.

    Returns:
        A dictionary with keys:
            ``obs``        — list of numpy observation arrays
            ``actions``    — list of numpy action arrays
            ``rewards``    — list of scalar rewards
            ``log_probs``  — list of scalar log-probabilities
            ``values``     — list of scalar value estimates
            ``dones``      — list of bool episode-termination flags
            ``last_value`` — scalar value estimate for the final state
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

        # Clip actions to the valid range before stepping.
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
    """Compute per-step advantages and discounted returns via GAE.

    Implements Generalized Advantage Estimation (GAE-lambda):
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t     = sum_{l=0}^{inf} (gamma * lam)^l * delta_{t+l}

    Episode boundaries (``done == True``) mask out the bootstrap value
    from the next state, ensuring advantages do not span episodes.

    Args:
        rewards:    List of per-step rewards from ``collect_experience``.
        values:     List of per-step value estimates from ``collect_experience``.
        dones:      List of per-step episode-end flags.
        last_value: Value estimate for the state after the final step.
        gamma:      Discount factor. Default: 0.99.
        lam:        GAE lambda (trade-off between bias and variance). Default: 0.95.

    Returns:
        advantages: Per-step advantage estimates.
        returns:    Per-step target returns (advantages + values), used as
                    regression targets for the value network.
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
    """Update the policy and value networks using the PPO-Clip objective.

    Performs ``update_epochs`` passes over the collected batch, sampling
    random mini-batches of size ``mini_batch_size`` each pass.

    Policy loss (PPO-Clip):
        L_CLIP = E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]
        where r_t = pi_theta(a|s) / pi_theta_old(a|s)

    An entropy bonus (coefficient 0.01) is subtracted from the policy
    loss to encourage exploration.

    Value loss (clipped MSE, returns normalised before regression):
        L_VF = 0.5 * E[ max((V(s_t) - R_t)^2,
                            (clip(V(s_t), V_old - eps, V_old + eps) - R_t)^2) ]

    Args:
        policy:            Actor network.
        value_net:         Critic network.
        policy_optimizer:  Optimizer for the actor.
        value_optimizer:   Optimizer for the critic.
        batch:             Dictionary produced by ``collect_experience``,
                           augmented with ``advantages`` and ``returns`` keys.
        clip_epsilon:      PPO clipping range. Default: 0.2.
        update_epochs:     Number of passes over the batch. Default: 10.
        mini_batch_size:   Mini-batch size. Default: 64.

    Returns:
        avg_policy_loss: Mean policy loss over all mini-batch updates.
        avg_value_loss:  Mean value loss over all mini-batch updates.
    """
    obs_tensor = torch.FloatTensor(np.array(batch["obs"]))
    act_tensor = torch.FloatTensor(np.array(batch["actions"]))
    old_log_prob_tensor = torch.FloatTensor(np.array(batch["log_probs"]))
    old_values_tensor = torch.FloatTensor(np.array(batch["values"]))
    adv_tensor = torch.FloatTensor(np.array(batch["advantages"]))
    ret_tensor = torch.FloatTensor(np.array(batch["returns"]))

    # Normalize advantages to zero mean and unit variance.
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    # Normalize returns so the value network regresses a well-scaled target,
    # which prevents the value loss from exploding when rewards are large or sparse.
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

            # --- Policy update (PPO-Clip) ---
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

            # --- Value update (clipped MSE regression) ---
            # Normalise old values to the same scale as the normalised return target
            # so the clip range is meaningful across different reward magnitudes.
            b_old_val_normalized = (b_old_val - returns_mean) / returns_std
            value_pred = value_net(b_obs)
            value_pred_clipped = b_old_val_normalized + torch.clamp(
                value_pred - b_old_val_normalized, -clip_epsilon, clip_epsilon
            )
            value_loss_unclipped = (value_pred - b_ret).pow(2)
            value_loss_clipped = (value_pred_clipped - b_ret).pow(2)
            # Take the pessimistic (max) loss to prevent the value from moving
            # too far from the old prediction in a single update.
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
