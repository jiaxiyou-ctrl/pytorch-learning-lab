"""Main training script for the PPO Reacher agent.

Trains a PPO agent on the MuJoCo Reacher-v5 environment and saves the
trained model checkpoint and reward curve to the results directory.

Example:
    python train.py --num_iterations 200 --steps_per_iter 2048
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
import gymnasium as gym

from networks import PolicyNetwork, ValueNetwork
from ppo import collect_experience, compute_advantages, ppo_update


def parse_args() -> argparse.Namespace:
    """Parse command-line hyperparameters."""
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on Reacher-v5."
    )
    parser.add_argument("--num_iterations", type=int, default=200,
                        help="Number of training iterations.")
    parser.add_argument("--steps_per_iter", type=int, default=2048,
                        help="Environment steps collected per iteration.")
    parser.add_argument("--policy_lr", type=float, default=3e-4,
                        help="Learning rate for the policy network.")
    parser.add_argument("--value_lr", type=float, default=3e-4,
                        help="Learning rate for the value network.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--lam", type=float, default=0.95,
                        help="GAE lambda.")
    parser.add_argument("--clip_epsilon", type=float, default=0.2,
                        help="PPO-Clip epsilon.")
    parser.add_argument("--update_epochs", type=int, default=4,
                        help="Number of update epochs per iteration.")
    parser.add_argument("--mini_batch_size", type=int, default=64,
                        help="Mini-batch size for PPO updates.")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"),
                        help="Directory for saving model and plots.")
    return parser.parse_args()


def save_reward_curve(
    avg_reward_history: list,
    total_reward_history: list,
    save_path: str,
) -> None:
    """Plot and save the training reward curve.

    Produces a two-panel figure:
    - Top panel: per-step average reward with 10-iteration smoothed trend.
    - Bottom panel: total episode reward per iteration (sum over steps_per_iter steps).

    Args:
        avg_reward_history:   Average per-step reward per iteration.
        total_reward_history: Total reward summed over all steps per iteration.
        save_path:            File path for the output PNG.
    """
    window = 10
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top panel: per-step average reward ---
    ax1.plot(avg_reward_history, alpha=0.4, color="steelblue", label="Raw Reward")
    if len(avg_reward_history) >= window:
        smoothed = np.convolve(
            avg_reward_history, np.ones(window) / window, mode="valid"
        )
        ax1.plot(
            range(window - 1, len(avg_reward_history)),
            smoothed,
            color="crimson",
            linewidth=2,
            label=f"Smoothed ({window}-iter avg)",
        )
    ax1.set_ylabel("Avg Reward per Step")
    ax1.set_title("PPO Training on Reacher-v5")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Bottom panel: total reward per iteration ---
    ax2.plot(total_reward_history, alpha=0.4, color="darkorange", label="Total Reward")
    if len(total_reward_history) >= window:
        smoothed_total = np.convolve(
            total_reward_history, np.ones(window) / window, mode="valid"
        )
        ax2.plot(
            range(window - 1, len(total_reward_history)),
            smoothed_total,
            color="darkgreen",
            linewidth=2,
            label=f"Smoothed ({window}-iter avg)",
        )
    ax2.set_xlabel("Training Iteration")
    ax2.set_ylabel("Total Reward (sum over steps)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run the full PPO training loop."""
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    env = gym.make("Reacher-v5")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = PolicyNetwork(obs_dim, act_dim)
    value_net = ValueNetwork(obs_dim)

    policy_optimizer = optim.Adam(policy.parameters(), lr=args.policy_lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=args.value_lr)

    # Linearly anneal both learning rates to 10 % of their starting value
    # by the final iteration, giving aggressive early updates and fine-tuning later.
    policy_scheduler = LinearLR(
        policy_optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=args.num_iterations,
    )
    value_scheduler = LinearLR(
        value_optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=args.num_iterations,
    )

    print("=" * 60)
    print("PPO Training — Reacher-v5")
    print("=" * 60)
    print(f"  Iterations   : {args.num_iterations}")
    print(f"  Steps / iter : {args.steps_per_iter}")
    print(f"  Total steps  : {args.num_iterations * args.steps_per_iter:,}")
    print(f"  Policy LR    : {args.policy_lr}")
    print(f"  Value LR     : {args.value_lr}")
    print(f"  Gamma / Lambda: {args.gamma} / {args.lam}")
    print(f"  Clip epsilon : {args.clip_epsilon}")
    print("=" * 60)

    avg_reward_history = []
    total_reward_history = []

    for iteration in range(args.num_iterations):
        # --- Rollout ---
        experience = collect_experience(
            env, policy, value_net, args.steps_per_iter
        )

        # --- GAE ---
        advantages, returns = compute_advantages(
            experience["rewards"],
            experience["values"],
            experience["dones"],
            experience["last_value"],
            gamma=args.gamma,
            lam=args.lam,
        )
        experience["advantages"] = advantages
        experience["returns"] = returns

        # --- PPO update ---
        avg_p_loss, avg_v_loss = ppo_update(
            policy,
            value_net,
            policy_optimizer,
            value_optimizer,
            experience,
            clip_epsilon=args.clip_epsilon,
            update_epochs=args.update_epochs,
            mini_batch_size=args.mini_batch_size,
        )

        avg_reward = float(np.mean(experience["rewards"]))
        total_reward = float(np.sum(experience["rewards"]))
        avg_reward_history.append(avg_reward)
        total_reward_history.append(total_reward)

        policy_scheduler.step()
        value_scheduler.step()

        if (iteration + 1) % 10 == 0:
            recent_avg = float(np.mean(avg_reward_history[-10:]))
            log_std_clamped = torch.clamp(policy.log_std, min=-2.0, max=0.0)
            std_values = log_std_clamped.exp().detach().numpy()
            print(
                f"Iteration {iteration + 1:3d}/{args.num_iterations} | "
                f"Reward: {recent_avg:.4f} | "
                f"P_Loss: {avg_p_loss:.4f} | "
                f"V_Loss: {avg_v_loss:.4f} | "
                f"Std: [{std_values[0]:.3f}, {std_values[1]:.3f}]"
            )

    env.close()

    # --- Save model ---
    model_path = os.path.join(args.save_dir, "trained_model.pth")
    torch.save(
        {
            "policy": policy.state_dict(),
            "value_net": value_net.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
        },
        model_path,
    )

    # --- Save reward curve ---
    curve_path = os.path.join(args.save_dir, "training_reward_curve.png")
    save_reward_curve(avg_reward_history, total_reward_history, curve_path)

    print()
    print(f"Training complete. Model saved to {model_path}")
    print(f"Reward curve saved to {curve_path}")


if __name__ == "__main__":
    main()
