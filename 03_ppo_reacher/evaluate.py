"""Evaluate a trained PPO Reacher agent. Optionally render or save GIF."""

import argparse
import os

import numpy as np
import torch
import gymnasium as gym

from networks import PolicyNetwork, ValueNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent on Reacher-v5."
    )
    parser.add_argument(
        "--model_path", type=str,
        default=os.path.join(os.path.dirname(__file__), "results", "trained_model.pth"),
    )
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument(
        "--gif_path", type=str,
        default=os.path.join(os.path.dirname(__file__), "results", "trained_agent.gif"),
    )
    return parser.parse_args()


def load_model(model_path: str):
    """Load checkpoint, return (policy, obs_dim, act_dim)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Run train.py first to generate a checkpoint."
        )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    obs_dim = checkpoint["obs_dim"]
    act_dim = checkpoint["act_dim"]

    policy = PolicyNetwork(obs_dim, act_dim)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    return policy, obs_dim, act_dim


def run_episode(env, policy: PolicyNetwork) -> tuple:
    """Run one greedy episode (deterministic mean action)."""
    observation, _ = env.reset()
    total_reward = 0.0
    steps = 0
    frames = []

    render_mode = getattr(env, "render_mode", None)

    done = False
    while not done:
        obs_tensor = torch.FloatTensor(observation)
        with torch.no_grad():
            mean, _ = policy(obs_tensor)
            action_np = mean.numpy()

        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
        observation, reward, terminated, truncated, _ = env.step(action_np)
        total_reward += reward
        steps += 1
        done = terminated or truncated

        if render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    return total_reward, steps, frames


def save_gif(frames: list, gif_path: str, fps: int = 30) -> None:
    """Save RGB frames as animated GIF."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow required for GIF: pip install Pillow")

    os.makedirs(os.path.dirname(os.path.abspath(gif_path)), exist_ok=True)

    pil_frames = [Image.fromarray(frame) for frame in frames]
    duration_ms = int(1000 / fps)

    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
    )


def main() -> None:
    args = parse_args()

    policy, obs_dim, act_dim = load_model(args.model_path)

    print("=" * 60)
    print("PPO Evaluation — Reacher-v5")
    print("=" * 60)
    print(f"  Model      : {args.model_path}")
    print(f"  Episodes   : {args.num_episodes}")
    print(f"  Obs / Act  : {obs_dim} / {act_dim}")
    print("=" * 60)

    if args.render:
        render_mode = "human"
    elif args.save_gif:
        render_mode = "rgb_array"
    else:
        render_mode = None

    env = gym.make("Reacher-v5", render_mode=render_mode)

    all_rewards = []
    all_frames = []

    for ep in range(args.num_episodes):
        reward, steps, frames = run_episode(env, policy)
        all_rewards.append(reward)
        all_frames.extend(frames)
        print(f"  Episode {ep + 1}/{args.num_episodes} | "
              f"Reward: {reward:.2f} | Steps: {steps}")

    env.close()

    avg_reward = float(np.mean(all_rewards))
    print(f"\n  Average Reward: {avg_reward:.2f} over {args.num_episodes} episodes")

    if args.save_gif and all_frames:
        save_gif(all_frames, args.gif_path)
        print(f"  GIF saved to {args.gif_path}")
    elif args.save_gif:
        print("  Warning: no frames captured.")


if __name__ == "__main__":
    main()
