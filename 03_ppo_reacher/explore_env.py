"""Explore Reacher-v5: print obs/action spaces and run fixed-action experiments."""

import argparse

import numpy as np
import gymnasium as gym


OBS_LABELS = [
    "cos(joint0_angle)",
    "cos(joint1_angle)",
    "sin(joint0_angle)",
    "sin(joint1_angle)",
    "target_x",
    "target_y",
    "joint0_angular_vel",
    "joint1_angular_vel",
    "fingertip_target_dx",
    "fingertip_target_dy",
    "fingertip_target_dz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore the Reacher-v5 observation and action spaces."
    )
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def print_observation_space(env) -> None:
    obs_space = env.observation_space
    print("=" * 60)
    print("Observation Space")
    print("=" * 60)
    print(f"  Type      : {obs_space}")
    print(f"  Dimension : {obs_space.shape[0]}")
    print(f"  Low       : {obs_space.low[:4]} ...")
    print(f"  High      : {obs_space.high[:4]} ...\n")

    observation, _ = env.reset()
    print("  Index  Value     Description")
    print("  " + "-" * 52)
    for i, label in enumerate(OBS_LABELS):
        if i < len(observation):
            print(f"  [{i:2d}]   {observation[i]:+.4f}   {label}")
    print()


def print_action_space(env) -> None:
    act_space = env.action_space
    print("=" * 60)
    print("Action Space")
    print("=" * 60)
    print(f"  Type      : {act_space}")
    print(f"  Dimension : {act_space.shape[0]}")
    print(f"  Action[0] : joint0 torque  range [{act_space.low[0]}, {act_space.high[0]}]")
    print(f"  Action[1] : joint1 torque  range [{act_space.low[1]}, {act_space.high[1]}]")
    print()


def run_experiments(env) -> None:
    """Run fixed-action experiments to characterize dynamics."""
    experiments = [
        {"name": "No movement (zero torque)",         "action": [0.0,  0.0]},
        {"name": "Shoulder only (clockwise)",          "action": [1.0,  0.0]},
        {"name": "Elbow only (clockwise)",             "action": [0.0,  1.0]},
        {"name": "Opposite directions (shoulder CW, elbow CCW)", "action": [1.0, -1.0]},
    ]
    steps_per_exp = 100

    print("=" * 60)
    print("Fixed-Action Experiments")
    print("=" * 60)

    for exp in experiments:
        observation, _ = env.reset()
        action = np.array(exp["action"], dtype=np.float32)
        total_reward = 0.0

        for _ in range(steps_per_exp):
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                observation, _ = env.reset()

        print(f"\n  [{exp['name']}]")
        print(f"    Action                : {exp['action']}")
        print(f"    joint0 angular vel    : {observation[6]:+.4f}")
        print(f"    joint1 angular vel    : {observation[7]:+.4f}")
        print(f"    fingertip-target dx   : {observation[8]:+.4f}")
        print(f"    fingertip-target dy   : {observation[9]:+.4f}")
        print(f"    Cumulative reward     : {total_reward:.2f}")

    print()


def compare_random_vs_still(env) -> None:
    """Random policy vs zero-action baseline, 50 steps each."""
    print("=" * 60)
    print("Random Policy vs. No-Action Baseline (50 steps each)")
    print("=" * 60)

    observation, _ = env.reset()
    random_reward = 0.0
    for _ in range(50):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        random_reward += reward
        if terminated or truncated:
            observation, _ = env.reset()

    observation, _ = env.reset()
    still_reward = 0.0
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(50):
        observation, reward, terminated, truncated, _ = env.step(zero_action)
        still_reward += reward
        if terminated or truncated:
            observation, _ = env.reset()

    print(f"\n  Random policy (50 steps)  : cumulative reward = {random_reward:.4f}")
    print(f"  Zero action  (50 steps)  : cumulative reward = {still_reward:.4f}")
    print()


def print_reward_explanation() -> None:
    print("=" * 60)
    print("Reward Function")
    print("=" * 60)
    print("""
  reward = - dist(fingertip, target) - ctrl_cost

  dist: Euclidean distance between fingertip and target.
  ctrl_cost: penalty proportional to squared torque norm.
  Optimal: reach target with minimal torque.
""")


def main() -> None:
    args = parse_args()
    render_mode = "human" if args.render else None

    env = gym.make("Reacher-v5", render_mode=render_mode)

    print_observation_space(env)
    print_action_space(env)
    run_experiments(env)
    compare_random_vs_still(env)
    print_reward_explanation()

    env.close()


if __name__ == "__main__":
    main()
