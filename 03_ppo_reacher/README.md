# Reacher-v5 PPO Agent

A Proximal Policy Optimization (PPO) agent trained to control a 2-DOF robotic arm to reach target positions in the MuJoCo Reacher-v5 environment.

---

## Overview

This project implements PPO from scratch using PyTorch to solve the Reacher-v5 continuous control task. The agent learns to coordinate two joint torques to move a fingertip to a randomly placed target.

The implementation follows the original PPO paper (Schulman et al., 2017) and uses Generalized Advantage Estimation (GAE) for variance reduction.

---

## Project Structure

```
03_ppo_reacher/
├── networks.py      # PolicyNetwork (actor) and ValueNetwork (critic)
├── ppo.py           # PPO algorithm: rollout collection, GAE, PPO-Clip update
├── train.py         # Training loop with argparse and result saving
├── evaluate.py      # Model evaluation and GIF recording
├── explore_env.py   # Environment exploration and reward structure analysis
└── results/
    ├── trained_model.pth           # Saved model checkpoint
    ├── training_reward_curve.png   # Reward curve from training
    └── trained_agent.gif           # Recorded evaluation episode
```

---

## Results

### Training Reward Curve

![Training Reward Curve](results/training_reward_curve.png)

### Trained Agent Demo

![Trained Agent](results/trained_agent.gif)

---

## Usage

### Explore the Environment

```bash
python explore_env.py            # headless (fast)
python explore_env.py --render   # open a render window
```

### Train the Agent

```bash
python train.py --num_iterations 200 --steps_per_iter 2048
```

All hyperparameters are configurable via command-line flags:

| Flag | Default | Description |
|---|---|---|
| `--num_iterations` | 200 | Number of training iterations |
| `--steps_per_iter` | 2048 | Environment steps per iteration |
| `--policy_lr` | 3e-4 | Actor learning rate |
| `--value_lr` | 1e-3 | Critic learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--lam` | 0.95 | GAE lambda |
| `--clip_epsilon` | 0.2 | PPO-Clip epsilon |
| `--update_epochs` | 10 | Update epochs per iteration |
| `--mini_batch_size` | 64 | Mini-batch size |
| `--save_dir` | `results/` | Output directory |

### Evaluate and Record GIF

```bash
# Silent evaluation
python evaluate.py --model_path results/trained_model.pth

# Live rendering
python evaluate.py --model_path results/trained_model.pth --render

# Save a GIF
python evaluate.py --model_path results/trained_model.pth --save_gif
```

---

## Requirements

```
Python 3.8+
gymnasium[mujoco]
torch
numpy
matplotlib
Pillow
```

---

## References

- Schulman et al. — [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) (2017)
- Schulman et al. — [*High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438) (2016)
- [Gymnasium Reacher-v5 Documentation](https://gymnasium.farama.org/environments/mujoco/reacher/)
