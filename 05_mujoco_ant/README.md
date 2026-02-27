# 05 — PPO Ant Walker (MuJoCo)

Training a quadruped ant to walk using PPO with observation/reward normalization and learning rate annealing.

---

## Overview

This project implements Proximal Policy Optimization (PPO) from scratch using PyTorch to solve the MuJoCo Ant-v5 continuous control task. The agent controls 8 joint torques to make a four-legged ant walk forward as fast as possible while staying upright.

Key techniques beyond vanilla PPO include running observation normalization (Welford's algorithm), reward normalization, linear learning-rate annealing, and optional domain randomization for sim-to-real robustness.

---

## Architecture

```
                       +----------------------------+
                       |        ActorCritic         |
                       |                            |
  obs (105-dim)        |  +--------------------+    |
  -------------------->|  | Actor Trunk        |    |--> action mean (8-dim)
                       |  | Linear(105, 256)   |    |    + learnable log_std
                       |  | Tanh               |    |
                       |  | Linear(256, 256)   |    |
                       |  | Tanh               |    |
                       |  | Linear(256, 8)     |    |
                       |  +--------------------+    |
                       |                            |
                       |  +--------------------+    |
                       |  | Critic Trunk       |    |--> state value (scalar)
                       |  | Linear(105, 256)   |    |
                       |  | Tanh               |    |
                       |  | Linear(256, 256)   |    |
                       |  | Tanh               |    |
                       |  | Linear(256, 1)     |    |
                       |  +--------------------+    |
                       +----------------------------+
```

---

## Results

### Training Reward Curve

![Training Reward Curve](results/training_reward_curve.png)

### Performance Summary

| Metric | Value |
|---|---|
| Total timesteps | 5,000,000 |
| Total episodes | ~24,400 |
| Best mean reward | ~626 |
| Training time | ~37 minutes |
| Throughput | ~2,200 FPS |

---

## Project Structure

```
05_mujoco_ant/
├── networks.py             # ActorCritic network (actor + critic heads)
├── ppo_buffer.py           # PPO rollout buffer with GAE computation
├── ppo_agent.py            # PPO agent (select_action, update)
├── obs_normalizer.py       # Running observation normalizer (Welford)
├── reward_normalizer.py    # Running reward normalizer
├── domain_random.py        # Domain randomization for sim-to-real
├── train.py                # Training loop with checkpoint & resume
├── record.py               # Record trained agent as MP4 video
├── plot_training_curve.py  # Plot training reward curve from log
├── training_log.txt        # Training output log
├── README.md               # Project documentation
├── results/
│   ├── training_reward_curve.png
│   └── ant_walking.gif
├── checkpoints/            # (gitignored) saved model checkpoints
└── videos/                 # (gitignored) recorded videos
```

---

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Agent

```bash
# Fresh start (1M steps by default)
python train.py

# Custom number of timesteps
python train.py --timesteps 5000000

# Resume from a checkpoint
python train.py --resume checkpoints/ant_ppo_final.pt

# Enable domain randomization
python train.py --domain-randomization
```

### Record Video

```bash
python record.py --checkpoint checkpoints/ant_ppo_final.pt --episodes 3
```

### Plot Training Curve

```bash
python plot_training_curve.py --log training_log.txt
```

### Explore the Environment

```bash
python explore_env.py
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 3e-4 (linearly annealed to 0) |
| Discount (γ) | 0.99 |
| GAE lambda (λ) | 0.95 |
| Clip epsilon (ε) | 0.2 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |
| Max gradient norm | 0.5 |
| Buffer size (steps/update) | 2,048 |
| Mini-batch size | 64 |
| Update epochs per rollout | 10 |
| Hidden layer width | 256 |
| Activation | Tanh |

---

## What I Learned

- **Observation normalization is essential** — without it, the ant flips over immediately because sensor channels span wildly different ranges
- **Reward normalization stabilizes training** — raw rewards in Ant-v5 vary from −200 to 600+; normalizing keeps gradients on a consistent scale
- **LR annealing prevents late-training instability** — linearly decaying the learning rate lets the policy fine-tune without overshooting
- **PPO is sample-efficient for locomotion** — the ant learns to walk forward in ~500K steps and reaches strong performance by ~2M steps
- **Domain randomization adds robustness** — randomizing gravity, friction, and mass during training produces policies that transfer better across conditions

---

## References

- Schulman et al. — [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) (2017)
- Schulman et al. — [*High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438) (2016)
- [Gymnasium Ant-v5 Documentation](https://gymnasium.farama.org/environments/mujoco/ant/)
