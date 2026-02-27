# PyTorch ML Portfolio

From-scratch implementations of core ML systems in PyTorch, covering reinforcement learning, NLP, and retrieval-augmented generation.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.0-4A90D9)](https://mujoco.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Overview

| Project | Task | Key Result |
| :-------- | :----- | :----------- |
| [Q-Learning Maze](#1-q-learning-maze-solver) | Tabular RL on 4x4 grid | Optimal policy in ~300 episodes |
| [Transformer Basics](#2-transformer-basics-pytorch--hf-integration) | Self-attention from scratch | Custom encoder block + HuggingFace inference |
| [PPO Reacher](#3-ppo-reacher-agent-mujoco) | 2-DOF robotic arm control | Converged policy via PPO-Clip + GAE |
| [RAG Assistant](#4-local-rag-assistant) | Local retrieval-augmented QA | End-to-end pipeline: ChromaDB + TinyLlama |
| [PPO Ant Walker](#5-ppo-ant-walker-mujoco) | Quadruped locomotion (8-DOF) | Peak reward +632, obs/reward norm, LR annealing |

---

## Projects

### [1) Q-Learning Maze Solver](./01_q_learning_maze/)

Tabular Q-learning agent with epsilon-greedy exploration that learns to navigate a 4x4 grid world with walls and sparse reward. Environment sim in `01_q_learning_maze/maze_env.py`, Q-value update and policy in `q_learning_agent.py`.

| Training Curve | Agent Demo |
| :---: | :---: |
| ![Reward Curve](01_q_learning_maze/results/training_reward_curve.png) | ![Maze Agent](01_q_learning_maze/results/maze_agent.gif) |

---

### [2) Transformer Basics (PyTorch + HF Integration)](./02_transformer_basics/)

From-scratch `SimpleSelfAttention` and `TransformerBlock` in `02_transformer_basics/transformer_block.py`, plus inference scripts using DistilBERT for sentiment (`sentiment_analysis.py`) and GPT-2 for generation (`text_generation.py`).

---

### [3) PPO Reacher Agent (MuJoCo)](./03_ppo_reacher/)

PPO-Clip + GAE training pipeline for a 2-DOF robotic arm in Reacher-v5. Separate actor/critic networks with Gaussian policy, clipped value loss, and LR annealing. Core PPO logic in `03_ppo_reacher/ppo.py`, networks in `networks.py`.

| Training Curve | Agent Demo |
| :---: | :---: |
| ![Reward Curve](03_ppo_reacher/results/training_reward_curve.png) | ![Trained Agent](03_ppo_reacher/results/trained_agent.gif) |

---

### [4) Local RAG Assistant](./04rag-assistant/)

Full RAG stack (load, chunk, embed, retrieve, generate) using LangChain + ChromaDB + TinyLlama, all running locally without API keys. Modular pipeline across `04rag-assistant/src/step*.py`. Indexes ~20 chunks from a source document, end-to-end in ~52s.

---

### [5) PPO Ant Walker (MuJoCo)](./05_mujoco_ant/)

PPO for high-dimensional locomotion (8 actuators, 105-dim observations) on Ant-v5. Reaches peak reward +632 after 5M steps. Key additions over vanilla PPO: Welford observation normalization (`obs_normalizer.py`), reward normalization (`reward_normalizer.py`), linear LR annealing, and optional domain randomization (`domain_random.py`) for sim-to-real robustness.

| Training Curve | Agent Demo |
| :---: | :---: |
| ![Reward Curve](05_mujoco_ant/results/training_reward_curve.png) | ![Ant Walking](05_mujoco_ant/results/ant_walking.gif) |

---

## Quick Start

**Prerequisites:** Python 3.9+

```bash
git clone https://github.com/YOUR_USERNAME/pytorch-learning-lab.git
cd pytorch-learning-lab
pip install -r requirements.txt
```

> `torch` installation can vary by OS/CUDA. Use the selector at [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Repository Layout

```text
pytorch-learning-lab/
├── 01_q_learning_maze/
│   ├── maze_env.py                # Grid world environment
│   ├── q_learning_agent.py        # Q-table agent with ε-greedy
│   ├── train.py                   # Training loop
│   └── results/                   # Reward curve + agent GIF
├── 02_transformer_basics/
│   ├── transformer_block.py       # Self-attention + encoder block
│   ├── sentiment_analysis.py      # DistilBERT inference
│   └── text_generation.py         # GPT-2 generation
├── 03_ppo_reacher/
│   ├── networks.py                # Actor-Critic networks
│   ├── ppo.py                     # PPO-Clip + GAE
│   ├── train.py                   # Training loop
│   ├── evaluate.py                # Evaluation + recording
│   └── results/                   # Reward curve + agent GIF
├── 04rag-assistant/
│   └── src/                       # Modular RAG pipeline steps
├── 05_mujoco_ant/
│   ├── networks.py                # ActorCritic (shared encoder)
│   ├── ppo_agent.py               # PPO agent with buffer
│   ├── obs_normalizer.py          # Welford running normalization
│   ├── reward_normalizer.py       # Reward variance scaling
│   ├── domain_random.py           # Sim-to-real randomization
│   ├── train.py                   # Training with checkpoint/resume
│   ├── record.py                  # MP4/GIF recording
│   └── results/                   # Reward curve + agent GIF
├── requirements.txt
└── README.md
```

---

## References

**Reinforcement Learning**
- Sutton & Barto — [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html)
- Schulman et al. — [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347)
- Schulman et al. — [*Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438)

**NLP & Transformers**
- Vaswani et al. — [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)

**Retrieval-Augmented Generation**
- Lewis et al. — [*Retrieval-Augmented Generation*](https://arxiv.org/abs/2005.11401)

**Documentation**
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [MuJoCo](https://mujoco.readthedocs.io/)
- [Gymnasium Ant-v5](https://gymnasium.farama.org/environments/mujoco/ant/)

---

## License

This repository is licensed under the [MIT License](https://opensource.org/licenses/MIT).
