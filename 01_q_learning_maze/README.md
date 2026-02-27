# 01 -- Q-Learning Maze Solver

Tabular Q-learning agent that learns to navigate a 4x4 grid world with two walls and sparse reward (+100 at goal). The agent uses epsilon-greedy exploration with a decaying schedule, and updates Q-values via the Bellman equation. Converges to the optimal policy in about 300 episodes.

---

## Architecture

```
+------------+  state, reward, done  +----------------+
|            |---------------------->|                |
| SimpleMaze |                       | QLearningAgent |
|            |<----------------------|                |
+------------+       action          +----------------+
                                            |
                                       Q-table update
                                      (Bellman equation)
```

The agent maintains a Q-table as a dictionary `{state: [q_values]}`, populated lazily on first visit. At each step it picks an action via epsilon-greedy, observes the outcome, and nudges the Q-value toward the TD target.

---

## Quick Start

```bash
cd 01_q_learning_maze
python train.py
```

This trains for 1000 episodes, then saves a reward curve and an animated GIF of the trained agent to `results/`.

---

## Files

```
01_q_learning_maze/
  maze_env.py          # SimpleMaze environment (4x4 grid, walls, rewards)
  q_learning_agent.py  # QLearningAgent (Q-table, eps-greedy, Bellman update)
  train.py             # Training loop + reward curve plot + GIF recording
  results/
    training_reward_curve.png
    maze_agent.gif
```

---

## Results

| Training Curve | Agent Demo |
| :---: | :---: |
| ![Reward Curve](results/training_reward_curve.png) | ![Maze Agent](results/maze_agent.gif) |
