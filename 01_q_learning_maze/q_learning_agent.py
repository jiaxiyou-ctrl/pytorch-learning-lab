"""Tabular Q-learning agent with epsilon-greedy exploration."""

import random


class QLearningAgent:
    """Q-table stored as {state: [q_values]}. States added lazily on first visit.

    Args:
        state_size: unused, table grows dynamically.
    """

    def __init__(
        self,
        state_size,
        num_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_q_values(self, state):
        """Return Q-values for state, init to zeros if unseen."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.num_actions
        return self.q_table[state]

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        q_values = self.get_q_values(state)
        return q_values.index(max(q_values))

    def learn(self, state, action, reward, next_state, done):
        """Q-table update: Q(s,a) <- Q(s,a) + lr * [target - Q(s,a)]."""
        q_values = self.get_q_values(state)
        old_q = q_values[action]

        if done:
            target = reward
        else:
            next_q_values = self.get_q_values(next_state)
            target = reward + self.gamma * max(next_q_values)

        q_values[action] = old_q + self.lr * (target - old_q)

    def decay_epsilon(self):
        """Decay exploration rate, clamped to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
