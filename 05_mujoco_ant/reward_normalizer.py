"""Running reward normalizer (divides by running std)."""

import numpy as np


class RewardNormalizer:
    """Normalizes scalar rewards by running standard deviation."""

    def __init__(self, clip: float = 10.0) -> None:
        self.clip = clip
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = 1e-4

    def update(self, reward: float) -> None:
        """Update running stats with a new reward sample."""
        delta = reward - self.mean
        self.count += 1
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, reward: float) -> float:
        """Return reward / running_std, clipped."""
        normalized = reward / np.sqrt(self.var + 1e-8)
        return float(np.clip(normalized, -self.clip, self.clip))
