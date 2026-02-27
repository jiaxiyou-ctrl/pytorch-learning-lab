"""Domain randomization for sim-to-real transfer (gravity, friction, mass)."""

from typing import Dict, Tuple

import gymnasium as gym
import numpy as np


class DomainRandomizer:
    """Randomizes MuJoCo physics params at episode start.

    Default values are cached lazily on first call to randomize().
    """

    def __init__(
        self,
        gravity_range: Tuple[float, float] = (-11.0, -8.0),
        friction_scale_range: Tuple[float, float] = (0.5, 1.5),
        mass_scale_range: Tuple[float, float] = (0.8, 1.2),
        randomize_gravity: bool = True,
        randomize_friction: bool = True,
        randomize_mass: bool = True,
    ) -> None:
        self.gravity_range = gravity_range
        self.friction_scale_range = friction_scale_range
        self.mass_scale_range = mass_scale_range
        self.randomize_gravity = randomize_gravity
        self.randomize_friction = randomize_friction
        self.randomize_mass = randomize_mass

        self._default_friction: np.ndarray | None = None
        self._default_mass: np.ndarray | None = None
        self._initialized = False

    def _save_default_values(self, model: object) -> None:
        self._default_friction = model.geom_friction.copy()
        self._default_mass = model.body_mass.copy()
        self._initialized = True

    def randomize(self, env: gym.Env) -> None:
        """Apply random physics perturbations. Call after env.reset()."""
        model = env.unwrapped.model

        if not self._initialized:
            self._save_default_values(model)

        if self.randomize_gravity:
            model.opt.gravity[2] = np.random.uniform(*self.gravity_range)

        if self.randomize_friction:
            scale = np.random.uniform(*self.friction_scale_range)
            model.geom_friction[:] = self._default_friction * scale

        if self.randomize_mass:
            scale = np.random.uniform(*self.mass_scale_range)
            model.body_mass[:] = self._default_mass * scale

    def get_current_params(self, env: gym.Env) -> Dict[str, float]:
        """Return current gravity_z, friction_0, mass_total."""
        model = env.unwrapped.model
        return {
            "gravity_z": float(model.opt.gravity[2]),
            "friction_0": float(model.geom_friction[0, 0]),
            "mass_total": float(model.body_mass.sum()),
        }
