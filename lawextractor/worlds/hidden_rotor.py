from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class HiddenRotorWorld:
    """A tiny world whose public sensors hide a simple controlled latent law.

    The scientist is never allowed to import this module. It only sees the
    sensor vectors emitted by ``observe`` and the interventions passed to
    ``step``. The hidden two-dimensional state exists only so Gate 0 has a
    receipt: after discovery we can ask whether the invented coordinates line
    up with the true state that generated the observations.
    """

    seed: int = 0
    sensor_dim: int = 12
    theta: float = 0.23
    damping: float = 0.985

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._rng = rng
        self._w1 = rng.normal(size=(self.sensor_dim, 2))
        self._w2 = rng.normal(size=(self.sensor_dim, 2))
        self._b1 = rng.uniform(-1.5, 1.5, size=self.sensor_dim)
        self._b2 = rng.uniform(-np.pi, np.pi, size=self.sensor_dim)
        self._gain = rng.uniform(0.65, 1.35, size=self.sensor_dim)
        self._state = np.zeros(2, dtype=np.float64)

        c, s = np.cos(self.theta), np.sin(self.theta)
        self._A = self.damping * np.array([[c, -s], [s, c]], dtype=np.float64)
        self._B = np.array([0.24, -0.11], dtype=np.float64)

    @property
    def action_dim(self) -> int:
        return 1

    def reset(self, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        radius = rng.uniform(0.25, 1.0)
        angle = rng.uniform(-np.pi, np.pi)
        self._state = radius * np.array([np.cos(angle), np.sin(angle)])
        return self.observe()

    def observe(self) -> np.ndarray:
        s = self._state
        # Deliberately awkward but smooth sensorium: no sensor is "x" or "y".
        a = self._w1 @ s + self._b1
        b = self._w2 @ s + self._b2
        y = self._gain * np.sin(2.4 * a) + 0.32 * np.sin(4.7 * b) + 0.08 * np.tanh(a * b)
        return y.astype(np.float32)

    def step(self, action: np.ndarray | float) -> np.ndarray:
        u = float(np.asarray(action).reshape(-1)[0])
        self._state = self._A @ self._state + self._B * u
        return self.observe()

    # The collector may record this into a physically separate evaluator file.
    # The scientist never receives it.
    def private_state(self) -> np.ndarray:
        return self._state.astype(np.float32).copy()
