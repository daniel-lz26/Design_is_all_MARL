"""Running mean/std normalization for value targets."""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Parallel-variance running statistics. The critic predicts normalized values."""

    def __init__(self, eps: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        b_mean, b_var, b_n = float(x.mean()), float(x.var()), x.size
        delta = b_mean - self.mean
        total = self.count + b_n
        self.mean += delta * b_n / total
        m2 = self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + 1e-8)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict[str, float]) -> None:
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]
