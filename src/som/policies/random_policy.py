"""Uniform random choice among feasible actions."""

from __future__ import annotations

import numpy as np

from som.core.decisions import WAIT, Choice, DecisionRequest
from som.core.history import CoreSnapshot


class RandomPolicy:
    """Picks uniformly among candidates. WAIT is included only if ``allow_wait`` is set,
    because a random WAIT can stall an episode that a dispatcher would finish."""

    name = "Random"

    def __init__(self, allow_wait: bool = False) -> None:
        self.allow_wait = allow_wait
        self._rng = np.random.default_rng(0)

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def decide(self, request: DecisionRequest, snapshot: CoreSnapshot) -> Choice:
        options: list[Choice] = list(request.candidates)
        if self.allow_wait and request.wait_allowed:
            options.append(WAIT)
        return options[int(self._rng.integers(len(options)))]
