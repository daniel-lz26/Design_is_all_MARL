"""Policy protocol: choose one agent's action for a decision request.

Policies are decision-level: they receive the agent's ``DecisionRequest`` and a read-only
``CoreSnapshot``. Learned policies build their observations from these with the shared encoders
in ``som.env.observations``, so sim, env and live runtime all call policies the same way.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from som.core.decisions import Choice, DecisionRequest
from som.core.history import CoreSnapshot


class Policy(Protocol):
    name: str

    def reset(self, rng: np.random.Generator) -> None:
        """Start a new episode. Stochastic policies draw only from ``rng``."""
        ...

    def decide(self, request: DecisionRequest, snapshot: CoreSnapshot) -> Choice: ...
