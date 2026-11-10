"""The controller interface for the cell env (spec §11.1). Rules, the auction, the central
scheduler and every learned method implement it.

``decentralized=True`` means ``act`` receives only each robot's own observation and mask:
the evaluation harness passes ``global_state=None``, and a test checks that it does. Only
centralized baselines (``decentralized=False``) get ``{"sim": CellSim}``.

``machine_rule`` (optional attribute) is the stations' sequencing rule (FIFO, EDD, SPT) the
controller is evaluated with; it configures the machines, not the robots.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class CellPolicy(Protocol):
    name: str
    decentralized: bool

    def reset(self, scenario_info: dict[str, Any]) -> None: ...

    def act(
        self,
        obs: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        global_state: dict[str, Any] | None = None,
    ) -> dict[str, int]: ...
