"""Family B baseline (spec §10): a self-organizing contract-net auction on the task board.

Every robot's bid for a task is its estimated time until it is free plus its travel time to
the pickup (``estimates.bid``); busy robots bid too, down robots do not. The lowest bid wins.
A deciding robot claims, among the tasks it wins, the one with its lowest bid; if it wins
none it waits, and is asked again when the board changes or its wait runs out, which is when
bids are recomputed (so a robot going down triggers a rebid).

Decentralized: a robot's own bid is its ``travel_to_pickup`` feature and the best competing
bid is its ``best_other_bid`` feature, both built from board and robot broadcasts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from som.cell_env.obs import T


class AuctionPolicy:
    decentralized = True

    def __init__(self, machine_rule: str = "FIFO") -> None:
        self.machine_rule = machine_rule
        self.name = "AUCTION" if machine_rule == "FIFO" else f"AUCTION/{machine_rule}"

    def reset(self, scenario_info: dict[str, Any]) -> None:
        self.layout = scenario_info["obs_layout"]
        self.acts = scenario_info["actions"]

    def act(
        self,
        obs: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        global_state: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        return {a: self._one(obs[a], masks[a]) for a in obs}

    def _one(self, o: np.ndarray, m: np.ndarray) -> int:
        if m[self.acts["noop"]]:
            return int(self.acts["noop"])
        rows = self.layout.tasks(o)
        valid = int(rows[:, T["valid"]].sum())
        mine = rows[:valid, T["travel_to_pickup"]]
        others = rows[:valid, T["best_other_bid"]]
        won = np.flatnonzero(mine <= others)
        if len(won) == 0:
            return int(self.acts["wait"])
        return int(won[np.argmin(mine[won])])
