"""Family B baseline (spec §10): a self-organizing contract-net auction on the task board.

A robot's bid for a task is its estimated time until it is free plus its travel time to the
pickup (``estimates.bid``). The lowest bid wins. A deciding robot claims, among the tasks it
wins, the one with its lowest bid; if it wins none it waits, and is asked again when the board
changes or its wait runs out, which is when bids are recomputed (so a robot going down
triggers a rebid). Down robots never bid.

``AUCTION`` (env-v1.2): only free robots bid, so a free robot never idles waiting for a busy
one. ``AUCTION_v0`` (env-v1): busy robots bid too (time-to-free + travel). On env-v1 that left
free robots idle 36-40% of the time against about 30% for the rules, which is why it was
changed; v0 stays registered for comparison.

Decentralized: a robot's own bid is its ``travel_to_pickup`` feature; the best competing bid
is its ``best_free_bid`` (free robots only) or ``best_other_bid`` (v0) feature, both built
from board and robot broadcasts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from som.cell_env.obs import T


class AuctionPolicy:
    decentralized = True

    def __init__(self, machine_rule: str = "FIFO", busy_bid: bool = False) -> None:
        self.machine_rule = machine_rule
        self.busy_bid = busy_bid
        base = "AUCTION_v0" if busy_bid else "AUCTION"
        self.name = base if machine_rule == "FIFO" else f"{base}/{machine_rule}"
        self.col = T["best_other_bid" if busy_bid else "best_free_bid"]

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
        others = rows[:valid, self.col]
        won = np.flatnonzero(mine <= others)
        if len(won) == 0:
            return int(self.acts["wait"])
        return int(won[np.argmin(mine[won])])
