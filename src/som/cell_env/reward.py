"""Team reward with named terms (spec §8), computed from simulator deltas over one step.

Terms: ``on_time`` (+10 per on-time delivery), ``late`` (-min(tardiness/30, 5) per late
delivery, and for every undelivered order when the episode is truncated, with tardiness up to
t_max), ``past_due`` (-0.01 per second per undelivered past-due order), and the shaping terms
``feed_idle``, ``blocked`` and ``empty_travel``. Every robot gets the sum; ``terms`` holds each
part so it can be logged on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from som.config.cell_schema import CellRewardCfg

if TYPE_CHECKING:
    from som.cell_env.des import CellSim


@dataclass
class Snapshot:
    t: float
    open: list[tuple[str, float]]  # (box, due) not delivered at t
    fed_idle: int
    blocked: int
    empty_s: float


class CellReward:
    def __init__(self, cfg: CellRewardCfg | None = None) -> None:
        self.cfg = cfg or CellRewardCfg()

    def snapshot(self, sim: CellSim) -> Snapshot:
        return Snapshot(
            sim.now,
            [(b.id, b.spec.due) for b in sim.boxes.values() if b.delivered is None],
            sim.fed_idle,
            len(sim.log.blocked_trips),
            sim.empty_travel_s,
        )

    def _late(self, tardiness: float) -> float:
        return -self.cfg.late_per_30s * min(tardiness / 30.0, self.cfg.late_cap)

    def step(self, sim: CellSim, before: Snapshot) -> dict[str, float]:
        cfg = self.cfg
        t0, t1 = before.t, sim.now
        terms = dict.fromkeys(
            ("on_time", "late", "past_due", "feed_idle", "blocked", "empty_travel"), 0.0
        )
        truncated = sim.truncated and not sim.done
        for bid, due in before.open:
            b = sim.boxes[bid]
            end = b.delivered if b.delivered is not None else t1
            terms["past_due"] -= cfg.past_due_per_s * max(0.0, end - max(t0, due))
            if b.delivered is not None:
                if b.delivered <= due:
                    terms["on_time"] += cfg.on_time
                else:
                    terms["late"] += self._late(b.delivered - due)
            elif truncated:
                terms["late"] += self._late(max(0.0, t1 - due))
        terms["feed_idle"] = cfg.feed_idle_machine * (sim.fed_idle - before.fed_idle)
        terms["blocked"] = -cfg.blocked_trip * (len(sim.log.blocked_trips) - before.blocked)
        terms["empty_travel"] = -cfg.empty_travel_per_s * (sim.empty_travel_s - before.empty_s)
        return terms
