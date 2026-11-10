"""RewardFn: team or local reward with at most four terms, computed from core deltas.

Over one decision interval [t0, t1]:

    r = w_deliver * (#delivered) - w_tardy * (overdue seconds accrued by open orders)
      + w_ontime * (#delivered on time) - w_time * (t1 - t0)

``team`` gives every agent the same scalar. ``local`` (ablation only) gives the delivery and
on-time terms to the robot that dropped the order at the sink and to the order's last machine;
the tardiness and time terms stay shared.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from som.config.schema import RewardCfg
from som.core.factory import FactoryCore
from som.core.ids import machine_id


@dataclass(frozen=True, slots=True)
class OpenOrder:
    id: str
    due: float


def open_orders(core: FactoryCore) -> list[OpenOrder]:
    """Orders not yet delivered; take this at t0 and pass it to ``RewardFn.compute``."""
    return [
        OpenOrder(o.id, o.spec.due_time)
        for o in core.orders.values()
        if o.delivered_time is None
    ]


class RewardFn:
    def __init__(self, cfg: RewardCfg) -> None:
        self.cfg = cfg

    def compute(
        self,
        core: FactoryCore,
        before: Sequence[OpenOrder],
        t0: float,
        t1: float,
        agents: Sequence[str],
    ) -> dict[str, float]:
        cfg = self.cfg
        shared = -cfg.w_time * (t1 - t0)
        credit: dict[str, float] = dict.fromkeys(agents, 0.0)
        team = 0.0
        for oo in before:
            o = core.orders[oo.id]  # type: ignore[index]
            end = o.delivered_time if o.delivered_time is not None else t1
            shared -= cfg.w_tardy * max(0.0, end - max(t0, oo.due))
            if o.delivered_time is None:
                continue
            gain = cfg.w_deliver + (cfg.w_ontime if o.delivered_time <= oo.due else 0.0)
            team += gain
            if cfg.mode == "local":
                for agent in self._deliverers(core, oo.id):
                    if agent in credit:
                        credit[agent] += gain
        if cfg.mode == "team":
            return dict.fromkeys(agents, shared + team)
        return {a: shared + credit[a] for a in agents}

    @staticmethod
    def _deliverers(core: FactoryCore, oid: str) -> list[str]:
        o = core.orders[oid]  # type: ignore[index]
        legs = [g for g in core.history().legs if g.order == oid and g.purpose == "dropoff"]
        out = [legs[-1].robot] if legs else []
        return [*out, machine_id(o.spec.job.route[-1])]
