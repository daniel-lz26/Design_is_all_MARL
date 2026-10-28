"""EpisodeMetrics computed from a core episode log.

On-time rate is over **all** orders (an undelivered order counts as late). Tardiness of an
undelivered order is measured against the episode end time, so it is a lower bound.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from som.core.history import EpisodeLog


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    n_orders: int
    delivered: int
    on_time: int
    on_time_rate: float
    mean_tardiness: float
    max_tardiness: float
    makespan: float
    robot_loaded_frac: float  # driving with an order
    robot_empty_frac: float  # driving to a pickup
    robot_dock_frac: float  # at a dock or waiting for one
    robot_idle_frac: float
    machine_util: float
    n_decisions: int
    n_waits: int
    terminated: bool

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def _clip(a: float, b: float | None, end: float) -> float:
    return max(0.0, min(b if b is not None else end, end) - a)


def compute_metrics(log: EpisodeLog) -> EpisodeMetrics:
    end = log.end_time
    tard = []
    on_time = 0
    for o in log.orders:
        finish = o.delivered if o.delivered is not None else end
        tard.append(max(0.0, finish - o.due))
        if o.delivered is not None and o.delivered <= o.due:
            on_time += 1
    delivered = [o.delivered for o in log.orders if o.delivered is not None]
    n = len(log.orders)
    makespan = max(delivered) if log.terminated and delivered else end
    span = max(makespan, 1e-9)
    n_r = max(len(log.robots), 1)
    loaded = sum(_clip(g.depart, g.arrive, makespan) for g in log.legs if g.purpose == "dropoff")
    empty = sum(_clip(g.depart, g.arrive, makespan) for g in log.legs if g.purpose == "pickup")
    dock = sum(_clip(d.request, d.end, makespan) for d in log.docks)
    busy = sum(_clip(p.start, p.end, makespan) for p in log.processes)
    return EpisodeMetrics(
        n_orders=n,
        delivered=len(delivered),
        on_time=on_time,
        on_time_rate=on_time / n if n else 0.0,
        mean_tardiness=sum(tard) / n if n else 0.0,
        max_tardiness=max(tard, default=0.0),
        makespan=makespan,
        robot_loaded_frac=loaded / (span * n_r),
        robot_empty_frac=empty / (span * n_r),
        robot_dock_frac=dock / (span * n_r),
        robot_idle_frac=max(0.0, 1.0 - (loaded + empty + dock) / (span * n_r)),
        machine_util=busy / (span * max(len(log.machines), 1)),
        n_decisions=sum(1 for d in log.decisions if d.order is not None),
        n_waits=sum(1 for d in log.decisions if d.order is None),
        terminated=log.terminated,
    )
