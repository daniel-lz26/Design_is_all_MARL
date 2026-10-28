"""EpisodeLog (what happened, for metrics and replay) and CoreSnapshot (immutable state view).

Both are plain data. ``EpisodeLog.to_dict``/``from_dict`` give a JSON-ready form; writing files
is the caller's job (the core does no I/O).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from som.core.layout import Layout, Pose2D


@dataclass(slots=True)
class OrderRecord:
    id: str
    job: str
    route: list[str]
    release: float
    due: float
    delivered: float | None = None
    timeline: list[tuple[float, str, str]] = field(default_factory=list)  # (t, status, location)


@dataclass(slots=True)
class ProcessRecord:
    machine: str
    order: str
    start: float
    end: float | None = None  # None while running; a pause closes the segment


@dataclass(slots=True)
class LegRecord:
    robot: str
    order: str | None
    purpose: str
    depart: float
    from_xy: tuple[float, float]
    to_station: str
    arrive: float | None = None  # None if the leg failed or the episode ended first


@dataclass(slots=True)
class DockRecord:
    robot: str
    station: str
    request: float  # arrival at the station
    start: float | None = None  # dock granted
    end: float | None = None


@dataclass(slots=True)
class DecisionRecord:
    t: float
    agent: str
    order: str | None  # None means WAIT
    n_candidates: int


@dataclass(slots=True)
class EpisodeLog:
    layout: str
    robots: list[str]
    machines: list[str]
    orders: list[OrderRecord]
    processes: list[ProcessRecord] = field(default_factory=list)
    legs: list[LegRecord] = field(default_factory=list)
    docks: list[DockRecord] = field(default_factory=list)
    decisions: list[DecisionRecord] = field(default_factory=list)
    failures: list[tuple[float, str, str]] = field(default_factory=list)  # (t, robot, reason)
    end_time: float = 0.0
    terminated: bool = False  # all orders delivered
    truncated: bool = False  # horizon reached first
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EpisodeLog:
        def tup(xs: list[Any]) -> list[tuple[Any, ...]]:
            return [tuple(x) for x in xs]

        return cls(
            layout=d["layout"],
            robots=list(d["robots"]),
            machines=list(d["machines"]),
            orders=[OrderRecord(**{**o, "timeline": tup(o["timeline"])}) for o in d["orders"]],
            processes=[ProcessRecord(**p) for p in d["processes"]],
            legs=[LegRecord(**{**g, "from_xy": tuple(g["from_xy"])}) for g in d["legs"]],
            docks=[DockRecord(**k) for k in d["docks"]],
            decisions=[DecisionRecord(**x) for x in d["decisions"]],
            failures=tup(d["failures"]),
            end_time=d["end_time"],
            terminated=d["terminated"],
            truncated=d["truncated"],
            meta=dict(d.get("meta", {})),
        )


# --------------------------------------------------------------------------- snapshot


@dataclass(frozen=True, slots=True)
class OrderView:
    id: str
    number: int
    job: str
    route: tuple[str, ...]
    process_times: tuple[float, ...]
    step_idx: int
    release: float
    due: float
    status: str
    location: str | None
    reserved_by: str | None
    available_since: float


@dataclass(frozen=True, slots=True)
class RobotView:
    id: str
    status: str
    pose: Pose2D
    station: str | None  # station it is at (None while traveling or before first dock)
    carrying: str | None
    task_order: str | None
    task_dest: str | None


@dataclass(frozen=True, slots=True)
class MachineView:
    id: str
    station: str
    process: str
    current: str | None
    busy_until: float | None
    paused_remaining: float | None
    down: bool
    queue: tuple[str, ...]
    inbound: int
    capacity: int


@dataclass(frozen=True, slots=True)
class CoreSnapshot:
    now: float
    layout: Layout
    robots: tuple[RobotView, ...]
    machines: tuple[MachineView, ...]
    orders: tuple[OrderView, ...]

    def order(self, oid: str) -> OrderView:
        for o in self.orders:
            if o.id == oid:
                return o
        raise KeyError(oid)

    def robot(self, rid: str) -> RobotView:
        for r in self.robots:
            if r.id == rid:
                return r
        raise KeyError(rid)

    def machine(self, mid: str) -> MachineView:
        for m in self.machines:
            if m.id == mid:
                return m
        raise KeyError(mid)
