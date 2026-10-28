"""Station, Pose2D and Layout: arena geometry built from LayoutCfg."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from som.config.schema import LayoutCfg, PoseTuple
from som.core.ids import MachineId, StationId, machine_id, station_id


@dataclass(frozen=True, slots=True)
class Pose2D:
    x: float
    y: float
    theta: float = 0.0

    @classmethod
    def of(cls, t: PoseTuple) -> Pose2D:
        return cls(float(t[0]), float(t[1]), float(t[2]))

    def distance_to(self, other: Pose2D) -> float:
        return math.hypot(other.x - self.x, other.y - self.y)


class StationKind(str, Enum):
    SOURCE = "source"
    MACHINE = "machine"
    SINK = "sink"


@dataclass(frozen=True, slots=True)
class Station:
    id: StationId
    key: str  # name in the layout YAML, e.g. "TURNING"
    kind: StationKind
    dock: Pose2D
    queue: Pose2D
    process: str | None = None
    machine: MachineId | None = None
    input_capacity: int | None = None


@dataclass(frozen=True, slots=True)
class Layout:
    name: str
    stations: tuple[Station, ...]  # layout file order
    robot_keys: tuple[str, ...]  # e.g. ("tb1", "tb2", ...); robot_i maps to robot_keys[i]
    robot_starts: tuple[Pose2D, ...]
    dock_time_s: float

    @classmethod
    def from_cfg(cls, cfg: LayoutCfg) -> Layout:
        stations = tuple(
            Station(
                id=station_id(key),
                key=key,
                kind=StationKind(s.kind),
                dock=Pose2D.of(s.dock),
                queue=Pose2D.of(s.queue),
                process=s.process,
                machine=machine_id(s.process) if s.process else None,
                input_capacity=s.input_capacity,
            )
            for key, s in cfg.stations.items()
        )
        return cls(
            name=cfg.name,
            stations=stations,
            robot_keys=tuple(cfg.robots),
            robot_starts=tuple(Pose2D.of(r.start) for r in cfg.robots.values()),
            dock_time_s=cfg.travel_calibration.dock_overhead_s,
        )

    def station(self, sid: StationId) -> Station:
        for s in self.stations:
            if s.id == sid:
                return s
        raise KeyError(sid)

    @property
    def source(self) -> Station:
        return next(s for s in self.stations if s.kind is StationKind.SOURCE)

    @property
    def sink(self) -> Station:
        return next(s for s in self.stations if s.kind is StationKind.SINK)

    @property
    def machines(self) -> tuple[Station, ...]:
        return tuple(s for s in self.stations if s.kind is StationKind.MACHINE)

    def station_for_process(self, process: str) -> Station:
        for s in self.machines:
            if s.process == process:
                return s
        raise KeyError(f"no machine station for process {process!r}")

    def station_for_machine(self, mid: MachineId) -> Station:
        for s in self.machines:
            if s.machine == mid:
                return s
        raise KeyError(mid)
