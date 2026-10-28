"""External events fed to FactoryCore.apply (arrivals, failures, process done, machine down/up).

Robot events carry the ``goal_id`` of the command that caused them and machine events carry the
run ``token``, so the core can ignore stale events (e.g. an arrival after RobotFailed, or a
ProcessDone for a run that was paused by MachineDown).
"""

from __future__ import annotations

from dataclasses import dataclass

from som.core.ids import MachineId, OrderId, RobotId, StationId


@dataclass(frozen=True, slots=True)
class OrderReleased:
    order: OrderId


@dataclass(frozen=True, slots=True)
class RobotArrived:
    robot: RobotId
    station: StationId
    goal_id: str


@dataclass(frozen=True, slots=True)
class DockReleased:
    """The robot finished its pickup/drop-off at the dock (fires after HoldDock's duration)."""

    robot: RobotId
    station: StationId
    goal_id: str


@dataclass(frozen=True, slots=True)
class RobotFailed:
    robot: RobotId
    reason: str
    goal_id: str | None = None  # None: whatever the robot is doing now


@dataclass(frozen=True, slots=True)
class ProcessDone:
    machine: MachineId
    order: OrderId
    token: int


@dataclass(frozen=True, slots=True)
class MachineDown:
    machine: MachineId


@dataclass(frozen=True, slots=True)
class MachineUp:
    machine: MachineId


@dataclass(frozen=True, slots=True)
class Tick:
    """No state change; lets agents that chose WAIT be queried again."""


Event = (
    OrderReleased
    | RobotArrived
    | DockReleased
    | RobotFailed
    | ProcessDone
    | MachineDown
    | MachineUp
    | Tick
)
