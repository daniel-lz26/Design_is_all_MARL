"""Commands FactoryCore returns for a runtime to execute (move robot, start process, cancel).

Commands are intent. The runtime turns them into future events:
MoveRobot -> RobotArrived, HoldDock -> DockReleased, StartProcess -> ProcessDone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from som.core.ids import MachineId, OrderId, RobotId, StationId

Purpose = Literal["pickup", "dropoff"]


@dataclass(frozen=True, slots=True)
class MoveRobot:
    robot: RobotId
    station: StationId
    purpose: Purpose
    goal_id: str


@dataclass(frozen=True, slots=True)
class HoldDock:
    """The robot occupies the station dock for ``duration`` seconds, then DockReleased fires."""

    robot: RobotId
    station: StationId
    duration: float
    goal_id: str


@dataclass(frozen=True, slots=True)
class StartProcess:
    """Start (or resume after MachineUp) a run; ProcessDone fires after ``duration``."""

    machine: MachineId
    order: OrderId
    duration: float
    token: int


@dataclass(frozen=True, slots=True)
class CancelRobot:
    robot: RobotId


Command = MoveRobot | HoldDock | StartProcess | CancelRobot
