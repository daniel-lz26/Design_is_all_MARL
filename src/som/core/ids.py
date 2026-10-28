"""Typed string IDs (robot, machine, station, order) and their prefix helpers."""

from __future__ import annotations

from typing import Literal, NewType

RobotId = NewType("RobotId", str)
MachineId = NewType("MachineId", str)
StationId = NewType("StationId", str)
OrderId = NewType("OrderId", str)
AgentId = str  # a RobotId or a MachineId
Role = Literal["robot", "machine"]

ROBOT_PREFIX = "robot_"
MACHINE_PREFIX = "machine_"
STATION_PREFIX = "station_"
ORDER_PREFIX = "order_"


def robot_id(index: int) -> RobotId:
    return RobotId(f"{ROBOT_PREFIX}{index}")


def machine_id(process: str) -> MachineId:
    return MachineId(f"{MACHINE_PREFIX}{process.lower()}")


def station_id(key: str) -> StationId:
    return StationId(f"{STATION_PREFIX}{key}")


def order_id(number: int) -> OrderId:
    return OrderId(f"{ORDER_PREFIX}{number}")


def role_of(agent: AgentId) -> Role:
    if agent.startswith(ROBOT_PREFIX):
        return "robot"
    if agent.startswith(MACHINE_PREFIX):
        return "machine"
    raise ValueError(f"{agent!r} is not an agent id")
