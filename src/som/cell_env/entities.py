"""Mutable cell entities (boxes, stations with arm and machine, robots) and the episode log."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from som.cell_env.orders import OrderSpec


class RS(str, Enum):
    """Robot state. Spec §4.1 plus IDLE (free, not parked), BLOCKED (waiting for an edge),
    WAIT_ARM (docked, queued for the arm), UNDOCKING and ABSENT (not yet joined)."""

    ABSENT = "ABSENT"
    IDLE_PARKED = "IDLE_PARKED"
    IDLE = "IDLE"
    TRAVEL_EMPTY = "TRAVEL_EMPTY"
    TRAVEL_LOADED = "TRAVEL_LOADED"
    BLOCKED = "BLOCKED"
    WAIT_DOCK = "WAIT_DOCK"
    DOCKING = "DOCKING"
    WAIT_ARM = "WAIT_ARM"
    BEING_LOADED = "BEING_LOADED"
    BEING_UNLOADED = "BEING_UNLOADED"
    UNDOCKING = "UNDOCKING"
    DOWN = "DOWN"


# Gantt / utilisation categories (spec §12)
UTIL = {
    RS.TRAVEL_LOADED: "loaded",
    RS.TRAVEL_EMPTY: "empty",
    RS.BLOCKED: "blocked",
    RS.WAIT_DOCK: "dock",
    RS.DOCKING: "dock",
    RS.WAIT_ARM: "dock",
    RS.BEING_LOADED: "dock",
    RS.BEING_UNLOADED: "dock",
    RS.UNDOCKING: "dock",
    RS.IDLE: "idle",
    RS.IDLE_PARKED: "idle",
    RS.DOWN: "down",
}


class Loc(str, Enum):
    PENDING = "PENDING"  # not released yet
    OUT = "OUT"  # in a station's out-buffer (an open transport task when unclaimed)
    ROBOT = "ROBOT"  # on a robot's top plate
    IN = "IN"  # in a machine's in-buffer
    MACHINE = "MACHINE"  # being processed, or finished and blocked in the machine
    DONE = "DONE"  # delivered to the sink


@dataclass(slots=True)
class Box:
    spec: OrderSpec
    step: int = 0  # index of the next process; == len(route) when only SINK is left
    loc: Loc = Loc.PENDING
    where: str = ""  # station key or robot id
    claimed_by: str | None = None
    t_avail: float = 0.0  # when it entered its current out-buffer
    delivered: float | None = None

    @property
    def id(self) -> str:
        return self.spec.id


@dataclass(slots=True)
class Station:
    key: str
    kind: str
    process: str | None
    dock: str
    approach: str
    in_cap: int | None
    out_cap: int | None
    load_s: float
    unload_s: float
    in_buf: list[str] = field(default_factory=list)
    out_buf: list[str] = field(default_factory=list)
    inbound: int = 0  # claimed boxes heading here
    dock_holder: str | None = None
    dock_queue: list[str] = field(default_factory=list)
    arm_busy: str | None = None  # robot being served
    arm_kind: str = ""  # "load" | "unload"
    arm_until: float = 0.0
    arm_queue: list[str] = field(default_factory=list)
    arm_down: bool = False
    arm_paused: float | None = None  # remaining handoff time while the arm is down
    arm_token: int = 0
    current: str | None = None  # box in the machine
    busy_until: float | None = None
    blocked_box: str | None = None  # finished, waiting for out-buffer space
    down: bool = False
    paused: float | None = None
    token: int = 0

    def in_free(self) -> float:
        if self.in_cap is None:
            return float("inf")
        return self.in_cap - len(self.in_buf) - self.inbound


@dataclass(slots=True)
class Robot:
    id: str  # "agv_1".."agv_4"
    index: int
    park: str  # home parking node
    state: RS = RS.IDLE_PARKED
    node: str = ""
    prev: str | None = None  # previous node (heading for turn penalties)
    path: list[str] = field(default_factory=list)
    goal: str = ""
    edge: Any = None  # Edge being traversed
    seg: tuple[float, float, str, str] | None = None  # (t0, t1, from, to) while moving
    purpose: str | None = None  # "pickup" | "dropoff" | "park" | "wait"
    task: str | None = None  # claimed box
    box: str | None = None  # carried box
    docked_at: str | None = None
    dock_target: str | None = None  # station whose dock it is heading for or holding
    token: int = 0
    needs_decision: bool = False
    seen_board: int = -1
    wait_until: float = 0.0
    blocked_now: bool = False  # already counted as a blocked trip at this hop
    down_xy: tuple[float, float] | None = None
    holds_spot: bool = False  # down and still blocking an edge or dock


@dataclass(slots=True)
class CellLog:
    """Everything the metrics, Gantt chart and replay need. Plain data, JSON-ready."""

    scenario: str
    seed: int
    robots: list[str]
    stations: list[str]
    orders: list[dict[str, Any]] = field(default_factory=list)
    robot_states: dict[str, list[tuple[float, str]]] = field(default_factory=dict)
    moves: dict[str, list[tuple[float, float, str, str]]] = field(default_factory=dict)
    carrying: dict[str, list[tuple[float, str | None]]] = field(default_factory=dict)
    claims: dict[str, list[tuple[float, str | None]]] = field(default_factory=dict)
    docks: list[tuple[str, str, float, float | None]] = field(default_factory=list)
    down_at: dict[str, list[tuple[float, tuple[float, float]]]] = field(default_factory=dict)
    machine_runs: list[tuple[str, str, float, float | None]] = field(default_factory=list)
    machine_blocked: list[tuple[str, float, float | None]] = field(default_factory=list)
    handoffs: list[tuple[str, str, str, float, float | None]] = field(default_factory=list)
    buffers: list[tuple[float, str, int, int]] = field(default_factory=list)  # t, st, in, out
    blocked_trips: list[tuple[float, str, str]] = field(default_factory=list)  # t, robot, node
    deadlocks: list[tuple[float, list[str]]] = field(default_factory=list)
    outages: list[tuple[float, str, str, str]] = field(default_factory=list)  # t, who, down|up, why
    decisions: list[tuple[float, str, int]] = field(default_factory=list)  # t, robot, action
    end_time: float = 0.0
    terminated: bool = False
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
