"""Job types (routes) and the order lifecycle state machine.

PENDING ──(OrderReleased)──► RELEASED@SRC ──(robot reserves)──► RESERVED
RESERVED ──(pickup dock done)──► IN_TRANSIT ──► QUEUED@machine | DELIVERED@SINK
QUEUED ──(machine selects)──► PROCESSING ──(ProcessDone)──► WAITING_PICKUP@machine
WAITING_PICKUP ──(robot reserves)──► RESERVED
RESERVED / IN_TRANSIT ──(RobotFailed)──► back to RELEASED or WAITING_PICKUP at the pickup station
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from som.core.ids import OrderId, RobotId, StationId


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    RELEASED = "RELEASED"
    RESERVED = "RESERVED"
    IN_TRANSIT = "IN_TRANSIT"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    WAITING_PICKUP = "WAITING_PICKUP"
    DELIVERED = "DELIVERED"


_ALLOWED: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.PENDING: frozenset({OrderStatus.RELEASED}),
    OrderStatus.RELEASED: frozenset({OrderStatus.RESERVED}),
    OrderStatus.WAITING_PICKUP: frozenset({OrderStatus.RESERVED}),
    OrderStatus.RESERVED: frozenset(
        {OrderStatus.IN_TRANSIT, OrderStatus.RELEASED, OrderStatus.WAITING_PICKUP}
    ),
    OrderStatus.IN_TRANSIT: frozenset(
        {
            OrderStatus.QUEUED,
            OrderStatus.DELIVERED,
            OrderStatus.RELEASED,
            OrderStatus.WAITING_PICKUP,
        }
    ),
    OrderStatus.QUEUED: frozenset({OrderStatus.PROCESSING}),
    OrderStatus.PROCESSING: frozenset({OrderStatus.WAITING_PICKUP}),
    OrderStatus.DELIVERED: frozenset(),
}

POOL_STATUSES = frozenset({OrderStatus.RELEASED, OrderStatus.WAITING_PICKUP})


class IllegalTransition(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class JobType:
    name: str
    part: str
    route: tuple[str, ...]  # process names, e.g. ("Turning", "Milling", "Drilling")


@dataclass(frozen=True, slots=True)
class OrderSpec:
    id: OrderId
    number: int  # numeric id, used for deterministic tie-breaking
    job: JobType
    release_time: float
    due_time: float
    process_times: tuple[float, ...]  # one per route step

    def __post_init__(self) -> None:
        if len(self.process_times) != len(self.job.route):
            raise ValueError(f"{self.id}: process_times must match route length")


@dataclass(slots=True)
class Order:
    spec: OrderSpec
    status: OrderStatus = OrderStatus.PENDING
    step_idx: int = 0  # index of the next process to perform; == len(route) when finished
    location: StationId | RobotId | None = None
    reserved_by: RobotId | None = None
    available_since: float = 0.0  # when it entered its current pool or queue
    delivered_time: float | None = None
    timeline: list[tuple[float, str, str]] = field(default_factory=list)

    @property
    def id(self) -> OrderId:
        return self.spec.id

    @property
    def route_done(self) -> bool:
        return self.step_idx >= len(self.spec.job.route)

    @property
    def next_process(self) -> str | None:
        return None if self.route_done else self.spec.job.route[self.step_idx]

    def transition(self, new: OrderStatus, now: float, location: StationId | RobotId) -> None:
        if new not in _ALLOWED[self.status]:
            raise IllegalTransition(f"{self.id}: {self.status.value} -> {new.value}")
        self.status = new
        self.location = location
        self.timeline.append((now, new.value, str(location)))
