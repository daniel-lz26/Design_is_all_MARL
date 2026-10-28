"""check_invariants(core): consistency checks run in tests and debug mode."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from som.core.order import OrderStatus

if TYPE_CHECKING:
    from som.core.factory import FactoryCore

_AT_STATION = {OrderStatus.RELEASED, OrderStatus.WAITING_PICKUP, OrderStatus.RESERVED}


def check_invariants(core: FactoryCore) -> list[str]:
    """Return human-readable violations (empty when the core is consistent)."""
    from som.core.factory import RobotStatus

    problems: list[str] = []

    # Each order is in exactly one place, and that place matches its status.
    places: Counter[str] = Counter()
    for sid, pool in core.pools.items():
        for oid in pool:
            places[oid] += 1
            o = core.orders[oid]
            if o.status not in _AT_STATION or o.location != sid:
                problems.append(f"{oid} in pool {sid} but {o.status.value}@{o.location}")
    for r in core.robots.values():
        if r.carrying is not None:
            places[r.carrying] += 1
            o = core.orders[r.carrying]
            if o.status is not OrderStatus.IN_TRANSIT or o.location != r.id:
                problems.append(f"{r.id} carries {o.id} which is {o.status.value}")
    for m in core.machines.values():
        for oid in m.queue:
            places[oid] += 1
            if core.orders[oid].status is not OrderStatus.QUEUED:
                problems.append(f"{oid} in {m.id} queue but {core.orders[oid].status.value}")
        if m.current is not None:
            places[m.current] += 1
            if core.orders[m.current].status is not OrderStatus.PROCESSING:
                problems.append(f"{m.id} runs {m.current} which is not PROCESSING")
    for oid in core.delivered:
        places[oid] += 1
        if core.orders[oid].status is not OrderStatus.DELIVERED:
            problems.append(f"{oid} delivered but {core.orders[oid].status.value}")
    for oid, o in core.orders.items():
        expected = 0 if o.status is OrderStatus.PENDING else 1
        if places[oid] != expected:
            problems.append(f"{oid} ({o.status.value}) is in {places[oid]} places")

    # Capacity and machine state.
    for m in core.machines.values():
        if m.inbound < 0:
            problems.append(f"{m.id} inbound count is negative")
        if len(m.queue) + m.inbound > m.capacity:
            problems.append(f"{m.id} over capacity: {len(m.queue)} queued + {m.inbound} inbound")
        if (m.current is None) != (m.busy_until is None and m.paused_remaining is None):
            problems.append(f"{m.id} busy state inconsistent")
        if m.down and m.busy_until is not None:
            problems.append(f"{m.id} is down but still running")

    # Reservations: robot task <-> order.reserved_by, and inbound counts match.
    inbound: Counter[str] = Counter()
    for r in core.robots.values():
        if r.task is None:
            if r.status is not RobotStatus.IDLE or r.carrying is not None:
                problems.append(f"{r.id} has no task but is {r.status.value}")
            continue
        o = core.orders[r.task.order]
        if o.reserved_by != r.id:
            problems.append(f"{r.id} task {o.id} is reserved by {o.reserved_by}")
        if core.machine_at(r.task.dest) is not None:
            inbound[r.task.dest] += 1
    reserved = Counter(o.reserved_by for o in core.orders.values() if o.reserved_by)
    for rid, n in reserved.items():
        if n > 1:
            problems.append(f"{rid} holds {n} reservations")
        if core.robots[rid].task is None:
            problems.append(f"{rid} holds a reservation without a task")
    for m in core.machines.values():
        if inbound[m.station] != m.inbound:
            problems.append(f"{m.id} inbound {m.inbound} != {inbound[m.station]} reserved trips")

    # Dock locks: one holder per dock, holders DOCKING there, waiters WAITING_DOCK there.
    for sid, dock in core.docks.items():
        if dock.holder is not None:
            h = core.robots[dock.holder]
            if h.status is not RobotStatus.DOCKING or h.station != sid:
                problems.append(f"{h.id} holds dock {sid} but is {h.status.value}@{h.station}")
        for w in dock.waiting:
            wr = core.robots[w]
            if wr.status is not RobotStatus.WAITING_DOCK or wr.station != sid:
                problems.append(f"{w} waits at {sid} but is {wr.status.value}@{wr.station}")
    for r in core.robots.values():
        holds = sum(d.holder == r.id for d in core.docks.values())
        waits = sum(r.id in d.waiting for d in core.docks.values())
        if r.status is RobotStatus.DOCKING and holds != 1:
            problems.append(f"{r.id} DOCKING but holds {holds} docks")
        if r.status is RobotStatus.WAITING_DOCK and waits != 1:
            problems.append(f"{r.id} WAITING_DOCK but in {waits} wait queues")

    # Time is monotone along every order timeline.
    for o in core.orders.values():
        ts = [t for t, _, _ in o.timeline]
        if ts != sorted(ts) or (ts and ts[-1] > core.now):
            problems.append(f"{o.id} timeline is not monotone")
    return problems
