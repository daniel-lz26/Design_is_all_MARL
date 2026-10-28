from __future__ import annotations

import pytest

from som.core.commands import HoldDock, MoveRobot, StartProcess
from som.core.decisions import WAIT, InfeasibleDecision
from som.core.events import (
    DockReleased,
    MachineDown,
    MachineUp,
    OrderReleased,
    ProcessDone,
    RobotArrived,
    RobotFailed,
)
from som.core.factory import FactoryCore, RobotStatus
from som.core.ids import MachineId, OrderId, RobotId, StationId
from som.core.invariants import check_invariants
from som.core.layout import Layout
from som.core.order import OrderStatus

from helpers import TURN, TURN_MILL, make_core, spec

R0, R1 = RobotId("robot_0"), RobotId("robot_1")
SRC, A, B, SINK = (StationId(f"station_{k}") for k in ("SRC", "A", "B", "SINK"))
TURNING, MILLING = MachineId("machine_turning"), MachineId("machine_milling")
O1, O2, O3 = OrderId("order_1"), OrderId("order_2"), OrderId("order_3")


def release_all(core: FactoryCore, t: float = 0.0) -> None:
    for oid in core.orders:
        core.apply(OrderReleased(oid), t)


def robot_req(core: FactoryCore, robot: RobotId, t: float):
    return next(r for r in core.pending_decisions(t) if r.agent == robot)


def dock_cycle(core: FactoryCore, robot: RobotId, station: StationId, t: float):
    """RobotArrived then DockReleased for the robot's current goal; returns the last commands."""
    goal = core.robots[robot].goal_id
    assert goal is not None
    cmds = core.apply(RobotArrived(robot, station, goal), t)
    assert any(isinstance(c, HoldDock) for c in cmds)
    return core.apply(DockReleased(robot, station, goal), t + 2.0)


def test_full_order_lifecycle(tiny_layout: Layout) -> None:
    core = make_core(tiny_layout, [spec(1, TURN)])
    assert core.pending_decisions(0.0) == []  # nothing released yet
    release_all(core)
    req = robot_req(core, R0, 0.0)
    (cmd,) = core.assign(R0, req.candidates[0], 0.0)
    assert cmd == MoveRobot(R0, SRC, "pickup", "robot_0#1")
    assert core.orders[O1].status is OrderStatus.RESERVED
    (move,) = dock_cycle(core, R0, SRC, 1.0)
    assert isinstance(move, MoveRobot) and move.station == A and move.purpose == "dropoff"
    assert core.orders[O1].status is OrderStatus.IN_TRANSIT
    assert dock_cycle(core, R0, A, 5.0) == []
    assert core.orders[O1].status is OrderStatus.QUEUED
    assert core.robots[R0].status is RobotStatus.IDLE
    (mreq,) = [r for r in core.pending_decisions(7.0) if r.agent == TURNING]
    (start,) = core.assign(TURNING, mreq.candidates[0], 7.0)
    assert start == StartProcess(TURNING, O1, 10.0, 1)
    core.apply(ProcessDone(TURNING, O1, 1), 17.0)
    assert core.orders[O1].status is OrderStatus.WAITING_PICKUP
    (cand,) = robot_req(core, R1, 17.0).candidates
    assert (cand.pickup, cand.dest) == (A, SINK)
    core.assign(R1, cand, 17.0)
    dock_cycle(core, R1, A, 18.0)
    dock_cycle(core, R1, SINK, 25.0)
    assert core.done
    assert core.orders[O1].delivered_time == 27.0
    log = core.history()
    assert log.terminated and log.orders[0].delivered == 27.0


def test_time_must_not_go_backwards(tiny_layout: Layout) -> None:
    core = make_core(tiny_layout, [spec(1), spec(2)])
    core.apply(OrderReleased(O1), 5.0)
    with pytest.raises(ValueError, match="backwards"):
        core.apply(OrderReleased(O2), 4.0)


def test_reservation_prevents_double_pick(tiny_layout: Layout) -> None:
    core = make_core(tiny_layout, [spec(1)])
    release_all(core)
    cand = robot_req(core, R0, 0.0).candidates[0]
    core.assign(R0, cand, 0.0)
    assert [r.agent for r in core.pending_decisions(0.0)] == []  # robot_1 has nothing left
    with pytest.raises(InfeasibleDecision):
        core.assign(R1, cand, 0.0)


def test_input_capacity_counts_inbound_reservations(tiny_layout: Layout) -> None:
    # Machine A has input_capacity 2: the third order bound for A is not offered.
    core = make_core(tiny_layout, [spec(1), spec(2), spec(3)], n_robots=2)
    release_all(core)
    core.assign(R0, robot_req(core, R0, 0.0).candidates[0], 0.0)
    core.assign(R1, robot_req(core, R1, 0.0).candidates[0], 0.0)
    assert core.machines[TURNING].inbound == 2
    assert not core.has_input_capacity(A)
    for r in core.robots.values():
        dock_cycle(core, r.id, SRC, 1.0 if r.id == R0 else 3.0)
    # Both deliver; the queue is now full, so order 3 still has no feasible destination.
    dock_cycle(core, R0, A, 10.0)
    dock_cycle(core, R1, A, 12.0)
    assert len(core.machines[TURNING].queue) == 2
    assert [r for r in core.pending_decisions(14.0) if r.role == "robot"] == []
    # Machine takes one -> capacity frees -> order 3 becomes a candidate.
    mreq = next(r for r in core.pending_decisions(14.0) if r.agent == TURNING)
    core.assign(TURNING, mreq.candidates[0], 14.0)
    assert robot_req(core, R0, 14.0).candidates[0].order == O3


def test_dock_lock_queues_second_robot(tiny_layout: Layout) -> None:
    core = make_core(tiny_layout, [spec(1), spec(2)])
    release_all(core)
    core.assign(R0, robot_req(core, R0, 0.0).candidates[0], 0.0)
    core.assign(R1, robot_req(core, R1, 0.0).candidates[0], 0.0)
    g0, g1 = core.robots[R0].goal_id, core.robots[R1].goal_id
    assert g0 and g1
    (hold,) = core.apply(RobotArrived(R0, SRC, g0), 1.0)
    assert isinstance(hold, HoldDock) and hold.duration == 2.0
    assert core.apply(RobotArrived(R1, SRC, g1), 1.5) == []
    assert core.robots[R1].status is RobotStatus.WAITING_DOCK
    assert core.docks[SRC].waiting == [R1]
    cmds = core.apply(DockReleased(R0, SRC, g0), 3.0)
    assert any(isinstance(c, HoldDock) and c.robot == R1 for c in cmds)
    assert core.docks[SRC].holder == R1
    log = core.history()
    second = next(d for d in log.docks if d.robot == R1)
    assert (second.request, second.start) == (1.5, 3.0)


def test_robot_failed_on_way_to_pickup_frees_order(tiny_layout: Layout) -> None:
    core = make_core(tiny_layout, [spec(1)])
    release_all(core)
    core.assign(R0, robot_req(core, R0, 0.0).candidates[0], 0.0)
    goal = core.robots[R0].goal_id
    core.apply(RobotFailed(R0, "aborted"), 3.0)
    o = core.orders[O1]
    assert (o.status, o.reserved_by, o.location) == (OrderStatus.RELEASED, None, SRC)
    assert core.machines[TURNING].inbound == 0
    assert core.robots[R0].status is RobotStatus.IDLE
    assert goal is not None and core.apply(RobotArrived(R0, SRC, goal), 4.0) == []  # stale
    assert robot_req(core, R1, 4.0).candidates[0].order == O1
    assert core.history().failures == [(3.0, R0, "aborted")]


def test_robot_failed_while_carrying_returns_order_to_pickup(tiny_layout: Layout) -> None:
    core = make_core(tiny_layout, [spec(1)])
    release_all(core)
    core.assign(R0, robot_req(core, R0, 0.0).candidates[0], 0.0)
    dock_cycle(core, R0, SRC, 1.0)
    assert core.robots[R0].carrying == O1
    core.apply(RobotFailed(R0, "timeout"), 5.0)
    o = core.orders[O1]
    assert (o.status, o.location) == (OrderStatus.RELEASED, SRC)
    assert O1 in core.pools[SRC] and core.robots[R0].carrying is None
    assert check_invariants(core) == []


def test_robot_failed_releases_held_dock(tiny_layout: Layout) -> None:
    core = make_core(tiny_layout, [spec(1), spec(2)])
    release_all(core)
    core.assign(R0, robot_req(core, R0, 0.0).candidates[0], 0.0)
    core.assign(R1, robot_req(core, R1, 0.0).candidates[0], 0.0)
    core.apply(RobotArrived(R0, SRC, core.robots[R0].goal_id or ""), 1.0)
    core.apply(RobotArrived(R1, SRC, core.robots[R1].goal_id or ""), 1.0)
    cmds = core.apply(RobotFailed(R0, "estop"), 2.0)
    assert any(isinstance(c, HoldDock) and c.robot == R1 for c in cmds)
    assert core.docks[SRC].holder == R1


def test_stale_failure_goal_id_is_ignored(tiny_layout: Layout) -> None:
    core = make_core(tiny_layout, [spec(1)])
    release_all(core)
    core.assign(R0, robot_req(core, R0, 0.0).candidates[0], 0.0)
    assert core.apply(RobotFailed(R0, "old", goal_id="robot_0#99"), 1.0) == []
    assert core.orders[O1].status is OrderStatus.RESERVED


def _queued_at_turning(core: FactoryCore) -> None:
    release_all(core)
    core.assign(R0, robot_req(core, R0, 0.0).candidates[0], 0.0)
    dock_cycle(core, R0, SRC, 0.0)
    dock_cycle(core, R0, A, 4.0)


def test_machine_down_pauses_and_up_resumes(tiny_layout: Layout) -> None:
    core = make_core(tiny_layout, [spec(1)])
    _queued_at_turning(core)
    mreq = next(r for r in core.pending_decisions(6.0) if r.agent == TURNING)
    (start,) = core.assign(TURNING, mreq.candidates[0], 6.0)  # runs 6 -> 16
    core.apply(MachineDown(TURNING), 10.0)
    assert core.machines[TURNING].paused_remaining == pytest.approx(6.0)
    assert core.apply(ProcessDone(TURNING, O1, start.token), 16.0) == []  # stale
    assert core.orders[O1].status is OrderStatus.PROCESSING
    (resume,) = core.apply(MachineUp(TURNING), 30.0)
    assert isinstance(resume, StartProcess) and resume.duration == pytest.approx(6.0)
    core.apply(ProcessDone(TURNING, O1, resume.token), 36.0)
    assert core.orders[O1].status is OrderStatus.WAITING_PICKUP
    runs = [(p.start, p.end) for p in core.history().processes]
    assert runs == [(6.0, 10.0), (30.0, 36.0)]


def test_down_machine_takes_no_new_work(tiny_layout: Layout) -> None:
    core = make_core(tiny_layout, [spec(1), spec(2)])
    core.apply(MachineDown(TURNING), 0.0)
    release_all(core)
    assert core.pending_decisions(0.0) == []  # all orders route to the down machine
    core.apply(MachineUp(TURNING), 5.0)
    assert {r.agent for r in core.pending_decisions(5.0)} == {R0, R1}


def test_down_machine_with_queue_makes_no_decision(tiny_layout: Layout) -> None:
    core = make_core(tiny_layout, [spec(1)])
    _queued_at_turning(core)
    core.apply(MachineDown(TURNING), 6.0)
    assert all(r.agent != TURNING for r in core.pending_decisions(6.0))
    core.apply(MachineUp(TURNING), 9.0)
    assert any(r.agent == TURNING for r in core.pending_decisions(9.0))


def test_wait_is_requeried_after_next_event(tiny_layout: Layout) -> None:
    core = make_core(tiny_layout, [spec(1), spec(2, release=5.0)])
    core.apply(OrderReleased(O1), 0.0)
    core.assign(R0, WAIT, 0.0)
    assert R0 not in {r.agent for r in core.pending_decisions(0.0)}
    core.apply(OrderReleased(O2), 5.0)
    assert R0 in {r.agent for r in core.pending_decisions(5.0)}
    assert core.history().decisions[0].order is None


def test_machine_wait_can_be_disabled(tiny_layout: Layout) -> None:
    from som.core.factory import CoreCfg

    core = FactoryCore(tiny_layout, [spec(1)], [R0], CoreCfg(machine_wait_allowed=False))
    _queued_at_turning(core)
    with pytest.raises(InfeasibleDecision, match="may not WAIT"):
        core.assign(TURNING, WAIT, 6.0)


def test_candidates_are_canonical_not_heuristic(tiny_layout: Layout) -> None:
    # Later due date released first -> it comes first (availability order, not EDD).
    core = make_core(tiny_layout, [spec(1, due=500.0), spec(2, due=50.0, release=1.0)])
    core.apply(OrderReleased(O1), 0.0)
    core.apply(OrderReleased(O2), 1.0)
    assert [c.order for c in robot_req(core, R0, 1.0).candidates] == [O1, O2]


def test_multi_step_route_goes_machine_to_machine(tiny_layout: Layout) -> None:
    core = make_core(tiny_layout, [spec(1, TURN_MILL)])
    _queued_at_turning(core)
    mreq = next(r for r in core.pending_decisions(6.0) if r.agent == TURNING)
    (start,) = core.assign(TURNING, mreq.candidates[0], 6.0)
    core.apply(ProcessDone(TURNING, O1, start.token), 16.0)
    cand = robot_req(core, R0, 16.0).candidates[0]
    assert (cand.pickup, cand.dest) == (A, B)
