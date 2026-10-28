"""FactoryCore: the single implementation of factory logic.

A deterministic state machine. Runtimes feed it events with ``apply`` and agent choices with
``assign``; both return commands for the runtime to execute. Time is always passed in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from som.core.commands import Command, HoldDock, MoveRobot, Purpose, StartProcess
from som.core.decisions import Candidate, Choice, DecisionRequest, InfeasibleDecision, Wait
from som.core.events import (
    DockReleased,
    Event,
    MachineDown,
    MachineUp,
    OrderReleased,
    ProcessDone,
    RobotArrived,
    RobotFailed,
    Tick,
)
from som.core.history import (
    CoreSnapshot,
    DecisionRecord,
    DockRecord,
    EpisodeLog,
    LegRecord,
    MachineView,
    OrderRecord,
    OrderView,
    ProcessRecord,
    RobotView,
)
from som.core.ids import AgentId, MachineId, OrderId, RobotId, StationId
from som.core.layout import Layout, Pose2D, StationKind
from som.core.order import Order, OrderSpec, OrderStatus


@dataclass(frozen=True, slots=True)
class CoreCfg:
    machine_wait_allowed: bool = True
    debug_invariants: bool = False  # run check_invariants after every apply/assign


class RobotStatus(str, Enum):
    IDLE = "IDLE"
    TO_PICKUP = "TO_PICKUP"
    TO_DROPOFF = "TO_DROPOFF"
    WAITING_DOCK = "WAITING_DOCK"  # at the station, dock held by another robot
    DOCKING = "DOCKING"  # holding the dock, transfer in progress


@dataclass(slots=True)
class TransportTask:
    order: OrderId
    pickup: StationId
    dest: StationId
    phase: Purpose


@dataclass(slots=True)
class RobotState:
    id: RobotId
    pose: Pose2D
    status: RobotStatus = RobotStatus.IDLE
    station: StationId | None = None
    carrying: OrderId | None = None
    task: TransportTask | None = None
    goal_id: str | None = None
    goal_seq: int = 0
    leg: LegRecord | None = None
    dock_rec: DockRecord | None = None


@dataclass(slots=True)
class MachineState:
    id: MachineId
    station: StationId
    process: str
    capacity: int
    queue: list[OrderId] = field(default_factory=list)
    inbound: int = 0  # reserved deliveries on their way here
    current: OrderId | None = None
    busy_until: float | None = None
    paused_remaining: float | None = None
    down: bool = False
    token: int = 0
    segment: ProcessRecord | None = None


@dataclass(slots=True)
class DockState:
    holder: RobotId | None = None
    waiting: list[RobotId] = field(default_factory=list)


class FactoryCore:
    def __init__(
        self,
        layout: Layout,
        orders: Sequence[OrderSpec],
        robots: Sequence[RobotId],
        cfg: CoreCfg | None = None,
    ) -> None:
        if len(robots) > len(layout.robot_starts):
            raise ValueError(f"layout has {len(layout.robot_starts)} start poses")
        self.layout = layout
        self.cfg = cfg or CoreCfg()
        self.now = 0.0
        self.orders: dict[OrderId, Order] = {}
        for spec in orders:
            for p in spec.job.route:
                layout.station_for_process(p)  # raises if the layout lacks this process
            if spec.id in self.orders:
                raise ValueError(f"duplicate order {spec.id}")
            self.orders[spec.id] = Order(spec)
        self.robots: dict[RobotId, RobotState] = {
            r: RobotState(r, layout.robot_starts[i]) for i, r in enumerate(robots)
        }
        self.machines: dict[MachineId, MachineState] = {}
        for s in layout.machines:
            assert s.machine is not None and s.process is not None and s.input_capacity
            self.machines[s.machine] = MachineState(s.machine, s.id, s.process, s.input_capacity)
        self.pools: dict[StationId, list[OrderId]] = {s.id: [] for s in layout.stations}
        self.delivered: list[OrderId] = []
        self.docks: dict[StationId, DockState] = {s.id: DockState() for s in layout.stations}
        self.waiting: set[AgentId] = set()  # chose WAIT since the last event
        self._log = EpisodeLog(
            layout=layout.name,
            robots=list(self.robots),
            machines=list(self.machines),
            orders=[],
        )

    # ------------------------------------------------------------------ queries

    @property
    def done(self) -> bool:
        return len(self.delivered) == len(self.orders)

    def dest_for(self, order: Order) -> StationId:
        p = order.next_process
        return self.layout.sink.id if p is None else self.layout.station_for_process(p).id

    def machine_at(self, sid: StationId) -> MachineState | None:
        st = self.layout.station(sid)
        return self.machines[st.machine] if st.machine else None

    def has_input_capacity(self, sid: StationId) -> bool:
        m = self.machine_at(sid)
        if m is None:
            return self.layout.station(sid).kind is StationKind.SINK
        return not m.down and len(m.queue) + m.inbound < m.capacity

    def _robot_candidates(self) -> tuple[Candidate, ...]:
        cands = []
        for sid, pool in self.pools.items():
            for oid in pool:
                o = self.orders[oid]
                if o.status in (OrderStatus.RELEASED, OrderStatus.WAITING_PICKUP):
                    dest = self.dest_for(o)
                    if self.has_input_capacity(dest):
                        cands.append(Candidate(oid, sid, dest, o.available_since))
        return self._canonical(cands)

    def _machine_candidates(self, m: MachineState) -> tuple[Candidate, ...]:
        return self._canonical(
            [Candidate(o, m.station, m.station, self.orders[o].available_since) for o in m.queue]
        )

    def _canonical(self, cands: list[Candidate]) -> tuple[Candidate, ...]:
        return tuple(
            sorted(cands, key=lambda c: (c.available_since, self.orders[c.order].spec.number))
        )

    def pending_decisions(self, now: float) -> list[DecisionRequest]:
        """Agents that are idle AND have >=1 feasible non-WAIT option. Deterministic order."""
        self._check_time(now)
        reqs: list[DecisionRequest] = []
        robot_cands = self._robot_candidates()
        if robot_cands:
            for r in self.robots.values():
                if r.status is RobotStatus.IDLE and r.id not in self.waiting:
                    reqs.append(DecisionRequest(r.id, "robot", robot_cands, True))
        for m in self.machines.values():
            idle = m.current is None and m.paused_remaining is None and not m.down
            if idle and m.queue and m.id not in self.waiting:
                cands = self._machine_candidates(m)
                reqs.append(DecisionRequest(m.id, "machine", cands, self.cfg.machine_wait_allowed))
        return reqs

    # ------------------------------------------------------------------ decisions

    def assign(self, agent: AgentId, choice: Choice, now: float) -> list[Command]:
        """Apply one agent's decision. Raises InfeasibleDecision if it isn't in its request."""
        req = next((r for r in self.pending_decisions(now) if r.agent == agent), None)
        if req is None:
            raise InfeasibleDecision(f"{agent} has no pending decision at t={now}")
        self.now = now
        if isinstance(choice, Wait):
            if not req.wait_allowed:
                raise InfeasibleDecision(f"{agent} may not WAIT")
            self.waiting.add(agent)
            self._log.decisions.append(DecisionRecord(now, agent, None, len(req.candidates)))
            return []
        if choice not in req.candidates:
            raise InfeasibleDecision(f"{agent}: {choice} is not a current candidate")
        self._log.decisions.append(DecisionRecord(now, agent, choice.order, len(req.candidates)))
        if req.role == "robot":
            cmds = self._assign_robot(RobotId(agent), choice)
        else:
            cmds = self._assign_machine(MachineId(agent), choice)
        self._debug_check()
        return cmds

    def _assign_robot(self, robot: RobotId, c: Candidate) -> list[Command]:
        r = self.robots[robot]
        o = self.orders[c.order]
        o.transition(OrderStatus.RESERVED, self.now, c.pickup)
        o.reserved_by = r.id
        m = self.machine_at(c.dest)
        if m is not None:
            m.inbound += 1
        r.task = TransportTask(c.order, c.pickup, c.dest, "pickup")
        r.status = RobotStatus.TO_PICKUP
        return [self._move(r, c.pickup, "pickup")]

    def _assign_machine(self, machine: MachineId, c: Candidate) -> list[Command]:
        m = self.machines[machine]
        o = self.orders[c.order]
        m.queue.remove(c.order)
        o.transition(OrderStatus.PROCESSING, self.now, m.station)
        duration = o.spec.process_times[o.step_idx]
        return [self._start_run(m, c.order, duration)]

    def _move(self, r: RobotState, sid: StationId, purpose: Purpose) -> MoveRobot:
        r.goal_seq += 1
        r.goal_id = f"{r.id}#{r.goal_seq}"
        r.station = None
        order = r.task.order if r.task else None
        r.leg = LegRecord(r.id, order, purpose, self.now, (r.pose.x, r.pose.y), sid)
        self._log.legs.append(r.leg)
        return MoveRobot(r.id, sid, purpose, r.goal_id)

    def _start_run(self, m: MachineState, oid: OrderId, duration: float) -> StartProcess:
        m.token += 1
        m.current = oid
        m.busy_until = self.now + duration
        m.paused_remaining = None
        m.segment = ProcessRecord(m.id, oid, self.now)
        self._log.processes.append(m.segment)
        return StartProcess(m.id, oid, duration, m.token)

    # ------------------------------------------------------------------ events

    def apply(self, event: Event, now: float) -> list[Command]:
        """Advance state for one external event. Returns commands the runtime must execute."""
        self._check_time(now)
        self.now = now
        self.waiting.clear()
        cmds: list[Command]
        if isinstance(event, OrderReleased):
            cmds = self._on_released(event)
        elif isinstance(event, RobotArrived):
            cmds = self._on_arrived(event)
        elif isinstance(event, DockReleased):
            cmds = self._on_dock_released(event)
        elif isinstance(event, ProcessDone):
            cmds = self._on_process_done(event)
        elif isinstance(event, RobotFailed):
            cmds = self._on_failed(event)
        elif isinstance(event, MachineDown):
            cmds = self._on_machine_down(event)
        elif isinstance(event, MachineUp):
            cmds = self._on_machine_up(event)
        elif isinstance(event, Tick):
            cmds = []
        else:  # pragma: no cover - exhaustive over Event
            raise TypeError(f"unknown event {event!r}")
        self._debug_check()
        return cmds

    def _on_released(self, e: OrderReleased) -> list[Command]:
        o = self.orders[e.order]
        src = self.layout.source.id
        o.transition(OrderStatus.RELEASED, self.now, src)
        o.available_since = self.now
        self.pools[src].append(o.id)
        return []

    def _current_goal(self, robot: RobotId, goal_id: str | None) -> RobotState | None:
        r = self.robots[robot]
        if r.goal_id is None or (goal_id is not None and goal_id != r.goal_id):
            return None  # stale: the goal was superseded or failed
        return r

    def _on_arrived(self, e: RobotArrived) -> list[Command]:
        r = self._current_goal(e.robot, e.goal_id)
        if r is None or r.status not in (RobotStatus.TO_PICKUP, RobotStatus.TO_DROPOFF):
            return []
        if r.leg is not None:
            r.leg.arrive = self.now
            r.leg = None
        r.pose = self.layout.station(e.station).dock
        r.station = e.station
        r.dock_rec = DockRecord(r.id, e.station, self.now)
        self._log.docks.append(r.dock_rec)
        dock = self.docks[e.station]
        if dock.holder is None:
            return [self._grant_dock(r, e.station)]
        r.status = RobotStatus.WAITING_DOCK
        dock.waiting.append(r.id)
        return []

    def _grant_dock(self, r: RobotState, sid: StationId) -> HoldDock:
        self.docks[sid].holder = r.id
        r.status = RobotStatus.DOCKING
        if r.dock_rec is not None:
            r.dock_rec.start = self.now
        assert r.goal_id is not None
        return HoldDock(r.id, sid, self.layout.dock_time_s, r.goal_id)

    def _release_dock(self, sid: StationId) -> list[Command]:
        dock = self.docks[sid]
        dock.holder = None
        if not dock.waiting:
            return []
        nxt = self.robots[dock.waiting.pop(0)]
        return [self._grant_dock(nxt, sid)]

    def _on_dock_released(self, e: DockReleased) -> list[Command]:
        r = self._current_goal(e.robot, e.goal_id)
        if r is None or r.status is not RobotStatus.DOCKING or r.task is None:
            return []
        if r.dock_rec is not None:
            r.dock_rec.end = self.now
            r.dock_rec = None
        task, o = r.task, self.orders[r.task.order]
        cmds: list[Command] = []
        if task.phase == "pickup":
            self.pools[task.pickup].remove(o.id)
            o.transition(OrderStatus.IN_TRANSIT, self.now, r.id)
            r.carrying = o.id
            task.phase = "dropoff"
            r.status = RobotStatus.TO_DROPOFF
            cmds.append(self._move(r, task.dest, "dropoff"))
        else:
            m = self.machine_at(task.dest)
            if m is None:
                o.transition(OrderStatus.DELIVERED, self.now, task.dest)
                o.delivered_time = self.now
                self.delivered.append(o.id)
            else:
                m.inbound -= 1
                o.transition(OrderStatus.QUEUED, self.now, task.dest)
                o.available_since = self.now
                m.queue.append(o.id)
            o.reserved_by = None
            r.carrying = None
            r.task = None
            r.goal_id = None
            r.status = RobotStatus.IDLE
        cmds.extend(self._release_dock(e.station))
        return cmds

    def _on_process_done(self, e: ProcessDone) -> list[Command]:
        m = self.machines[e.machine]
        if e.token != m.token or m.current != e.order or m.paused_remaining is not None:
            return []  # stale: the run was paused or replaced
        o = self.orders[e.order]
        o.step_idx += 1
        o.transition(OrderStatus.WAITING_PICKUP, self.now, m.station)
        o.available_since = self.now
        self.pools[m.station].append(o.id)
        if m.segment is not None:
            m.segment.end = self.now
            m.segment = None
        m.current = None
        m.busy_until = None
        return []

    def _on_failed(self, e: RobotFailed) -> list[Command]:
        r = self._current_goal(e.robot, e.goal_id)
        if r is None or r.task is None:
            return []
        self._log.failures.append((self.now, r.id, e.reason))
        task, o = r.task, self.orders[r.task.order]
        back = (
            OrderStatus.RELEASED
            if task.pickup == self.layout.source.id
            else (OrderStatus.WAITING_PICKUP)
        )
        if r.carrying is not None:  # order returns to the station it was picked up from
            self.pools[task.pickup].append(o.id)
            r.carrying = None
        o.transition(back, self.now, task.pickup)
        o.reserved_by = None
        o.available_since = self.now
        m = self.machine_at(task.dest)
        if m is not None:
            m.inbound -= 1
        if r.leg is not None:
            r.leg = None  # arrive stays None: the leg never completed
        cmds: list[Command] = []
        for sid, dock in self.docks.items():
            if r.id in dock.waiting:
                dock.waiting.remove(r.id)
            if dock.holder == r.id:
                cmds.extend(self._release_dock(sid))
        if r.dock_rec is not None:
            r.dock_rec.end = self.now
            r.dock_rec = None
        r.task = None
        r.goal_id = None
        r.status = RobotStatus.IDLE
        return cmds

    def _on_machine_down(self, e: MachineDown) -> list[Command]:
        m = self.machines[e.machine]
        if m.down:
            return []
        m.down = True
        if m.current is not None and m.busy_until is not None:
            m.paused_remaining = m.busy_until - self.now
            m.busy_until = None
            m.token += 1  # invalidates the pending ProcessDone
            if m.segment is not None:
                m.segment.end = self.now
                m.segment = None
        return []

    def _on_machine_up(self, e: MachineUp) -> list[Command]:
        m = self.machines[e.machine]
        if not m.down:
            return []
        m.down = False
        if m.current is not None and m.paused_remaining is not None:
            return [self._start_run(m, m.current, m.paused_remaining)]
        return []

    # ------------------------------------------------------------------ views

    def snapshot(self) -> CoreSnapshot:
        return CoreSnapshot(
            now=self.now,
            layout=self.layout,
            robots=tuple(
                RobotView(
                    r.id,
                    r.status.value,
                    r.pose,
                    r.station,
                    r.carrying,
                    r.task.order if r.task else None,
                    r.task.dest if r.task else None,
                )
                for r in self.robots.values()
            ),
            machines=tuple(
                MachineView(
                    m.id,
                    m.station,
                    m.process,
                    m.current,
                    m.busy_until,
                    m.paused_remaining,
                    m.down,
                    tuple(m.queue),
                    m.inbound,
                    m.capacity,
                )
                for m in self.machines.values()
            ),
            orders=tuple(
                OrderView(
                    o.id,
                    o.spec.number,
                    o.spec.job.name,
                    o.spec.job.route,
                    o.spec.process_times,
                    o.step_idx,
                    o.spec.release_time,
                    o.spec.due_time,
                    o.status.value,
                    o.location,
                    o.reserved_by,
                    o.available_since,
                )
                for o in self.orders.values()
            ),
        )

    def history(self) -> EpisodeLog:
        """Per-order timeline plus per-agent decisions, legs, docks and process runs."""
        self._log.orders = [
            OrderRecord(
                o.id,
                o.spec.job.name,
                list(o.spec.job.route),
                o.spec.release_time,
                o.spec.due_time,
                o.delivered_time,
                list(o.timeline),
            )
            for o in self.orders.values()
        ]
        self._log.end_time = self.now
        self._log.terminated = self.done
        return self._log

    # ------------------------------------------------------------------ internals

    def _check_time(self, now: float) -> None:
        if now < self.now:
            raise ValueError(f"time went backwards: {now} < {self.now}")

    def _debug_check(self) -> None:
        if self.cfg.debug_invariants:
            from som.core.invariants import check_invariants

            problems = check_invariants(self)
            if problems:
                raise AssertionError("; ".join(problems))
