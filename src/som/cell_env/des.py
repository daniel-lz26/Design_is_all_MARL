"""CellSim: discrete-event simulation of the 4-TurtleBot assembly cell (spec §2-§7).

Continuous time, one event heap, no physics. Robots claim transport tasks from a shared task
board; everything after a claim is automatic: path planning on the roadmap with edge
reservations, queueing for the dock, waiting for the station arm, the handoff, undocking and
delivery. Machines take boxes from their in-buffer by ``machine_seq_rule``, and block when
their out-buffer is full.

The policy-facing API is small: ``advance()`` runs to the next decision point and returns the
robots that must decide; ``tasks(robot)`` / ``mask(robot)`` describe the options;
``act(robot, action)`` applies one choice. Actions: ``0..K-1`` claim the k-th open task (sorted
by due time, then order number), ``K`` go park, ``K+1`` wait.

Every random draw comes from one ``SeedSequence(seed)``: orders, disruptions, travel, arm and
process noise each get a child stream, and disruptions are drawn at reset.
"""

from __future__ import annotations

import heapq
import math
from pathlib import Path

import numpy as np

from som.cell_env import disruptions as dz
from som.cell_env.entities import RS, Box, CellLog, Loc, Robot, Station
from som.cell_env.orders import OrderSpec, build_orders
from som.cell_env.roadmap import Edge, Roadmap
from som.config.cell_schema import CellLayoutCfg, CellScenarioCfg
from som.config.loader import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

# event kinds
(
    RELEASE,
    ARRIVE,
    EDGE_RETRY,
    DOCK_DONE,
    HANDOFF_DONE,
    UNDOCK_DONE,
    MACHINE_DONE,
    WAIT_END,
    CLEAR,
    RESCUE,
    DISRUPT,
) = range(11)

IDLE_STATES = (RS.IDLE, RS.IDLE_PARKED)


class CellSim:
    def __init__(
        self, scenario: CellScenarioCfg, layout: CellLayoutCfg, root: Path = REPO_ROOT
    ) -> None:
        self.cfg = scenario
        self.layout = layout
        self.root = root
        self.roadmap = Roadmap(layout)
        self.k = scenario.k_tasks
        self.machine_rule = scenario.machine_seq_rule  # baselines may override per run
        self.machine_priority: dict[str, float] | None = None  # set by planners: lowest first
        self.h = scenario.handling or layout.handling
        self.debug = False  # run check_invariants after every event and action
        self.robot_ids = [f"agv_{i + 1}" for i in range(scenario.n_robots)]
        if scenario.n_robots > len(layout.parking):
            raise ValueError("more robots than parking spots")
        self.machine_of = {s.process: k for k, s in layout.stations.items() if s.process}
        self.src = next(k for k, s in layout.stations.items() if s.kind == "source")
        self.sink = next(k for k, s in layout.stations.items() if s.kind == "sink")

    # ------------------------------------------------------------------ reset

    def reset(self, seed: int) -> list[str]:
        cfg, lay = self.cfg, self.layout
        ss = np.random.SeedSequence(seed).spawn(5)
        orders_rng, dis_rng, self.travel_rng, self.arm_rng, self.proc_rng = (
            np.random.default_rng(s) for s in ss
        )
        for e in self.roadmap.edges.values():
            e.occupants.clear()
            e.waiters.clear()
            e.closed = False
        orders = build_orders(cfg, lay, self.h, self.roadmap, orders_rng, self.root)
        self.schedule = dz.draw(
            cfg.disruptions, self.robot_ids, lay, self.h, self.roadmap, orders, dis_rng
        )
        orders = orders + self.schedule.rush_orders
        self.t_max = cfg.t_max or 3.0 * max(o.due for o in orders)
        self.now = 0.0
        self._heap: list[tuple[float, int, int, str, int]] = []
        self._seq = 0
        self.board = 0  # task-board version
        self.stuck: dict[frozenset[str], str] = {}  # edge -> broken robot blocking it
        self.stations = {
            k: Station(
                k,
                s.kind,
                s.process,
                s.dock,
                s.approach,
                s.in_buffer_cap,
                s.out_buffer_cap,
                s.arm_load_s or self.h.arm_load_s,
                s.arm_unload_s or self.h.arm_unload_s,
            )
            for k, s in lay.stations.items()
        }
        self.boxes = {o.id: Box(o) for o in orders}
        self.robots = {
            rid: Robot(rid, i, lay.parking[i], node=lay.parking[i])
            for i, rid in enumerate(self.robot_ids)
        }
        self.delivered = 0
        self.truncated = False
        self.fed_idle = 0  # boxes dropped into the in-buffer of an idle machine (reward shaping)
        self.empty_travel_s = 0.0
        self._since: dict[str, float] = {}
        self.log = CellLog(cfg.name, seed, list(self.robot_ids), list(self.stations))
        for r in self.robots.values():
            self.log.robot_states[r.id] = []
            self.log.moves[r.id] = []
            self.log.carrying[r.id] = [(0.0, None)]
            self.log.claims[r.id] = [(0.0, None)]
            self.log.down_at[r.id] = []
            absent = r.id in self.schedule.absent
            self._set(r, RS.ABSENT if absent else RS.IDLE_PARKED)
            r.needs_decision = False
        for o in orders:
            self._push(o.release, RELEASE, o.id)
        for t, kind, target in self.schedule.events:
            self._push(t, DISRUPT, f"{kind}|{target}")
        self.log.meta["coordinator_outages"] = list(self.schedule.coordinator_outages)
        self.log.meta["t_max"] = self.t_max
        return self.advance()

    # ------------------------------------------------------------------ queries

    @property
    def done(self) -> bool:
        return self.delivered == len(self.boxes)

    def next_station(self, box: Box) -> Station:
        route = box.spec.route
        key = self.sink if box.step >= len(route) else self.machine_of[route[box.step]]
        return self.stations[key]

    def open_tasks(self) -> list[Box]:
        """Unclaimed boxes in out-buffers whose next station has room, by (due, number)."""
        out = []
        for st in self.stations.values():
            for bid in st.out_buf:
                b = self.boxes[bid]
                if b.claimed_by is None and self.next_station(b).in_free() > 0:
                    out.append(b)
        out.sort(key=lambda b: (b.spec.due, b.spec.number))
        return out

    def tasks(self, rid: str) -> list[Box]:
        return self.open_tasks()[: self.k] if self._can_act(self.robots[rid]) else []

    def mask(self, rid: str) -> np.ndarray:
        """K claim slots, then park, then wait. A robot that cannot act gets all zeros."""
        m = np.zeros(self.k + 2, dtype=np.int8)
        r = self.robots[rid]
        if not self._can_act(r):
            return m
        m[: len(self.tasks(rid))] = 1
        m[self.k] = int(not (r.state is RS.IDLE_PARKED and r.node == r.park))
        m[self.k + 1] = 1
        return m

    def location(self, rid: str) -> str:
        """Roadmap node a robot would start from if it set off now."""
        r = self.robots[rid]
        if r.docked_at is not None:
            return self.stations[r.docked_at].approach
        return r.node

    def _can_act(self, r: Robot) -> bool:
        return r.state in IDLE_STATES and r.task is None and r.box is None

    def pending(self) -> list[str]:
        """Robots at a decision point, in id order."""
        tasks_exist = None
        out = []
        for r in self.robots.values():
            if not self._can_act(r):
                continue
            if r.needs_decision:
                out.append(r.id)
                continue
            if r.seen_board < self.board:  # something new happened since it last decided
                if tasks_exist is None:
                    tasks_exist = bool(self.open_tasks())
                if tasks_exist:
                    out.append(r.id)
        return out

    # ------------------------------------------------------------------ decisions

    def act(self, rid: str, action: int) -> None:
        r = self.robots[rid]
        mask = self.mask(rid)
        if not 0 <= action < len(mask) or not mask[action]:
            raise ValueError(f"{rid}: action {action} is masked out")
        self.log.decisions.append((self.now, rid, action))
        r.needs_decision = False
        r.seen_board = self.board
        if action < self.k:
            self._claim(r, self.tasks(rid)[action])
        elif action == self.k:
            r.purpose = "park"
            self._leave(r)
        else:
            r.purpose = "wait"
            r.wait_until = self.now + self.cfg.wait_cap_s
            self._push(r.wait_until, WAIT_END, r.id, r.token)
            self._leave(r)
        self._check()

    def _check(self) -> None:
        if self.debug:
            from som.cell_env.invariants import check_invariants

            problems = check_invariants(self)
            if problems:
                raise AssertionError(f"t={self.now:.2f}: " + "; ".join(problems))

    def _claim(self, r: Robot, box: Box) -> None:
        box.claimed_by = r.id
        self.next_station(box).inbound += 1
        r.task = box.id
        self.log.claims[r.id].append((self.now, box.id))
        r.purpose = "pickup"
        if r.docked_at == box.where:
            self._request_arm(r, self.stations[box.where])
        else:
            self._leave(r)

    # ------------------------------------------------------------------ main loop

    def advance(self) -> list[str]:
        """Process events until a robot must decide; [] when the episode is over."""
        heap = self._heap
        while True:
            if self.done:
                self._finish()
                return []
            if not heap or heap[0][0] > self.now:
                pend = self.pending()
                if pend:
                    return pend
            if not heap or heap[0][0] > self.t_max:
                self.now = self.t_max
                self.truncated = True
                self._finish()
                return []
            t, _, kind, arg, token = heapq.heappop(heap)
            self.now = t
            self._dispatch(kind, arg, token)
            self._check()

    def _finish(self) -> None:
        log = self.log
        log.end_time = self.now
        log.terminated = self.done
        log.truncated = self.truncated and not self.done
        log.orders = [
            {
                "id": b.id,
                "job": b.spec.job,
                "route": list(b.spec.route),
                "release": b.spec.release,
                "due": b.spec.due,
                "delivered": b.delivered,
                "rush": b.spec.rush,
            }
            for b in self.boxes.values()
        ]

    def _push(self, t: float, kind: int, arg: str, token: int = 0) -> None:
        heapq.heappush(self._heap, (t, self._seq, kind, arg, token))
        self._seq += 1

    def _dispatch(self, kind: int, arg: str, token: int) -> None:
        if kind == RELEASE:
            self._put_out(self.stations[self.src], self.boxes[arg])
        elif kind == MACHINE_DONE:
            self._machine_done(self.stations[arg], token)
        elif kind == HANDOFF_DONE:
            self._handoff_done(self.stations[arg], token)
        elif kind == DISRUPT:
            self._disrupt(*arg.split("|", 1))
        else:
            r = self.robots[arg]
            if token != r.token:
                return  # stale: the robot broke down after this was scheduled
            if kind == ARRIVE:
                self._arrive(r)
            elif kind == EDGE_RETRY:
                if r.state is RS.BLOCKED and r.edge is None:
                    self._step(r)
            elif kind == DOCK_DONE:
                self._dock_done(r)
            elif kind == UNDOCK_DONE:
                self._undock_done(r)
            elif kind == WAIT_END:
                if r.purpose == "wait" and self._can_act(r):
                    r.needs_decision = True
            elif kind == CLEAR:
                self._clear(r)
            elif kind == RESCUE:
                self._rescue(r)

    # ------------------------------------------------------------------ movement

    def _set(self, r: Robot, state: RS) -> None:
        if r.state is not state or not self.log.robot_states[r.id]:
            if r.state is RS.TRAVEL_EMPTY:
                self.empty_travel_s += self.now - self._since.get(r.id, self.now)
            self._since[r.id] = self.now
            r.state = state
            self.log.robot_states[r.id].append((self.now, state.value))

    def _leave(self, r: Robot) -> None:
        """Undock first if docked, then carry on with the robot's purpose."""
        if r.docked_at is not None:
            st = self.stations[r.docked_at]
            self._set(r, RS.UNDOCKING)
            dur = self.h.undock_time_s + self.roadmap.edge_time(st.approach, st.dock)
            self._push(self.now + dur, UNDOCK_DONE, r.id, r.token)
        else:
            self._continue(r)

    def _continue(self, r: Robot) -> None:
        if r.purpose == "pickup":
            assert r.task is not None
            self._go(r, self.stations[self.boxes[r.task].where].approach)  # where = pickup station
        elif r.purpose == "dropoff":
            assert r.box is not None
            self._go(r, self.next_station(self.boxes[r.box]).approach)
        elif r.purpose == "park":
            self._go(r, r.park)
        else:  # wait where it is
            self._set(r, RS.IDLE)

    def _avoid(self) -> frozenset[frozenset[str]]:
        return self.roadmap.closed_edges() | frozenset(self.stuck)

    def _go(self, r: Robot, goal: str) -> None:
        r.goal = goal
        _, r.path = self.roadmap.path(r.node, goal, self._avoid())
        if r.node != goal and not r.path:
            _, r.path = self.roadmap.path(r.node, goal)  # no way round: wait for the blockage
        self._step(r)

    def _step(self, r: Robot) -> None:
        if not r.path:
            self._arrived(r)
            return
        nxt = r.path[0]
        e = self.roadmap.edge(r.node, nxt)
        if e.closed or e.key in self.stuck:
            _, alt = self.roadmap.path(r.node, r.goal, self._avoid())
            if alt and alt[0] != nxt:
                r.path = alt
                self._step(r)
                return
        if e.closed or e.key in self.stuck or len(e.occupants) >= e.capacity:
            if r.id not in e.waiters:
                e.waiters.append(r.id)
            if not r.blocked_now:
                r.blocked_now = True
                self.log.blocked_trips.append((self.now, r.id, r.node))
                self._check_deadlock(r, e)
            self._set(r, RS.BLOCKED)
            return
        r.blocked_now = False
        e.occupants.append(r.id)
        r.edge = e
        dur = e.length / self.roadmap.v
        sigma = self.cfg.noise.speed_sigma
        if sigma > 0:
            dur *= float(np.exp(self.travel_rng.normal(-0.5 * sigma * sigma, sigma)))
        if r.prev is not None and self.roadmap.is_turn(r.prev, r.node, nxt):
            dur += self.roadmap.turn_s
        self._set(r, RS.TRAVEL_LOADED if r.box else RS.TRAVEL_EMPTY)
        r.seg = (self.now, self.now + dur, r.node, nxt)
        self.log.moves[r.id].append(r.seg)
        self._push(self.now + dur, ARRIVE, r.id, r.token)

    def _arrive(self, r: Robot) -> None:
        e: Edge = r.edge
        e.occupants.remove(r.id)
        r.prev, r.node = r.node, r.path.pop(0)
        r.edge = None
        r.seg = None
        self._wake(e)
        self._step(r)

    def _wake(self, e: Edge) -> None:
        free = e.capacity - len(e.occupants)
        if e.closed or e.key in self.stuck or free <= 0:
            return
        for rid in e.waiters[:free]:
            self._push(self.now, EDGE_RETRY, rid, self.robots[rid].token)
        del e.waiters[:free]

    def _check_deadlock(self, r: Robot, e: Edge) -> None:
        """A wait cycle cannot form here (waiting robots hold nothing); log one if it does."""
        seen, frontier = {r.id}, [o for o in e.occupants]
        while frontier:
            o = self.robots[frontier.pop()]
            if o.id in seen:
                self.log.deadlocks.append((self.now, sorted(seen)))
                return
            seen.add(o.id)
            if o.state is RS.BLOCKED and o.path:
                frontier += self.roadmap.edge(o.node, o.path[0]).occupants

    def _arrived(self, r: Robot) -> None:
        if r.purpose in ("pickup", "dropoff"):
            if r.purpose == "pickup":
                st = self.stations[self.boxes[r.task or ""].where]
            else:
                st = self.next_station(self.boxes[r.box or ""])
            r.dock_target = st.key
            if st.dock_holder is None and not st.dock_queue:
                self._grant_dock(r, st)
            else:
                st.dock_queue.append(r.id)
                self._set(r, RS.WAIT_DOCK)
        elif r.purpose == "park":
            self._set(r, RS.IDLE_PARKED)
            r.purpose = None
        else:
            self._set(r, RS.IDLE)

    # ------------------------------------------------------------------ docks and arms

    def _grant_dock(self, r: Robot, st: Station) -> None:
        st.dock_holder = r.id
        self._set(r, RS.DOCKING)
        dur = self.h.dock_time_s + self.roadmap.edge_time(st.approach, st.dock)
        self.log.docks.append((r.id, st.key, self.now, None))
        self._push(self.now + dur, DOCK_DONE, r.id, r.token)

    def _dock_done(self, r: Robot) -> None:
        st = self.stations[r.dock_target or ""]
        r.docked_at = st.key
        r.prev, r.node = st.approach, st.dock
        self._request_arm(r, st)

    def _undock_done(self, r: Robot) -> None:
        st = self.stations[r.docked_at or ""]
        r.docked_at = None
        r.prev, r.node = st.dock, st.approach
        self._close_dock(r.id)
        self._release_dock(st)
        self._continue(r)

    def _close_dock(self, rid: str) -> None:
        for i in range(len(self.log.docks) - 1, -1, -1):
            d = self.log.docks[i]
            if d[0] == rid and d[3] is None:
                self.log.docks[i] = (*d[:3], self.now)
                return

    def _release_dock(self, st: Station) -> None:
        st.dock_holder = None
        while st.dock_queue:
            nxt = self.robots[st.dock_queue.pop(0)]
            if nxt.state is RS.WAIT_DOCK:
                self._grant_dock(nxt, st)
                return

    def _request_arm(self, r: Robot, st: Station) -> None:
        st.arm_queue.append(r.id)
        self._set(r, RS.WAIT_ARM)
        self._arm_try(st)

    def _arm_try(self, st: Station) -> None:
        if st.arm_busy is not None or st.arm_down or not st.arm_queue:
            return
        r = self.robots[st.arm_queue.pop(0)]
        st.arm_busy = r.id
        st.arm_kind = "unload" if r.box else "load"
        dur = st.unload_s if r.box else st.load_s
        tri = self.cfg.noise.arm_tri
        if tri > 0:
            dur *= float(self.arm_rng.triangular(1 - tri, 1, 1 + tri))
        st.arm_until = self.now + dur
        st.arm_token += 1
        self._set(r, RS.BEING_UNLOADED if r.box else RS.BEING_LOADED)
        self.log.handoffs.append((st.key, r.id, st.arm_kind, self.now, None))
        self._push(st.arm_until, HANDOFF_DONE, st.key, st.arm_token)

    def _close_handoff(self, st: Station, rid: str) -> None:
        for i in range(len(self.log.handoffs) - 1, -1, -1):
            h = self.log.handoffs[i]
            if h[0] == st.key and h[1] == rid and h[4] is None:
                self.log.handoffs[i] = (*h[:4], self.now)
                return

    def _handoff_done(self, st: Station, token: int) -> None:
        if token != st.arm_token or st.arm_busy is None:
            return
        r = self.robots[st.arm_busy]
        st.arm_busy = None
        self._close_handoff(st, r.id)
        if st.arm_kind == "load":
            box = self.boxes[r.task or ""]
            st.out_buf.remove(box.id)
            box.loc, box.where = Loc.ROBOT, r.id
            r.box = box.id
            self.log.carrying[r.id].append((self.now, box.id))
            self._log_buffers(st)
            self._unblock_machine(st)
            r.purpose = "dropoff"
            self._leave(r)
        else:
            box = self.boxes[r.box or ""]
            r.box = r.task = None
            r.purpose = None
            self.log.carrying[r.id].append((self.now, None))
            self.log.claims[r.id].append((self.now, None))
            st.inbound -= 1
            box.claimed_by = None
            if st.kind == "sink":
                box.loc, box.where, box.delivered = Loc.DONE, st.key, self.now
                self.delivered += 1
            else:
                box.loc, box.where = Loc.IN, st.key
                if st.current is None and st.blocked_box is None and not st.down:
                    self.fed_idle += 1
                st.in_buf.append(box.id)
                self._log_buffers(st)
                self._machine_try(st)
            self._set(r, RS.IDLE)
            r.needs_decision = True
        self._arm_try(st)

    # ------------------------------------------------------------------ machines

    def _log_buffers(self, st: Station) -> None:
        self.log.buffers.append((self.now, st.key, len(st.in_buf), len(st.out_buf)))

    def _bump_board(self) -> None:
        self.board += 1

    def _put_out(self, st: Station, box: Box) -> None:
        st.out_buf.append(box.id)
        box.loc, box.where, box.t_avail = Loc.OUT, st.key, self.now
        self._log_buffers(st)
        self._bump_board()

    def _machine_try(self, st: Station) -> None:
        if st.kind != "machine" or st.down or st.current or st.blocked_box or not st.in_buf:
            return
        rule = self.machine_rule
        if self.machine_priority is not None:
            pri = self.machine_priority
            bid = min(st.in_buf, key=lambda b: (pri.get(b, math.inf), self.boxes[b].spec.number))
        elif rule == "FIFO":
            bid = st.in_buf[0]
        elif rule == "EDD":
            bid = min(st.in_buf, key=lambda b: (self.boxes[b].spec.due, self.boxes[b].spec.number))
        else:
            bid = min(
                st.in_buf, key=lambda b: (self._proc(self.boxes[b]), self.boxes[b].spec.number)
            )
        st.in_buf.remove(bid)
        box = self.boxes[bid]
        box.loc, box.where = Loc.MACHINE, st.key
        st.current = bid
        dur = self.h.conveyor_transfer_s + self._proc(box)
        sigma = self.cfg.noise.process_sigma
        if sigma > 0:
            dur *= float(np.exp(self.proc_rng.normal(-0.5 * sigma * sigma, sigma)))
        self._start_run(st, dur)
        self._log_buffers(st)
        self._bump_board()  # an in-buffer slot opened

    def _proc(self, box: Box) -> float:
        return box.spec.proc[box.step]

    def _start_run(self, st: Station, dur: float) -> None:
        st.token += 1
        st.busy_until = self.now + dur
        st.paused = None
        self.log.machine_runs.append((st.key, st.current or "", self.now, None))
        self._push(st.busy_until, MACHINE_DONE, st.key, st.token)

    def _close_run(self, st: Station) -> None:
        for i in range(len(self.log.machine_runs) - 1, -1, -1):
            run = self.log.machine_runs[i]
            if run[0] == st.key and run[3] is None:
                self.log.machine_runs[i] = (*run[:3], self.now)
                return

    def _machine_done(self, st: Station, token: int) -> None:
        if token != st.token or st.current is None:
            return
        box = self.boxes[st.current]
        st.current, st.busy_until = None, None
        self._close_run(st)
        box.step += 1
        if st.out_cap is None or len(st.out_buf) < st.out_cap:
            self._put_out(st, box)
        else:
            st.blocked_box = box.id
            self.log.machine_blocked.append((st.key, self.now, None))
        self._machine_try(st)

    def _unblock_machine(self, st: Station) -> None:
        if st.blocked_box is None:
            return
        box = self.boxes[st.blocked_box]
        st.blocked_box = None
        for i in range(len(self.log.machine_blocked) - 1, -1, -1):
            if self.log.machine_blocked[i][0] == st.key and self.log.machine_blocked[i][2] is None:
                k, t0, _ = self.log.machine_blocked[i]
                self.log.machine_blocked[i] = (k, t0, self.now)
                break
        self._put_out(st, box)
        self._machine_try(st)

    # ------------------------------------------------------------------ disruptions

    def _disrupt(self, kind: str, target: str) -> None:
        self.log.outages.append(
            (self.now, target, "down" if kind.endswith(("DOWN", "CLOSE")) else "up", kind)
        )
        if kind == dz.ROBOT_DOWN:
            self._robot_down(self.robots[target])
        elif kind == dz.ROBOT_UP:
            self._robot_up(self.robots[target])
        elif kind == dz.JOIN:
            r = self.robots[target]
            self._set(r, RS.IDLE_PARKED)
            r.needs_decision = True
        elif kind in (dz.ARM_DOWN, dz.ARM_UP):
            self._arm_fault(self.stations[target], kind == dz.ARM_DOWN)
        elif kind in (dz.MACHINE_DOWN, dz.MACHINE_UP):
            self._machine_fault(self.stations[target], kind == dz.MACHINE_DOWN)
        elif kind in (dz.AISLE_CLOSE, dz.AISLE_OPEN):
            a, b = target.split("|")
            e = self.roadmap.edge(a, b)
            e.closed = kind == dz.AISLE_CLOSE
            for rid in list(e.waiters):  # waiting robots replan (or keep waiting)
                self._push(self.now, EDGE_RETRY, rid, self.robots[rid].token)
            e.waiters.clear()
        self._bump_board()

    def _robot_down(self, r: Robot) -> None:
        if r.state in (RS.DOWN, RS.ABSENT):
            return
        r.token += 1  # everything scheduled for this robot is now stale
        dz_cfg = self.cfg.disruptions
        for e in self.roadmap.edges.values():
            if r.id in e.waiters:
                e.waiters.remove(r.id)
        if r.edge is not None:  # broken on an aisle: it blocks that lane for a while
            t0, t1, a, b = r.seg or (self.now, self.now, r.node, r.node)
            f = (self.now - t0) / max(t1 - t0, 1e-9)
            (x0, y0), (x1, y1) = self.roadmap.xy[a], self.roadmap.xy[b]
            r.down_xy = (x0 + f * (x1 - x0), y0 + f * (y1 - y0))
            moves = self.log.moves[r.id]
            if moves and moves[-1][1] > self.now:
                moves[-1] = (moves[-1][0], self.now, a, b)  # truncated at the breakdown point
            self.stuck[r.edge.key] = r.id
            r.holds_spot = True
        for st in self.stations.values():
            if r.id in st.dock_queue:
                st.dock_queue.remove(r.id)
            if r.id in st.arm_queue:
                st.arm_queue.remove(r.id)
            if st.arm_busy == r.id:  # handoff interrupted: nothing moved
                st.arm_busy = None
                st.arm_token += 1
                self._close_handoff(st, r.id)
                self._arm_try(st)
            if st.dock_holder == r.id:
                r.holds_spot = True
        if r.task is not None and r.box is None:  # claimed but not picked up: free the claim
            box = self.boxes[r.task]
            box.claimed_by = None
            self.next_station(box).inbound -= 1
            r.task = None
            self.log.claims[r.id].append((self.now, None))
        self.log.down_at[r.id].append((self.now, r.down_xy or self.roadmap.xy[r.node]))
        if r.box is not None and dz_cfg.rescue:
            self._push(self.now + dz_cfg.rescue_delay_s, RESCUE, r.id, r.token)
        if r.holds_spot:
            self._push(self.now + dz_cfg.clear_delay_s, CLEAR, r.id, r.token)
        r.needs_decision = False
        r.purpose = "dropoff" if r.box else None
        self._set(r, RS.DOWN)

    def _clear(self, r: Robot) -> None:
        """Someone pushes the broken robot aside: it stops blocking its edge or dock."""
        r.holds_spot = False
        if r.edge is not None:
            e = r.edge
            e.occupants.remove(r.id)
            self.stuck.pop(e.key, None)
            r.edge, r.seg, r.path = None, None, []
            self._wake(e)
            for w in list(e.waiters):
                self._push(self.now, EDGE_RETRY, w, self.robots[w].token)
            e.waiters.clear()
        for st in self.stations.values():
            if st.dock_holder == r.id:
                r.docked_at = None
                r.prev, r.node = st.dock, st.approach
                self._close_dock(r.id)
                self._release_dock(st)

    def _rescue(self, r: Robot) -> None:
        """Move the box off a broken robot to the nearest station with out-buffer room."""
        if r.state is not RS.DOWN or r.box is None:
            return
        box = self.boxes[r.box]
        x, y = r.down_xy or self.roadmap.xy[r.node]

        def dist(st: Station) -> float:
            sx, sy = self.roadmap.xy[st.dock]
            return math.hypot(sx - x, sy - y)

        room = [
            s
            for s in self.stations.values()
            if s.kind != "sink" and (s.out_cap is None or len(s.out_buf) < s.out_cap)
        ]
        st = min(room, key=dist)
        self.next_station(box).inbound -= 1
        box.claimed_by = None
        r.box = r.task = None
        r.purpose = None
        self.log.carrying[r.id].append((self.now, None))
        self.log.claims[r.id].append((self.now, None))
        self._put_out(st, box)

    def _robot_up(self, r: Robot) -> None:
        if r.state is not RS.DOWN:
            return
        if r.holds_spot:
            self._clear(r)
        r.down_xy = None
        if r.edge is None and r.node in self.roadmap.leaves and r.node != r.park:
            r.node = next(s.approach for s in self.stations.values() if s.dock == r.node)
        self._set(r, RS.IDLE)
        if r.box is not None:  # not rescued: finish the delivery
            r.purpose = "dropoff"
            r.docked_at = None
            self._continue(r)
        else:
            r.needs_decision = True

    def _arm_fault(self, st: Station, down: bool) -> None:
        if down and not st.arm_down:
            st.arm_down = True
            if st.arm_busy is not None:
                st.arm_paused = st.arm_until - self.now
                st.arm_token += 1
        elif not down and st.arm_down:
            st.arm_down = False
            if st.arm_busy is not None and st.arm_paused is not None:
                st.arm_until = self.now + st.arm_paused
                st.arm_paused = None
                st.arm_token += 1
                self._push(st.arm_until, HANDOFF_DONE, st.key, st.arm_token)
            else:
                self._arm_try(st)

    def _machine_fault(self, st: Station, down: bool) -> None:
        if down and not st.down:
            st.down = True
            if st.current is not None and st.busy_until is not None:
                st.paused = st.busy_until - self.now
                st.busy_until = None
                st.token += 1
                self._close_run(st)
        elif not down and st.down:
            st.down = False
            if st.current is not None and st.paused is not None:
                self._start_run(st, st.paused)
            else:
                self._machine_try(st)


def make_sim(scenario_path: str | Path, root: Path = REPO_ROOT) -> CellSim:
    """Load a ``configs/cell/*.yaml`` scenario (path relative to ``root``) and its layout."""
    p = Path(scenario_path)
    sc = load_yaml(p if p.is_absolute() else root / p, CellScenarioCfg)
    return CellSim(sc, load_yaml(root / sc.layout, CellLayoutCfg), root)


__all__ = ["CellSim", "OrderSpec", "make_sim"]
