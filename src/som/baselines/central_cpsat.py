"""Family C baselines (spec §10): a central scheduler that plans the whole fleet with CP-SAT.

``CENTRAL_STATIC`` solves once at t = 0 and executes the plan open-loop. ``CENTRAL_REPLAN``
re-solves (5 s limit, warm-started from the previous plan) whenever a disruption happens or an
unplanned order appears. Both are centralized: they read the simulator (``global_state``), and
while the coordinator is unreachable (scenario coordinator outages) robots finish their
current task and then wait.

The model (integer deciseconds) covers what the spec lists: transport legs assigned to robots
with sequence-dependent empty travel between them (one circuit per robot), arm handoffs and
dock time, one robot per dock (no-overlap per station), machine sequencing (no-overlap per
machine) and release times. It leaves out aisle traffic and buffer capacities, and it does not
know about future disruptions or rush orders. Objective: most orders on time, then least total
tardiness. Above ``MAX_CPSAT_LEGS`` remaining legs (20+ orders) CP-SAT finds no plan within its
budget, so the scheduler uses its greedy list schedule instead (status ``GREEDY``); when CP-SAT
times out on a smaller problem the same schedule is the fallback (status ``FALLBACK``).

Execution goes through the normal action interface: a robot's next planned leg is claimed when
that box is on the board (only the board's top K tasks are claimable, as for every
controller); otherwise the robot waits. Machines follow the planned processing order.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

from ortools.sat.python import cp_model

from som.cell_env.entities import RS, Loc
from som.cell_env.estimates import free_at

SCALE = 10  # deciseconds
FIXED_LIMIT_S = 60.0  # static solve on fixed (deterministic) order books, per spec
RANDOM_LIMIT_S = 10.0  # static solve on generated order books (100 test seeds per scenario)
REPLAN_LIMIT_S = 5.0
MAX_CPSAT_LEGS = 35  # above this (about 13 orders) CP-SAT finds no plan in budget: use greedy


@dataclass
class Leg:
    box: str
    step: int  # box.step when this leg is picked up
    src: str  # station keys
    dst: str
    ready: int  # earliest time the box can be loaded (deciseconds), if not tied to an op
    after_op: int | None = None  # index of the op this leg waits for
    before_op: int | None = None  # index of the op this leg feeds


@dataclass
class Op:
    box: str
    station: str
    dur: int
    ready: int = 0  # boxes already in the in-buffer
    fixed_end: int | None = None  # a run in progress


@dataclass
class Plan:
    queues: dict[str, list[tuple[str, int]]] = field(default_factory=dict)  # robot -> (box, step)
    priority: dict[str, float] = field(default_factory=dict)  # box -> planned op start
    starts: dict[tuple[str, int], int] = field(default_factory=dict)  # warm-start hints
    status: str = ""
    objective: float = math.nan


class CentralPolicy:
    decentralized = False

    def __init__(self, replan: bool, workers: int = 4) -> None:
        self.replan = replan
        self.name = "CENTRAL_REPLAN" if replan else "CENTRAL_STATIC"
        self.machine_rule = None  # machines follow the plan
        self.workers = workers
        self._cache: dict[Any, Plan] = {}

    def reset(self, scenario_info: dict[str, Any]) -> None:
        self.acts = scenario_info["actions"]
        self.plan: Plan | None = None
        self.seen_outages = 0
        self.known: set[str] = set()
        self.solves: list[tuple[float, str, float]] = []  # (t, status, objective)

    # ------------------------------------------------------------------ acting

    def act(self, obs: dict, masks: dict, global_state: dict | None = None) -> dict[str, int]:
        assert global_state is not None, "the central scheduler needs global state"
        sim = global_state["sim"]
        out = {}
        for rid, m in masks.items():
            out[rid] = int(self.acts["noop"]) if m[self.acts["noop"]] else self._decide(sim, rid)
        return out

    def _coordinator_down(self, sim: Any) -> bool:
        return any(t0 <= sim.now < t1 for t0, t1 in sim.schedule.coordinator_outages)

    def _decide(self, sim: Any, rid: str) -> int:
        wait = int(self.acts["wait"])
        if self._coordinator_down(sim):
            return wait
        if self.plan is None:
            self._solve(sim, first=True)
        elif self.replan and self._needs_replan(sim):
            self._solve(sim, first=False)
        assert self.plan is not None
        queue = self.plan.queues.get(rid, [])
        while queue:
            bid, step = queue[0]
            b = sim.boxes[bid]
            taken = b.step == step and b.claimed_by not in (None, rid)
            if b.delivered is not None or b.step > step or taken:
                queue.pop(0)  # done, or another robot took this very leg: skip it
                continue
            board = [t.id for t in sim.tasks(rid)]
            if bid in board and b.step == step:
                queue.pop(0)
                return board.index(bid)
            return wait  # not ready yet (or not in the top K of the board)
        return wait

    def _needs_replan(self, sim: Any) -> bool:
        released = {b.id for b in sim.boxes.values() if b.loc is not Loc.PENDING}
        new_orders = bool(released - self.known)
        return len(sim.log.outages) != self.seen_outages or new_orders

    # ------------------------------------------------------------------ solving

    def _solve(self, sim: Any, first: bool) -> None:
        if first:
            limit = FIXED_LIMIT_S if sim.cfg.orders_from is not None else RANDOM_LIMIT_S
        else:
            limit = REPLAN_LIMIT_S
        key = None
        if first:  # identical order books (e.g. cell_10 for every seed) share one solve
            key = (
                tuple(
                    (b.id, b.spec.release, b.spec.due, b.spec.proc)
                    for b in sim.boxes.values()
                    if not b.spec.rush
                ),
                tuple(sim.robot_ids),
                sim.cfg.layout,
                sim.h,
            )
        if key is not None and key in self._cache:
            plan = self._cache[key]
            self.plan = Plan(
                {r: list(q) for r, q in plan.queues.items()},
                dict(plan.priority),
                dict(plan.starts),
                plan.status,
                plan.objective,
            )
        else:
            self.plan = solve(sim, limit, self.workers, hints=self.plan)
            if key is not None:
                self._cache[key] = self.plan
        sim.machine_priority = dict(self.plan.priority)
        self.seen_outages = len(sim.log.outages)
        self.known = {
            b.id for b in sim.boxes.values() if b.loc is not Loc.PENDING or not b.spec.rush
        }
        self.solves.append((sim.now, self.plan.status, self.plan.objective))


def _d(seconds: float) -> int:
    return math.ceil(seconds * SCALE)


def solve(sim: Any, limit_s: float, workers: int = 4, hints: Plan | None = None) -> Plan:
    """Plan every remaining leg and machine run from the simulator's current state."""
    now = _d(sim.now)
    h = sim.h
    st = sim.stations
    appr = {k: s.approach for k, s in st.items()}
    spur = {k: sim.roadmap.edge_time(s.approach, s.dock) for k, s in st.items()}
    dock_in = {k: _d(h.dock_time_s + spur[k]) for k in st}
    dock_out = {k: _d(h.undock_time_s + spur[k]) for k in st}
    load = {k: _d(s.load_s) for k, s in st.items()}
    unload = {k: _d(s.unload_s) for k, s in st.items()}

    def travel(a: str, b: str) -> int:
        return _d(sim.roadmap.time(a, b))

    # ---- what is left to do -------------------------------------------------------------
    legs: list[Leg] = []
    ops: list[Op] = []
    busy: dict[str, tuple[int, str]] = {}  # robot -> (free at, node) for robots mid-task
    ready_box: dict[str, int] = {}  # box -> when its committed leg delivers it
    for r in sim.robots.values():
        if r.state in (RS.DOWN, RS.ABSENT):
            continue
        if r.task is not None or r.box is not None:
            t_free, node = free_at(sim, r)
            busy[r.id] = (now + _d(t_free), node)
            ready_box[r.box or r.task or ""] = now + _d(t_free)

    def route_from(box: Any, step: int, src: str, ready: int, first_op: Op | None) -> None:
        """Ops and legs for ``box`` from station ``src`` (out-buffer) at route index ``step``."""
        prev_op = None
        if first_op is not None:
            ops.append(first_op)
            prev_op = len(ops) - 1
        route = box.spec.route
        s, cur = step, src
        while True:
            dst = sim.sink if s >= len(route) else sim.machine_of[route[s]]
            leg = Leg(box.id, s, cur, dst, ready, after_op=prev_op)
            legs.append(leg)
            if dst == sim.sink:
                return
            ops.append(Op(box.id, dst, _d(h.conveyor_transfer_s + box.spec.proc[s])))
            leg.before_op = len(ops) - 1
            prev_op = len(ops) - 1
            s, cur, ready = s + 1, dst, 0

    for b in sim.boxes.values():
        if b.delivered is not None:
            continue
        if b.loc is Loc.PENDING:
            if b.spec.rush:
                continue  # rush orders are a surprise to the planner
            route_from(b, 0, sim.src, _d(b.spec.release), None)
        elif b.loc is Loc.OUT and b.claimed_by is None:
            route_from(b, b.step, b.where, now, None)
        elif b.id in ready_box:  # on a robot, or claimed: the committed leg is not planned
            dst = sim.next_station(b)
            t = ready_box[b.id]
            if dst.key == sim.sink:
                continue
            first = Op(b.id, dst.key, _d(h.conveyor_transfer_s + b.spec.proc[b.step]), ready=t)
            route_from(b, b.step + 1, dst.key, 0, first)
        elif b.loc is Loc.OUT:  # claimed by a robot that went down: plan it fresh
            route_from(b, b.step, b.where, now, None)
        elif b.loc in (Loc.IN, Loc.MACHINE):
            s = st[b.where]
            if b.loc is Loc.MACHINE and b.id == s.blocked_box:  # done, waiting for room
                route_from(b, b.step, s.key, now, None)
                continue
            if b.loc is Loc.MACHINE and s.busy_until is not None:
                op = Op(b.id, s.key, 0, fixed_end=_d(s.busy_until))
            elif b.loc is Loc.MACHINE:  # paused by a fault: optimistic, resumes now
                op = Op(b.id, s.key, _d(s.paused or 0.0), ready=now)
            else:
                op = Op(b.id, s.key, _d(h.conveyor_transfer_s + b.spec.proc[b.step]), ready=now)
            route_from(b, b.step + 1, s.key, 0, op)
        # boxes stuck on a down robot (no rescue) cannot be planned

    robots = [r for r in sim.robot_ids if sim.robots[r].state not in (RS.DOWN, RS.ABSENT)]
    plan = Plan(queues={r: [] for r in sim.robot_ids})
    if not legs or not robots:
        plan.status = "EMPTY"
        return plan

    dur = [
        dock_in[g.src]
        + load[g.src]
        + dock_out[g.src]
        + travel(appr[g.src], appr[g.dst])
        + dock_in[g.dst]
        + unload[g.dst]
        + dock_out[g.dst]
        for g in legs
    ]
    op_len = [max(op.fixed_end - now, 0) if op.fixed_end is not None else op.dur for op in ops]
    starts0 = {r: busy.get(r, (now, sim.location(r))) for r in robots}
    greedy = _greedy(
        legs, ops, op_len, dur, robots, starts0, now, dock_in, load, dock_out, appr, sim
    )
    seqs, g_go, g_start, g_op = greedy
    if len(legs) > MAX_CPSAT_LEGS:
        plan.status = "GREEDY"
        for r in robots:
            plan.queues[r] = [(legs[i].box, legs[i].step) for i in seqs[r]]
        for j, op in enumerate(ops):
            plan.priority[op.box] = float(g_op[j])
        return plan
    horizon = max(_d(sim.t_max), max(g_start) + max(dur)) + 10 * SCALE * len(legs)
    m = cp_model.CpModel()
    # go[i]: the robot sets off for the pickup, which needs the box on the board and the
    # robot free; start[i]: it starts docking at the pickup; end[i]: undocked at the drop-off.
    go = [m.new_int_var(0, horizon, f"g{i}") for i in range(len(legs))]
    start = [m.new_int_var(0, horizon, f"s{i}") for i in range(len(legs))]
    end = [m.new_int_var(0, horizon, f"e{i}") for i in range(len(legs))]
    for i, g in enumerate(legs):
        m.add(end[i] == start[i] + dur[i])
        m.add(go[i] >= g.ready)
        m.add(start[i] >= go[i])
    op_s = [m.new_int_var(0, horizon, f"os{j}") for j in range(len(ops))]
    op_e = [m.new_int_var(0, horizon, f"oe{j}") for j in range(len(ops))]
    for j, op in enumerate(ops):
        m.add(op_e[j] == op_s[j] + op_len[j])
        if op.fixed_end is not None:
            m.add(op_s[j] == now)
        else:
            m.add(op_s[j] >= max(op.ready, now))
    for i, g in enumerate(legs):
        if g.after_op is not None:  # loading starts once the box is out of the machine
            m.add(go[i] >= op_e[g.after_op])
        if g.before_op is not None:  # the machine starts once the box is unloaded
            m.add(op_s[g.before_op] >= end[i] - dock_out[g.dst])

    # robots: one circuit each over its legs, empty travel as transition time
    pres: dict[tuple[str, int], Any] = {}
    arc_lit: dict[tuple[str, int, int], Any] = {}
    idle: dict[str, Any] = {}
    for r in robots:
        t0, node = busy.get(r, (now, sim.location(r)))
        arcs = []
        for i, g in enumerate(legs):
            p = m.new_bool_var(f"p{r}_{i}")
            pres[r, i] = p
            arcs.append((i + 1, i + 1, ~p))
            first = m.new_bool_var(f"f{r}_{i}")
            arc_lit[r, 0, i + 1] = first
            arcs.append((0, i + 1, first))
            m.add(go[i] >= t0).only_enforce_if(first)
            m.add(start[i] >= go[i] + _d(sim.roadmap.time(node, appr[g.src]))).only_enforce_if(
                first
            )
            last = m.new_bool_var(f"l{r}_{i}")
            arc_lit[r, i + 1, 0] = last
            arcs.append((i + 1, 0, last))
        for i, gi in enumerate(legs):
            for k, gk in enumerate(legs):
                if i == k or (gk.box == gi.box and gk.step < gi.step):
                    continue
                lit = m.new_bool_var(f"a{r}_{i}_{k}")
                arc_lit[r, i + 1, k + 1] = lit
                arcs.append((i + 1, k + 1, lit))
                m.add(go[k] >= end[i]).only_enforce_if(lit)
                m.add(start[k] >= go[k] + travel(appr[gi.dst], appr[gk.src])).only_enforce_if(lit)
        idle[r] = m.new_bool_var(f"idle{r}")
        arcs.append((0, 0, idle[r]))
        m.add_circuit(arcs)
    for i in range(len(legs)):
        m.add_exactly_one(pres[r, i] for r in robots)

    # one robot per dock (covers the one-arm rule too), one job per machine
    docks: dict[str, list[Any]] = {k: [] for k in st}
    for i, g in enumerate(legs):
        a = dock_in[g.src] + load[g.src] + dock_out[g.src]
        docks[g.src].append(m.new_interval_var(start[i], a, start[i] + a, f"da{i}"))
        b = dock_in[g.dst] + unload[g.dst] + dock_out[g.dst]
        docks[g.dst].append(m.new_interval_var(end[i] - b, b, end[i], f"db{i}"))
    for ivs in docks.values():
        if len(ivs) > 1:
            m.add_no_overlap(ivs)
    machines: dict[str, list[Any]] = {}
    for j, op in enumerate(ops):
        machines.setdefault(op.station, []).append(
            m.new_interval_var(op_s[j], op_len[j], op_e[j], f"op{j}")
        )
    for ivs in machines.values():
        if len(ivs) > 1:
            m.add_no_overlap(ivs)

    # objective: on-time count first, then total tardiness
    last_leg = {g.box: i for i, g in enumerate(legs) if g.dst == sim.sink}
    n = len(last_leg)
    big = n * horizon + 1
    late_terms, tard_terms = [], []
    for bid, i in last_leg.items():
        due = _d(sim.boxes[bid].spec.due)
        done = end[i] - dock_out[sim.sink]
        late = m.new_bool_var(f"late{bid}")
        m.add(done <= due).only_enforce_if(~late)
        tard = m.new_int_var(0, horizon, f"t{bid}")
        m.add(tard >= done - due)
        late_terms.append(late)
        tard_terms.append(tard)
    m.minimize(big * sum(late_terms) + sum(tard_terms))

    # warm start: the greedy list schedule, with the previous plan's start times where known
    for r in robots:
        seq = seqs[r]
        m.add_hint(idle[r], not seq)
        for i in range(len(legs)):
            m.add_hint(pres[r, i], i in seq)
        chain = [0, *(i + 1 for i in seq), 0] if seq else []
        used = set(itertools.pairwise(chain))
        for (rr, a, b), lit in arc_lit.items():
            if rr == r:
                m.add_hint(lit, (a, b) in used)
    for i in range(len(legs)):
        m.add_hint(go[i], g_go[i])
        prev = hints.starts.get((legs[i].box, legs[i].step)) if hints is not None else None
        m.add_hint(start[i], prev if prev is not None else g_start[i])
    for j in range(len(ops)):
        m.add_hint(op_s[j], g_op[j])

    solver = cp_model.CpSolver()
    # Deterministic: interleaved workers and a deterministic-time budget give the same plan
    # for the same state on any machine (wall-clock limits do not). Wall time is a backstop.
    solver.parameters.num_workers = workers
    solver.parameters.interleave_search = True
    solver.parameters.max_deterministic_time = limit_s
    solver.parameters.max_time_in_seconds = 4 * limit_s
    solver.parameters.random_seed = 0
    status = solver.solve(m)
    plan.status = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        plan.status = "FALLBACK"  # no solution within budget: execute the greedy schedule
        for r in robots:
            plan.queues[r] = [(legs[i].box, legs[i].step) for i in seqs[r]]
        for j, op in enumerate(ops):
            plan.priority[op.box] = float(g_op[j])
        return plan
    plan.objective = solver.objective_value
    order = sorted(range(len(legs)), key=lambda i: (solver.value(go[i]), solver.value(start[i])))
    for i in order:
        r = next(r for r in robots if solver.value(pres[r, i]))
        plan.queues[r].append((legs[i].box, legs[i].step))
        plan.starts[legs[i].box, legs[i].step] = solver.value(start[i])
    for j, op in enumerate(ops):
        plan.priority[op.box] = float(solver.value(op_s[j]))
    return plan


__all__ = ["CentralPolicy", "Plan", "solve"]


def _greedy(
    legs: list[Leg],
    ops: list[Op],
    op_len: list[int],
    dur: list[int],
    robots: list[str],
    starts: dict[str, tuple[int, str]],
    now: int,
    dock_in: dict[str, int],
    load: dict[str, int],
    dock_out: dict[str, int],
    appr: dict[str, str],
    sim: Any,
) -> tuple[dict[str, list[int]], list[int], list[int], list[int]]:
    """List schedule: repeatedly take the ready leg with the earliest (ready, due) and give it
    to the robot that can start docking first; machines run boxes in arrival order. Returns
    per-robot leg sequences, set-off and dock-start times per leg, and op start times."""
    robot_t = {r: starts[r][0] for r in robots}
    robot_at = {r: starts[r][1] for r in robots}
    machine_t: dict[str, int] = {}
    dock_t: dict[str, int] = {}
    op_s = [now] * len(ops)
    op_done: dict[int, int] = {}
    fed = {g.before_op for g in legs if g.before_op is not None}

    def run_op(j: int, ready: int) -> None:
        if ops[j].fixed_end is not None:
            op_s[j] = now
        else:
            op_s[j] = max(ready, ops[j].ready, now, machine_t.get(ops[j].station, now))
        op_done[j] = op_s[j] + op_len[j]
        machine_t[ops[j].station] = op_done[j]

    for j, op in enumerate(ops):  # runs not fed by a planned leg: in progress or queued
        if j not in fed:
            run_op(j, op.ready)
    go, start = [now] * len(legs), [now] * len(legs)
    seqs: dict[str, list[int]] = {r: [] for r in robots}
    todo = set(range(len(legs)))
    while todo:
        ready = {}
        for i in todo:
            after = legs[i].after_op
            if after is None:
                ready[i] = legs[i].ready
            elif after in op_done:
                ready[i] = op_done[after]
        if not ready:
            break
        i = min(ready, key=lambda i: (ready[i], sim.boxes[legs[i].box].spec.due, i))
        g = legs[i]
        best: tuple[int, int, str] | None = None
        for r in robots:
            t_go = max(ready[i], robot_t[r])
            t_start = max(
                t_go + _d(sim.roadmap.time(robot_at[r], appr[g.src])), dock_t.get(g.src, 0)
            )
            if best is None or t_start < best[0]:
                best = (t_start, t_go, r)
        assert best is not None
        t_start, t_go, r = best
        go[i], start[i] = t_go, t_start
        end = t_start + dur[i]
        dock_t[g.src] = t_start + dock_in[g.src] + load[g.src] + dock_out[g.src]
        dock_t[g.dst] = end
        robot_t[r], robot_at[r] = end, appr[g.dst]
        seqs[r].append(i)
        todo.discard(i)
        if g.before_op is not None:
            run_op(g.before_op, end - dock_out[g.dst])
    return seqs, go, start, op_s
