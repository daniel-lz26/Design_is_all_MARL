"""Versioned per-role observation encoders (OBS_SCHEMA_VERSION). Shared with the live runtime.

Every vector has a fixed length that depends only on the layout and on K, never on the number of
robots or orders, so one policy works with 2 or 4 robots. The scales below are part of the schema:
changing any of them, or the feature order, means bumping ``OBS_SCHEMA_VERSION``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from som.core.decisions import Candidate
from som.core.history import CoreSnapshot, MachineView, OrderView
from som.core.layout import Layout
from som.policies.heuristics import remaining_work
from som.sim.travel import TravelModel

OBS_SCHEMA_VERSION = 1

TIME_SCALE_S = 200.0  # seconds-valued features are divided by this, then clipped
TIME_CLIP = 3.0
OPS_SCALE = 4.0  # remaining route steps
COUNT_SCALE = 5.0  # orders waiting at a station

# Orders sitting at a station (not on a robot, not being processed).
WAITING_STATUSES = ("RELEASED", "WAITING_PICKUP", "QUEUED", "RESERVED")


@dataclass(frozen=True, slots=True)
class ObsContext:
    """Everything an encoder needs besides the snapshot. ``travel`` must be noise-free."""

    layout: Layout
    travel: TravelModel
    horizon: float
    k_robot: int
    k_machine: int
    max_robots: int

    @property
    def n_stations(self) -> int:
        return len(self.layout.stations)

    @property
    def n_machines(self) -> int:
        return len(self.layout.machines)

    def station_index(self, sid: str | None) -> int | None:
        for i, s in enumerate(self.layout.stations):
            if s.id == sid:
                return i
        return None


def scaled_time(seconds: float) -> float:
    return float(np.clip(seconds / TIME_SCALE_S, -TIME_CLIP, TIME_CLIP))


def onehot(i: int | None, n: int) -> list[float]:
    v = [0.0] * n
    if i is not None:
        v[i] = 1.0
    return v


def _remaining_ops(o: OrderView) -> float:
    return (len(o.route) - o.step_idx) / OPS_SCALE


def factory_summary(ctx: ObsContext, snap: CoreSnapshot) -> list[float]:
    """Per-station waiting counts, then per machine: free input share, busy, time left, down."""
    counts = [0.0] * ctx.n_stations
    for o in snap.orders:
        i = ctx.station_index(o.location) if o.status in WAITING_STATUSES else None
        if i is not None:
            counts[i] += 1.0 / COUNT_SCALE
    out = counts
    for m in snap.machines:
        out += _machine_state(m, snap.now)
    return out


def _machine_state(m: MachineView, now: float) -> list[float]:
    free = 0.0 if m.down else (m.capacity - len(m.queue) - m.inbound) / m.capacity
    if m.busy_until is not None:
        left = scaled_time(m.busy_until - now)
    elif m.paused_remaining is not None:
        left = scaled_time(m.paused_remaining)
    else:
        left = 0.0
    return [free, float(m.current is not None), left, float(m.down)]


def _free_share(snap: CoreSnapshot, ctx: ObsContext, process: str | None) -> float:
    """Free input capacity share at the station doing ``process`` (1.0 for the sink)."""
    if process is None:
        return 1.0
    mid = ctx.layout.station_for_process(process).machine
    m = snap.machine(mid) if mid else None
    if m is None or m.down:
        return 0.0
    return (m.capacity - len(m.queue) - m.inbound) / m.capacity


# --------------------------------------------------------------------------- robot


def robot_obs_dim(ctx: ObsContext) -> int:
    s, m = ctx.n_stations, ctx.n_machines
    return s + 2 + ctx.k_robot * (1 + 2 * s + 5) + (s + 4 * m) + s


def robot_obs(
    ctx: ObsContext, snap: CoreSnapshot, agent: str, slots: Sequence[Candidate]
) -> np.ndarray:
    s = ctx.n_stations
    me = snap.robot(agent)
    out = onehot(ctx.station_index(me.station), s)
    out += [float(me.carrying is not None), snap.now / ctx.horizon]
    for k in range(ctx.k_robot):
        if k >= len(slots):
            out += [0.0] * (1 + 2 * s + 5)
            continue
        c = slots[k]
        o = snap.order(c.order)
        pickup = ctx.layout.station(c.pickup).dock
        dest = ctx.layout.station(c.dest).dock
        work = remaining_work(o, me.pose, c.pickup, ctx.layout, ctx.travel)
        out += [1.0]
        out += onehot(ctx.station_index(c.pickup), s)
        out += onehot(ctx.station_index(c.dest), s)
        out += [
            scaled_time(ctx.travel.estimate(me.pose, pickup)),
            scaled_time(ctx.travel.estimate(pickup, dest)),
            scaled_time(o.due - snap.now - work),
            _remaining_ops(o),
            scaled_time(snap.now - c.available_since),
        ]
    out += factory_summary(ctx, snap)
    heading = [0.0] * s
    for r in snap.robots:
        if r.id == agent or r.task_order is None:
            continue
        target = snap.order(r.task_order).location if r.status == "TO_PICKUP" else r.task_dest
        i = ctx.station_index(target)
        if i is not None:
            heading[i] += 1.0 / ctx.max_robots
    out += heading
    return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- machine


def machine_obs_dim(ctx: ObsContext) -> int:
    s, m = ctx.n_stations, ctx.n_machines
    return m + 3 + ctx.k_machine * 6 + (s + 4 * m)


def machine_obs(
    ctx: ObsContext, snap: CoreSnapshot, agent: str, slots: Sequence[Candidate]
) -> np.ndarray:
    me = snap.machine(agent)
    station = ctx.layout.station(me.station)
    idx = next(i for i, st in enumerate(ctx.layout.machines) if st.id == me.station)
    out = onehot(idx, ctx.n_machines)
    out += _machine_state(me, snap.now)[1:3] + [snap.now / ctx.horizon]
    for k in range(ctx.k_machine):
        if k >= len(slots):
            out += [0.0] * 6
            continue
        o = snap.order(slots[k].order)
        work = remaining_work(o, station.dock, station.id, ctx.layout, ctx.travel)
        nxt = o.route[o.step_idx + 1] if o.step_idx + 1 < len(o.route) else None
        out += [
            1.0,
            scaled_time(o.process_times[o.step_idx]),
            scaled_time(o.due - snap.now - work),
            (len(o.route) - o.step_idx - 1) / OPS_SCALE,
            scaled_time(snap.now - slots[k].available_since),
            _free_share(snap, ctx, nxt),
        ]
    out += factory_summary(ctx, snap)
    return np.asarray(out, dtype=np.float32)


def obs_dim(ctx: ObsContext, role: str) -> int:
    return robot_obs_dim(ctx) if role == "robot" else machine_obs_dim(ctx)


def encode(
    ctx: ObsContext, snap: CoreSnapshot, agent: str, role: str, slots: Sequence[Candidate]
) -> np.ndarray:
    fn = robot_obs if role == "robot" else machine_obs
    return fn(ctx, snap, agent, slots)
