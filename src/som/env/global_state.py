"""Fixed-size centralized critic state, padded to max_robots and max_orders.

``base_state`` describes the whole factory; ``agent_state`` appends a role one-hot and an agent
slot one-hot, so one critic per role can tell agents apart (agent-specific global state, as in
MAPPO). Orders beyond ``max_orders`` are left out, undelivered ones first in line.
"""

from __future__ import annotations

import numpy as np

from som.core.history import CoreSnapshot
from som.core.order import OrderStatus
from som.env.observations import OPS_SCALE, ObsContext, factory_summary, onehot, scaled_time

ROBOT_STATUSES = ("IDLE", "TO_PICKUP", "TO_DROPOFF", "WAITING_DOCK", "DOCKING")
ORDER_STATUSES = tuple(s.value for s in OrderStatus)

_ROBOT_FEATS = 1 + len(ROBOT_STATUSES) + 3
_ORDER_FEATS = 1 + len(ORDER_STATUSES) + 4


def base_state_dim(ctx: ObsContext, max_orders: int) -> int:
    return (
        1
        + ctx.n_stations
        + 4 * ctx.n_machines
        + ctx.max_robots * _ROBOT_FEATS
        + max_orders * _ORDER_FEATS
    )


def agent_id_dim(ctx: ObsContext) -> int:
    return 2 + ctx.max_robots + ctx.n_machines


def state_dim(ctx: ObsContext, max_orders: int) -> int:
    return base_state_dim(ctx, max_orders) + agent_id_dim(ctx)


def base_state(ctx: ObsContext, snap: CoreSnapshot, max_orders: int) -> np.ndarray:
    xs = [s.dock.x for s in ctx.layout.stations]
    ys = [s.dock.y for s in ctx.layout.stations]
    w, h = max(max(xs), 1e-6), max(max(ys), 1e-6)
    out = [snap.now / ctx.horizon]
    out += factory_summary(ctx, snap)
    for i in range(ctx.max_robots):
        if i >= len(snap.robots):
            out += [0.0] * _ROBOT_FEATS
            continue
        r = snap.robots[i]
        out += [1.0, *onehot(ROBOT_STATUSES.index(r.status), len(ROBOT_STATUSES))]
        out += [float(r.carrying is not None), r.pose.x / w, r.pose.y / h]
    orders = sorted(snap.orders, key=lambda o: (o.status == "DELIVERED", o.number))
    for i in range(max_orders):
        if i >= len(orders):
            out += [0.0] * _ORDER_FEATS
            continue
        o = orders[i]
        left = sum(o.process_times[o.step_idx :])
        out += [1.0, *onehot(ORDER_STATUSES.index(o.status), len(ORDER_STATUSES))]
        out += [
            scaled_time(o.due - snap.now - left),
            o.step_idx / OPS_SCALE,
            (len(o.route) - o.step_idx) / OPS_SCALE,
            scaled_time(snap.now - o.available_since),
        ]
    return np.asarray(out, dtype=np.float32)


def agent_onehot(ctx: ObsContext, role: str, slot: int) -> np.ndarray:
    """Role one-hot, then the agent's slot: robot index, or max_robots + machine index."""
    v = np.zeros(agent_id_dim(ctx), dtype=np.float32)
    v[0 if role == "robot" else 1] = 1.0
    v[2 + (slot if role == "robot" else ctx.max_robots + slot)] = 1.0
    return v
