"""Draw a scenario's disruptions once per episode, before any policy acts (spec §6).

Drawing everything at reset means every policy faces the same disruptions for a given seed.
Policies never see this schedule; they only see current up/down status.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from som.cell_env.orders import OrderSpec, work_estimate
from som.cell_env.roadmap import Roadmap
from som.config.cell_schema import CellDisruptionCfg, CellLayoutCfg

# Event kinds a schedule may contain (handled by CellSim).
ROBOT_DOWN, ROBOT_UP, ARM_DOWN, ARM_UP = "ROBOT_DOWN", "ROBOT_UP", "ARM_DOWN", "ARM_UP"
MACHINE_DOWN, MACHINE_UP = "MACHINE_DOWN", "MACHINE_UP"
AISLE_CLOSE, AISLE_OPEN, JOIN = "AISLE_CLOSE", "AISLE_OPEN", "JOIN"


@dataclass
class Schedule:
    events: list[tuple[float, str, str]] = field(default_factory=list)  # (t, kind, target)
    rush_orders: list[OrderSpec] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)  # robots that join later
    coordinator_outages: list[tuple[float, float]] = field(default_factory=list)


def draw(
    cfg: CellDisruptionCfg,
    robots: list[str],
    layout: CellLayoutCfg,
    roadmap: Roadmap,
    orders: list[OrderSpec],
    rng: np.random.Generator,
) -> Schedule:
    s = Schedule()

    def window() -> float:
        return float(rng.uniform(*cfg.fault_window_s))

    if cfg.robot_down == "permanent" and len(robots) > 1:
        r = robots[int(rng.integers(len(robots)))]
        s.events.append((float(rng.uniform(*cfg.robot_down_at_s)), ROBOT_DOWN, r))
    elif cfg.robot_down == "mtbf":
        horizon = 2 * max(o.due for o in orders)
        for r in robots:
            t = float(rng.exponential(cfg.robot_mtbf_s))
            while t < horizon:
                up = t + float(rng.uniform(*cfg.robot_repair_s))
                s.events += [(t, ROBOT_DOWN, r), (up, ROBOT_UP, r)]
                t = up + float(rng.exponential(cfg.robot_mtbf_s))
    stations = list(layout.stations)
    machines = [k for k, st in layout.stations.items() if st.kind == "machine"]
    if rng.random() < cfg.arm_fault_p:
        st, t = stations[int(rng.integers(len(stations)))], window()
        s.events += [(t, ARM_DOWN, st), (t + float(rng.uniform(*cfg.arm_fault_s)), ARM_UP, st)]
    if rng.random() < cfg.machine_fault_p:
        m, t = machines[int(rng.integers(len(machines)))], window()
        dur = float(rng.uniform(*cfg.machine_fault_s))
        s.events += [(t, MACHINE_DOWN, m), (t + dur, MACHINE_UP, m)]
    if rng.random() < cfg.aisle_block_p:
        lanes = sorted(
            f"{e.a}|{e.b}"
            for e in roadmap.edges.values()
            if e.capacity == 1 and e.a not in roadmap.leaves and e.b not in roadmap.leaves
        )
        if lanes:
            lane, t = lanes[int(rng.integers(len(lanes)))], window()
            dur = float(rng.uniform(*cfg.aisle_block_s))
            s.events += [(t, AISLE_CLOSE, lane), (t + dur, AISLE_OPEN, lane)]
    if rng.random() < cfg.rush_order_p:
        like = orders[int(rng.integers(len(orders)))]
        t = window()
        k = float(rng.uniform(*cfg.rush_slack_k))
        due = t + k * work_estimate(like.route, like.proc, layout, roadmap)
        n = max(o.number for o in orders) + 1
        s.rush_orders.append(
            OrderSpec(
                f"order_{n}",
                n,
                like.job,
                like.route,
                like.proc,
                round(t, 3),
                round(due, 3),
                rush=True,
            )
        )
    if cfg.robot_join_at_s is not None and len(robots) > 1:
        s.absent.append(robots[-1])
        s.events.append((float(rng.uniform(*cfg.robot_join_at_s)), JOIN, robots[-1]))
    if rng.random() < cfg.coordinator_outage_p:
        t = window()
        s.coordinator_outages.append((t, t + float(rng.uniform(*cfg.coordinator_outage_s))))
    s.events.sort(key=lambda e: e[0])
    return s
