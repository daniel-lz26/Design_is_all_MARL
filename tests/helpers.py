"""Shared builders for tests: a tiny 4-station layout and order/core factories."""

from __future__ import annotations

from som.core.factory import CoreCfg, FactoryCore
from som.core.ids import order_id, robot_id
from som.core.layout import Layout
from som.core.order import JobType, OrderSpec

TINY_LAYOUT = {
    "name": "tiny",
    "arena": {"size_m": [3.0, 1.0]},
    "stations": {
        "SRC": {"kind": "source", "dock": [0.0, 0.5, 0.0], "queue": [0.0, 0.2, 0.0]},
        "A": {
            "kind": "machine",
            "process": "Turning",
            "dock": [1.0, 0.5, 0.0],
            "queue": [1.0, 0.2, 0.0],
            "input_capacity": 2,
        },
        "B": {
            "kind": "machine",
            "process": "Milling",
            "dock": [2.0, 0.5, 0.0],
            "queue": [2.0, 0.2, 0.0],
        },
        "SINK": {"kind": "sink", "dock": [3.0, 0.5, 0.0], "queue": [3.0, 0.2, 0.0]},
    },
    "robots": {"tb1": {"start": [0.0, 0.5, 0.0]}, "tb2": {"start": [0.0, 0.5, 0.0]}},
    "travel_calibration": {"v_eff": 0.5, "w_eff": 1.0, "dock_overhead_s": 2.0},
}

TURN = JobType("J1", "Shaft", ("Turning",))
TURN_MILL = JobType("J2", "Flange", ("Turning", "Milling"))


def spec(n: int, job: JobType = TURN, release: float = 0.0, due: float = 100.0) -> OrderSpec:
    return OrderSpec(order_id(n), n, job, release, due, tuple(10.0 for _ in job.route))


def make_core(layout: Layout, orders: list[OrderSpec], n_robots: int = 2) -> FactoryCore:
    robots = [robot_id(i) for i in range(n_robots)]
    return FactoryCore(layout, orders, robots, CoreCfg(debug_invariants=True))
