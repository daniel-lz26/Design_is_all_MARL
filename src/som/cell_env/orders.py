"""Orders (boxes) for the cell: fixed from an abstract scenario, or generated (spec §9).

Due dates follow the calibrated rule from the env-v1 review: due = release + k x the order's
minimum flow time in an empty cell (``min_flow_time``). For fixed orders ``due_k`` rescales
the source scenario's due dates; without it they are kept as given (``cell_10_raw``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from som.cell_env.roadmap import Roadmap
from som.config.cell_schema import CellGeneratorCfg, CellLayoutCfg, CellScenarioCfg, HandlingCfg
from som.config.loader import load_yaml
from som.config.schema import ScenarioCfg


@dataclass(frozen=True, slots=True)
class OrderSpec:
    id: str  # "order_<n>"
    number: int
    job: str
    route: tuple[str, ...]  # processes, e.g. ("Turning", "Milling")
    proc: tuple[float, ...]
    release: float
    due: float
    rush: bool = False
    min_flow: float = 0.0  # seconds from release to delivery in an empty cell


def min_flow_time(
    route: tuple[str, ...],
    proc: tuple[float, ...],
    layout: CellLayoutCfg,
    handling: HandlingCfg,
    roadmap: Roadmap,
) -> float:
    """Release-to-delivery time with no other orders and a robot already docked wherever the
    box is ready: per leg, load + undock + drive + dock + unload; per machine, conveyor +
    process. A lower bound on what any controller can achieve."""
    by_proc = {s.process: s for s in layout.stations.values() if s.process}
    src = next(s for s in layout.stations.values() if s.kind == "source")
    sink = next(s for s in layout.stations.values() if s.kind == "sink")
    stops = [src, *(by_proc[p] for p in route), sink]
    h = handling

    def spur(st: object) -> float:
        return roadmap.edge_time(st.approach, st.dock)  # type: ignore[attr-defined]

    t = sum(p + h.conveyor_transfer_s for p in proc)
    for a, b in itertools.pairwise(stops):
        t += (a.arm_load_s or h.arm_load_s) + h.undock_time_s + spur(a)
        t += roadmap.time(a.approach, b.approach)
        t += spur(b) + h.dock_time_s + (b.arm_unload_s or h.arm_unload_s)
    return t


def fixed_orders(path: Path) -> list[OrderSpec]:
    """Orders of a fixed abstract scenario: same routes, release, due and process times."""
    sc = load_yaml(path, ScenarioCfg)
    if sc.kind != "fixed":
        raise ValueError(f"{path}: orders_from must point at a fixed scenario")
    out = []
    for o in sc.orders:
        route = tuple(sc.job_types[o.job].route)
        proc = tuple(o.process_times or [sc.default_process_times[p] for p in route])
        out.append(OrderSpec(f"order_{o.id}", o.id, o.job, route, proc, o.release, o.due))
    return out


def generated_orders(
    gen: CellGeneratorCfg,
    flow: object,
    rng: np.random.Generator,
) -> list[OrderSpec]:
    """``flow(route, proc)`` returns the minimum flow time used for the due date."""
    names = sorted(gen.job_mix)
    w = np.array([gen.job_mix[n] for n in names], dtype=float)
    w /= w.sum()
    out: list[OrderSpec] = []
    t = gen.first_release_s
    i = 0
    while gen.n_orders is None or i < gen.n_orders:
        if i > 0:
            gap = gen.mean_interarrival_s
            t += float(rng.exponential(gap)) if gen.arrivals == "poisson" else gap
        if gen.release_until_s is not None and t > gen.release_until_s:
            break
        i += 1
        job = names[int(rng.choice(len(names), p=w))]
        route = tuple(gen.job_types[job])
        jit = rng.uniform(-gen.process_jitter, gen.process_jitter, len(route))
        proc = tuple(
            round(gen.mean_process_s[p] * (1 + float(j)), 3)
            for p, j in zip(route, jit, strict=True)
        )
        mf = flow(route, proc)  # type: ignore[operator]
        due = t + float(rng.uniform(*gen.due_k)) * mf
        out.append(OrderSpec(f"order_{i}", i, job, route, proc, round(t, 3), round(due, 3),
                             min_flow=mf))  # fmt: skip
    return out


def build_orders(
    cfg: CellScenarioCfg,
    layout: CellLayoutCfg,
    handling: HandlingCfg,
    roadmap: Roadmap,
    rng: np.random.Generator,
    root: Path,
) -> list[OrderSpec]:
    def flow(route: tuple[str, ...], proc: tuple[float, ...]) -> float:
        return min_flow_time(route, proc, layout, handling, roadmap)

    if cfg.orders_from is None:
        assert cfg.generator is not None
        return generated_orders(cfg.generator, flow, rng)
    out = []
    for o in fixed_orders(root / cfg.orders_from):
        mf = flow(o.route, o.proc)
        due = o.release + cfg.due_k * mf if cfg.due_k is not None else o.due
        out.append(replace(o, due=round(due, 3), min_flow=mf))
    return out
