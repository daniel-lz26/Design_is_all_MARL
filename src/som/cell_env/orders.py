"""Orders (boxes) for the cell: fixed from an abstract scenario, or generated (spec §9)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from som.cell_env.roadmap import Roadmap
from som.config.cell_schema import CellGeneratorCfg, CellLayoutCfg, CellScenarioCfg
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


def work_estimate(
    route: tuple[str, ...], proc: tuple[float, ...], layout: CellLayoutCfg, roadmap: Roadmap
) -> float:
    """Process + travel between approach nodes + dock/undock and arm handoffs per leg."""
    stations = {s.process: s for s in layout.stations.values() if s.process}
    src = next(s for s in layout.stations.values() if s.kind == "source")
    sink = next(s for s in layout.stations.values() if s.kind == "sink")
    stops = [src, *(stations[p] for p in route), sink]
    r = layout.robot
    per_leg = 2 * (r.dock_time_s + r.undock_time_s) + layout.arm_load_s + layout.arm_unload_s
    travel = sum(roadmap.time(a.approach, b.approach) for a, b in itertools.pairwise(stops))
    return sum(proc) + travel + per_leg * (len(stops) - 1)


def generated_orders(
    gen: CellGeneratorCfg, layout: CellLayoutCfg, roadmap: Roadmap, rng: np.random.Generator
) -> list[OrderSpec]:
    names = sorted(gen.job_mix)
    w = np.array([gen.job_mix[n] for n in names], dtype=float)
    w /= w.sum()
    out: list[OrderSpec] = []
    t = gen.first_release_s
    for i in range(1, gen.n_orders + 1):
        if i > 1:
            gap = gen.mean_interarrival_s
            t += float(rng.exponential(gap)) if gen.arrivals == "poisson" else gap
        job = names[int(rng.choice(len(names), p=w))]
        route = tuple(gen.job_types[job])
        jit = rng.uniform(-gen.process_jitter, gen.process_jitter, len(route))
        proc = tuple(
            round(gen.mean_process_s[p] * (1 + float(j)), 3)
            for p, j in zip(route, jit, strict=True)
        )
        k = float(rng.uniform(*gen.due_k))
        due = t + k * work_estimate(route, proc, layout, roadmap)
        out.append(OrderSpec(f"order_{i}", i, job, route, proc, round(t, 3), round(due, 3)))
    return out


def build_orders(
    cfg: CellScenarioCfg,
    layout: CellLayoutCfg,
    roadmap: Roadmap,
    rng: np.random.Generator,
    root: Path,
) -> list[OrderSpec]:
    if cfg.orders_from is not None:
        return fixed_orders(root / cfg.orders_from)
    assert cfg.generator is not None
    return generated_orders(cfg.generator, layout, roadmap, rng)
