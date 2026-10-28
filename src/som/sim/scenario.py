"""Build order specs from a ScenarioCfg, either fixed or generated with an rng."""

from __future__ import annotations

import numpy as np

from som.config.schema import EnvCfg, LayoutCfg, ScenarioCfg
from som.core.factory import CoreCfg, FactoryCore
from som.core.ids import order_id, robot_id
from som.core.layout import Layout
from som.core.order import JobType, OrderSpec


def job_types(cfg: ScenarioCfg) -> dict[str, JobType]:
    return {name: JobType(name, jt.part, tuple(jt.route)) for name, jt in cfg.job_types.items()}


def build_orders(cfg: ScenarioCfg, rng: np.random.Generator | None = None) -> list[OrderSpec]:
    """Fixed scenarios ignore ``rng``. Generated scenarios require it."""
    jobs = job_types(cfg)
    defaults = cfg.default_process_times
    if cfg.kind == "fixed":
        return [
            OrderSpec(
                id=order_id(o.id),
                number=o.id,
                job=jobs[o.job],
                release_time=o.release,
                due_time=o.due,
                process_times=tuple(o.process_times or [defaults[p] for p in jobs[o.job].route]),
            )
            for o in cfg.orders
        ]
    if rng is None:
        raise ValueError(f"scenario {cfg.name!r} is generated and needs an rng")
    gen = cfg.generator
    assert gen is not None
    names = sorted(gen.job_mix)
    weights = np.array([gen.job_mix[n] for n in names], dtype=float)
    weights /= weights.sum()
    specs: list[OrderSpec] = []
    t = 0.0
    for i in range(1, gen.n_orders + 1):
        if i > 1:
            t += float(rng.uniform(*gen.interarrival_s))
        job = jobs[names[int(rng.choice(len(names), p=weights))]]
        jitter = rng.uniform(-gen.process_time_jitter, gen.process_time_jitter, len(job.route))
        specs.append(
            OrderSpec(
                id=order_id(i),
                number=i,
                job=job,
                release_time=round(t, 3),
                due_time=round(t + float(rng.uniform(*gen.due_slack_s)), 3),
                process_times=tuple(
                    round(defaults[p] * (1.0 + float(j)), 3)
                    for p, j in zip(job.route, jitter, strict=True)
                ),
            )
        )
    return specs


def build_core(
    layout_cfg: LayoutCfg,
    scenario_cfg: ScenarioCfg,
    env_cfg: EnvCfg,
    rng: np.random.Generator | None = None,
    debug_invariants: bool = False,
) -> FactoryCore:
    layout = Layout.from_cfg(layout_cfg)
    robots = [robot_id(i) for i in range(env_cfg.n_robots)]
    cfg = CoreCfg(
        machine_wait_allowed=env_cfg.machine_wait_allowed, debug_invariants=debug_invariants
    )
    return FactoryCore(layout, build_orders(scenario_cfg, rng), robots, cfg)
