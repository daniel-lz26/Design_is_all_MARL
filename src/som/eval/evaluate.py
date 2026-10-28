"""Run a policy over scenarios and seeds, producing per-episode metric rows."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from som.config.schema import EnvCfg, LayoutCfg, ScenarioCfg
from som.core.history import EpisodeLog
from som.eval.metrics import compute_metrics
from som.policies.base import Policy
from som.sim.runtime import SimRuntime, run_episode
from som.sim.scenario import build_core
from som.sim.travel import StraightLineTravel


def run_one(
    layout_cfg: LayoutCfg,
    scenario_cfg: ScenarioCfg,
    env_cfg: EnvCfg,
    policy: Policy,
    seed: int,
    debug_invariants: bool = False,
) -> EpisodeLog:
    """One episode. ``seed`` drives the order generator, travel noise and the policy's rng
    through independent streams, so a fixed seed always reproduces the same log."""
    scen_ss, travel_ss, policy_ss = np.random.SeedSequence(seed).spawn(3)
    core = build_core(
        layout_cfg, scenario_cfg, env_cfg, np.random.default_rng(scen_ss), debug_invariants
    )
    travel = StraightLineTravel.from_cfg(layout_cfg.travel_calibration, env_cfg.travel_noise_sigma)
    policy.reset(np.random.default_rng(policy_ss))
    rt = SimRuntime(core, travel, np.random.default_rng(travel_ss), scenario_cfg.horizon)
    log = run_episode(rt, policy)
    log.meta.update(
        policy=policy.name,
        scenario=scenario_cfg.name,
        seed=seed,
        robot_starts=[[p.x, p.y, p.theta] for p in core.layout.robot_starts[: len(core.robots)]],
    )
    return log


def evaluate(
    policies: Sequence[Policy],
    layout_cfg: LayoutCfg,
    scenario_cfg: ScenarioCfg,
    env_cfg: EnvCfg,
    seeds: Sequence[int],
    log_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """policy x seed rows. If ``log_dir`` is given, each episode log is saved as
    ``<log_dir>/<policy>/ep_<index>.json`` (index into ``seeds``)."""
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for ep, seed in enumerate(seeds):
            log = run_one(layout_cfg, scenario_cfg, env_cfg, policy, seed)
            if log_dir is not None:
                save_log(log, log_dir / safe_name(policy.name) / f"ep_{ep:03d}.json")
            rows.append(
                {
                    "policy": policy.name,
                    "scenario": scenario_cfg.name,
                    "seed": seed,
                    "ep": ep,
                    **compute_metrics(log).as_row(),
                }
            )
    return rows


def safe_name(name: str) -> str:
    return name.replace("/", "__")


def save_log(log: EpisodeLog, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log.to_dict()), encoding="utf-8")


def load_log(path: Path) -> EpisodeLog:
    return EpisodeLog.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_rows_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
