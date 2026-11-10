"""Evaluation harness (spec §11.1): run any CellPolicy on a scenario and score it.

Robots at a decision point act one at a time in id order, each with a fresh observation, so a
claim is on the board before the next robot looks (first claim wins, as on a real job board).
Decentralized policies get ``global_state=None``; that is the C1 guarantee, checked in tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from som.cell_env.cell_env import CellEnv
from som.cell_env.entities import CellLog
from som.cell_env.metrics import compute, recovery_time
from som.policies.cell_base import CellPolicy


def run_episode(env: CellEnv, policy: CellPolicy, seed: int) -> CellLog:
    sim = env.sim
    rule = getattr(policy, "machine_rule", None)
    sim.machine_rule = rule or sim.cfg.machine_seq_rule
    sim.machine_priority = None
    env.reset(seed)
    policy.reset(env.scenario_info())
    pending = env._pending
    while pending:
        for rid in pending:
            if rid not in sim.pending():
                continue
            obs = {rid: env.observe(rid)}
            masks = {rid: env.mask(rid)}
            gs = None if policy.decentralized else env.global_state()
            env.apply(rid, int(policy.act(obs, masks, gs)[rid]))
        pending = env.advance()
    return sim.log


def evaluate(
    make_policy: Callable[[], CellPolicy],
    scenario: str,
    seeds: list[int],
    n_robots: int | None = None,
    on_log: Callable[[int, CellLog], None] | None = None,
) -> list[dict[str, Any]]:
    env = CellEnv(scenario, n_robots=n_robots)
    policy = make_policy()
    rows = []
    for seed in seeds:
        t0 = time.perf_counter()
        log = run_episode(env, policy, seed)
        row = {
            "policy": policy.name,
            "scenario": env.sim.cfg.name,
            "seed": seed,
            "n_robots": len(env.possible_agents),
            **compute(log).as_row(),
            "recovery_time": recovery_time(log),
            "wall_s": time.perf_counter() - t0,
        }
        rows.append(row)
        if on_log is not None:
            on_log(seed, log)
    return rows
