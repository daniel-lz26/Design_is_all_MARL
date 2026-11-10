"""Calibrate the due-date factor k for the cell scenarios (env-v1 decision 1).

due = release + k x the order's minimum flow time in an empty cell. For each k on a grid, run
every family-A rule pair and report the best pair's on-time %. Fixed scenarios (cell_10) need
one seed (no noise); generated ones use the validation seeds and a k range of +/- spread.

    python scripts/calibrate_cell_dues.py --scenario configs/cell/cell_10.yaml
    python scripts/calibrate_cell_dues.py --scenario configs/cell/cell_random_10.yaml --spread 0.25
"""

from __future__ import annotations

import argparse

import numpy as np

from som.baselines.cell_rules import all_rule_pairs, run_rule_episode
from som.cell_env.des import CellSim, make_sim
from som.cell_env.metrics import compute


def best_rule(sim: CellSim, seeds: list[int]) -> tuple[str, float]:
    scores = []
    for pol in all_rule_pairs():
        on = []
        for s in seeds:
            run_rule_episode(sim, pol, s)
            on.append(compute(sim.log).on_time_pct)
        scores.append((float(np.mean(on)), pol.name))
    pct, name = max(scores)
    return name, pct


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scenario", default="configs/cell/cell_10.yaml")
    ap.add_argument(
        "--k", type=float, nargs="+", default=[round(x, 2) for x in np.arange(1.2, 2.61, 0.1)]
    )
    ap.add_argument("--spread", type=float, default=0.25, help="generated: k range = k +/- spread")
    args = ap.parse_args()

    base = make_sim(args.scenario)
    cfg = base.cfg
    lo, hi = cfg.seeds.val
    seeds = [0] if cfg.generator is None else list(range(int(lo), int(hi) + 1))
    print(f"{cfg.name}: best family-A rule on-time % by k ({len(seeds)} seeds)")
    for k in args.k:
        if cfg.generator is None:
            sc = cfg.model_copy(update={"due_k": k})
        else:
            gen = cfg.generator.model_copy(update={"due_k": (k - args.spread, k + args.spread)})
            sc = cfg.model_copy(update={"generator": gen})
        name, pct = best_rule(CellSim(sc, base.layout, base.root), seeds)
        print(f"  k={k:4.2f}  best {name:<16} {pct:5.1f}%")


if __name__ == "__main__":
    main()
