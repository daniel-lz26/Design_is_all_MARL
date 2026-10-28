"""Evaluate Random and every heuristic pair on a scenario and print the baseline table.

Writes episode logs and metrics to runs/<timestamp>_baselines_<scenario>/, and the table, csv and
a Gantt chart of the best heuristic to reports/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from som.config.loader import dump_resolved, load_yaml
from som.config.schema import EnvCfg, LayoutCfg, ScenarioCfg
from som.core.layout import Layout
from som.eval.compare import best_policy, summarize, to_markdown, write_summary_csv
from som.eval.evaluate import evaluate, load_log, safe_name, write_rows_csv
from som.eval.plots import gantt
from som.policies.base import Policy
from som.policies.heuristics import all_heuristics
from som.policies.random_policy import RandomPolicy
from som.sim.travel import StraightLineTravel

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="abstract_10")
    ap.add_argument("--layout", default="arena_13x6ft")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument(
        "--seed-base", type=int, default=1000, help="eval seeds, disjoint from training"
    )
    ap.add_argument("--robots", type=int, default=2)
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    args = ap.parse_args()

    layout_cfg = load_yaml(ROOT / "configs" / "layouts" / f"{args.layout}.yaml", LayoutCfg)
    scenario_cfg = load_yaml(ROOT / "configs" / "scenarios" / f"{args.scenario}.yaml", ScenarioCfg)
    env_cfg = EnvCfg(n_robots=args.robots)
    layout = Layout.from_cfg(layout_cfg)
    # Heuristics plan with the noise-free estimate; the simulator applies the noise.
    planner = StraightLineTravel.from_cfg(layout_cfg.travel_calibration)
    policies: list[Policy] = [RandomPolicy(), *all_heuristics(layout, planner)]
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))

    run = ROOT / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_baselines_{args.scenario}"
    dump_resolved(layout_cfg, run / "config" / "layout.yaml")
    dump_resolved(scenario_cfg, run / "config" / "scenario.yaml")
    dump_resolved(env_cfg, run / "config" / "env.yaml")

    rows = evaluate(policies, layout_cfg, scenario_cfg, env_cfg, seeds, log_dir=run / "logs")
    write_rows_csv(rows, run / "eval" / "rows.csv")
    summary = summarize(rows)
    best = best_policy(summary, exclude=["Random"])
    notes = [
        f"Scenario `{args.scenario}`, layout `{args.layout}`, {args.robots} robots, "
        f"travel noise sigma {env_cfg.travel_noise_sigma}, seeds {seeds[0]}-{seeds[-1]}.",
        "**Inputs contain PLACEHOLDER values** (station coordinates, release and process "
        "times). Do not quote these numbers as final.",
        "On-time % is over all orders, not only delivered ones.",
    ]
    table = to_markdown(summary, f"Baselines on {args.scenario}", notes)
    (run / "eval" / "table.md").write_text(table, encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps({"scenario": args.scenario, "best_heuristic": best.policy, "seeds": seeds}),
        encoding="utf-8",
    )

    args.reports.mkdir(parents=True, exist_ok=True)
    stem = f"baselines_{args.scenario}"
    (args.reports / f"{stem}.md").write_text(table, encoding="utf-8")
    write_summary_csv(summary, args.reports / f"{stem}.csv")
    best_log = load_log(run / "logs" / safe_name(best.policy) / "ep_000.json")
    gantt(best_log, args.reports / f"gantt_{args.scenario}_{safe_name(best.policy)}_ep0.png")

    if (sys.stdout.encoding or "").lower().replace("-", "") != "utf8":
        table = table.replace("±", "+/-")  # consoles without UTF-8 (e.g. cp1252)
    print(table)
    print(f"best heuristic: {best.policy}")
    print(f"run dir: {run.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
