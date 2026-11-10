"""Evaluate trained MAPPO policies next to every baseline on the same scenario and seeds.

    python scripts/evaluate.py --name mappo_2r                         # newest seeds of a config
    python scripts/evaluate.py --name mappo_4r mappo_4r_disrupt --robots 4 \\
        --scenario abstract_10_disrupt                                 # compare two families
    python scripts/evaluate.py --runs runs/A runs/B                     # explicit runs

Each training run contributes its ``policy/`` artifact (best on held-out seeds of its training
scenario, chosen without looking at the eval scenario). Runs are grouped into families by config
name. Rows: Random, the 12 heuristic pairs, then per family one row per training seed, the
family's seeds pooled, and its learned robots with EDD machines (seed that did best on held-out
seeds). Writes runs/<timestamp>_eval_<scenario>_<n>r/ and, in reports/, the results table and
csv, a learning curve per family and a Gantt chart of each family's first seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from som.config.loader import dump_resolved, load_yaml
from som.config.schema import EnvCfg, LayoutCfg, ScenarioCfg
from som.core.layout import Layout
from som.eval.compare import best_policy, summarize, to_markdown, write_summary_csv
from som.eval.evaluate import evaluate, load_log, safe_name, write_rows_csv
from som.eval.plots import gantt, learning_curves
from som.policies.base import Policy
from som.policies.heuristics import all_heuristics, make_policy
from som.policies.learned import LearnedPolicy, RoleMixPolicy
from som.policies.random_policy import RandomPolicy
from som.sim.travel import StraightLineTravel

ROOT = Path(__file__).resolve().parents[1]


def meta(run: Path) -> dict[str, Any]:
    return dict(json.loads((run / "policy" / "policy_meta.json").read_text(encoding="utf-8")))


def newest_runs(name: str) -> list[Path]:
    """Runs of config ``name`` that share the newest timestamp (one per training seed)."""
    runs = sorted(
        p
        for p in (ROOT / "runs").glob(f"*_{name}*")
        if (p / "policy").is_dir() and meta(p)["run"] == name
    )
    if not runs:
        raise SystemExit(f"no trained runs named {name!r} under runs/")
    stamp = runs[-1].name[:15]
    return [p for p in runs if p.name.startswith(stamp)]


def family(run: Path) -> str:
    return str(meta(run)["run"])


def seed_label(run: Path) -> str:
    tail = run.name.rsplit("_", 1)[-1]
    seed = tail if tail.startswith("s") and tail[1:].isdigit() else run.name
    return f"{family(run)} {seed}"


def heldout_score(run: Path) -> tuple[float, float]:
    """Best (on-time, -tardiness) on held-out training-distribution seeds, as the trainer ranks."""
    with (run / "eval.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["scenario"] == meta(run)["trained_on"]]
    return max((float(r["on_time_rate"]), -float(r["mean_tardiness"])) for r in rows)


def eval_curve(run: Path, scenario: str) -> list[tuple[float, float]]:
    path = run / "eval.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [
            (float(r["env_steps"]), 100.0 * float(r["on_time_rate"]))
            for r in csv.DictReader(f)
            if r["scenario"] == scenario
        ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="*", help="training run dirs")
    ap.add_argument("--name", nargs="+", default=["mappo_2r"], help="config names (no --runs)")
    ap.add_argument("--scenario", default="abstract_10")
    ap.add_argument("--layout", default="arena_13x6ft")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--robots", type=int, default=2)
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    args = ap.parse_args()

    found = args.runs or [r for n in args.name for r in newest_runs(n)]
    runs = [r if r.is_absolute() else ROOT / r for r in found]
    families: dict[str, list[Path]] = {}
    for r in runs:
        families.setdefault(family(r), []).append(r)

    layout_cfg = load_yaml(ROOT / "configs" / "layouts" / f"{args.layout}.yaml", LayoutCfg)
    scenario_cfg = load_yaml(ROOT / "configs" / "scenarios" / f"{args.scenario}.yaml", ScenarioCfg)
    layout = Layout.from_cfg(layout_cfg)
    planner = StraightLineTravel.from_cfg(layout_cfg.travel_calibration)
    learned = {
        seed_label(r): LearnedPolicy.from_artifact(r / "policy", layout_cfg, name=seed_label(r))
        for r in runs
    }
    mixes: list[Policy] = []
    for fam, fruns in families.items():
        pick = max(fruns, key=heldout_score)
        mixes.append(
            RoleMixPolicy(
                learned[seed_label(pick)],
                make_policy("FIFO/EDD", layout, planner),
                f"{fam} robots ({seed_label(pick).split()[-1]}) + EDD machines",
            )
        )
    first = learned[seed_label(runs[0])]
    env_cfg = EnvCfg.model_validate({**first.env_cfg.model_dump(), "n_robots": args.robots})
    policies: list[Policy] = [
        RandomPolicy(),
        *all_heuristics(layout, planner),
        *learned.values(),
        *mixes,
    ]
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))

    tag = f"{args.scenario}_{args.robots}r"
    run_dir = ROOT / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_eval_{tag}"
    dump_resolved(layout_cfg, run_dir / "config" / "layout.yaml")
    dump_resolved(scenario_cfg, run_dir / "config" / "scenario.yaml")
    dump_resolved(env_cfg, run_dir / "config" / "env.yaml")
    rows = evaluate(policies, layout_cfg, scenario_cfg, env_cfg, seeds, log_dir=run_dir / "logs")
    pooled = {}
    for fam, fruns in families.items():
        pooled[fam] = f"{fam} (all seeds)"
        labels = {seed_label(r) for r in fruns}
        rows += [{**r, "policy": pooled[fam]} for r in rows if r["policy"] in labels]
    write_rows_csv(rows, run_dir / "eval" / "rows.csv")

    summary = summarize(rows)
    learned_names = {*learned, *pooled.values(), *(m.name for m in mixes)}

    def rank(name: str) -> tuple[int, int, int]:
        if name == "Random":
            return (0, 0, 0)
        if name not in learned_names:
            return (1, 0, 0)
        fam = next(f for f in families if name.startswith(f + " "))
        return (2, list(families).index(fam), 0 if name == pooled[fam] else 1)

    summary.sort(key=lambda s: rank(s.policy))
    best_h = best_policy(summary, exclude=["Random", *learned_names])
    trained_on = sorted({meta(r)["trained_on"] for r in runs})
    notes = [
        f"Scenario `{args.scenario}`, layout `{args.layout}`, {args.robots} robots, "
        f"travel noise sigma {env_cfg.travel_noise_sigma}, seeds {seeds[0]}-{seeds[-1]}.",
        f"MAPPO families {', '.join(f'`{f}`' for f in families)} trained on "
        f"{', '.join(f'`{t}`' for t in trained_on)} (never on `{args.scenario}`). "
        "`(all seeds)` rows pool each family's training seeds.",
        "**Inputs contain PLACEHOLDER values** (station coordinates, release and process "
        "times). Do not quote these numbers as final.",
        "On-time % is over all orders, not only delivered ones.",
    ]
    if scenario_cfg.disruptions is not None:
        notes.append(
            "Disruptions on: blocked and delayed trips, machine breakdowns, robot outages "
            "(rates are scenario design choices, not measurements)."
        )
    title = f"MAPPO vs baselines on {args.scenario}, {args.robots} robots"
    table = to_markdown(summary, title, notes)
    (run_dir / "eval" / "table.md").write_text(table, encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "scenario": args.scenario,
                "best_heuristic": best_h.policy,
                "learned": list(learned),
                "training_runs": [r.name for r in runs],
                "seeds": seeds,
            }
        ),
        encoding="utf-8",
    )

    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / f"results_{tag}.md").write_text(table, encoding="utf-8")
    write_summary_csv(summary, args.reports / f"results_{tag}.csv")
    for fam, fruns in families.items():
        lab = seed_label(fruns[0])
        gantt(
            load_log(run_dir / "logs" / safe_name(lab) / "ep_000.json"),
            args.reports / f"gantt_{tag}_{safe_name(lab).replace(' ', '_')}_ep0.png",
        )
        if args.robots == meta(fruns[0])["env"]["n_robots"]:
            learning_curves(
                {seed_label(r): eval_curve(r, args.scenario) for r in fruns},
                args.reports / f"learning_curve_{fam}_{args.scenario}.png",
                f"{fam} on {args.scenario} during training (10 eval seeds)",
                baseline=(f"best heuristic ({best_h.policy}, 20 seeds)", best_h.on_time_pct),
            )

    if (sys.stdout.encoding or "").lower().replace("-", "") != "utf8":
        table = table.replace("±", "+/-")
    print(table)
    print(f"best heuristic: {best_h.policy}")
    print(f"run dir: {run_dir.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
