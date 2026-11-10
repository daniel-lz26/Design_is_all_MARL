"""Evaluate cell controllers through the shared harness (spec §11.1).

    python scripts/evaluate_cell.py --policy NEAREST/FIFO --scenarios cell_10 cell_10_disrupt
    python scripts/evaluate_cell.py --policy baselines --workers 10        # every A, B, C baseline
    python scripts/evaluate_cell.py --policy my_method --checkpoint runs/x/policy --split test
    python scripts/evaluate_cell.py --policy baselines --skip-existing  # resume, e.g. on another PC

Writes results/<policy>/<scenario>.csv (one row per seed; ``_<n>r`` is appended when
``--robots`` overrides the scenario) and a Gantt chart of the first seed. Every row records the
env version (git tag) it was produced with.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = [
    "cell_10",
    "cell_10_raw",
    "cell_10_disrupt",
    "cell_random_10",
    "cell_random_10_disrupt",
    "cell_random_10_flex",
    "cell_random_20",
    "cell_random_30",
    "cell_stream",
]


def env_version() -> str:
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def safe(name: str) -> str:
    return name.replace("/", "__")


def seeds_for(scenario: str, split: str) -> list[int]:
    from som.config.cell_schema import CellScenarioCfg
    from som.config.loader import load_yaml

    cfg = load_yaml(ROOT / "configs" / "cell" / f"{scenario}.yaml", CellScenarioCfg)
    lo, hi = getattr(cfg.seeds, split)
    return list(range(int(lo), int(hi) + 1))


def run_job(
    policy: str,
    checkpoint: str | None,
    scenario: str,
    split: str,
    robots: int | None,
    out: str,
    version: str,
    n_seeds: int | None,
) -> tuple[str, str, int]:
    from som.cell_env.harness import evaluate
    from som.cell_env.render_gantt import gantt
    from som.policies.registry import family, make

    seeds = seeds_for(scenario, split)[:n_seeds]
    folder = Path(out) / safe(policy)
    tag = f"{scenario}_{robots}r" if robots else scenario

    def on_log(seed: int, log: Any) -> None:
        if seed == seeds[0]:
            gantt(log, folder / f"gantt_{tag}_s{seed}.png", policy)

    rows = evaluate(
        lambda: make(policy, checkpoint), f"configs/cell/{scenario}.yaml", seeds, robots, on_log
    )
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["env_version", "family", "split", *rows[0]])
        w.writeheader()
        for r in rows:
            w.writerow({"env_version": version, "family": family(policy), "split": split, **r})
    return policy, tag, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--policy",
        nargs="+",
        required=True,
        help="registry names, or 'baselines' for every A, B and C baseline",
    )
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--robots", type=int, default=None, help="override the robot count")
    ap.add_argument("--seeds", type=int, default=None, help="only the first N seeds of the split")
    ap.add_argument(
        "--workers", type=int, default=1, help="parallel jobs; central jobs use 4 threads each"
    )
    ap.add_argument(
        "--skip-existing", action="store_true", help="skip jobs whose csv already exists"
    )
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    from som.policies.registry import REGISTRY, family

    policies = []
    for p in args.policy:
        policies += sorted(n for n in REGISTRY if family(n) in "ABC") if p == "baselines" else [p]
    version = env_version()
    jobs = [
        (p, args.checkpoint, s, args.split, args.robots, args.out, version, args.seeds)
        for p in policies
        for s in args.scenarios
    ]
    if args.skip_existing:

        def done(job: tuple) -> bool:
            tag = f"{job[2]}_{args.robots}r" if args.robots else job[2]
            return (Path(args.out) / safe(job[0]) / f"{tag}.csv").exists()

        skipped = [j for j in jobs if done(j)]
        jobs = [j for j in jobs if not done(j)]
        print(f"skipping {len(skipped)} jobs with existing results", flush=True)
    print(
        f"{len(jobs)} jobs ({len(policies)} policies x {len(args.scenarios)} scenarios), "
        f"env {version}",
        flush=True,
    )
    if args.workers <= 1:
        for j in jobs:
            print(*run_job(*j), flush=True)
        return
    # spawn on every OS: Linux's default fork can hang once OR-Tools or torch are loaded
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
        futures = [pool.submit(run_job, *j) for j in jobs]
        for f in as_completed(futures):
            print(*f.result(), flush=True)


if __name__ == "__main__":
    main()
