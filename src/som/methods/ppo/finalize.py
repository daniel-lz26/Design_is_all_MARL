"""Select and evaluate finished PPO runs (spec §11.2: select on validation, report on test).

    python -m som.methods.ppo.finalize                     # IPPO and MAPPO, seeds 0-2
    python -m som.methods.ppo.finalize --workers 6 --seeds 0 1 2

For each method, reads runs/ppo/<METHOD>_s<seed>/summary.json, picks the seed with the best
validation score (mean on-time %, then lower tardiness), and evaluates that run's best/
checkpoint on the test seeds of every scenario plus cell_random_10 with 2 and 3 robots, writing
results/<METHOD>/. Every seed is also evaluated into results/ppo/seeds/ so the spread across
seeds can be reported. Writes results/ppo/selection.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def evaluate(method: str, ckpt: Path, out: Path, workers: int) -> None:
    base = [sys.executable, str(ROOT / "scripts" / "evaluate_cell.py"), "--policy", method]
    base += ["--checkpoint", str(ckpt), "--out", str(out), "--workers", str(workers)]
    subprocess.run(base, check=True)
    for n in (2, 3):
        subprocess.run([*base, "--scenarios", "cell_random_10", "--robots", str(n)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs" / "ppo")
    ap.add_argument("--methods", nargs="+", default=["IPPO", "MAPPO"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--skip-seeds", action="store_true", help="only evaluate the selected runs")
    args = ap.parse_args()
    results = args.results
    selection = {}
    for method in args.methods:
        runs = {}
        for s in args.seeds:
            summ = args.runs / f"{method}_s{s}" / "summary.json"
            if not summ.exists():
                raise SystemExit(f"{summ} missing: is {method} seed {s} finished?")
            runs[s] = json.loads(summ.read_text(encoding="utf-8"))
        best = max(runs, key=lambda s: tuple(runs[s]["best_val"]))
        selection[method] = {
            "selected_seed": best,
            "val": {s: runs[s]["best_val"] for s in runs},
            "minutes": {s: round(runs[s]["seconds"] / 60, 1) for s in runs},
            "decisions": {s: runs[s]["decisions"] for s in runs},
            "hardware": runs[best].get("hardware"),
        }
        print(f"{method}: validation {selection[method]['val']} -> seed {best}", flush=True)
        evaluate(method, args.runs / f"{method}_s{best}" / "best", results, args.workers)
        if not args.skip_seeds:
            for s in args.seeds:
                out = results / "ppo" / "seeds" / f"s{s}"
                evaluate(method, args.runs / f"{method}_s{s}" / "best", out, args.workers)
    (results / "ppo").mkdir(parents=True, exist_ok=True)
    sel_path = results / "ppo" / "selection.json"
    if sel_path.exists():  # keep methods finalized in an earlier call (e.g. IPPO before MAPPO)
        selection = {**json.loads(sel_path.read_text(encoding="utf-8")), **selection}
    sel_path.write_text(json.dumps(selection, indent=2))
    compare = [sys.executable, str(ROOT / "scripts" / "compare_cell.py"), "--results", str(results)]
    subprocess.run(compare, check=True)


if __name__ == "__main__":
    main()
