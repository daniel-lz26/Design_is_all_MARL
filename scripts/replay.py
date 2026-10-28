"""Render one saved episode of a run as a gif.

Defaults to the latest baselines run and its best heuristic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from som.config.loader import load_yaml
from som.config.schema import LayoutCfg
from som.core.layout import Layout
from som.eval.evaluate import load_log, safe_name
from som.viz.replay import animate

ROOT = Path(__file__).resolve().parents[1]


def latest_run() -> Path:
    runs = sorted((ROOT / "runs").glob("*_baselines_*"))
    if not runs:
        raise SystemExit("no baselines run found; run `make baselines` first")
    return runs[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None, help="run directory (default: latest)")
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--policy", default=None, help="default: best heuristic of the run")
    ap.add_argument("--out", type=Path, default=None, help="default: reports/replay_*.gif")
    args = ap.parse_args()

    run = args.run or latest_run()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    policy = args.policy or summary["best_heuristic"]
    log = load_log(run / "logs" / safe_name(policy) / f"ep_{args.ep:03d}.json")
    layout = Layout.from_cfg(load_yaml(run / "config" / "layout.yaml", LayoutCfg))
    out = args.out or (
        ROOT / "reports" / f"replay_{summary['scenario']}_{safe_name(policy)}_ep{args.ep}.gif"
    )
    animate(log, layout, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
