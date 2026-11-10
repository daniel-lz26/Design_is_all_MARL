"""Train MAPPO from a train config: ``python scripts/train.py --cfg configs/train/mappo_2r.yaml``.

Writes runs/<timestamp>_<name>[_s<seed>]/ with the resolved config, progress.csv, eval.csv,
TensorBoard logs, checkpoints/last.pt, policy/ (best on held-out training-distribution seeds) and
policy_final/.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from som.config.loader import dump_resolved, load_yaml
from som.config.schema import LayoutCfg, ScenarioCfg, TrainCfg
from som.rl.mappo import MAPPOTrainer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--seed", type=int, help="override the config seed")
    ap.add_argument("--steps", type=int, help="override total_steps")
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = load_yaml(args.cfg, TrainCfg)
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.steps is not None:
        overrides["total_steps"] = args.steps
    if overrides:
        cfg = TrainCfg.model_validate({**cfg.model_dump(), **overrides})
    layout_cfg = load_yaml(ROOT / cfg.layout, LayoutCfg)
    scenario_cfg = load_yaml(ROOT / cfg.scenario, ScenarioCfg)
    evals = [load_yaml(ROOT / p, ScenarioCfg) for p in cfg.eval_scenarios]

    suffix = f"_s{cfg.seed}" if args.seed is not None else ""
    run = ROOT / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{cfg.name}{suffix}"
    dump_resolved(cfg, run / "config.yaml")
    dump_resolved(layout_cfg, run / "config" / "layout.yaml")
    dump_resolved(scenario_cfg, run / "config" / "scenario.yaml")
    print(f"run dir: {run.relative_to(ROOT).as_posix()}", flush=True)

    trainer = MAPPOTrainer(cfg, layout_cfg, scenario_cfg, evals, run)
    best = trainer.train(on_log=lambda s: print(s, flush=True))
    (run / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
