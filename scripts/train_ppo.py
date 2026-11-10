"""Train the PPO method on the cell env: IPPO (--critic local) or MAPPO (--critic central).

    python scripts/train_ppo.py --critic central --seed 0                  # MAPPO, full budget
    python scripts/train_ppo.py --critic local --seed 1 --out runs/ppo     # IPPO
    python scripts/train_ppo.py --budget 250000 --name smoke               # short smoke run
    python scripts/train_ppo.py --bc-only                                  # clone the teacher, stop

Writes runs/ppo/<name>/: config.json, progress.csv (per rollout), eval.csv (validation),
best/actor.pt (best on validation; this is what gets evaluated), final/actor.pt, summary.json.
Evaluate on the test seeds with:

    python scripts/evaluate_cell.py --policy MAPPO --checkpoint runs/ppo/<name>/best
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import time
from pathlib import Path

import torch

from som.methods.ppo.trainer import PPOConfig, PPOTrainer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--critic", choices=["central", "local"], default="central")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", help="run name (default <METHOD>_s<seed>)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "ppo")
    ap.add_argument("--budget", type=int, help="env decisions incl. demonstrations (1.5M)")
    ap.add_argument("--bc-decisions", type=int, help="0 disables the warm-start")
    ap.add_argument("--bc-only", action="store_true", help="stop after behaviour cloning")
    ap.add_argument("--scenarios", nargs="+", help="training mix")
    ap.add_argument("--n-envs", type=int)
    ap.add_argument("--eval-every", type=int)
    ap.add_argument("--threads", type=int, help="torch threads (1 per run when running several)")
    args = ap.parse_args()

    method = "MAPPO" if args.critic == "central" else "IPPO"
    over = {
        "critic": args.critic,
        "seed": args.seed,
        "name": args.name or f"{method}_s{args.seed}",
        "budget": args.budget,
        "bc_decisions": args.bc_decisions,
        "scenarios": args.scenarios,
        "n_envs": args.n_envs,
        "eval_every": args.eval_every,
        "threads": args.threads,
    }
    cfg = PPOConfig(**{k: v for k, v in over.items() if v is not None})
    run_dir = args.out / cfg.name
    if (run_dir / "summary.json").exists():
        raise SystemExit(f"{run_dir} already holds a finished run; pick another --name")
    trainer = PPOTrainer(cfg, run_dir)
    hw = {"machine": platform.node(), "cpu": platform.processor(), "torch": torch.__version__}
    print(f"{cfg.name}: {cfg.method}, budget {cfg.budget:,} decisions, mix {cfg.scenarios}, {hw}")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with (run_dir / "train.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if args.bc_only:
        t0 = time.time()
        ws = trainer.warm_start(log)
        trainer._evaluate(_null(), 0, log, time.time() - t0)
        from som.methods.ppo.policy import ACTOR_FILE, save_actor

        save_actor(run_dir / "bc" / ACTOR_FILE, trainer.actor, trainer.meta(bc_only=True))
        (run_dir / "summary.json").write_text(json.dumps({**ws, "hardware": hw}, indent=2))
        return
    summary = trainer.train(log)
    summary["hardware"] = hw
    summary["config"] = dataclasses.asdict(cfg)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"done: {summary['decisions']:,} decisions in {summary['seconds'] / 60:.1f} min")


def _null() -> object:
    class Null:
        def write(self, row: dict) -> None:
            pass

    return Null()


if __name__ == "__main__":
    main()
