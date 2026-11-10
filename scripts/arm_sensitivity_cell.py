"""Arm handoff sensitivity (env-v1 decision 3): run a controller with arm load/unload times of
3, 6 and 9 s and report whether robot transport is still the bottleneck.

    python scripts/arm_sensitivity_cell.py --policy NEAREST/SPT --scenarios cell_10 cell_random_10

Due dates stay calibrated at the default 6 s, so on-time % shows how much the guess matters.
Bottleneck = the resource with the highest busy share over the makespan: the robot fleet
(time not idle), the busiest machine, or the busiest arm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from som.cell_env.cell_env import CellEnv
from som.cell_env.harness import run_episode
from som.cell_env.metrics import compute
from som.policies.registry import make

ROOT = Path(__file__).resolve().parents[1]


def busiest(log: object, kind: str, span: float) -> tuple[str, float]:
    busy: dict[str, float] = {}
    rows = log.machine_runs if kind == "machine" else log.handoffs  # type: ignore[attr-defined]
    for row in rows:
        key, t0, t1 = row[0], row[-2], row[-1]
        busy[key] = busy.get(key, 0.0) + max(0.0, min(t1 if t1 is not None else span, span) - t0)
    key = max(busy, key=busy.__getitem__)
    return key, busy[key] / span


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--policy", default="NEAREST/SPT")
    ap.add_argument("--scenarios", nargs="+", default=["cell_10", "cell_random_10"])
    ap.add_argument("--arm", type=float, nargs="+", default=[3.0, 6.0, 9.0])
    ap.add_argument("--seeds", type=int, default=100, help="first N test seeds")
    args = ap.parse_args()

    lines = [
        "| Scenario | Arm s | On-time % | Robots busy | Robot time: dock / travel / idle | "
        "Busiest machine | Busiest arm | Bottleneck |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in args.scenarios:
        for arm in args.arm:
            base = CellEnv(ROOT / "configs" / "cell" / f"{name}.yaml")
            h = base.sim.layout.handling.model_copy(update={"arm_load_s": arm, "arm_unload_s": arm})
            env = CellEnv(base.sim.cfg.model_copy(update={"handling": h}), base.sim.layout)
            lo = int(env.sim.cfg.seeds.test[0])
            seeds = (
                [0]
                if env.sim.cfg.orders_from and env.sim.cfg.noise.speed_sigma == 0
                else list(range(lo, lo + args.seeds))
            )
            on, busy, dock, travel, idle, mach, arms = [], [], [], [], [], [], []
            for seed in seeds:
                pol = make(args.policy)
                log = run_episode(env, pol, seed)
                m = compute(log)
                u = m.robot_util
                on.append(m.on_time_pct)
                busy.append(1 - u["idle"])
                dock.append(u["dock"])
                travel.append(u["loaded"] + u["empty"] + u["blocked"])
                idle.append(u["idle"])
                mach.append(busiest(log, "machine", m.makespan))
                arms.append(busiest(log, "arm", m.makespan))
            mk = max({k for k, _ in mach}, key=lambda k: np.mean([v for kk, v in mach if kk == k]))
            mv = np.mean([v for k, v in mach if k == mk])
            ak = max({k for k, _ in arms}, key=lambda k: np.mean([v for kk, v in arms if kk == k]))
            av = np.mean([v for k, v in arms if k == ak])
            shares = {"robot transport": np.mean(busy), f"machine {mk}": mv, f"arm {ak}": av}
            neck = max(shares, key=shares.__getitem__)
            lines.append(
                f"| {name} | {arm:g} | {np.mean(on):.1f} | {100 * np.mean(busy):.0f}% | "
                f"{100 * np.mean(dock):.0f}% / {100 * np.mean(travel):.0f}% / "
                f"{100 * np.mean(idle):.0f}% | {mk.lower()} {100 * mv:.0f}% | "
                f"{ak.lower()} {100 * av:.0f}% | {neck} |"
            )
            print(lines[-1], flush=True)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
