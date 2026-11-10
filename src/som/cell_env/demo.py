"""Run a dispatching rule end to end on a cell scenario and print a summary.

python -m som.cell_env.demo --config configs/cell/cell_10.yaml [--policy NEAREST/FIFO]
"""

from __future__ import annotations

import argparse
import time

from som.baselines.cell_rules import CellRulePolicy, run_rule_episode
from som.cell_env.des import make_sim
from som.cell_env.metrics import compute


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/cell/cell_10.yaml")
    ap.add_argument("--policy", default="NEAREST/FIFO")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", type=int, default=200, help="episodes for the speed benchmark")
    args = ap.parse_args()

    robot_rule, _, machine_rule = args.policy.upper().partition("/")
    policy = CellRulePolicy(robot_rule, machine_rule or "FIFO")
    sim = make_sim(args.config)
    run_rule_episode(sim, policy, args.seed)
    m = compute(sim.log)
    print(f"{policy.name} on {sim.cfg.name}, seed {args.seed}, {sim.cfg.n_robots} robots")
    print(f"  delivered {m.delivered}/{m.n_orders}, on time {m.on_time} ({m.on_time_pct:.1f}%)")
    print(f"  mean tardiness {m.mean_tardiness:.1f} s, max {m.max_tardiness:.1f} s")
    print(f"  makespan {m.makespan:.1f} s, throughput {m.throughput_per_h:.1f} orders/h")
    util = "  ".join(f"{k} {100 * v:.0f}%" for k, v in m.robot_util.items())
    print(f"  robot time: {util}")
    print(
        f"  machine util {100 * m.machine_util:.0f}%, machine blocked {m.machine_blocked_s:.0f} s,"
        f" arm util {100 * m.arm_util:.0f}%"
    )
    print(f"  blocked trips {m.blocked_trips}, deadlocks {m.deadlocks}, decisions {m.decisions}")
    print("  orders:")
    for o in sim.log.orders:
        d = o["delivered"]
        late = "on time" if d is not None and d <= o["due"] else "late"
        dtxt = f"{d:6.1f}" if d is not None else "   ---"
        print(
            f"    {o['id']:>9} {'>'.join(p[0] for p in o['route']):<6} due {o['due']:5.0f}"
            f"  delivered {dtxt}  {late}"
        )

    t0 = time.perf_counter()
    decisions = 0
    for s in range(args.bench):
        run_rule_episode(sim, policy, s)
        decisions += len(sim.log.decisions)
    dt = time.perf_counter() - t0
    print(f"  speed: {decisions / dt:,.0f} decision points/s ({args.bench} episodes, one core)")


if __name__ == "__main__":
    main()
