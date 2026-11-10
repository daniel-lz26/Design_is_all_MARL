"""Per-episode metrics for the cell (spec §12). On-time % counts every order, delivered or not."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from som.cell_env.entities import RS, UTIL, CellLog

UTIL_KEYS = ("loaded", "empty", "dock", "blocked", "idle", "down")


@dataclass(frozen=True, slots=True)
class CellMetrics:
    n_orders: int
    delivered: int
    on_time: int
    on_time_pct: float
    mean_tardiness: float
    max_tardiness: float
    makespan: float
    throughput_per_h: float
    robot_util: dict[str, float]  # share of present robot-time per category
    machine_util: float
    machine_blocked_s: float
    arm_util: float
    blocked_trips: int
    deadlocks: int
    decisions: int
    terminated: bool

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        util = row.pop("robot_util")
        row.update({f"robot_{k}": v for k, v in util.items()})
        return row


def _overlap(a: float, b: float | None, end: float) -> float:
    return max(0.0, min(end if b is None else b, end) - a)


def compute(log: CellLog) -> CellMetrics:
    end = log.end_time
    tard, on_time, delivered = [], 0, []
    for o in log.orders:
        finish = o["delivered"] if o["delivered"] is not None else end
        tard.append(max(0.0, finish - o["due"]))
        if o["delivered"] is not None:
            delivered.append(o["delivered"])
            on_time += o["delivered"] <= o["due"]
    n = len(log.orders)
    makespan = max(delivered) if log.terminated and delivered else end
    span = max(makespan, 1e-9)
    util = dict.fromkeys(UTIL_KEYS, 0.0)
    present = 0.0
    for states in log.robot_states.values():
        for (t, s), nxt in zip(states, [*states[1:], (span, "")], strict=True):
            dur = _overlap(t, nxt[0], span)
            if s == RS.ABSENT.value:
                continue
            present += dur
            util[UTIL[RS(s)]] += dur
    util = {k: v / max(present, 1e-9) for k, v in util.items()}
    machines = {st for st, *_ in log.machine_runs}
    n_m = max(len([s for s in log.stations if s not in ("SRC", "SINK")]), 1)
    busy = sum(_overlap(t0, t1, span) for _, _, t0, t1 in log.machine_runs)
    blocked = sum(_overlap(t0, t1, span) for _, t0, t1 in log.machine_blocked)
    arm = sum(_overlap(t0, t1, span) for *_, t0, t1 in log.handoffs)
    del machines
    return CellMetrics(
        n_orders=n,
        delivered=len(delivered),
        on_time=on_time,
        on_time_pct=100.0 * on_time / n if n else 0.0,
        mean_tardiness=sum(tard) / n if n else 0.0,
        max_tardiness=max(tard, default=0.0),
        makespan=makespan,
        throughput_per_h=3600.0 * len(delivered) / span,
        robot_util=util,
        machine_util=busy / (span * n_m),
        machine_blocked_s=blocked,
        arm_util=arm / (span * len(log.stations)),
        blocked_trips=len(log.blocked_trips),
        deadlocks=len(log.deadlocks),
        decisions=len(log.decisions),
        terminated=log.terminated,
    )


RECOVERY_WINDOW_S = 120.0
RECOVERY_STEP_S = 5.0


def recovery_time(log: CellLog, window: float = RECOVERY_WINDOW_S) -> float:
    """Mean seconds, over disruption events, until rolling throughput is back within 10% of
    its level just before the event (spec §12). Throughput is deliveries in the trailing
    ``window``. An event the line never dips after counts 0; one it never recovers from counts
    until the episode end. Events with no deliveries before them are skipped. NaN if none."""
    done = sorted(o["delivered"] for o in log.orders if o["delivered"] is not None)

    def rate(t: float) -> int:
        return sum(t - window < d <= t for d in done)

    out = []
    for t_e, _, kind, _ in log.outages:
        if kind != "down":
            continue
        pre = rate(t_e)
        if pre == 0:
            continue
        t, dip = t_e, None
        while t <= min(t_e + window, log.end_time):
            if rate(t) < 0.9 * pre:
                dip = t
                break
            t += RECOVERY_STEP_S
        if dip is None:
            out.append(0.0)
            continue
        t = dip
        while t <= log.end_time and rate(t) < 0.9 * pre:
            t += RECOVERY_STEP_S
        out.append(min(t, log.end_time) - t_e)
    return float(sum(out) / len(out)) if out else float("nan")
