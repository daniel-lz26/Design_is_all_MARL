"""Baseline comparison table: mean with 95% CI, as markdown and csv."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Two-sided 95% Student-t critical values by degrees of freedom; 1.96 beyond the table.
_T975 = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
]  # fmt: skip


def mean_ci95(xs: Sequence[float]) -> tuple[float, float]:
    """Mean and half-width of the 95% t confidence interval."""
    n = len(xs)
    if n == 0:
        return math.nan, math.nan
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    t = _T975[n - 2] if n - 1 <= len(_T975) else 1.96
    return m, t * sd / math.sqrt(n)


@dataclass(frozen=True, slots=True)
class SummaryRow:
    policy: str
    n: int
    on_time_pct: float
    on_time_pct_ci: float
    mean_tardiness: float
    mean_tardiness_ci: float
    makespan: float
    makespan_ci: float
    delivered: float
    completed_pct: float  # episodes where every order was delivered


def summarize(rows: Sequence[dict[str, Any]]) -> list[SummaryRow]:
    """One row per policy, in first-seen order."""
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_policy.setdefault(r["policy"], []).append(r)
    out = []
    for name, rs in by_policy.items():
        ot, ot_ci = mean_ci95([100.0 * float(r["on_time_rate"]) for r in rs])
        td, td_ci = mean_ci95([float(r["mean_tardiness"]) for r in rs])
        mk, mk_ci = mean_ci95([float(r["makespan"]) for r in rs])
        out.append(
            SummaryRow(
                policy=name,
                n=len(rs),
                on_time_pct=ot,
                on_time_pct_ci=ot_ci,
                mean_tardiness=td,
                mean_tardiness_ci=td_ci,
                makespan=mk,
                makespan_ci=mk_ci,
                delivered=sum(float(r["delivered"]) for r in rs) / len(rs),
                completed_pct=100.0 * sum(bool(r["terminated"]) for r in rs) / len(rs),
            )
        )
    return out


def best_policy(summary: Sequence[SummaryRow], exclude: Sequence[str] = ()) -> SummaryRow:
    """Highest on-time %, ties broken by lower mean tardiness."""
    pool = [s for s in summary if s.policy not in exclude]
    return min(pool, key=lambda s: (-s.on_time_pct, s.mean_tardiness))


def to_markdown(summary: Sequence[SummaryRow], title: str, notes: Sequence[str] = ()) -> str:
    best = best_policy(summary).policy if summary else None
    lines = [f"### {title}", ""]
    lines += [f"> {n}" for n in notes]
    if notes:
        lines.append("")
    lines += [
        "| Policy (robot/machine) | on-time % | mean tardiness (s) | makespan (s) "
        "| delivered | all delivered % |",
        "|---|---|---|---|---|---|",
    ]
    for s in summary:
        name = f"**{s.policy}**" if s.policy == best else s.policy
        lines.append(
            f"| {name} | {s.on_time_pct:.1f} ± {s.on_time_pct_ci:.1f} "
            f"| {s.mean_tardiness:.1f} ± {s.mean_tardiness_ci:.1f} "
            f"| {s.makespan:.1f} ± {s.makespan_ci:.1f} | {s.delivered:.1f} "
            f"| {s.completed_pct:.0f} |"
        )
    n = summary[0].n if summary else 0
    lines += ["", f"Mean ± 95% CI over {n} seeds. Bold = highest on-time %."]
    return "\n".join(lines) + "\n"


def write_summary_csv(summary: Sequence[SummaryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(summary[0])))
        w.writeheader()
        w.writerows(asdict(s) for s in summary)
