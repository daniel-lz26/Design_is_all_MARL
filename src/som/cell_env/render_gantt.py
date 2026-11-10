"""Gantt chart of a cell episode, in the style of ``som.eval.plots.gantt`` (spec §12, plot 1).

Machine rows (Turning, Grinding, Milling, Drilling), then one row per robot. Colours are order
ids. Robots: solid = loaded travel, hatched = empty travel, dotted outline = at a dock or arm
(queueing, docking, handoff, undocking), grey cross-hatch = down, red x = blocked trip.
Machines: solid = processing, grey dots = blocked by a full out-buffer, grey cross-hatch = down.
Dotted vertical lines are due times. Title: ``<policy> s<seed> · <scenario>``.
"""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from som.cell_env.entities import CellLog

DOCK_STATES = {"WAIT_DOCK", "DOCKING", "WAIT_ARM", "BEING_LOADED", "BEING_UNLOADED", "UNDOCKING"}


def order_colors(log: CellLog) -> dict[str, Any]:
    cmap = plt.get_cmap("tab10" if len(log.orders) <= 10 else "tab20")
    n = cmap.N
    return {o["id"]: cmap(i % n) for i, o in enumerate(log.orders)}


def value_at(timeline: list[tuple[float, Any]], t: float) -> Any:
    """Value of a step timeline [(t, value), ...] at time ``t`` (right-continuous)."""
    i = bisect_right([x[0] for x in timeline], t) - 1
    return timeline[i][1] if i >= 0 else None


def intervals(states: list[tuple[float, str]], end: float) -> list[tuple[float, float, str]]:
    out = []
    for (t, s), nxt in zip(states, [*states[1:], (end, "")], strict=True):
        if nxt[0] > t:
            out.append((t, nxt[0], s))
    return out


def outage_spans(log: CellLog, who: str, end: float) -> list[tuple[float, float]]:
    spans, start = [], None
    for t, target, kind, _ in log.outages:
        if target != who:
            continue
        if kind == "down" and start is None:
            start = t
        elif kind == "up" and start is not None:
            spans.append((start, t))
            start = None
    if start is not None:
        spans.append((start, end))
    return spans


def gantt(log: CellLog, path: Path, policy: str = "") -> Path:
    colors = order_colors(log)
    end = log.end_time
    machines = [s for s in log.stations if s not in ("SRC", "SINK")]
    lanes = machines + list(log.robots)
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(lanes) + 1.8))

    def bar(y: int, t0: float, t1: float, **kw: Any) -> None:
        if t1 > t0:
            ax.barh(y, t1 - t0, left=t0, height=0.8, **kw)

    for st, box, t0, t1 in log.machine_runs:
        y = lanes.index(st)
        bar(y, t0, t1 or end, color=colors[box], edgecolor="k", linewidth=0.6)
        ax.text(t0 + 0.5, y, box.removeprefix("order_"), va="center", fontsize=7)
    for st, t0, t1 in log.machine_blocked:
        bar(lanes.index(st), t0, t1 or end, color="0.9", hatch="..", edgecolor="0.5", linewidth=0.4)
    for who in lanes:
        for t0, t1 in outage_spans(log, who, end):
            bar(lanes.index(who), t0, t1, color="0.85", hatch="xx", edgecolor="0.5", linewidth=0.3)

    for rid in log.robots:
        y = lanes.index(rid)
        carrying, claims = log.carrying[rid], log.claims[rid]
        for t0, t1, s in intervals(log.robot_states[rid], end):
            box = value_at(carrying, t0) or value_at(claims, t0)
            c = colors.get(box, "0.6") if box else "0.6"
            if s == "TRAVEL_LOADED":
                bar(y, t0, t1, color=c, edgecolor="k", linewidth=0.4)
            elif s == "TRAVEL_EMPTY":
                bar(y, t0, t1, color=c, alpha=0.45, hatch="//", edgecolor="k", linewidth=0.4)
            elif s in DOCK_STATES:
                bar(y, t0, t1, facecolor="none", edgecolor=c, linestyle=":", linewidth=1.2)
        for t, r, _ in log.blocked_trips:
            if r == rid:
                ax.plot(t, y, marker="x", color="red", ms=7, zorder=3)

    for o in log.orders:
        ax.axvline(o["due"], color=colors[o["id"]], ls=":", lw=0.8)
    ax.set_yticks(range(len(lanes)), [n.lower() if n in machines else n for n in lanes])
    ax.invert_yaxis()
    ax.set_xlim(0, end)
    ax.set_xlabel(
        "time (s)  ·  solid = loaded  ·  hatched = empty  ·  dotted outline = dock/arm  ·  "
        "grey x-hatch = down  ·  red x = blocked trip  ·  dotted lines = due times"
    )
    ax.set_title(f"{policy} s{log.seed} · {log.scenario}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
