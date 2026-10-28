"""Learning curves, on-time bars, and per-episode Gantt charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from som.core.history import EpisodeLog


def order_colors(log: EpisodeLog) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {o.id: cmap(i % 10) for i, o in enumerate(log.orders)}


def gantt(log: EpisodeLog, path: Path, title: str | None = None) -> None:
    """One lane per machine and per robot. Machine bars are process runs; robot bars are
    driving legs (solid = loaded, hatched = empty) colored by order."""
    colors = order_colors(log)
    lanes = list(log.machines) + list(log.robots)
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(lanes) + 1.6))
    end = log.end_time
    for p in log.processes:
        y = lanes.index(p.machine)
        ax.barh(y, (p.end or end) - p.start, left=p.start, color=colors[p.order], edgecolor="k")
        ax.text(p.start + 0.5, y, p.order.removeprefix("order_"), va="center", fontsize=7)
    for g in log.legs:
        y = lanes.index(g.robot)
        loaded = g.purpose == "dropoff"
        ax.barh(
            y,
            (g.arrive or end) - g.depart,
            left=g.depart,
            color=colors.get(g.order or "", "0.7"),
            alpha=1.0 if loaded else 0.45,
            hatch=None if loaded else "//",
            edgecolor="k",
            linewidth=0.4,
        )
    for o in log.orders:
        ax.axvline(o.due, color=colors[o.id], ls=":", lw=0.8)
    ax.set_yticks(range(len(lanes)), [n.split("_", 1)[1] for n in lanes])
    ax.invert_yaxis()
    ax.set_xlabel("time (s)  ·  dotted lines = due times")
    ax.set_title(title or f"{log.meta.get('policy', '')} · {log.meta.get('scenario', '')}")
    ax.set_xlim(0, end)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
