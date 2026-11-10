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
    driving legs (solid = loaded, hatched = empty) colored by order. Gray cross-hatched spans
    are breakdowns and outages; a red x is a failed (blocked) trip."""
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
            g.end(end) - g.depart,
            left=g.depart,
            color=colors.get(g.order or "", "0.7"),
            alpha=1.0 if loaded else 0.45,
            hatch=None if loaded else "//",
            edgecolor="k",
            linewidth=0.4,
        )
    down_since: dict[str, float] = {}
    for t, agent, kind, _ in [*log.outages, *((end, a, "up", "") for a in lanes)]:
        if kind == "down":
            down_since.setdefault(agent, t)
        elif agent in down_since:
            start = down_since.pop(agent)
            ax.barh(lanes.index(agent), t - start, left=start, height=0.9, color="0.85",
                    hatch="xx", edgecolor="0.5", linewidth=0.3, zorder=0)  # fmt: skip
    for t, robot, _ in log.failures:
        ax.plot(t, lanes.index(robot), marker="x", color="red", ms=7, zorder=3)
    for o in log.orders:
        ax.axvline(o.due, color=colors[o.id], ls=":", lw=0.8)
    ax.set_yticks(range(len(lanes)), [n.split("_", 1)[1] for n in lanes])
    ax.invert_yaxis()
    xlabel = "time (s)  ·  dotted lines = due times"
    if log.outages or log.failures:
        xlabel += "  ·  gray = down  ·  red x = blocked trip"
    ax.set_xlabel(xlabel)
    ax.set_title(title or f"{log.meta.get('policy', '')} · {log.meta.get('scenario', '')}")
    ax.set_xlim(0, end)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def learning_curves(
    curves: dict[str, list[tuple[float, float]]],
    path: Path,
    title: str,
    baseline: tuple[str, float] | None = None,
) -> None:
    """On-time % against training env steps, one line per training seed.

    ``curves`` maps a label to (env_steps, on_time_pct) points; ``baseline`` draws a horizontal
    reference line such as the best heuristic on the same scenario and seeds.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, pts in curves.items():
        xs, ys = zip(*pts, strict=True) if pts else ((), ())
        ax.plot(xs, ys, marker="o", ms=2.5, lw=1.2, label=label)
    if baseline is not None:
        ax.axhline(baseline[1], color="k", ls="--", lw=1, label=baseline[0])
    ax.set_xlabel("training steps (decision points)")
    ax.set_ylabel("on-time %")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
