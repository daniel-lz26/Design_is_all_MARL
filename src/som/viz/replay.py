"""Animate an episode log over the arena layout (matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from som.core.history import EpisodeLog, OrderRecord
from som.core.layout import Layout, StationKind
from som.eval.plots import order_colors

_KIND_COLOR = {
    StationKind.SOURCE: "#4c9f70",
    StationKind.MACHINE: "#5b7db1",
    StationKind.SINK: "#b1645b",
}


def status_at(o: OrderRecord, t: float) -> tuple[str, str]:
    """(status, location) of an order at time ``t``."""
    cur = ("PENDING", "")
    for ts, status, loc in o.timeline:
        if ts > t:
            break
        cur = (status, loc)
    return cur


def robot_xy(
    log: EpisodeLog, layout: Layout, robot: str, t: float
) -> tuple[float, float, str | None]:
    """Interpolated position of ``robot`` at ``t`` and the order it carries (if any)."""
    i = log.robots.index(robot)
    starts = log.meta.get("robot_starts") or [[p.x, p.y, p.theta] for p in layout.robot_starts]
    x, y = float(starts[i][0]), float(starts[i][1])
    for g in log.legs:
        if g.robot != robot or g.depart > t:
            continue
        dock = next(s.dock for s in layout.stations if s.id == g.to_station)
        arrive = g.arrive if g.arrive is not None else log.end_time
        if t >= arrive:
            if g.arrive is not None:
                x, y = dock.x, dock.y
        else:
            f = (t - g.depart) / max(arrive - g.depart, 1e-9)
            x = g.from_xy[0] + f * (dock.x - g.from_xy[0])
            y = g.from_xy[1] + f * (dock.y - g.from_xy[1])
    carrying = next((o.id for o in log.orders if status_at(o, t) == ("IN_TRANSIT", robot)), None)
    return x, y, carrying


def animate(
    log: EpisodeLog,
    layout: Layout,
    path: Path,
    sim_dt: float = 2.0,
    fps: int = 12,
    title: str | None = None,
) -> Path:
    """Render the episode to a gif. One frame per ``sim_dt`` seconds of episode time."""
    colors = order_colors(log)
    width = max(s.dock.x for s in layout.stations) + 0.4
    height = max(s.dock.y for s in layout.stations) + 0.4
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [2.4, 1]})
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    for s in layout.stations:
        ax.add_patch(
            Rectangle(
                (s.dock.x - 0.12, s.dock.y - 0.12),
                0.24,
                0.24,
                color=_KIND_COLOR[s.kind],
                alpha=0.35,
            )
        )
        ax.text(s.dock.x, s.dock.y + 0.17, s.key, ha="center", fontsize=8)
    queue_txt = {
        s.id: ax.text(s.dock.x, s.dock.y - 0.2, "", ha="center", fontsize=7)
        for s in layout.stations
    }
    dots = {r: ax.plot([], [], "o", ms=13, mec="k")[0] for r in log.robots}
    labels = {r: ax.text(0, 0, "", ha="center", va="center", fontsize=7) for r in log.robots}
    tx.axis("off")
    table = tx.text(0, 1, "", va="top", family="monospace", fontsize=7.5)
    head = fig.suptitle("")
    n_frames = int(log.end_time / sim_dt) + 2
    name = title or f"{log.meta.get('policy', '')} · {log.meta.get('scenario', '')}"

    def draw(k: int) -> list[object]:
        t = min(k * sim_dt, log.end_time)
        counts: dict[str, int] = {}
        rows = ["order  status          due"]
        done = ontime = 0
        for o in log.orders:
            status, loc = status_at(o, t)
            if status in ("RELEASED", "WAITING_PICKUP", "RESERVED", "QUEUED", "PROCESSING"):
                counts[loc] = counts.get(loc, 0) + 1
            if status == "DELIVERED":
                done += 1
                ontime += int(o.delivered is not None and o.delivered <= o.due)
            late = "!" if status != "DELIVERED" and t > o.due else " "
            rows.append(f"{o.id.removeprefix('order_'):>5}  {status:<14} {o.due:5.0f}{late}")
        for sid, txt in queue_txt.items():
            txt.set_text(f"{counts[sid]} here" if counts.get(sid) else "")
        for r in log.robots:
            x, y, carrying = robot_xy(log, layout, r, t)
            dots[r].set_data([x], [y])
            dots[r].set_color(colors[carrying] if carrying else "white")
            dots[r].set_markeredgecolor("k")
            labels[r].set_position((x, y))
            labels[r].set_text(r.removeprefix("robot_"))
        table.set_text("\n".join(rows) + "\n\n! = past due")
        head.set_text(
            f"{name}   t = {t:5.0f} s   delivered {done}/{len(log.orders)}   on time {ontime}"
        )
        return [*dots.values(), *labels.values(), table, head]

    anim = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)
    return path
