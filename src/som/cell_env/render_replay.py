"""2D replay of a cell episode (spec §12, plot 4), in the style of ``som.viz.replay``.

Floor plan with the roadmap (thick = two lanes, thin dashed = single lane, red = closed),
stations with their arm (green idle, orange busy, red down) and buffer counts, parking spots,
and robots coloured by the box on top (white = empty, grey x = down). The order table on the
right shows each order's status and due time ("!" = past due).

    python -m som.cell_env.render_replay --config configs/cell/cell_10.yaml --policy NEAREST/FIFO
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

from som.cell_env.des import CellSim
from som.cell_env.entities import CellLog
from som.cell_env.render_gantt import order_colors, outage_spans, value_at

ARM_IDLE, ARM_BUSY, ARM_DOWN = "#4c9f70", "#e08a2c", "#c0392b"


def robot_xy(sim: CellSim, log: CellLog, rid: str, t: float) -> tuple[float, float]:
    xy = sim.roadmap.xy
    for t_down, pos in log.down_at[rid]:
        if t >= t_down and any(t0 <= t < t1 for t0, t1 in outage_spans(log, rid, log.end_time)):
            return pos
    for r, st, t0, t1 in log.docks:
        if r == rid and t0 <= t < (t1 if t1 is not None else log.end_time):
            return xy[sim.stations[st].dock]
    moves = log.moves[rid]
    i = bisect_right([m[0] for m in moves], t) - 1
    if i < 0:
        return xy[sim.robots[rid].park]
    t0, t1, a, b = moves[i]
    if t >= t1:
        return xy[b]
    f = (t - t0) / max(t1 - t0, 1e-9)
    (x0, y0), (x1, y1) = xy[a], xy[b]
    return x0 + f * (x1 - x0), y0 + f * (y1 - y0)


def order_status(log: CellLog, o: dict, t: float) -> str:
    if o["delivered"] is not None and o["delivered"] <= t:
        return "DELIVERED"
    if t < o["release"]:
        return "PENDING"
    for rid, tl in log.carrying.items():
        if value_at(tl, t) == o["id"]:
            return f"ON {rid}"
    for st, box, t0, t1 in log.machine_runs:
        if box == o["id"] and t0 <= t < (t1 if t1 is not None else log.end_time):
            return f"IN {st[:4]}"
    return "WAITING"


def animate(sim: CellSim, path: Path, policy: str = "", sim_dt: float = 2.0, fps: int = 12) -> Path:
    log = sim.log
    colors = order_colors(log)
    xy = sim.roadmap.xy
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [2.3, 1]})
    w, h = sim.layout.floor_m
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    edge_lines = {}
    for e in sim.roadmap.edges.values():
        (x0, y0), (x1, y1) = xy[e.a], xy[e.b]
        style = {"lw": 3.0, "ls": "-"} if e.capacity > 1 else {"lw": 1.2, "ls": "--"}
        (edge_lines[e.key],) = ax.plot([x0, x1], [y0, y1], color="0.75", zorder=0, **style)
    for p in sim.layout.parking:
        ax.add_patch(
            Rectangle(
                (xy[p][0] - 0.12, xy[p][1] - 0.1), 0.24, 0.2, fill=False, edgecolor="0.5", ls=":"
            )
        )
    arms, buf_txt = {}, {}
    for st in sim.stations.values():
        x, y = xy[st.dock]
        ax.add_patch(Rectangle((x - 0.13, y - 0.13), 0.26, 0.26, color="#5b7db1", alpha=0.25))
        above = y >= xy[st.approach][1]
        ty = y + 0.22 if above else y - 0.3
        ax.text(x, ty, st.key, ha="center", fontsize=7, weight="bold")
        arms[st.key] = Circle((x + 0.2, y), 0.06, color=ARM_IDLE)
        ax.add_patch(arms[st.key])
        buf_txt[st.key] = ax.text(x, ty + (0.13 if above else -0.13), "", ha="center", fontsize=6)
    dots = {r: ax.plot([], [], "o", ms=12, mec="k", zorder=4)[0] for r in log.robots}
    labels = {
        r: ax.text(0, 0, "", ha="center", va="center", fontsize=6, zorder=5) for r in log.robots
    }
    tx.axis("off")
    table = tx.text(0, 1, "", va="top", family="monospace", fontsize=7)
    head = fig.suptitle("")
    buffers: dict[str, list[tuple[float, int, int]]] = {}
    for t, st, n_in, n_out in log.buffers:
        buffers.setdefault(st, []).append((t, n_in, n_out))

    def draw(k: int) -> list[object]:
        t = min(k * sim_dt, log.end_time)
        closed = {
            target
            for target in {o[1] for o in log.outages if "|" in o[1]}
            if any(t0 <= t < t1 for t0, t1 in outage_spans(log, target, log.end_time))
        }
        for key, line in edge_lines.items():
            a, b = sorted(key)
            line.set_color("#c0392b" if f"{a}|{b}" in closed or f"{b}|{a}" in closed else "0.75")
        for st in sim.stations.values():
            down = any(t0 <= t < t1 for t0, t1 in outage_spans(log, st.key, log.end_time))
            busy = any(
                h[0] == st.key and h[3] <= t < (h[4] if h[4] is not None else log.end_time)
                for h in log.handoffs
            )
            arms[st.key].set_color(ARM_DOWN if down else ARM_BUSY if busy else ARM_IDLE)
            bt = buffers.get(st.key, [])
            i = bisect_right([b[0] for b in bt], t) - 1
            n_in, n_out = (bt[i][1], bt[i][2]) if i >= 0 else (0, 0)
            if st.kind == "machine":
                buf_txt[st.key].set_text(f"in {n_in}/{st.in_cap} out {n_out}/{st.out_cap}")
            elif st.kind == "source":
                buf_txt[st.key].set_text(f"{n_out} waiting")
        for rid in log.robots:
            x, y = robot_xy(sim, log, rid, t)
            down = any(t0 <= t < t1 for t0, t1 in outage_spans(log, rid, log.end_time))
            box = value_at(log.carrying[rid], t)
            dots[rid].set_data([x], [y])
            dots[rid].set_color("0.6" if down else colors[box] if box else "white")
            dots[rid].set_markeredgecolor("k")
            labels[rid].set_position((x, y))
            labels[rid].set_text("x" if down else rid.removeprefix("agv_"))
        rows = ["order  status        due"]
        done = ontime = 0
        for o in log.orders:
            s = order_status(log, o, t)
            done += s == "DELIVERED"
            ontime += s == "DELIVERED" and o["delivered"] <= o["due"]
            late = "!" if s != "DELIVERED" and t > o["due"] else " "
            rush = "*" if o["rush"] else " "
            rows.append(f"{o['id'].removeprefix('order_'):>4}{rush} {s:<12} {o['due']:5.0f}{late}")
        table.set_text("\n".join(rows) + "\n\n! = past due   * = rush order")
        head.set_text(
            f"{policy} s{log.seed} · {log.scenario}   t = {t:5.0f} s   "
            f"delivered {done}/{len(log.orders)}   on time {ontime}"
        )
        return [*dots.values(), *labels.values(), table, head]

    anim = FuncAnimation(fig, draw, frames=int(log.end_time / sim_dt) + 2, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)
    return path


def main() -> None:
    from som.baselines.cell_rules import CellRulePolicy, run_rule_episode
    from som.cell_env.des import make_sim

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/cell/cell_10.yaml")
    ap.add_argument("--policy", default="NEAREST/FIFO")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    robot_rule, _, machine_rule = args.policy.upper().partition("/")
    sim = make_sim(args.config)
    run_rule_episode(sim, CellRulePolicy(robot_rule, machine_rule or "FIFO"), args.seed)
    name = args.policy.replace("/", "__")
    out = args.out or Path("reports") / f"replay_{sim.cfg.name}_{name}_s{args.seed}.gif"
    print(f"wrote {animate(sim, out, args.policy)}")


if __name__ == "__main__":
    main()
