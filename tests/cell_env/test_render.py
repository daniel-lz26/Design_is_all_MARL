from __future__ import annotations

from pathlib import Path

from PIL import Image

from som.baselines.cell_rules import CellRulePolicy, run_rule_episode
from som.cell_env.des import make_sim
from som.cell_env.render_gantt import gantt
from som.cell_env.render_replay import animate, robot_xy


def test_gantt_and_replay_render(tmp_path: Path) -> None:
    sim = make_sim("configs/cell/cell_10_disrupt.yaml")
    run_rule_episode(sim, CellRulePolicy("NEAREST", "FIFO"), 20003)
    png = gantt(sim.log, tmp_path / "g.png", "NEAREST/FIFO")
    assert Image.open(png).size[0] > 1000
    gif = animate(sim, tmp_path / "r.gif", "NEAREST/FIFO", sim_dt=60.0)
    assert Image.open(gif).n_frames == int(sim.log.end_time / 60.0) + 2


def test_robot_positions_stay_on_the_floor() -> None:
    sim = make_sim("configs/cell/cell_random_10_disrupt.yaml")
    run_rule_episode(sim, CellRulePolicy("EDD", "FIFO"), 20001)
    w, h = sim.layout.floor_m
    for rid in sim.log.robots:
        for k in range(0, int(sim.log.end_time), 5):
            x, y = robot_xy(sim, sim.log, rid, float(k))
            assert 0 <= x <= w and 0 <= y <= h
