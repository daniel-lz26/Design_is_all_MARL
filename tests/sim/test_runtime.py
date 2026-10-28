from __future__ import annotations

import math

import numpy as np
import pytest

from som.config.schema import EnvCfg, LayoutCfg, ScenarioCfg
from som.core.events import MachineDown, MachineUp, RobotFailed
from som.core.history import EpisodeLog
from som.core.ids import MachineId, RobotId
from som.core.layout import Layout, Pose2D
from som.eval.evaluate import run_one
from som.policies.heuristics import make_policy
from som.policies.random_policy import RandomPolicy
from som.sim.runtime import SimRuntime, run_episode
from som.sim.scenario import build_orders
from som.sim.travel import StraightLineTravel

from helpers import TURN_MILL, make_core, spec

ENV = EnvCfg(n_robots=2, travel_noise_sigma=0.1)


# --------------------------------------------------------------------------- travel


def test_travel_estimate_straight_and_rotating() -> None:
    tr = StraightLineTravel(v_eff=0.5, w_eff=1.0)
    a = Pose2D(0, 0, 0)
    assert tr.estimate(a, a) == 0.0
    assert tr.estimate(a, Pose2D(1, 0)) == pytest.approx(2.0)
    assert tr.estimate(a, Pose2D(0, 1)) == pytest.approx(math.pi / 2 + 2.0)
    assert tr.estimate(a, Pose2D(-1, 0)) == pytest.approx(math.pi + 2.0)


def test_travel_noise_has_mean_one(rng: np.random.Generator) -> None:
    tr = StraightLineTravel(v_eff=0.5, w_eff=1.0, noise_sigma=0.2)
    samples = [tr.sample(Pose2D(0, 0), Pose2D(1, 0), rng) for _ in range(20000)]
    assert np.mean(samples) == pytest.approx(2.0, rel=0.01)
    assert np.std(samples) > 0.2


# --------------------------------------------------------------------------- scenario


def test_fixed_scenario_uses_default_process_times(abstract_10: ScenarioCfg) -> None:
    specs = build_orders(abstract_10)
    assert len(specs) == 10
    o2 = specs[1]
    assert o2.job.route == ("Turning", "Milling", "Drilling")
    assert o2.process_times == (30, 40, 10)
    assert (o2.release_time, o2.due_time) == (31, 166)


def test_generated_scenario_is_seeded(random_10: ScenarioCfg) -> None:
    a = build_orders(random_10, np.random.default_rng(3))
    b = build_orders(random_10, np.random.default_rng(3))
    c = build_orders(random_10, np.random.default_rng(4))
    assert a == b and a != c
    assert len(a) == 10
    releases = [o.release_time for o in a]
    assert releases == sorted(releases) and releases[0] == 0.0
    for o in a:
        assert 115.0 <= o.due_time - o.release_time <= 155.0
    with pytest.raises(ValueError, match="needs an rng"):
        build_orders(random_10)


# --------------------------------------------------------------------------- runtime


def _rt(layout: Layout, orders, horizon: float = 1000.0, sigma: float = 0.0) -> SimRuntime:
    core = make_core(layout, orders)
    tr = StraightLineTravel(0.5, 1.0, sigma)
    return SimRuntime(core, tr, np.random.default_rng(0), horizon)


def test_runtime_completes_tiny_scenario(tiny_layout: Layout) -> None:
    rt = _rt(tiny_layout, [spec(1, TURN_MILL), spec(2, TURN_MILL, release=3.0)])
    policy = make_policy("FIFO/FIFO", tiny_layout, StraightLineTravel(0.5, 1.0))
    log = run_episode(rt, policy)
    assert log.terminated and not log.truncated
    assert all(o.delivered is not None for o in log.orders)
    assert log.end_time == max(o.delivered for o in log.orders if o.delivered)


def test_runtime_truncates_at_horizon(tiny_layout: Layout) -> None:
    rt = _rt(tiny_layout, [spec(1, TURN_MILL)], horizon=15.0)
    log = run_episode(rt, make_policy("EDD/EDD", tiny_layout, StraightLineTravel(0.5, 1.0)))
    assert log.truncated and not log.terminated and log.end_time == 15.0


def test_robot_failure_mid_episode_still_completes(tiny_layout: Layout) -> None:
    rt = _rt(tiny_layout, [spec(1, TURN_MILL), spec(2, TURN_MILL)])
    rt.schedule(3.0, RobotFailed(RobotId("robot_0"), "injected"))  # mid-drive to A
    log = run_episode(rt, make_policy("FIFO/FIFO", tiny_layout, StraightLineTravel(0.5, 1.0)))
    assert log.terminated
    assert [f[1] for f in log.failures] == ["robot_0"]
    assert any(g.arrive is None for g in log.legs)  # the failed leg never arrived


def test_machine_down_window_delays_but_completes(tiny_layout: Layout) -> None:
    orders = [spec(1, TURN_MILL)]
    pol = make_policy("FIFO/FIFO", tiny_layout, StraightLineTravel(0.5, 1.0))
    base = run_episode(_rt(tiny_layout, orders), pol)
    rt = _rt(tiny_layout, orders)
    turning = MachineId("machine_turning")
    rt.schedule(8.0, MachineDown(turning))  # mid-run: turning starts ~6 s and runs 10 s
    rt.schedule(40.0, MachineUp(turning))
    log = run_episode(rt, pol)
    assert log.terminated
    assert log.end_time > base.end_time
    segs = [p for p in log.processes if p.machine == turning]
    assert len(segs) == 2 and segs[0].end == 8.0 and segs[1].start == 40.0
    total = sum((p.end or 0) - p.start for p in segs)
    assert total == pytest.approx(10.0)


def _dump(log: EpisodeLog) -> dict:
    return log.to_dict()


def test_same_seed_gives_identical_log(arena_cfg: LayoutCfg, random_10: ScenarioCfg) -> None:
    a = run_one(arena_cfg, random_10, ENV, RandomPolicy(), seed=7)
    b = run_one(arena_cfg, random_10, ENV, RandomPolicy(), seed=7)
    c = run_one(arena_cfg, random_10, ENV, RandomPolicy(), seed=8)
    assert _dump(a) == _dump(b)
    assert _dump(a) != _dump(c)


def test_log_round_trips_through_dict(arena_cfg: LayoutCfg, abstract_10: ScenarioCfg) -> None:
    log = run_one(arena_cfg, abstract_10, ENV, RandomPolicy(), seed=1)
    assert EpisodeLog.from_dict(log.to_dict()) == log


@pytest.mark.parametrize("seed", range(25))
def test_random_policy_preserves_invariants(
    arena_cfg: LayoutCfg, random_10: ScenarioCfg, seed: int
) -> None:
    """Property-style: random decisions (including WAIT) never break core invariants.
    debug_invariants makes the core assert after every apply/assign."""
    log = run_one(
        arena_cfg, random_10, ENV, RandomPolicy(allow_wait=True), seed, debug_invariants=True
    )
    delivered = [o.id for o in log.orders if o.delivered is not None]
    assert len(delivered) == len(set(delivered))
    assert log.terminated or log.truncated
