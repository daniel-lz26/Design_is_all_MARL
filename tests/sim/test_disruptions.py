from __future__ import annotations

import numpy as np
import pytest

from som.config.schema import DisruptionCfg, EnvCfg, LayoutCfg, ScenarioCfg
from som.core.events import MachineDown, RobotDown
from som.core.ids import MachineId, RobotId
from som.core.layout import Layout
from som.eval.evaluate import run_one
from som.eval.metrics import compute_metrics
from som.policies.heuristics import make_policy
from som.policies.random_policy import RandomPolicy
from som.sim.disruptions import Disruptions
from som.sim.travel import StraightLineTravel

ROBOTS = [RobotId(f"robot_{i}") for i in range(4)]
MACHINES = [MachineId("machine_turning"), MachineId("machine_milling")]


def test_outages_always_leave_one_robot(rng: np.random.Generator) -> None:
    d = Disruptions(DisruptionCfg(robot_outages=9, robot_outage_permanent_prob=1.0), rng)
    events = [e for _, e in d.scheduled(ROBOTS[:2], MACHINES, 1000.0)]
    assert [type(e) for e in events] == [RobotDown]  # one of two robots, never comes back


def test_temporary_outage_and_breakdowns_come_back_up(rng: np.random.Generator) -> None:
    cfg = DisruptionCfg(robot_outages=1, machine_mtbf_s=200.0)
    sched = Disruptions(cfg, rng).scheduled(ROBOTS, MACHINES, 1000.0)
    times = [t for t, _ in sched]
    assert times == sorted(times)
    kinds = [type(e).__name__ for _, e in sched]
    assert kinds.count("RobotDown") == kinds.count("RobotUp") == 1
    assert kinds.count("MachineDown") == kinds.count("MachineUp") >= 1
    assert all(t < 1000.0 for t, e in sched if isinstance(e, MachineDown))


def test_trip_outcomes(rng: np.random.Generator) -> None:
    blocked = Disruptions(DisruptionCfg(trip_block_prob=1.0, trip_block_timeout_s=7.0), rng)
    assert blocked.trip(3.0).blocked and blocked.trip(3.0).duration == 7.0
    assert not blocked.trip(0.0).blocked  # already at the station: nothing to block
    delayed = Disruptions(DisruptionCfg(trip_delay_prob=1.0, trip_delay_s=(5.0, 5.0)), rng)
    assert delayed.trip(3.0).duration == pytest.approx(8.0)


@pytest.mark.parametrize("n_robots", [2, 4])
def test_heuristic_finishes_disrupted_schedule(
    arena_cfg: LayoutCfg, abstract_10_disrupt: ScenarioCfg, n_robots: int
) -> None:
    """Outages, breakdowns and blocked trips never lose an order; the core checks every step."""
    layout = Layout.from_cfg(arena_cfg)
    pol = make_policy(
        "NEAREST/EDD", layout, StraightLineTravel.from_cfg(arena_cfg.travel_calibration)
    )
    env = EnvCfg(n_robots=n_robots)
    saw_failure = saw_outage = False
    for seed in range(1000, 1010):
        log = run_one(arena_cfg, abstract_10_disrupt, env, pol, seed, debug_invariants=True)
        m = compute_metrics(log)
        assert m.delivered == 10 and log.terminated
        saw_failure |= m.n_failed_trips > 0
        saw_outage |= any(o[1].startswith("robot_") for o in log.outages)
    assert saw_failure and saw_outage


def test_disruptions_are_reproducible(
    arena_cfg: LayoutCfg, abstract_10_disrupt: ScenarioCfg
) -> None:
    env = EnvCfg(n_robots=4)
    a = run_one(arena_cfg, abstract_10_disrupt, env, RandomPolicy(), 7)
    b = run_one(arena_cfg, abstract_10_disrupt, env, RandomPolicy(), 7)
    assert a.to_dict() == b.to_dict()


def test_permanently_offline_robot_stops_working(
    arena_cfg: LayoutCfg, abstract_10_disrupt: ScenarioCfg
) -> None:
    sc = abstract_10_disrupt.model_copy(
        update={
            "disruptions": DisruptionCfg(
                robot_outages=1, robot_outage_start_s=(50.0, 50.0), robot_outage_permanent_prob=1.0
            )
        }
    )
    log = run_one(arena_cfg, sc, EnvCfg(n_robots=2), RandomPolicy(), 3, debug_invariants=True)
    ((t, robot, kind, _),) = log.outages
    assert (t, kind) == (50.0, "down")
    assert all(d.t <= 50.0 for d in log.decisions if d.agent == robot)
    assert compute_metrics(log).delivered == 10  # the other robot picks up the slack
