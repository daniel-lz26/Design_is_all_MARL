from __future__ import annotations

from dataclasses import asdict

import pytest

import som.baselines.central_cpsat as cc
from som.cell_env.cell_env import CellEnv
from som.cell_env.harness import run_episode
from som.cell_env.metrics import compute
from som.policies.registry import make


@pytest.fixture(autouse=True)
def quick_solves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cc, "FIXED_LIMIT_S", 3.0)
    monkeypatch.setattr(cc, "RANDOM_LIMIT_S", 2.0)
    monkeypatch.setattr(cc, "REPLAN_LIMIT_S", 1.0)


def test_central_plans_cell_10_and_is_reproducible() -> None:
    env = CellEnv("configs/cell/cell_10.yaml")
    logs = []
    for _ in range(2):
        pol = make("CENTRAL_STATIC")
        assert not pol.decentralized
        logs.append(asdict(run_episode(env, pol, 0)))
        assert pol.solves[0][1] in ("OPTIMAL", "FEASIBLE")
        assert compute(env.sim.log).delivered == 10
    assert logs[0] == logs[1]


def test_static_never_replans_and_replan_reacts_to_disruptions() -> None:
    env = CellEnv("configs/cell/cell_10_disrupt.yaml")
    static, replan = make("CENTRAL_STATIC"), make("CENTRAL_REPLAN")
    run_episode(env, static, 20001)
    run_episode(env, replan, 20001)
    assert len(static.solves) == 1
    assert len(replan.solves) > 1
    assert compute(env.sim.log).delivered == len(env.sim.boxes)


def test_coordinator_outage_stops_new_assignments() -> None:
    env = CellEnv("configs/cell/cell_10_disrupt.yaml")
    for seed in range(20000, 20010):
        run_episode(env, make("CENTRAL_REPLAN"), seed)
        windows = env.sim.schedule.coordinator_outages
        if windows:
            break
    else:
        pytest.fail("no coordinator outage in the first ten seeds")
    t0, t1 = windows[0]
    claims = [a for t, _, a in env.sim.log.decisions if t0 <= t < t1 and a < env.k]
    assert claims == []


def test_large_order_books_use_the_greedy_schedule() -> None:
    env = CellEnv("configs/cell/cell_random_30.yaml")
    pol = make("CENTRAL_STATIC")
    run_episode(env, pol, 20000)
    assert pol.solves[0][1] == "GREEDY"
    assert compute(env.sim.log).delivered == 30


def test_one_policy_reused_across_seeds_keeps_its_plan() -> None:
    """evaluate_cell reuses one policy per worker; a cached cell_10 plan must not be consumed."""
    env = CellEnv("configs/cell/cell_10.yaml")
    pol = make("CENTRAL_STATIC")
    done = []
    for seed in (20000, 20001, 20002):
        run_episode(env, pol, seed)
        done.append(compute(env.sim.log).delivered)
    assert done == [10, 10, 10]
    assert len(pol._cache) == 1


def test_a_deadlocked_plan_is_broken_and_every_order_delivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model ignores out-buffer capacity, so with the production 10 s solve on this seed
    every robot's next leg waited on a box stuck behind full out buffers and the cell stalled
    until t_max."""
    monkeypatch.setattr(cc, "RANDOM_LIMIT_S", 10.0)
    env = CellEnv("configs/cell/cell_random_10.yaml")
    pol = make("CENTRAL_REPLAN")
    run_episode(env, pol, 20044)
    assert compute(env.sim.log).delivered == len(env.sim.boxes)
    assert pol.stall_breaks > 0
