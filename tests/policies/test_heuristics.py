from __future__ import annotations

import pytest

from som.config.schema import EnvCfg, LayoutCfg, ScenarioCfg
from som.core.decisions import Candidate, DecisionRequest
from som.core.events import OrderReleased
from som.core.layout import Layout
from som.eval.evaluate import run_one
from som.policies.heuristics import (
    MACHINE_RULES,
    ROBOT_RULES,
    all_heuristics,
    make_policy,
    remaining_work,
)
from som.policies.random_policy import RandomPolicy
from som.sim.travel import StraightLineTravel

from helpers import TURN, make_core, spec

TRAVEL = StraightLineTravel(0.5, 1.0)


def _robot_choice(layout: Layout, rule: str, orders, now: float = 0.0) -> str:
    core = make_core(layout, orders, n_robots=1)
    for o in orders:
        core.apply(OrderReleased(o.id), max(core.now, o.release_time))
    (req,) = core.pending_decisions(core.now)
    choice = make_policy(f"{rule}/FIFO", layout, TRAVEL).decide(req, core.snapshot())
    assert isinstance(choice, Candidate)
    return choice.order


def test_robot_fifo_and_edd(tiny_layout: Layout) -> None:
    orders = [spec(1, due=300.0), spec(2, due=50.0, release=1.0)]
    assert _robot_choice(tiny_layout, "FIFO", orders) == "order_1"
    assert _robot_choice(tiny_layout, "EDD", orders) == "order_2"


def test_min_slack_prefers_tighter_order(tiny_layout: Layout) -> None:
    orders = [spec(1, due=100.0), spec(2, due=60.0)]
    assert _robot_choice(tiny_layout, "MIN_SLACK", orders) == "order_2"


def test_remaining_work_counts_legs_docks_and_processing(tiny_layout: Layout) -> None:
    core = make_core(tiny_layout, [spec(1, TURN)])
    core.apply(OrderReleased(core.orders["order_1"].id), 0.0)  # type: ignore[index]
    o = core.snapshot().orders[0]
    src = tiny_layout.source
    # at SRC already: SRC->A (2 s) + A->SINK (4 s) + 2 legs x 2 docks x 2 s + 10 s turning
    assert remaining_work(o, src.dock, src.id, tiny_layout, TRAVEL) == pytest.approx(24.0)


def test_machine_spt_and_edd(tiny_layout: Layout) -> None:
    snap_orders = [spec(1, due=300.0), spec(2, due=50.0)]
    core = make_core(tiny_layout, snap_orders)
    snap = core.snapshot()
    sid = tiny_layout.machines[0].id
    req = DecisionRequest(
        "machine_turning",
        "machine",
        (Candidate(snap_orders[0].id, sid, sid, 0.0), Candidate(snap_orders[1].id, sid, sid, 1.0)),
        True,
    )
    assert make_policy("FIFO/FIFO", tiny_layout, TRAVEL).decide(req, snap).order == "order_1"  # type: ignore[union-attr]
    assert make_policy("FIFO/EDD", tiny_layout, TRAVEL).decide(req, snap).order == "order_2"  # type: ignore[union-attr]


def test_bad_rule_names_rejected(tiny_layout: Layout) -> None:
    with pytest.raises(ValueError, match="robot rule"):
        make_policy("LIFO/FIFO", tiny_layout, TRAVEL)
    with pytest.raises(ValueError, match="ROBOT_RULE/MACHINE_RULE"):
        make_policy("EDD", tiny_layout, TRAVEL)


def test_suite_has_every_pair(tiny_layout: Layout) -> None:
    names = [p.name for p in all_heuristics(tiny_layout, TRAVEL)]
    assert len(names) == len(ROBOT_RULES) * len(MACHINE_RULES) == 12


@pytest.mark.parametrize(
    "name", ["Random"] + [f"{r}/{m}" for r in ROBOT_RULES for m in MACHINE_RULES]
)
def test_every_baseline_completes_abstract_10(
    arena_cfg: LayoutCfg, abstract_10: ScenarioCfg, name: str
) -> None:
    layout = Layout.from_cfg(arena_cfg)
    policy = RandomPolicy() if name == "Random" else make_policy(name, layout, TRAVEL)
    log = run_one(arena_cfg, abstract_10, EnvCfg(), policy, seed=1000, debug_invariants=True)
    assert log.terminated, f"{name} left orders undelivered"
    assert all(o.delivered is not None for o in log.orders)
    assert sum(d.order is None for d in log.decisions) == 0  # dispatchers never WAIT
