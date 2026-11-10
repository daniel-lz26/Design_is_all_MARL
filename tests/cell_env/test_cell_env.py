from __future__ import annotations

import time
from dataclasses import asdict

import pytest

import som.cell_env.des as des
from som.baselines.cell_rules import ROBOT_RULES, CellRulePolicy, run_rule_episode
from som.cell_env.des import CellSim, make_sim
from som.cell_env.entities import RS
from som.cell_env.metrics import compute
from som.cell_env.orders import fixed_orders, min_flow_time
from som.config.cell_schema import CellDisruptionCfg, HandlingCfg

SCENARIOS = [
    "cell_10",
    "cell_10_raw",
    "cell_10_disrupt",
    "cell_random_10",
    "cell_random_10_disrupt",
    "cell_random_10_flex",
    "cell_random_20",
    "cell_random_30",
    "cell_stream",
]


def sim_for(name: str) -> CellSim:
    return make_sim(f"configs/cell/{name}.yaml")


@pytest.mark.parametrize("name", SCENARIOS)
def test_invariants_hold_after_every_event(name: str) -> None:
    """Box conservation, edge/dock/arm/buffer limits and claims, checked at every event."""
    sim = sim_for(name)
    sim.debug = True
    for rule in ("NEAREST", "MST", "RANDOM"):
        for seed in (20000, 20001):
            run_rule_episode(sim, CellRulePolicy(rule, "EDD"), seed)
            assert sim.log.terminated or sim.log.truncated


def test_every_robot_rule_finishes_cell_10() -> None:
    sim = sim_for("cell_10")
    for rule in ROBOT_RULES:
        run_rule_episode(sim, CellRulePolicy(rule, "FIFO"), 0)
        assert compute(sim.log).delivered == 10, rule


@pytest.mark.parametrize("name", ["cell_10", "cell_random_10_disrupt", "cell_random_10_flex"])
def test_same_seed_same_trajectory(name: str) -> None:
    sim = sim_for(name)
    runs = []
    for _ in range(2):
        run_rule_episode(sim, CellRulePolicy("NEAREST", "SPT"), 20005)
        runs.append(asdict(sim.log))
    assert runs[0] == runs[1]


def test_heuristics_are_deterministic_on_cell_10() -> None:
    """cell_10 has no noise or disruptions: the seed must not matter."""
    sim = sim_for("cell_10")
    logs = []
    for seed in (0, 20000, 20099):
        run_rule_episode(sim, CellRulePolicy("EDD", "EDD"), seed)
        log = asdict(sim.log)
        log.pop("seed")
        logs.append(log)
    assert logs[0] == logs[1] == logs[2]


def test_masked_action_is_never_executed() -> None:
    sim = sim_for("cell_10")
    pending = sim.reset(0)
    rid = pending[0]
    mask = sim.mask(rid)
    bad = int((mask == 0).nonzero()[0][0])
    with pytest.raises(ValueError, match="masked"):
        sim.act(rid, bad)
    with pytest.raises(ValueError, match="masked"):
        sim.act(rid, len(mask))
    assert sim.log.decisions == []


def test_down_robots_get_no_actions() -> None:
    sim = sim_for("cell_10_disrupt")
    policy = CellRulePolicy("NEAREST", "FIFO")
    saw_down = False
    for seed in range(20000, 20010):
        policy.reset(sim, seed)
        pending = sim.reset(seed)
        while pending:
            for r in sim.robots.values():
                if r.state is RS.DOWN:
                    saw_down = True
                    assert not sim.mask(r.id).any() and r.id not in pending
            for rid in pending:
                if rid in sim.pending():
                    sim.act(rid, policy.choose(sim, rid))
            pending = sim.advance()
    assert saw_down


def test_decision_speed() -> None:
    """Spec: at least 2,000 decision points per second on one core."""
    sim = sim_for("cell_10")
    policy = CellRulePolicy("NEAREST", "FIFO")
    run_rule_episode(sim, policy, 0)  # warm caches
    t0, decisions = time.perf_counter(), 0
    for seed in range(40):
        run_rule_episode(sim, policy, seed)
        decisions += len(sim.log.decisions)
    assert decisions / (time.perf_counter() - t0) >= 2000


# --------------------------------------------------------------------------- due dates and limits


def test_due_dates_follow_the_calibrated_rule() -> None:
    sim, raw = sim_for("cell_10"), sim_for("cell_10_raw")
    sim.reset(0)
    raw.reset(0)
    abstract = {o.id: o for o in fixed_orders(sim.root / "configs/scenarios/abstract_10.yaml")}
    for b in sim.boxes.values():
        mf = min_flow_time(b.spec.route, b.spec.proc, sim.layout, sim.h, sim.roadmap)
        assert b.spec.due == pytest.approx(b.spec.release + 1.5 * mf, abs=1e-3)
        assert raw.boxes[b.id].spec.due == abstract[b.id].due
    assert sim.t_max == pytest.approx(3 * max(b.spec.due for b in sim.boxes.values()))


def test_min_flow_time_is_a_lower_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = sim_for("cell_10_raw")
    for spec in fixed_orders(sim.root / "configs/scenarios/abstract_10.yaml"):
        monkeypatch.setattr(des, "build_orders", lambda *a, spec=spec, **k: [spec])
        run_rule_episode(sim, CellRulePolicy("NEAREST", "FIFO"), 0)
        mf = min_flow_time(spec.route, spec.proc, sim.layout, sim.h, sim.roadmap)
        assert sim.log.orders[0]["delivered"] - spec.release >= mf - 1e-6


def test_undelivered_orders_count_late_up_to_t_max() -> None:
    base = sim_for("cell_10")
    sim = CellSim(base.cfg.model_copy(update={"t_max": 250.0}), base.layout, base.root)
    run_rule_episode(sim, CellRulePolicy("NEAREST", "FIFO"), 0)
    m = compute(sim.log)
    assert sim.log.truncated and sim.log.end_time == 250.0 and m.delivered < m.n_orders
    late = [max(0.0, (o["delivered"] or 250.0) - o["due"]) for o in sim.log.orders]
    assert m.mean_tardiness == pytest.approx(sum(late) / len(late))
    assert m.on_time == sum(
        o["delivered"] is not None and o["delivered"] <= o["due"] for o in sim.log.orders
    )


# --------------------------------------------------------------------------- disruptions


def _disrupt(**kw: object) -> CellSim:
    base = sim_for("cell_10")
    return CellSim(
        base.cfg.model_copy(update={"disruptions": CellDisruptionCfg(**kw)}), base.layout, base.root
    )


def test_box_on_a_broken_robot_is_stuck_or_rescued() -> None:
    stuck, rescued = 0, 0
    for rescue in (False, True):
        sim = _disrupt(robot_down="permanent", rescue=rescue)
        sim.debug = True
        for seed in range(20000, 20030):
            run_rule_episode(sim, CellRulePolicy("NEAREST", "FIFO"), seed)
            down = next(r for r in sim.robots.values() if r.state is RS.DOWN)
            if rescue:
                rescued += compute(sim.log).delivered == 10
            elif down.box is not None:
                stuck += 1
                assert sim.log.truncated and compute(sim.log).delivered == 9
    assert stuck > 0 and rescued == 30


def test_closed_aisle_is_avoided_when_possible() -> None:
    sim = sim_for("cell_10")
    t_open, path = sim.roadmap.path("A0", "A3")
    blocked = frozenset({frozenset(("A1", "A2"))})
    t_closed, detour = sim.roadmap.path("A0", "A3", blocked)
    assert "A2" in path and t_closed > t_open
    assert all(frozenset(e) not in blocked for e in zip(["A0", *detour], detour, strict=False))


def test_full_out_buffer_blocks_the_machine() -> None:
    base = sim_for("cell_random_20")
    slow = HandlingCfg(arm_load_s=60.0, arm_unload_s=6.0)  # pickups crawl, out-buffers fill
    sim = CellSim(base.cfg.model_copy(update={"handling": slow}), base.layout, base.root)
    sim.debug = True
    blocked = 0
    for seed in range(10000, 10003):
        run_rule_episode(sim, CellRulePolicy("NEAREST", "FIFO"), seed)
        blocked += len(sim.log.machine_blocked)
    assert blocked > 0


def test_robot_joins_mid_shift() -> None:
    sim = sim_for("cell_random_10_flex")
    run_rule_episode(sim, CellRulePolicy("NEAREST", "FIFO"), 20000)
    joins = [o for o in sim.log.outages if o[3] == "JOIN"]
    assert len(joins) == 1
    t_join, rid = joins[0][0], joins[0][1]
    assert all(t >= t_join for t, r, _ in sim.log.decisions if r == rid)
    assert any(r == rid for _, r, _ in sim.log.decisions)
