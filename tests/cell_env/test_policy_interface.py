from __future__ import annotations

import csv
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from som.baselines.cell_rules import ROBOT_RULES, CellRulePolicy, run_rule_episode
from som.cell_env.cell_env import CellEnv
from som.cell_env.des import CellSim
from som.cell_env.entities import RS
from som.cell_env.harness import evaluate, run_episode
from som.config.cell_schema import CellRewardCfg
from som.policies.registry import REGISTRY, family, make

ROOT = Path(__file__).resolve().parents[2]


class Spy:
    """Claims the first task (or waits) and records what the harness handed it."""

    def __init__(self, decentralized: bool) -> None:
        self.decentralized = decentralized
        self.name = "SPY"
        self.seen: list[Any] = []

    def reset(self, scenario_info: dict[str, Any]) -> None:
        self.acts = scenario_info["actions"]

    def act(self, obs: dict, masks: dict, global_state: dict | None = None) -> dict[str, int]:
        self.seen.append(global_state)
        return {a: 0 if masks[a][0] else int(self.acts["wait"]) for a in obs}


@pytest.mark.parametrize("decentralized", [True, False])
def test_decentralized_policies_never_get_global_state(decentralized: bool) -> None:
    spy = Spy(decentralized)
    run_episode(CellEnv("configs/cell/cell_random_10_disrupt.yaml"), spy, 20000)
    assert spy.seen
    if decentralized:
        assert all(gs is None for gs in spy.seen)
    else:
        assert all(isinstance(gs, dict) and "sim" in gs for gs in spy.seen)


def test_registry_holds_every_baseline() -> None:
    rules = [n for n in REGISTRY if family(n) == "A"]
    auctions = {"AUCTION", "AUCTION/EDD", "AUCTION/SPT"}
    assert len(rules) == 21 and auctions | {f"AUCTION_v0{a[7:]}" for a in auctions} <= set(REGISTRY)
    assert all(make(n).decentralized for n in REGISTRY if family(n) in "AB")
    with pytest.raises(KeyError, match="unknown policy"):
        make("NOT_A_POLICY")


@pytest.mark.parametrize("scenario", ["cell_10", "cell_random_10_disrupt"])
def test_observation_rules_match_simulator_rules(scenario: str) -> None:
    env = CellEnv(f"configs/cell/{scenario}.yaml")
    ref = CellSim(env.sim.cfg, env.sim.layout, env.sim.root)
    for rule in ROBOT_RULES:
        log = asdict(run_episode(env, make(f"{rule}/EDD"), 20002))
        run_rule_episode(ref, CellRulePolicy(rule, "EDD"), 20002)
        assert log == asdict(ref.log), rule


def test_pettingzoo_parallel_api() -> None:
    parallel_api_test(CellEnv("configs/cell/cell_random_10.yaml", strict=False), num_cycles=300)


def test_noop_rules() -> None:
    env = CellEnv("configs/cell/cell_10_disrupt.yaml")
    rng = np.random.default_rng(0)
    _, infos = env.reset(seed=20000)
    saw_down = False
    while env.agents:
        acts = {}
        for a in env.agents:
            m = infos[a]["action_mask"]
            if infos[a]["decision"]:
                assert not m[env.NOOP] and m[: env.NOOP].any()
            else:
                assert m.tolist() == [0] * env.NOOP + [1]
            if env.sim.robots[a].state is RS.DOWN:
                saw_down = True
                assert m.tolist() == [0] * env.NOOP + [1]
            acts[a] = int(rng.choice(np.flatnonzero(m)))
        _, _, _, _, infos = env.step(acts)
    assert saw_down


def test_masked_action_is_rejected() -> None:
    env = CellEnv("configs/cell/cell_10.yaml")
    _, infos = env.reset(seed=0)
    agent = next(a for a in env.agents if infos[a]["decision"])
    with pytest.raises(ValueError, match="masked"):
        env.step({agent: env.NOOP})


def test_team_reward_adds_up_to_the_episode_objective() -> None:
    cfg = CellRewardCfg(feed_idle_machine=0.0, blocked_trip=0.0, empty_travel_per_s=0.0)
    for scenario, t_max in (("cell_random_10", None), ("cell_10", 300.0)):
        env = CellEnv(f"configs/cell/{scenario}.yaml", reward=cfg)
        if t_max:
            env.sim.cfg = env.sim.cfg.model_copy(update={"t_max": t_max})
        rng = np.random.default_rng(1)
        _, infos = env.reset(seed=20003)
        total = 0.0
        while env.agents:
            acts = {a: int(rng.choice(np.flatnonzero(infos[a]["action_mask"]))) for a in env.agents}
            _, rew, _, _, infos = env.step(acts)
            assert len(set(rew.values())) == 1
            total += next(iter(rew.values()))
        end = env.sim.log.end_time
        expected = 0.0
        for o in env.sim.log.orders:
            finish = o["delivered"] if o["delivered"] is not None else end
            tard = max(0.0, finish - o["due"])
            expected += 10.0 if o["delivered"] is not None and tard == 0 else -min(tard / 30, 5)
            expected -= 0.01 * tard
        assert total == pytest.approx(expected, abs=1e-6), scenario


@pytest.mark.parametrize("mode", ["board", "local", "global"])
def test_observation_size_does_not_depend_on_robot_count(mode: str) -> None:
    dims = set()
    for n in (2, 3, 4):
        env = CellEnv("configs/cell/cell_random_10.yaml", obs_mode=mode, n_robots=n)
        obs, _ = env.reset(seed=20000)
        dims |= {o.shape for o in obs.values()}
        assert all(np.isfinite(o).all() for o in obs.values())
    assert len(dims) == 1


def test_local_mode_hides_far_robots() -> None:
    env = CellEnv("configs/cell/cell_random_10.yaml", obs_mode="local")
    env.enc.radius = 0.0
    obs, _ = env.reset(seed=20000)
    lay = env.enc.layout
    for o in obs.values():
        assert not lay.others(o)[:, 0].any()


def test_evaluate_and_compare_scripts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import compare_cell
    import evaluate_cell

    for p in ("NEAREST/SPT", "AUCTION"):
        evaluate_cell.run_job(p, None, "cell_random_10", "test", None, str(tmp_path), "t", 5)
    rows = list(csv.DictReader((tmp_path / "AUCTION" / "cell_random_10.csv").open()))
    assert len(rows) == 5 and rows[0]["env_version"] == "t" and rows[0]["family"] == "B"
    assert (tmp_path / "NEAREST__SPT" / "gantt_cell_random_10_s20000.png").exists()
    monkeypatch.setattr(sys, "argv", ["compare_cell.py", "--results", str(tmp_path)])
    compare_cell.main()
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "**NEAREST/SPT** (best A)" in text and "AUCTION" in text
    assert (
        evaluate(lambda: make("AUCTION"), "configs/cell/cell_10.yaml", [0])[0]["policy"]
        == "AUCTION"
    )


class SimAuction:
    """Reference contract-net auction computed from the simulator (centralized, test only)."""

    decentralized = False

    def __init__(self, busy_bid: bool) -> None:
        self.busy_bid, self.name, self.machine_rule = busy_bid, "SIM_AUCTION", "FIFO"

    def reset(self, info: dict) -> None:
        self.acts = info["actions"]

    def act(self, obs: dict, masks: dict, global_state: dict | None = None) -> dict[str, int]:
        from som.cell_env.estimates import bid

        sim = global_state["sim"]  # type: ignore[index]
        out = {}
        for rid in masks:
            r = sim.robots[rid]
            others = [
                o
                for o in sim.robots.values()
                if o.id != rid and o.state not in (RS.DOWN, RS.ABSENT)
                and (self.busy_bid or (o.task is None and o.box is None))
            ]  # fmt: skip
            won = []
            for k, b in enumerate(sim.tasks(rid)):
                mine = bid(sim, r, b)
                if mine <= min((bid(sim, o, b) for o in others), default=float("inf")):
                    won.append((mine, k))
            out[rid] = min(won)[1] if won else int(self.acts["wait"])
        return out


@pytest.mark.parametrize("busy_bid", [False, True])
@pytest.mark.parametrize("scenario", ["cell_10", "cell_random_10_disrupt"])
def test_auction_matches_a_simulator_reference(scenario: str, busy_bid: bool) -> None:
    """The observation-based auction makes exactly the reference's decisions. The time scale
    used by the features is fine enough that no two bids collapse onto the same value."""
    env = CellEnv(f"configs/cell/{scenario}.yaml")
    name = "AUCTION_v0" if busy_bid else "AUCTION"
    run_episode(env, make(name), 20003)
    ours = list(env.sim.log.decisions)
    run_episode(env, SimAuction(busy_bid), 20003)
    assert ours == list(env.sim.log.decisions)
