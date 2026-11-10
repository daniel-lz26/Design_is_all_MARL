"""Family A baselines (spec §10): robot dispatching rule x machine sequencing rule.

A robot rule scores the open tasks a free robot sees on the board and claims the best one;
ties keep the board order (due time, then order number). With no task it waits where it is
(``idle="wait"``, the default: it undocks and stays near the station) or goes to its parking
spot (``idle="park"``). The machine rule is the cell's ``machine_seq_rule`` (FIFO, EDD or SPT).
"""

from __future__ import annotations

import numpy as np

from som.cell_env.des import CellSim
from som.cell_env.entities import Box, Station
from som.cell_env.estimates import remaining_proc, slack
from som.cell_env.obs import S, T

ROBOT_RULES = ("NEAREST", "FIFO", "EDD", "SPT", "MST", "BOTTLENECK", "RANDOM")
MACHINE_RULES = ("FIFO", "EDD", "SPT")
BOTTLENECK_PROCESS = "Milling"
BOTTLENECK_STATION = "MILLING"


def starving(st: Station) -> bool:
    """The machine has nothing queued or coming and will be free soon."""
    return not st.in_buf and st.inbound == 0 and st.blocked_box is None


class CellRulePolicy:
    """``CellRulePolicy("NEAREST", "FIFO")`` acts for one robot at a time via ``choose``."""

    def __init__(
        self, robot_rule: str, machine_rule: str = "FIFO", seed: int = 0, idle: str = "wait"
    ) -> None:
        if robot_rule not in ROBOT_RULES:
            raise ValueError(f"unknown robot rule {robot_rule!r}; choose from {ROBOT_RULES}")
        if machine_rule not in MACHINE_RULES:
            raise ValueError(f"unknown machine rule {machine_rule!r}; choose from {MACHINE_RULES}")
        self.robot_rule = robot_rule
        self.machine_rule = machine_rule
        self.name = f"{robot_rule}/{machine_rule}"
        if idle not in ("wait", "park"):
            raise ValueError("idle must be 'wait' or 'park'")
        self.idle = idle
        self.rng = np.random.default_rng(seed)

    def reset(self, sim: CellSim, seed: int) -> None:
        sim.machine_rule = self.machine_rule
        self.rng = np.random.default_rng(seed)

    def choose(self, sim: CellSim, rid: str) -> int:
        tasks = sim.tasks(rid)
        mask = sim.mask(rid)
        if not tasks:
            return sim.k if self.idle == "park" and mask[sim.k] else sim.k + 1
        here = sim.location(rid)
        rule = self.robot_rule
        if rule == "RANDOM":
            return int(self.rng.integers(len(tasks)))
        if rule == "BOTTLENECK":
            mill = sim.stations[sim.machine_of[BOTTLENECK_PROCESS]]
            feeds = [i for i, b in enumerate(tasks) if sim.next_station(b) is mill]
            if feeds and starving(mill):
                return min(feeds, key=lambda i: self._travel(sim, here, tasks[i]))
            rule = "NEAREST"
        scores = [self._score(rule, sim, here, b) for b in tasks]
        return int(np.argmin(scores))  # first minimum keeps board order on ties

    def _travel(self, sim: CellSim, here: str, b: Box) -> float:
        return sim.roadmap.time(here, sim.stations[b.where].approach)

    def _score(self, rule: str, sim: CellSim, here: str, b: Box) -> float:
        if rule == "NEAREST":
            return self._travel(sim, here, b)
        if rule == "FIFO":
            return b.t_avail
        if rule == "EDD":
            return b.spec.due
        if rule == "SPT":
            return remaining_proc(b)
        return slack(sim, b, here)  # MST


def all_rule_pairs() -> list[CellRulePolicy]:
    return [CellRulePolicy(r, m) for r in ROBOT_RULES for m in MACHINE_RULES]


def run_rule_episode(sim: CellSim, policy: CellRulePolicy, seed: int) -> None:
    """Drive one episode with a rule policy; robots deciding together go in id order."""
    policy.reset(sim, seed)
    pending = sim.reset(seed)
    while pending:
        for rid in pending:
            if sim.mask(rid).any() and rid in sim.pending():
                sim.act(rid, policy.choose(sim, rid))
        pending = sim.advance()


class ObsRulePolicy:
    """The same rules, decided from each robot's observation alone (C1: no global state).

    Gives exactly the decisions of ``CellRulePolicy`` (a test checks this), so the
    simulator-side version can serve as a fast reference for calibration.
    """

    decentralized = True

    def __init__(self, robot_rule: str, machine_rule: str = "FIFO", idle: str = "wait") -> None:
        if robot_rule not in ROBOT_RULES:
            raise ValueError(f"unknown robot rule {robot_rule!r}; choose from {ROBOT_RULES}")
        self.robot_rule = robot_rule
        self.machine_rule = machine_rule
        self.idle = idle
        self.name = f"{robot_rule}/{machine_rule}"
        self.rng = np.random.default_rng(0)

    def reset(self, scenario_info: dict) -> None:
        self.layout = scenario_info["obs_layout"]
        self.acts = scenario_info["actions"]
        self.rng = np.random.default_rng(scenario_info["seed"])
        self.mill = scenario_info["stations"].index(BOTTLENECK_STATION)

    def act(self, obs: dict, masks: dict, global_state: dict | None = None) -> dict[str, int]:
        return {a: self._one(obs[a], masks[a]) for a in obs}

    def _one(self, o: np.ndarray, m: np.ndarray) -> int:
        if m[self.acts["noop"]]:
            return int(self.acts["noop"])
        tasks = self.layout.tasks(o)
        valid = int(tasks[:, T["valid"]].sum())
        if valid == 0:
            park = self.acts["park"]
            return int(park if self.idle == "park" and m[park] else self.acts["wait"])
        rows = tasks[:valid]
        rule = self.robot_rule
        if rule == "RANDOM":
            return int(self.rng.integers(valid))
        if rule == "BOTTLENECK":
            st = self.layout.stations(o)[self.mill]
            starving = st[S["in_occupancy"]] == 0 and st[S["inbound"]] == 0
            starving = starving and st[S["machine_blocked"]] == 0
            feeds = np.flatnonzero(rows[:, T["dest_is_bottleneck"]] > 0)
            if len(feeds) and starving:
                return int(feeds[np.argmin(rows[feeds, T["travel_to_pickup"]])])
            rule = "NEAREST"
        col = {
            "NEAREST": rows[:, T["travel_to_pickup"]],
            "FIFO": -rows[:, T["waiting_time"]],
            "EDD": rows[:, T["due_in"]],
            "SPT": rows[:, T["remaining_proc"]],
            "MST": rows[:, T["slack"]],
        }[rule]
        return int(np.argmin(col))
