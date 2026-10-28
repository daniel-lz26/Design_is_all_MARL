"""Dispatching rules: robot FIFO/EDD/NEAREST/MIN_SLACK, machine FIFO/EDD/SPT.

A heuristic never chooses WAIT while it has a candidate. Ties break on (available_since, order
number), which is the canonical candidate order.
"""

from __future__ import annotations

import itertools

import numpy as np

from som.core.decisions import Candidate, Choice, DecisionRequest
from som.core.history import CoreSnapshot, OrderView
from som.core.ids import StationId
from som.core.layout import Layout, Pose2D
from som.sim.travel import TravelModel

ROBOT_RULES = ("FIFO", "EDD", "NEAREST", "MIN_SLACK")
MACHINE_RULES = ("FIFO", "EDD", "SPT")


def remaining_work(
    order: OrderView, start: Pose2D, pickup: StationId, layout: Layout, travel: TravelModel
) -> float:
    """Estimated seconds to finish ``order`` if a robot at ``start`` picks it up at ``pickup``.

    Drive to pickup, then every remaining transport leg (drive + two dock holds) and process step.
    """
    dock = layout.dock_time_s
    t = travel.estimate(start, layout.station(pickup).dock)
    stops = [layout.station(pickup)]
    stops += [layout.station_for_process(p) for p in order.route[order.step_idx :]]
    stops.append(layout.sink)
    for a, b in itertools.pairwise(stops):
        t += 2 * dock + travel.estimate(a.dock, b.dock)
    t += sum(order.process_times[order.step_idx :])
    return t


class DispatchPolicy:
    """A robot rule paired with a machine rule, e.g. ``DispatchPolicy("MIN_SLACK", "EDD", ...)``."""

    def __init__(self, robot_rule: str, machine_rule: str, layout: Layout, travel: TravelModel):
        if robot_rule not in ROBOT_RULES:
            raise ValueError(f"unknown robot rule {robot_rule!r}; choose from {ROBOT_RULES}")
        if machine_rule not in MACHINE_RULES:
            raise ValueError(f"unknown machine rule {machine_rule!r}; choose from {MACHINE_RULES}")
        self.robot_rule = robot_rule
        self.machine_rule = machine_rule
        self.name = f"{robot_rule}/{machine_rule}"
        self.layout = layout
        self.travel = travel

    def reset(self, rng: np.random.Generator) -> None:
        pass  # deterministic

    def decide(self, request: DecisionRequest, snapshot: CoreSnapshot) -> Choice:
        rule = self.robot_rule if request.role == "robot" else self.machine_rule
        score = getattr(self, f"_{request.role}_{rule.lower()}")

        def key(c: Candidate) -> tuple[float, float, int]:
            o = snapshot.order(c.order)
            return (score(c, o, snapshot, request.agent), c.available_since, o.number)

        return min(request.candidates, key=key)

    # Robot rules ---------------------------------------------------------------

    def _robot_fifo(self, c: Candidate, o: OrderView, s: CoreSnapshot, agent: str) -> float:
        return c.available_since

    def _robot_edd(self, c: Candidate, o: OrderView, s: CoreSnapshot, agent: str) -> float:
        return o.due

    def _robot_nearest(self, c: Candidate, o: OrderView, s: CoreSnapshot, agent: str) -> float:
        return self.travel.estimate(s.robot(agent).pose, self.layout.station(c.pickup).dock)

    def _robot_min_slack(self, c: Candidate, o: OrderView, s: CoreSnapshot, agent: str) -> float:
        work = remaining_work(o, s.robot(agent).pose, c.pickup, self.layout, self.travel)
        return o.due - s.now - work

    # Machine rules -------------------------------------------------------------

    def _machine_fifo(self, c: Candidate, o: OrderView, s: CoreSnapshot, agent: str) -> float:
        return c.available_since

    def _machine_edd(self, c: Candidate, o: OrderView, s: CoreSnapshot, agent: str) -> float:
        return o.due

    def _machine_spt(self, c: Candidate, o: OrderView, s: CoreSnapshot, agent: str) -> float:
        return o.process_times[o.step_idx]


def make_policy(name: str, layout: Layout, travel: TravelModel) -> DispatchPolicy:
    """Parse ``"ROBOT_RULE/MACHINE_RULE"``, e.g. ``"MIN_SLACK/EDD"``."""
    robot_rule, sep, machine_rule = name.upper().partition("/")
    if not sep:
        raise ValueError(f"expected ROBOT_RULE/MACHINE_RULE, got {name!r}")
    return DispatchPolicy(robot_rule, machine_rule, layout, travel)


def all_heuristics(layout: Layout, travel: TravelModel) -> list[DispatchPolicy]:
    return [DispatchPolicy(r, m, layout, travel) for r in ROBOT_RULES for m in MACHINE_RULES]
