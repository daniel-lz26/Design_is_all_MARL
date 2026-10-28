"""SimRuntime: event heap, virtual clock, advance to the next decision point."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

import numpy as np

from som.core.commands import CancelRobot, Command, HoldDock, MoveRobot, StartProcess
from som.core.decisions import Choice, DecisionRequest
from som.core.events import DockReleased, Event, OrderReleased, ProcessDone, RobotArrived
from som.core.factory import FactoryCore
from som.core.history import EpisodeLog
from som.sim.travel import TravelModel

if TYPE_CHECKING:
    from som.policies.base import Policy


class SimRuntime:
    def __init__(
        self,
        core: FactoryCore,
        travel: TravelModel,
        rng: np.random.Generator,
        horizon: float,
    ) -> None:
        self.core = core
        self.travel = travel
        self.rng = rng
        self.horizon = horizon
        self.now = 0.0
        self.truncated = False
        self._heap: list[tuple[float, int, Event]] = []
        self._seq = 0
        for o in core.orders.values():
            self.schedule(o.spec.release_time, OrderReleased(o.id))

    def schedule(self, t: float, event: Event) -> None:
        """Queue an event. Also used by tests and disruption scenarios to inject events."""
        if t < self.now:
            raise ValueError(f"cannot schedule in the past: {t} < {self.now}")
        heapq.heappush(self._heap, (t, self._seq, event))
        self._seq += 1

    def execute(self, cmds: list[Command]) -> None:
        for c in cmds:
            if isinstance(c, MoveRobot):
                start = self.core.robots[c.robot].pose
                goal = self.core.layout.station(c.station).dock
                dur = self.travel.sample(start, goal, self.rng)
                self.schedule(self.now + dur, RobotArrived(c.robot, c.station, c.goal_id))
            elif isinstance(c, HoldDock):
                self.schedule(self.now + c.duration, DockReleased(c.robot, c.station, c.goal_id))
            elif isinstance(c, StartProcess):
                self.schedule(self.now + c.duration, ProcessDone(c.machine, c.order, c.token))
            elif isinstance(c, CancelRobot):
                pass  # sim robots stop instantly; the core already dropped the goal

    @property
    def done(self) -> bool:
        return self.core.done or self.truncated

    def advance_to_next_decision(self) -> list[DecisionRequest]:
        """Pop events until some agent must decide. Returns [] when the episode is over."""
        while True:
            if self.core.done:
                return []
            reqs = self.core.pending_decisions(self.now)
            if reqs:
                return reqs
            if not self._heap or self._heap[0][0] > self.horizon:
                # Nothing more can happen before the horizon: the episode is truncated.
                self.now = self.horizon
                self.truncated = True
                return []
            t, _, event = heapq.heappop(self._heap)
            self.now = t
            self.execute(self.core.apply(event, t))

    def assign(self, agent: str, choice: Choice) -> None:
        self.execute(self.core.assign(agent, choice, self.now))

    def log(self) -> EpisodeLog:
        log = self.core.history()
        if self.truncated:
            log.end_time = self.horizon
            log.truncated = True
        return log


def run_episode(runtime: SimRuntime, policy: Policy) -> EpisodeLog:
    """Drive one episode with a decision-level policy.

    Agents deciding at the same instant are resolved one at a time in the core's order, and each
    agent sees a freshly computed request, so two robots never pick the same order.
    """
    core = runtime.core
    while True:
        reqs = runtime.advance_to_next_decision()
        if not reqs:
            break
        for agent in [r.agent for r in reqs]:
            req = next((r for r in core.pending_decisions(runtime.now) if r.agent == agent), None)
            if req is not None:
                runtime.assign(agent, policy.decide(req, core.snapshot()))
    return runtime.log()
