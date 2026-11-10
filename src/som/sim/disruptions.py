"""Disruptions: draw machine breakdowns, robot outages and blocked or delayed trips.

Scheduled disturbances (breakdowns, outages) are drawn once per episode as plain events. Trip
outcomes are drawn per MoveRobot as the runtime executes it. Everything comes from one rng, so a
seed reproduces the same disruptions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from som.config.schema import DisruptionCfg
from som.core.events import Event, MachineDown, MachineUp, RobotDown, RobotUp
from som.core.ids import MachineId, RobotId


@dataclass(frozen=True, slots=True)
class TripOutcome:
    blocked: bool  # True: the goal aborts after ``duration``; False: arrival after ``duration``
    duration: float


class Disruptions:
    def __init__(self, cfg: DisruptionCfg, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

    def scheduled(
        self, robots: Sequence[RobotId], machines: Sequence[MachineId], horizon: float
    ) -> list[tuple[float, Event]]:
        cfg, rng = self.cfg, self.rng
        out: list[tuple[float, Event]] = []
        if cfg.machine_mtbf_s is not None:
            for m in machines:
                t = float(rng.exponential(cfg.machine_mtbf_s))
                while t < horizon:
                    up = t + float(rng.uniform(*cfg.machine_repair_s))
                    out += [(t, MachineDown(m)), (up, MachineUp(m))]
                    t = up + float(rng.exponential(cfg.machine_mtbf_s))
        n = min(cfg.robot_outages, max(len(robots) - 1, 0))
        for i in rng.choice(len(robots), size=n, replace=False) if n else []:
            t = float(rng.uniform(*cfg.robot_outage_start_s))
            out.append((t, RobotDown(robots[int(i)], "outage")))
            if rng.random() >= cfg.robot_outage_permanent_prob:
                out.append((t + float(rng.uniform(*cfg.robot_outage_s)), RobotUp(robots[int(i)])))
        return sorted(out, key=lambda e: e[0])

    def trip(self, travel_s: float) -> TripOutcome:
        cfg, rng = self.cfg, self.rng
        if travel_s > 0 and rng.random() < cfg.trip_block_prob:
            return TripOutcome(True, cfg.trip_block_timeout_s)
        if travel_s > 0 and rng.random() < cfg.trip_delay_prob:
            return TripOutcome(False, travel_s + float(rng.uniform(*cfg.trip_delay_s)))
        return TripOutcome(False, travel_s)
