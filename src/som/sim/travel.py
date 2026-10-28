"""TravelModel protocol and the StraightLineTravel estimate with optional noise.

Travel time covers driving only: ``rotate(dtheta)/w_eff + dist/v_eff``. Time spent at the dock is
modeled by the core (HoldDock with the layout's ``dock_overhead_s``), so it is not counted here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from som.config.schema import TravelCalibrationCfg
from som.core.layout import Pose2D


class TravelModel(Protocol):
    def estimate(self, start: Pose2D, goal: Pose2D) -> float:
        """Deterministic expected travel time in seconds."""
        ...

    def sample(self, start: Pose2D, goal: Pose2D, rng: np.random.Generator) -> float:
        """One noisy realization of the travel time."""
        ...


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True, slots=True)
class StraightLineTravel:
    v_eff: float  # m/s
    w_eff: float  # rad/s
    noise_sigma: float = 0.0  # lognormal sigma; the multiplier has mean 1

    @classmethod
    def from_cfg(cls, cal: TravelCalibrationCfg, noise_sigma: float = 0.0) -> StraightLineTravel:
        return cls(cal.v_eff, cal.w_eff, noise_sigma)

    def estimate(self, start: Pose2D, goal: Pose2D) -> float:
        dist = start.distance_to(goal)
        if dist < 1e-9:
            return 0.0
        bearing = math.atan2(goal.y - start.y, goal.x - start.x)
        return abs(wrap_angle(bearing - start.theta)) / self.w_eff + dist / self.v_eff

    def sample(self, start: Pose2D, goal: Pose2D, rng: np.random.Generator) -> float:
        t = self.estimate(start, goal)
        if t == 0.0 or self.noise_sigma == 0.0:
            return t
        s = self.noise_sigma
        return t * float(np.exp(rng.normal(-0.5 * s * s, s)))
