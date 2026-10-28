"""Decision requests, candidates, and the WAIT sentinel offered to agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from som.core.ids import AgentId, OrderId, Role, StationId


@dataclass(frozen=True, slots=True)
class Candidate:
    """One feasible choice. Robots: transport ``order`` from ``pickup`` to ``dest``.
    Machines: process ``order`` (``pickup == dest ==`` the machine's station)."""

    order: OrderId
    pickup: StationId
    dest: StationId
    available_since: float


class _Wait:
    __slots__ = ()

    def __repr__(self) -> str:
        return "WAIT"


WAIT: Final = _Wait()
Wait = _Wait
Choice = Candidate | _Wait


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    agent: AgentId
    role: Role
    candidates: tuple[Candidate, ...]  # canonical order: available_since, then order number
    wait_allowed: bool


class InfeasibleDecision(ValueError):
    pass
