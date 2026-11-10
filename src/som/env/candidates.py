"""Place candidates into K fixed action slots per agent, in canonical order.

Slot 0 is WAIT (or NOOP for an agent with no decision); slots 1..K hold the first K candidates of
the request. The core already sorts candidates by (available_since, order number), so slot order
carries no heuristic preference. Extra candidates beyond K are dropped and counted.
"""

from __future__ import annotations

import numpy as np

from som.core.decisions import Candidate, DecisionRequest


def slot_candidates(request: DecisionRequest | None, k: int) -> tuple[list[Candidate], int]:
    """The candidates shown in slots 1..k, and how many were truncated."""
    if request is None:
        return [], 0
    cands = list(request.candidates)
    return cands[:k], max(0, len(cands) - k)


def action_mask(
    request: DecisionRequest | None, k: int, wait_allowed: bool = True
) -> np.ndarray:
    """Boolean mask of length k + 1. An agent without a request may only NOOP (slot 0)."""
    mask = np.zeros(k + 1, dtype=np.int8)
    if request is None:
        mask[0] = 1
        return mask
    shown, _ = slot_candidates(request, k)
    mask[1 : len(shown) + 1] = 1
    mask[0] = int(request.wait_allowed and wait_allowed)
    return mask
