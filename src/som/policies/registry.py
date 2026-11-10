"""Name -> factory for every cell controller (spec §11.1, §15).

Method branches add their policy with a single ``register(...)`` line in the block at the end,
kept in alphabetical order so merges conflict trivially at most.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from som.policies.cell_base import CellPolicy

Factory = Callable[[str | None], CellPolicy]
REGISTRY: dict[str, Factory] = {}


def register(name: str, factory: Factory) -> None:
    if name in REGISTRY:
        raise ValueError(f"policy {name!r} is already registered")
    REGISTRY[name] = factory


def make(name: str, checkpoint: str | None = None) -> CellPolicy:
    if name not in REGISTRY:
        raise KeyError(f"unknown policy {name!r}; known: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name](checkpoint)


def family(name: str) -> str:
    """A = dispatching rules, B = auction, C = central scheduler, L = learned methods."""
    if name.startswith("AUCTION"):
        return "B"
    if name.startswith("CENTRAL"):
        return "C"
    robot = name.split("/")[0]
    from som.baselines.cell_rules import ROBOT_RULES

    return "A" if robot in ROBOT_RULES else "L"


def _baselines() -> None:
    from som.baselines.auction import AuctionPolicy
    from som.baselines.cell_rules import MACHINE_RULES, ROBOT_RULES, ObsRulePolicy

    def rule(r: str, m: str) -> Factory:
        return lambda ckpt: ObsRulePolicy(r, m)

    for r in ROBOT_RULES:
        for m in MACHINE_RULES:
            register(f"{r}/{m}", rule(r, m))

    def auction(m: str, busy_bid: bool = False) -> Factory:
        return lambda ckpt: AuctionPolicy(m, busy_bid)

    register("AUCTION", auction("FIFO"))
    register("AUCTION/EDD", auction("EDD"))
    register("AUCTION/SPT", auction("SPT"))
    register("AUCTION_v0", auction("FIFO", busy_bid=True))  # env-v1 auction: busy robots bid
    register("AUCTION_v0/EDD", auction("EDD", busy_bid=True))
    register("AUCTION_v0/SPT", auction("SPT", busy_bid=True))


def _central() -> None:
    try:
        from som.baselines.central_cpsat import CentralPolicy
    except ImportError:  # ortools missing: the central baselines are simply unavailable
        return

    def central(replan: bool) -> Factory:
        return lambda ckpt: CentralPolicy(replan=replan)

    register("CENTRAL_REPLAN", central(True))
    register("CENTRAL_STATIC", central(False))


_baselines()
_central()

# --- learned methods: one line per method branch, alphabetical ---------------------------
_LEARNED: dict[str, Any] = {}

__import__("som.methods.ppo", fromlist=["x"]).register_all()  # method/ppo: IPPO, MAPPO
