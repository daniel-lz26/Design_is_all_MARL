"""LearnedPolicy: decision-level wrapper around exported MAPPO actors.

It encodes the request with the same observation encoders as the env, so a policy trained in
``SomParallelEnv`` acts identically under ``run_episode`` and the live runtime. Greedy (argmax)
by default; ``stochastic=True`` samples from the masked distribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from som.config.schema import EnvCfg, LayoutCfg
from som.core.decisions import WAIT, Choice, DecisionRequest
from som.core.history import CoreSnapshot
from som.core.layout import Layout
from som.env.candidates import action_mask, slot_candidates
from som.env.observations import ObsContext, encode
from som.policies.base import Policy
from som.rl.artifact import ArtifactMismatch, load_artifact
from som.rl.networks import RoleActor
from som.sim.travel import StraightLineTravel


class LearnedPolicy:
    def __init__(
        self,
        actors: dict[str, RoleActor],
        ctx: ObsContext,
        env_cfg: EnvCfg,
        stochastic: bool = False,
        name: str = "MAPPO",
    ) -> None:
        self.actors = actors
        self.ctx = ctx
        self.env_cfg = env_cfg
        self.k = {"robot": ctx.k_robot, "machine": ctx.k_machine}
        self.wait = {"robot": env_cfg.robot_wait_allowed, "machine": env_cfg.machine_wait_allowed}
        self.stochastic = stochastic
        self.name = name
        self._gen = torch.Generator().manual_seed(0)

    @classmethod
    def from_artifact(
        cls, path: Path, layout_cfg: LayoutCfg, stochastic: bool = False, name: str = "MAPPO"
    ) -> LearnedPolicy:
        actors, meta = load_artifact(path)
        if meta["layout"] != layout_cfg.name:
            raise ArtifactMismatch(f"trained on layout {meta['layout']!r}, got {layout_cfg.name!r}")
        env_cfg = EnvCfg.model_validate(meta["env"])
        ctx = ObsContext(
            layout=Layout.from_cfg(layout_cfg),
            travel=StraightLineTravel.from_cfg(layout_cfg.travel_calibration),
            horizon=meta["horizon"],
            k_robot=meta["K_robot"],
            k_machine=meta["K_machine"],
            max_robots=env_cfg.max_robots,
        )
        return cls(actors, ctx, env_cfg, stochastic, name)

    def reset(self, rng: np.random.Generator) -> None:
        self._gen = torch.Generator().manual_seed(int(rng.integers(2**31 - 1)))

    @torch.no_grad()
    def decide(self, request: DecisionRequest, snapshot: CoreSnapshot) -> Choice:
        role = request.role
        slots, _ = slot_candidates(request, self.k[role])
        obs = encode(self.ctx, snapshot, request.agent, role, slots)
        mask = action_mask(request, self.k[role], self.wait[role])
        logits = self.actors[role](torch.as_tensor(obs)[None], torch.as_tensor(mask)[None])[0]
        if self.stochastic:
            act = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=self._gen))
        else:
            act = int(torch.argmax(logits))
        return WAIT if act == 0 else slots[act - 1]


class RoleMixPolicy:
    """Robots follow one policy and machines another, e.g. learned robots with EDD machines."""

    def __init__(self, robot: Policy, machine: Policy, name: str) -> None:
        self.by_role = {"robot": robot, "machine": machine}
        self.name = name

    def reset(self, rng: np.random.Generator) -> None:
        for p in self.by_role.values():
            p.reset(rng)

    def decide(self, request: DecisionRequest, snapshot: CoreSnapshot) -> Choice:
        return self.by_role[request.role].decide(request, snapshot)
