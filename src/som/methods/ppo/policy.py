"""PPOPolicy: the trained actor behind the CellPolicy interface (spec §11.1).

Decentralized: each robot acts from its own observation and mask (the harness passes
``global_state=None``). Evaluation is greedy (argmax over the masked logits). The checkpoint
holds the actor, its frozen observation normalization and a ``meta`` dict; critics and
optimizer state are never needed to run the policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from som.cell_env.obs import CELL_OBS_VERSION, ObsLayout
from som.methods.ppo.model import CandidateActor

ACTOR_FILE = "actor.pt"


class PPOPolicy:
    decentralized = True

    def __init__(
        self, actor: CandidateActor, meta: dict[str, Any], name: str | None = None
    ) -> None:
        self.actor = actor.eval()
        self.meta = meta
        self.name = name or meta.get("name", "PPO")
        self.machine_rule = meta.get("machine_rule", "SPT")

    @classmethod
    def load(cls, path: str | Path, name: str | None = None) -> PPOPolicy:
        p = Path(path)
        if p.is_dir():
            p = p / ACTOR_FILE
        ck = torch.load(p, map_location="cpu", weights_only=True)
        meta = ck["meta"]
        if meta["obs_version"] != CELL_OBS_VERSION:
            raise ValueError(
                f"{p}: trained on cell obs v{meta['obs_version']}, code is v{CELL_OBS_VERSION}"
            )
        if name is not None and meta.get("method") not in (None, name):
            raise ValueError(f"{p} is a {meta.get('method')} checkpoint, not {name}")
        lay = ObsLayout(**meta["layout"])
        actor = CandidateActor(lay, meta["hidden"])
        actor.load_state_dict(ck["actor"])
        return cls(actor, meta, name)

    def reset(self, scenario_info: dict[str, Any]) -> None:
        lay: ObsLayout = scenario_info["obs_layout"]
        if lay != self.actor.lay:
            raise ValueError(f"observation layout {lay} differs from the trained {self.actor.lay}")

    @torch.no_grad()
    def act(
        self,
        obs: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        global_state: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        names = list(obs)
        o = torch.as_tensor(np.stack([obs[a] for a in names]))
        m = torch.as_tensor(np.stack([masks[a] for a in names]))
        a = self.actor.logits(o, m).argmax(-1).tolist()
        return dict(zip(names, (int(x) for x in a), strict=True))


def save_actor(path: Path, actor: CandidateActor, meta: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": actor.state_dict(), "meta": meta}, path)
    return path
