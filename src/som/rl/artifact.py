"""Export and load the policy artifact (policy.pt + policy_meta.json).

The artifact holds actors only, never critics or optimizer state, so deployment does not need the
trainer. ``policy_meta.json`` carries the contract keys from ARCHITECTURE.md (obs schema version,
K per role, net, layout, training scenario, git sha, creation time) plus what is needed to rebuild
the observation context: per-role obs dims, the env config and the scenario horizon.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from som.env.observations import OBS_SCHEMA_VERSION
from som.rl.networks import RoleActor

POLICY_FILE = "policy.pt"
META_FILE = "policy_meta.json"


class ArtifactMismatch(ValueError):
    pass


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def export_artifact(path: Path, actors: dict[str, RoleActor], meta: dict[str, Any]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({role: a.state_dict() for role, a in actors.items()}, path / POLICY_FILE)
    full = {
        "obs_schema_version": OBS_SCHEMA_VERSION,
        "git_sha": git_sha(),
        "created": datetime.now().isoformat(timespec="seconds"),
        **meta,
    }
    (path / META_FILE).write_text(json.dumps(full, indent=2), encoding="utf-8")
    return path


def load_artifact(path: Path) -> tuple[dict[str, RoleActor], dict[str, Any]]:
    meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
    if meta["obs_schema_version"] != OBS_SCHEMA_VERSION:
        raise ArtifactMismatch(
            f"{path}: obs schema v{meta['obs_schema_version']}, code is v{OBS_SCHEMA_VERSION}"
        )
    states = torch.load(path / POLICY_FILE, map_location="cpu", weights_only=True)
    hidden = meta["net"]["hidden"]
    k = {"robot": meta["K_robot"], "machine": meta["K_machine"]}
    actors = {}
    for role, sd in states.items():
        actor = RoleActor(meta["net"]["obs_dim"][role], k[role] + 1, hidden)
        actor.load_state_dict(sd)
        actor.eval()
        actors[role] = actor
    return actors, meta
