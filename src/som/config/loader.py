"""Load YAML files into schema models and dump resolved configs to run directories."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from som.config.schema import DeployCfg, LayoutCfg, ScenarioCfg, TrainCfg

M = TypeVar("M", bound=BaseModel)

# configs/<dir>/*.yaml -> schema, so every config file can be validated without guessing.
SCHEMA_BY_DIR: dict[str, type[BaseModel]] = {
    "layouts": LayoutCfg,
    "scenarios": ScenarioCfg,
    "train": TrainCfg,
    "deploy": DeployCfg,
}


def load_yaml(path: str | Path, model: type[M]) -> M:
    """Parse a YAML file and validate it against ``model``."""
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return model.model_validate(data)


def load_config(path: str | Path) -> BaseModel:
    """Load a file under ``configs/`` using the schema implied by its directory."""
    p = Path(path)
    schema = SCHEMA_BY_DIR.get(p.parent.name)
    if schema is None:
        raise ValueError(f"{path}: no schema registered for directory {p.parent.name!r}")
    return load_yaml(p, schema)


def dump_resolved(cfg: BaseModel, path: str | Path) -> None:
    """Write the fully resolved config (defaults filled in) next to a run's outputs."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False)
