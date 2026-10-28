from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from som.config.loader import load_yaml
from som.config.schema import LayoutCfg, ScenarioCfg
from som.core.layout import Layout

from helpers import TINY_LAYOUT

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


@pytest.fixture
def configs_dir() -> Path:
    return CONFIGS


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def tiny_layout_cfg() -> LayoutCfg:
    return LayoutCfg.model_validate(TINY_LAYOUT)


@pytest.fixture
def tiny_layout(tiny_layout_cfg: LayoutCfg) -> Layout:
    return Layout.from_cfg(tiny_layout_cfg)


@pytest.fixture
def abstract_10() -> ScenarioCfg:
    return load_yaml(CONFIGS / "scenarios" / "abstract_10.yaml", ScenarioCfg)


@pytest.fixture
def random_10() -> ScenarioCfg:
    return load_yaml(CONFIGS / "scenarios" / "random_10.yaml", ScenarioCfg)


@pytest.fixture
def arena_cfg() -> LayoutCfg:
    return load_yaml(CONFIGS / "layouts" / "arena_13x6ft.yaml", LayoutCfg)
