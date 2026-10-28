from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


@pytest.fixture
def configs_dir() -> Path:
    return CONFIGS


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
