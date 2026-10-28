from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from som.config.loader import SCHEMA_BY_DIR, load_config, load_yaml
from som.config.schema import (
    DeployCfg,
    EnvCfg,
    LayoutCfg,
    RewardCfg,
    ScenarioCfg,
    StationCfg,
    TrainCfg,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
ALL_CONFIGS = sorted(CONFIGS.rglob("*.yaml"))


def test_configs_exist() -> None:
    names = {p.relative_to(CONFIGS).as_posix() for p in ALL_CONFIGS}
    assert {"layouts/arena_13x6ft.yaml", "scenarios/abstract_10.yaml"} <= names


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.relative_to(CONFIGS).as_posix())
def test_every_config_loads_through_its_schema(path: Path) -> None:
    assert path.parent.name in SCHEMA_BY_DIR, f"no schema for configs/{path.parent.name}/"
    cfg = load_config(path)
    assert isinstance(cfg, SCHEMA_BY_DIR[path.parent.name])


def test_arena_layout_contents() -> None:
    layout = load_yaml(CONFIGS / "layouts" / "arena_13x6ft.yaml", LayoutCfg)
    assert set(layout.stations) == {"SRC", "TURNING", "GRINDING", "MILLING", "DRILLING", "SINK"}
    assert set(layout.robots) == {"tb1", "tb2", "tb3", "tb4"}
    assert all(s.input_capacity == 5 for s in layout.stations.values() if s.kind == "machine")


def test_abstract_10_contents() -> None:
    sc = load_yaml(CONFIGS / "scenarios" / "abstract_10.yaml", ScenarioCfg)
    assert [o.id for o in sc.orders] == list(range(1, 11))
    assert [o.due for o in sc.orders] == [146, 166, 196, 236, 266, 266, 296, 326, 346, 376]
    assert sc.horizon == 1600


def test_placeholders_are_marked() -> None:
    for rel in ("layouts/arena_13x6ft.yaml", "scenarios/abstract_10.yaml"):
        assert "# PLACEHOLDER" in (CONFIGS / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "model", [LayoutCfg, ScenarioCfg, EnvCfg, RewardCfg, TrainCfg, DeployCfg, StationCfg]
)
def test_unknown_keys_rejected(model: type) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate({"definitely_not_a_field": 1})


def test_machine_station_requires_process() -> None:
    with pytest.raises(ValidationError, match="need a process"):
        StationCfg(kind="machine", dock=(0, 0, 0), queue=(0, 0, 0))


def test_machine_capacity_defaults_to_five() -> None:
    st = StationCfg.model_validate(
        {"kind": "machine", "process": "Turning", "dock": [0, 0, 0], "queue": [0, 0, 0]}
    )
    assert st.input_capacity == 5


def test_scenario_rejects_unknown_job() -> None:
    with pytest.raises(ValidationError, match="unknown job type"):
        ScenarioCfg.model_validate(
            {
                "name": "x",
                "kind": "fixed",
                "horizon": 10,
                "default_process_times": {"Turning": 1},
                "job_types": {"P1": {"part": "Shaft", "route": ["Turning"]}},
                "orders": [{"id": 1, "job": "P9", "release": 0, "due": 5}],
            }
        )


def test_env_robot_bounds() -> None:
    with pytest.raises(ValidationError, match="max_robots"):
        EnvCfg(n_robots=5, max_robots=4)
