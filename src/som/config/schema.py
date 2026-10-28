"""Pydantic models for every YAML config: layout, scenario, env, reward, train, deploy.

All models reject unknown keys (``extra="forbid"``) so typos and stale flags fail loudly.
Units are meters, radians and seconds throughout.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PoseTuple = tuple[float, float, float]  # (x, y, theta) in the arena frame
DEFAULT_INPUT_CAPACITY = 5


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- layout


class ArenaCfg(StrictModel):
    frame: str = "arena"
    size_m: tuple[float, float]


class StationCfg(StrictModel):
    kind: Literal["source", "machine", "sink"]
    process: str | None = None
    dock: PoseTuple
    queue: PoseTuple
    input_capacity: int | None = Field(default=None, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _default_capacity(cls, data: object) -> object:
        # Legacy eligible_buffer_capacity was 5; machines get it unless the layout says otherwise.
        if isinstance(data, dict) and data.get("kind") == "machine":
            data = {"input_capacity": DEFAULT_INPUT_CAPACITY, **data}
        return data

    @model_validator(mode="after")
    def _machine_fields(self) -> StationCfg:
        if self.kind == "machine":
            if self.process is None:
                raise ValueError("machine stations need a process")
        elif self.process is not None or self.input_capacity is not None:
            raise ValueError(f"{self.kind} stations take no process or input_capacity")
        return self


class RobotStartCfg(StrictModel):
    start: PoseTuple


class TravelCalibrationCfg(StrictModel):
    v_eff: float = Field(gt=0)  # m/s
    w_eff: float = Field(gt=0)  # rad/s
    dock_overhead_s: float = Field(ge=0)


class LayoutCfg(StrictModel):
    name: str
    arena: ArenaCfg
    stations: dict[str, StationCfg]
    robots: dict[str, RobotStartCfg]
    travel_calibration: TravelCalibrationCfg

    @model_validator(mode="after")
    def _one_source_one_sink(self) -> LayoutCfg:
        kinds = [s.kind for s in self.stations.values()]
        if kinds.count("source") != 1 or kinds.count("sink") != 1:
            raise ValueError("layout needs exactly one source and one sink station")
        processes = [s.process for s in self.stations.values() if s.kind == "machine"]
        if len(processes) != len(set(processes)):
            raise ValueError("each process may appear on only one machine station")
        return self


# --------------------------------------------------------------------------- scenario


class JobTypeCfg(StrictModel):
    part: str
    route: list[str] = Field(min_length=1)


class OrderCfg(StrictModel):
    id: int = Field(ge=0)
    job: str
    release: float = Field(ge=0)
    due: float
    process_times: list[float] | None = None  # defaults to default_process_times per step


class GeneratorCfg(StrictModel):
    """Random order streams for training (random_10, stress_20)."""

    n_orders: int = Field(ge=1)
    interarrival_s: tuple[float, float]  # uniform range between consecutive releases
    due_slack_s: tuple[float, float]  # uniform range for due - release
    job_mix: dict[str, float]  # relative weights per job type
    process_time_jitter: float = Field(default=0.0, ge=0, lt=1)  # +/- fraction, uniform


class ScenarioCfg(StrictModel):
    name: str
    kind: Literal["fixed", "generated"]
    horizon: float = Field(gt=0)
    default_process_times: dict[str, float]
    job_types: dict[str, JobTypeCfg]
    orders: list[OrderCfg] = Field(default_factory=list)
    generator: GeneratorCfg | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ScenarioCfg:
        if self.kind == "fixed" and (not self.orders or self.generator is not None):
            raise ValueError("fixed scenarios need orders and no generator")
        if self.kind == "generated" and (self.generator is None or self.orders):
            raise ValueError("generated scenarios need a generator and no orders")
        for jt in self.job_types.values():
            missing = set(jt.route) - set(self.default_process_times)
            if missing:
                raise ValueError(f"no default process time for {sorted(missing)}")
        ids = [o.id for o in self.orders]
        if len(ids) != len(set(ids)):
            raise ValueError("order ids must be unique")
        for o in self.orders:
            if o.job not in self.job_types:
                raise ValueError(f"order {o.id}: unknown job type {o.job!r}")
            route = self.job_types[o.job].route
            if o.process_times is not None and len(o.process_times) != len(route):
                raise ValueError(f"order {o.id}: process_times must match route length")
        if self.generator is not None:
            unknown = set(self.generator.job_mix) - set(self.job_types)
            if unknown:
                raise ValueError(f"job_mix references unknown job types {sorted(unknown)}")
        return self


# --------------------------------------------------------------------------- env / reward


class EnvCfg(StrictModel):
    n_robots: int = Field(default=2, ge=1)
    max_robots: int = Field(default=4, ge=1)
    max_orders: int = Field(default=24, ge=1)
    k_robot: int = Field(default=6, ge=1)
    k_machine: int = Field(default=5, ge=1)
    machine_wait_allowed: bool = True
    travel_noise_sigma: float = Field(default=0.1, ge=0)  # lognormal sigma on travel times

    @model_validator(mode="after")
    def _robot_bounds(self) -> EnvCfg:
        if self.n_robots > self.max_robots:
            raise ValueError("n_robots exceeds max_robots")
        return self


class RewardCfg(StrictModel):
    mode: Literal["team", "local"] = "team"
    w_deliver: float = 1.0
    w_tardy: float = 0.01
    w_ontime: float = 1.0
    w_time: float = 0.0


# --------------------------------------------------------------------------- train / deploy


class TrainCfg(StrictModel):
    name: str
    layout: str  # path to a layout YAML
    scenario: str  # training scenario YAML
    eval_scenarios: list[str] = Field(default_factory=list)
    env: EnvCfg = Field(default_factory=EnvCfg)
    reward: RewardCfg = Field(default_factory=RewardCfg)
    critic_mode: Literal["central", "local"] = "central"
    total_steps: int = Field(ge=1)
    n_envs: int = Field(default=8, ge=1)
    rollout_steps: int = Field(default=256, ge=1)
    update_epochs: int = Field(default=4, ge=1)
    minibatches: int = Field(default=4, ge=1)
    lr: float = Field(default=3e-4, gt=0)
    gamma: float = Field(default=0.995, gt=0, le=1)  # per second; discount is gamma**dt
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    clip: float = Field(default=0.2, gt=0)
    entropy_coef: float = Field(default=0.01, ge=0)
    value_coef: float = Field(default=0.5, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    hidden: list[int] = Field(default_factory=lambda: [128, 128])
    seed: int = 0
    eval_every: int = Field(default=10, ge=1)  # updates
    eval_seeds: list[int] = Field(default_factory=lambda: list(range(1000, 1010)))


class DeployCfg(StrictModel):
    name: str
    layout: str
    scenario: str
    backend: Literal["kinematic", "ros2"]
    pose_source: Literal["odom", "arena"] = "odom"
    robots: list[str] = Field(min_length=1)  # layout robot keys / ROS namespaces, e.g. tb1, tb2
    policy: str  # artifact directory, or a heuristic name such as "MIN_SLACK/EDD"
    time_scale: float = Field(default=1.0, gt=0)  # applies to machine timers and due dates only
    loop_hz: float = Field(default=10.0, gt=0)
    env: EnvCfg = Field(default_factory=EnvCfg)
