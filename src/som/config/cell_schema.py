"""Pydantic models for the cell env: floor layout (roadmap, stations, arms) and scenarios.

Kept apart from ``schema.py`` so the abstract env's configs are untouched. Units are meters,
seconds and m/s. See ``Context/CELL_ENV_SPEC.md`` §3-§6 and §9.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from som.config.schema import StrictModel

XY = tuple[float, float]
Range = tuple[float, float]


# --------------------------------------------------------------------------- layout


class EdgeCfg(StrictModel):
    a: str
    b: str
    capacity: int = Field(default=2, ge=1)  # 1 = single lane
    directed: bool = False  # True: only a -> b
    length_m: float | None = Field(default=None, gt=0)  # default: straight-line distance


class CellStationCfg(StrictModel):
    kind: Literal["source", "machine", "sink"]
    process: str | None = None
    dock: str  # dock node (a leaf off the approach node)
    approach: str  # aisle node where robots queue for the dock
    in_buffer_cap: int | None = Field(default=None, ge=1)  # None = unlimited (sink)
    out_buffer_cap: int | None = Field(default=None, ge=1)  # None = unlimited (source)
    arm_load_s: float | None = Field(default=None, gt=0)  # None: the handling block's value
    arm_unload_s: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _kind_fields(self) -> CellStationCfg:
        if (self.kind == "machine") != (self.process is not None):
            raise ValueError("machine stations, and only they, need a process")
        if self.kind == "machine" and (self.in_buffer_cap is None or self.out_buffer_cap is None):
            raise ValueError("machine stations need in_buffer_cap and out_buffer_cap")
        return self


class CellRobotCfg(StrictModel):
    v_cruise: float = Field(default=0.18, gt=0)
    turn_penalty_s: float = Field(default=1.5, ge=0)  # per heading change above the threshold
    turn_threshold_deg: float = Field(default=45.0, ge=0, le=180)


class HandlingCfg(StrictModel):
    """Every arm, dock and conveyor time in one place. All are guesses until measured in
    Isaac Sim; scenarios can override the whole block for sensitivity runs."""

    arm_load_s: float = Field(default=6.0, gt=0)  # arm puts a box on a robot
    arm_unload_s: float = Field(default=6.0, gt=0)  # arm takes a box off a robot
    dock_time_s: float = Field(default=3.0, ge=0)  # align under the arm (plus spur travel)
    undock_time_s: float = Field(default=1.5, ge=0)
    conveyor_transfer_s: float = Field(default=2.0, ge=0)  # in-buffer -> machine


class CellLayoutCfg(StrictModel):
    name: str
    floor_m: XY
    nodes: dict[str, XY]
    edges: list[EdgeCfg]
    stations: dict[str, CellStationCfg]
    parking: list[str] = Field(min_length=1)
    robot: CellRobotCfg = Field(default_factory=CellRobotCfg)
    handling: HandlingCfg = Field(default_factory=HandlingCfg)

    @model_validator(mode="after")
    def _consistent(self) -> CellLayoutCfg:
        for e in self.edges:
            for n in (e.a, e.b):
                if n not in self.nodes:
                    raise ValueError(f"edge {e.a}-{e.b}: unknown node {n!r}")
        kinds = [s.kind for s in self.stations.values()]
        if kinds.count("source") != 1 or kinds.count("sink") != 1:
            raise ValueError("layout needs exactly one source and one sink")
        procs = [s.process for s in self.stations.values() if s.process]
        if len(procs) != len(set(procs)):
            raise ValueError("each process may appear on only one station")
        linked = {frozenset((e.a, e.b)) for e in self.edges}
        for key, s in self.stations.items():
            for n in (s.dock, s.approach):
                if n not in self.nodes:
                    raise ValueError(f"station {key}: unknown node {n!r}")
            if frozenset((s.dock, s.approach)) not in linked:
                raise ValueError(f"station {key}: no edge between dock and approach node")
        for p in self.parking:
            if p not in self.nodes:
                raise ValueError(f"unknown parking node {p!r}")
        return self


# --------------------------------------------------------------------------- scenario


class CellGeneratorCfg(StrictModel):
    """Random orders (spec §9): job type from the pool, times around the means. Due dates use
    the calibrated rule: due = release + k * the order's minimum flow time in an empty cell."""

    n_orders: int | None = Field(default=None, ge=1)
    release_until_s: float | None = Field(default=None, gt=0)  # streams: stop releasing here
    job_types: dict[str, list[str]]  # name -> route of processes
    job_mix: dict[str, float]
    mean_process_s: dict[str, float]
    process_jitter: float = Field(default=0.25, ge=0, lt=1)  # +/- uniform fraction
    arrivals: Literal["poisson", "even"] = "poisson"
    mean_interarrival_s: float = Field(default=25.6, gt=0)
    first_release_s: float = Field(default=10.0, ge=0)
    due_k: Range = (1.5, 3.0)  # uniform k; due = release + k * minimum flow time

    @model_validator(mode="after")
    def _known_jobs(self) -> CellGeneratorCfg:
        if self.n_orders is None and self.release_until_s is None:
            raise ValueError("give n_orders, release_until_s, or both")
        if set(self.job_mix) - set(self.job_types):
            raise ValueError("job_mix references unknown job types")
        for route in self.job_types.values():
            if set(route) - set(self.mean_process_s):
                raise ValueError(f"no mean process time for some of {route}")
        return self


class CellNoiseCfg(StrictModel):
    speed_sigma: float = Field(default=0.0, ge=0)  # lognormal, mean 1, per edge traversal
    arm_tri: float = Field(default=0.0, ge=0, lt=1)  # +/- triangular fraction on handoffs
    process_sigma: float = Field(default=0.0, ge=0)  # lognormal on machine process times


class CellDisruptionCfg(StrictModel):
    """Spec §6. Every rate is a scenario design choice, not a measurement."""

    robot_down: Literal["none", "permanent", "mtbf"] = "none"
    robot_down_at_s: Range = (80.0, 220.0)  # permanent mode: one robot, for the rest of the run
    robot_mtbf_s: float = Field(default=600.0, gt=0)
    robot_repair_s: Range = (30.0, 90.0)
    rescue: bool = False  # True: a box on a broken robot goes to the nearest station
    rescue_delay_s: float = Field(default=20.0, ge=0)
    clear_delay_s: float = Field(default=20.0, ge=0)  # a broken robot blocks its spot this long
    arm_fault_p: float = Field(default=0.0, ge=0, le=1)
    arm_fault_s: Range = (20.0, 60.0)
    machine_fault_p: float = Field(default=0.0, ge=0, le=1)
    machine_fault_s: Range = (30.0, 90.0)
    aisle_block_p: float = Field(default=0.0, ge=0, le=1)
    aisle_block_s: Range = (30.0, 120.0)
    rush_order_p: float = Field(default=0.0, ge=0, le=1)
    rush_slack_k: Range = (1.1, 1.4)  # rush due = release + k * minimum flow time
    robot_join_at_s: Range | None = None  # _flex: one absent robot joins at this time
    robot_join_absent: tuple[int, int] = (1, 1)  # robots absent at the start (one joins)
    coordinator_outage_p: float = Field(default=0.0, ge=0, le=1)  # central baselines only
    coordinator_outage_s: Range = (30.0, 90.0)
    fault_window_s: Range = (30.0, 300.0)  # when one-off faults start


class SeedSplitCfg(StrictModel):
    train: Range = (0, 9999)
    val: Range = (10000, 10019)
    test: Range = (20000, 20099)


class CellScenarioCfg(StrictModel):
    name: str
    layout: str  # path to a configs/cell_layouts/*.yaml, relative to the repo root
    n_robots: int = Field(default=4, ge=1)
    max_robots: int = Field(default=4, ge=1)
    orders_from: str | None = None  # path to an abstract ScenarioCfg with fixed orders
    generator: CellGeneratorCfg | None = None
    due_k: float | None = Field(default=None, gt=0)  # fixed orders: due = release + k * min flow
    t_max: float | None = Field(default=None, gt=0)  # default: 3 x the last due time
    handling: HandlingCfg | None = None  # overrides the layout's handling block
    machine_seq_rule: Literal["FIFO", "EDD", "SPT"] = "FIFO"
    k_tasks: int = Field(default=8, ge=1)
    wait_cap_s: float = Field(default=10.0, gt=0)  # a WAIT is re-asked after this long
    battery: bool = False
    noise: CellNoiseCfg = Field(default_factory=CellNoiseCfg)
    disruptions: CellDisruptionCfg = Field(default_factory=CellDisruptionCfg)
    seeds: SeedSplitCfg = Field(default_factory=SeedSplitCfg)

    @model_validator(mode="after")
    def _consistent(self) -> CellScenarioCfg:
        if (self.orders_from is None) == (self.generator is None):
            raise ValueError("give exactly one of orders_from and generator")
        if self.n_robots > self.max_robots:
            raise ValueError("n_robots exceeds max_robots")
        if self.battery:
            raise ValueError("battery is not implemented yet (spec: optional, phase 2)")
        return self


class CellRewardCfg(StrictModel):
    """Team reward terms (spec §8). Set a weight to 0 to switch the term off."""

    on_time: float = 10.0  # per order delivered on time
    late_per_30s: float = 1.0  # per late order: -late_per_30s x min(tardiness / 30, late_cap)
    late_cap: float = 5.0
    past_due_per_s: float = 0.01  # per second, per undelivered order past its due time
    feed_idle_machine: float = 0.2  # shaping: a box lands in an idle machine's in-buffer
    blocked_trip: float = 0.05  # shaping, subtracted per blocked trip
    empty_travel_per_s: float = 0.02  # shaping, subtracted per second of empty travel
