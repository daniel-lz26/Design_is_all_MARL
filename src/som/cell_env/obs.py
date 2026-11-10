"""Per-robot observations (spec §7.3), fixed size and independent of how many robots exist.

Every feature is something a TurtleBot can get from its own state, the task board, or the
station and robot broadcasts allowed by spec §1a. Layout (``ObsLayout`` holds the offsets):

    self      x, y, state one-hot, battery, agent-id one-hot (max_robots), time, docked
    tasks     K rows (the board's top K, by due then order number), TASK_FEATURES each
    stations  one row per station (layout order), STATION_FEATURES each
    others    max_robots - 1 rows (id order, self skipped), OTHER_FEATURES each
    global    orders done, on time so far, past due, time to next release
    [orders]  obs_mode "global" only: max_orders x ORDER_FEATURES, an upper-reference ablation

Times are seconds / TIME_SCALE (no clipping below CLIP), counts / COUNT_SCALE. Changing any
feature, order or scale means bumping CELL_OBS_VERSION.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from som.cell_env.entities import RS, Loc
from som.cell_env.estimates import bid, free_at, remaining_proc, slack

if TYPE_CHECKING:
    from som.cell_env.des import CellSim

CELL_OBS_VERSION = 1
TIME_SCALE = 200.0
CLIP = 10.0  # +/- 2,000 s
COUNT_SCALE = 4.0
OPS_SCALE = 4.0
BOTTLENECK_PROCESS = "Milling"
STATES = tuple(RS)

TASK_FEATURES = (
    "valid",
    "travel_to_pickup",
    "travel_pickup_to_dest",
    "slack",
    "remaining_ops",
    "remaining_proc",
    "dest_in_occupancy",
    "dest_busy",
    "dest_time_to_free",
    "dest_arm_queue",
    "waiting_time",
    "due_in",
    "dest_is_bottleneck",
    "best_other_bid",
    "pickup_here",
)
STATION_FEATURES = (
    "in_occupancy",
    "out_occupancy",
    "machine_busy",
    "machine_blocked",
    "machine_down",
    "time_to_free",
    "arm_busy",
    "arm_down",
    "robots_at_dock",
    "inbound",
    "queue_work",
)
OTHER_FEATURES = (
    "present",
    "dx",
    "dy",
    *(f"state_{s.value}" for s in STATES),
    *(f"dest_{i}" for i in range(6)),
    "eta_free",
    "up",
)
GLOBAL_FEATURES = ("done", "on_time", "past_due", "next_release")
ORDER_FEATURES = ("present", "stage", *(f"loc_{x.value}" for x in Loc), "slack")
T = {name: i for i, name in enumerate(TASK_FEATURES)}
S = {name: i for i, name in enumerate(STATION_FEATURES)}


def ts(seconds: float) -> float:
    if math.isinf(seconds):
        return CLIP
    return max(-CLIP, min(CLIP, seconds / TIME_SCALE))


@dataclass(frozen=True)
class ObsLayout:
    k: int
    n_stations: int
    max_robots: int
    mode: str = "board"
    max_orders: int = 40

    @property
    def self_dim(self) -> int:
        return 2 + len(STATES) + 1 + self.max_robots + 2

    @property
    def tasks_at(self) -> int:
        return self.self_dim

    @property
    def stations_at(self) -> int:
        return self.tasks_at + self.k * len(TASK_FEATURES)

    @property
    def others_at(self) -> int:
        return self.stations_at + self.n_stations * len(STATION_FEATURES)

    @property
    def global_at(self) -> int:
        return self.others_at + (self.max_robots - 1) * len(OTHER_FEATURES)

    @property
    def orders_at(self) -> int:
        return self.global_at + len(GLOBAL_FEATURES)

    @property
    def dim(self) -> int:
        extra = self.max_orders * len(ORDER_FEATURES) if self.mode == "global" else 0
        return self.orders_at + extra

    def tasks(self, obs: np.ndarray) -> np.ndarray:
        return obs[self.tasks_at : self.stations_at].reshape(self.k, len(TASK_FEATURES))

    def stations(self, obs: np.ndarray) -> np.ndarray:
        return obs[self.stations_at : self.others_at].reshape(
            self.n_stations, len(STATION_FEATURES)
        )

    def others(self, obs: np.ndarray) -> np.ndarray:
        return obs[self.others_at : self.global_at].reshape(
            self.max_robots - 1, len(OTHER_FEATURES)
        )


class ObsEncoder:
    def __init__(
        self, sim: CellSim, mode: str = "board", sense_radius_m: float = 1.5, max_orders: int = 40
    ) -> None:
        if mode not in ("board", "local", "global"):
            raise ValueError(f"obs_mode must be board, local or global, not {mode!r}")
        self.sim = sim
        self.mode = mode
        self.radius = sense_radius_m
        self.station_keys = list(sim.layout.stations)
        if len(self.station_keys) != 6:
            raise ValueError("observation schema v1 expects 6 stations (SRC, 4 machines, SINK)")
        self.layout = ObsLayout(sim.k, len(self.station_keys), sim.cfg.max_robots, mode, max_orders)
        self.w, self.h = sim.layout.floor_m

    # ------------------------------------------------------------------ parts

    def _xy(self, rid: str) -> tuple[float, float]:
        sim = self.sim
        r = sim.robots[rid]
        if r.down_xy is not None:
            return r.down_xy
        if r.docked_at is not None:
            return sim.roadmap.xy[sim.stations[r.docked_at].dock]
        return sim.roadmap.xy[r.node]

    def _station_rows(self) -> np.ndarray:
        sim = self.sim
        out = np.zeros((len(self.station_keys), len(STATION_FEATURES)), np.float32)
        for i, key in enumerate(self.station_keys):
            st = sim.stations[key]
            cap_in = st.in_cap or COUNT_SCALE
            if st.busy_until is not None:
                left = st.busy_until - sim.now
            elif st.paused is not None:
                left = st.paused
            else:
                left = 0.0
            queued = sum(sim.boxes[b].spec.proc[sim.boxes[b].step] for b in st.in_buf)
            out[i] = (
                len(st.in_buf) / cap_in if st.in_cap else 0.0,
                len(st.out_buf) / (st.out_cap or COUNT_SCALE),
                float(st.current is not None),
                float(st.blocked_box is not None),
                float(st.down),
                ts(left),
                float(st.arm_busy is not None),
                float(st.arm_down),
                (len(st.dock_queue) + (st.dock_holder is not None)) / COUNT_SCALE,
                st.inbound / cap_in,
                ts(queued + left),
            )
        return out

    def _global(self) -> list[float]:
        sim = self.sim
        n = len(sim.boxes)
        done = [b for b in sim.boxes.values() if b.delivered is not None]
        ontime = sum(b.delivered <= b.spec.due for b in done)  # type: ignore[operator]
        past = sum(b.delivered is None and sim.now > b.spec.due for b in sim.boxes.values())
        future = [b.spec.release - sim.now for b in sim.boxes.values() if b.loc is Loc.PENDING]
        nxt = ts(min(future)) if future else CLIP
        return [len(done) / n, ontime / n, past / n, nxt]

    def _orders(self) -> np.ndarray:
        sim = self.sim
        out = np.zeros((self.layout.max_orders, len(ORDER_FEATURES)), np.float32)
        locs = list(Loc)
        for i, b in enumerate(sorted(sim.boxes.values(), key=lambda b: b.spec.number)):
            if i >= self.layout.max_orders:
                break
            row = [1.0, b.step / len(b.spec.route)] + [float(b.loc is x) for x in locs]
            left = remaining_proc(b) if b.delivered is None else 0.0
            row.append(ts(b.spec.due - sim.now - left))
            out[i] = row
        return out

    # ------------------------------------------------------------------ public

    def encode(
        self, rid: str, stations: np.ndarray | None = None, glob: list[float] | None = None
    ) -> np.ndarray:
        sim = self.sim
        lay = self.layout
        r = sim.robots[rid]
        x, y = self._xy(rid)
        out = np.zeros(lay.dim, np.float32)
        selfv = [x / self.w, y / self.h]
        selfv += [float(r.state is s) for s in STATES]
        selfv.append(1.0)  # battery (not modelled yet)
        selfv += [float(r.index == i) for i in range(lay.max_robots)]
        selfv += [sim.now / sim.t_max, float(r.docked_at is not None)]
        out[: lay.self_dim] = selfv

        here = sim.location(rid)
        tasks = sim.tasks(rid)
        others = [
            o for o in sim.robots.values() if o.id != rid and o.state not in (RS.DOWN, RS.ABSENT)
        ]
        rows = out[lay.tasks_at : lay.stations_at].reshape(lay.k, len(TASK_FEATURES))
        for k, b in enumerate(tasks):
            pickup, dest = sim.stations[b.where], sim.next_station(b)
            cap = dest.in_cap or COUNT_SCALE
            dest_left = (dest.busy_until - sim.now) if dest.busy_until is not None else 0.0
            rows[k] = (
                1.0,
                ts(sim.roadmap.time(here, pickup.approach)),
                ts(sim.roadmap.time(pickup.approach, dest.approach)),
                ts(slack(sim, b, here)),
                (len(b.spec.route) - b.step) / OPS_SCALE,
                ts(remaining_proc(b)),
                (len(dest.in_buf) + dest.inbound) / cap if dest.in_cap else 0.0,
                float(dest.current is not None),
                ts(dest_left),
                (len(dest.arm_queue) + (dest.arm_busy is not None)) / COUNT_SCALE,
                ts(sim.now - b.t_avail),
                ts(b.spec.due - sim.now),
                float(dest.process == BOTTLENECK_PROCESS),
                ts(min((bid(sim, o, b) for o in others), default=math.inf)),
                float(r.docked_at == b.where),
            )
        out[lay.stations_at : lay.others_at] = (
            stations if stations is not None else self._station_rows()
        ).ravel()

        orows = out[lay.others_at : lay.global_at].reshape(lay.max_robots - 1, len(OTHER_FEATURES))
        j = 0
        for o in sim.robots.values():
            if o.id == rid:
                continue
            if j >= lay.max_robots - 1:
                break
            if o.state is not RS.ABSENT:
                ox, oy = self._xy(o.id)
                if self.mode != "local" or math.hypot(ox - x, oy - y) <= self.radius:
                    claimed = o.box or o.task
                    dest_i = (
                        self.station_keys.index(sim.next_station(sim.boxes[claimed]).key)
                        if claimed
                        else None
                    )
                    t_free, _ = free_at(sim, o)
                    orows[j, :3] = [1.0, (ox - x) / self.w, (oy - y) / self.h]
                    orows[j, 3 : 3 + len(STATES)] = [float(o.state is s) for s in STATES]
                    if dest_i is not None:
                        orows[j, 3 + len(STATES) + dest_i] = 1.0
                    orows[j, -2] = ts(t_free)
                    orows[j, -1] = float(o.state is not RS.DOWN)
            j += 1
        out[lay.global_at : lay.orders_at] = glob if glob is not None else self._global()
        if self.mode == "global":
            out[lay.orders_at :] = self._orders().ravel()
        return out

    def encode_all(self, rids: list[str]) -> dict[str, np.ndarray]:
        stations, glob = self._station_rows(), self._global()
        return {rid: self.encode(rid, stations, glob) for rid in rids}

    def critic_state(self) -> np.ndarray:
        """MAPPO critic input (training only): every robot's obs plus the order matrix."""
        parts = [self.encode(rid) for rid in self.sim.robot_ids]
        pad = self.layout.max_robots - len(parts)
        parts += [np.zeros(self.layout.dim, np.float32)] * pad
        return np.concatenate([*parts, self._orders().ravel()])
