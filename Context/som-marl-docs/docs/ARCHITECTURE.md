# Architecture

This document is the contract. If code disagrees with it, either the code is wrong or this doc needs
updating in the same change. Interfaces marked **[contract]** are depended on by another team or by
saved artifacts. Changing one needs human sign-off.

---

## 1. System overview

```
                         ┌───────────────────────────── configs/ (YAML, pydantic-validated) ─────────────────────────────┐
                         │  layouts/arena_13x6ft.yaml   scenarios/abstract_10.yaml   train/mappo_2r.yaml   deploy/lab_*.yaml │
                         └───────────────┬──────────────────────────────┬──────────────────────────────┬────────────────┘
                                         │                              │                              │
 TRAINING / EVAL (any machine, no ROS)   ▼                              ▼                              │   LAB (Ubuntu PC + robots)
 ┌──────────────────────────────────────────────────────────────┐                                     ▼
 │  som.rl (MAPPO)       ◄──►  som.env.SomParallelEnv            │      ┌─────────────────────────────────────────────────┐
 │        ▲                         │  obs / masks / reward      │      │ som.runtime.LiveRuntime  (wall clock)           │
 │        │                         ▼                            │      │   ├─ Policy (learned or heuristic)              │
 │  som.policies  ──────────►  som.sim.SimRuntime (event heap,   │      │   ├─ FactoryCore  ◄── same class as training     │
 │  (heuristics, learned)        virtual clock, TravelModel)     │      │   ├─ virtual machines (timers)                   │
 │                                  │                            │      │   └─ RobotBackend ──► ros2 | kinematic           │
 │                                  ▼                            │      └───────────────┬─────────────────────────────────┘
 │                         som.core.FactoryCore  (pure)          │                      │ NavigateToPose action per robot
 └──────────────────────────────────────────────────────────────┘                      ▼
                                                                         ┌────────────────────────────────┐
                                                                         │ ros2_ws/som_robot goto_controller│ ×N  (on each TurtleBot)
                                                                         │  pose: /tbN/odom  or arena pose │
                                                                         │  out:  /tbN/cmd_vel             │
                                                                         └────────────────────────────────┘
```

Three layers, one rule: **the factory logic exists once** (`som.core`). Everything else is either a
*time source* (sim heap vs wall clock), a *decision source* (learned vs heuristic policy), or a
*motion source* (travel model vs kinematic sim vs real robots).

---

## 2. Repository layout

```
som-marl/
├── CLAUDE.md
├── README.md                     # human quickstart, demo instructions
├── pyproject.toml                # package "som", src layout, extras: [dev], [viz], [ros]
├── Makefile
├── .pre-commit-config.yaml       # ruff, ruff-format, end-of-file, yaml check
├── .gitignore                    # runs/, *.pt outside reports/, __pycache__, .venv
│
├── configs/
│   ├── layouts/
│   │   └── arena_13x6ft.yaml     # stations, dock poses, robot start poses, arena bounds   [contract]
│   ├── scenarios/
│   │   ├── abstract_10.yaml      # the abstract's 10-order schedule (headline eval)
│   │   ├── random_10.yaml        # generator params: jittered arrivals/dues (training distribution)
│   │   └── stress_20.yaml        # heavier load, generalization check
│   ├── train/
│   │   ├── mappo_2r.yaml         # primary
│   │   ├── mappo_4r.yaml
│   │   ├── mappo_2r_disrupt.yaml # stretch (D15)
│   │   └── ippo_2r.yaml          # optional ablation: same trainer, critic_mode: local
│   └── deploy/
│       ├── lab_2r_odom.yaml      # backend=ros2, pose_source=odom, robots tb1,tb2
│       ├── lab_2r_camera.yaml
│       └── desk_kinematic.yaml   # backend=kinematic, for demo rehearsal with no robots
│
├── src/som/
│   ├── __init__.py
│   ├── config/
│   │   ├── schema.py             # pydantic models: LayoutCfg, ScenarioCfg, EnvCfg, RewardCfg, TrainCfg, DeployCfg
│   │   └── loader.py             # load_yaml(path) -> model; resolves includes; dumps resolved config to run dir
│   ├── core/                     # PURE. no torch/ros/io/clock
│   │   ├── ids.py                # RobotId, MachineId, StationId, OrderId (NewType str) + prefix helpers
│   │   ├── layout.py             # Station, Pose2D, Layout (from LayoutCfg)
│   │   ├── order.py              # JobType(route), Order(state machine: WAITING/RESERVED/IN_TRANSIT/QUEUED/PROCESSING/DONE)
│   │   ├── events.py             # OrderReleased, RobotArrived, RobotFailed, ProcessDone, MachineDown, MachineUp, Tick
│   │   ├── commands.py           # MoveRobot, StartProcess, CancelRobot
│   │   ├── decisions.py          # DecisionRequest, Candidate, the WAIT sentinel
│   │   ├── factory.py            # FactoryCore
│   │   └── invariants.py         # check_invariants(core) -> list[str]; called in tests + debug mode
│   ├── sim/
│   │   ├── travel.py             # TravelModel protocol + StraightLineTravel(speed, turn_rate, dock_overhead)
│   │   ├── runtime.py            # SimRuntime: event heap, advance_to_next_decision()
│   │   └── scenario.py           # build orders from ScenarioCfg (fixed or generated with rng)
│   ├── env/
│   │   ├── parallel_env.py       # SomParallelEnv(pettingzoo.ParallelEnv)
│   │   ├── candidates.py         # build K candidate slots per agent from DecisionRequest
│   │   ├── observations.py       # versioned obs encoders per role (OBS_SCHEMA_VERSION)        [contract]
│   │   ├── global_state.py       # fixed-size critic state, padded to max_robots/max_orders
│   │   └── rewards.py            # RewardFn: team/local, ≤4 terms, computed from core deltas
│   ├── policies/
│   │   ├── base.py               # Policy protocol: act(obs_by_agent, mask_by_agent) -> actions
│   │   ├── heuristics.py         # robot: FIFO, EDD, NEAREST, MIN_SLACK; machine: FIFO, EDD, SPT
│   │   ├── random_policy.py
│   │   └── learned.py            # LearnedPolicy.from_artifact(dir)
│   ├── rl/
│   │   ├── networks.py           # RoleActor, Critic (MLP; LayerNorm; orthogonal init)
│   │   ├── buffer.py             # flat numpy buffer over (env, agent) with dt, masks, decision flag
│   │   ├── mappo.py              # trainer; critic_mode: "central" (MAPPO) | "local" (IPPO)
│   │   ├── vec.py                # SyncVecEnv over N SomParallelEnv copies (simple, no subprocess at first)
│   │   ├── normalization.py      # running mean/std for value targets (PopArt-lite)
│   │   └── artifact.py           # export/load policy artifact                               [contract]
│   ├── eval/
│   │   ├── metrics.py            # EpisodeMetrics from core history
│   │   ├── evaluate.py           # run policy × scenario × seeds -> DataFrame-like rows (csv)
│   │   ├── compare.py            # baseline table: mean ± 95% CI, markdown + csv
│   │   └── plots.py              # learning curves, on-time bars, Gantt chart per episode
│   ├── runtime/
│   │   ├── live.py               # LiveRuntime: wall-clock loop around FactoryCore + Policy + RobotBackend
│   │   ├── recorder.py           # writes the same episode log format as SimRuntime (replayable)
│   │   └── backends/
│   │       ├── base.py           # RobotBackend protocol                                       [contract]
│   │       ├── kinematic.py      # unicycle sim at real time (desk rehearsal, dashboard dev)
│   │       └── ros2.py           # rclpy action clients → /tbN/go_to_pose (lazy import)
│   └── viz/
│       ├── replay.py             # matplotlib animation from an episode log
│       └── dashboard.py          # live view: arena, robots, station queues, machine timers, order table
│
├── ros2_ws/src/som_robot/        # ament_python package; runs ON each TurtleBot
│   ├── package.xml
│   ├── setup.py / setup.cfg
│   ├── som_robot/
│   │   ├── goto_controller.py    # NavigateToPose action server → cmd_vel
│   │   ├── safety.py             # LiDAR front-arc stop, velocity caps, watchdog
│   │   └── pose_sources.py       # OdomPose (+start-pose offset) | ArenaPose (camera topic)
│   ├── launch/robot.launch.py    # namespace from arg, params file
│   └── config/goto_params.yaml
│
├── scripts/                      # thin CLIs; all logic lives in src/som
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_baselines.py
│   ├── replay.py
│   └── run_lab.py
├── tests/
│   ├── core/  sim/  env/  rl/  runtime/  policies/
│   └── conftest.py               # fixtures: tiny layout, 3-order scenario, rng
├── docs/
├── reports/                      # committed: figures + tables for SCCUR
└── runs/                         # gitignored
```

Packaging: `pip install -e .` gives `som`. Extras: `[dev]` (pytest, ruff, mypy), `[viz]` (pygame). There is
no `[ros]` pip extra: rclpy comes from the system ROS install (`source /opt/ros/humble/setup.bash`).
`ros2_ws` is built separately with `colcon build` on the robots. It **does not depend on `som`**, so the
robots never need torch or this package.

---

## 3. Domain model (`som.core`)

### 3.1 Entities

- **Station**: `id`, `kind ∈ {SOURCE, MACHINE, SINK}`, `dock_pose: Pose2D`, `queue_pose: Pose2D`,
  `input_capacity` (machines only; default 5, from legacy `eligible_buffer_capacity`).
- **Machine**: one per MACHINE station. `process ∈ {Turning, Grinding, Milling, Drilling}`,
  `busy_until: float | None`, `current: OrderId | None`. Machines are virtual: processing is a timer.
- **JobType**: `route: list[Process]` (e.g. P2 = [Turning, Milling, Drilling]).
- **Order**: `id`, `job_type`, `release_time`, `due_time`, `process_times: list[float]`, `step_idx`,
  `location` (station id, robot id, or machine), and `status`:

```
RELEASED@SRC ──(robot reserves)──► RESERVED ──(robot docks at pickup)──► IN_TRANSIT
     ▲                                                                        │
     │                                                 (robot docks at dest)  ▼
 QUEUED@machine ◄──────────────────────────────────────────────────────── (dest is machine)
     │ (machine selects)                                                      │ (dest is SINK)
     ▼                                                                        ▼
 PROCESSING ──(ProcessDone)──► WAITING_PICKUP@machine (back into transport pool)   DELIVERED
```

- **Robot**: `id`, `status ∈ {IDLE, TO_PICKUP, TO_DROPOFF, WAITING_DOCK}`, `carrying: OrderId | None`,
  `task: TransportTask | None`, `last_station`, `pose` (an estimate, informational only).
  Capacity is 1. A robot holds a **reservation** on its order from assignment until drop-off.
- **Dock lock**: each station dock admits one robot at a time. Others wait (`WAITING_DOCK`). This is
  modeled in training too, so learned policies see congestion.

### 3.2 FactoryCore API [contract]

```python
class FactoryCore:
    def __init__(self, layout: Layout, orders: Sequence[OrderSpec], robots: Sequence[RobotId],
                 cfg: CoreCfg) -> None: ...

    # time is always passed in, never read
    def apply(self, event: Event, now: float) -> list[Command]:
        """Advance state for one external event. Returns commands the runtime must execute
        (e.g. after RobotArrived at pickup → MoveRobot to drop-off)."""

    def pending_decisions(self, now: float) -> list[DecisionRequest]:
        """Agents that are idle AND have ≥1 feasible non-WAIT option. Deterministic order."""

    def assign(self, agent: AgentId, choice: Candidate | Wait, now: float) -> list[Command]:
        """Apply one agent's decision. Raises InfeasibleDecision if the choice isn't in its request."""

    def snapshot(self) -> CoreSnapshot      # immutable view for obs/global-state/dashboard
    def history(self) -> EpisodeLog          # per-order timeline + per-agent decisions, for metrics/replay
    @property
    def done(self) -> bool                   # all orders delivered
```

- `Command`s are **intent** (`MoveRobot(robot, station, purpose)`, `StartProcess(machine, order, duration)`).
  Runtimes turn them into future events: SimRuntime schedules `RobotArrived` at `now + travel_time`;
  LiveRuntime sends a ROS goal and waits for the result.
- `RobotFailed(robot, reason)` (goal aborted/timeout) returns the order to its previous station and
  frees the reservation. The core must handle this; the real lab will produce it.
- `MachineDown(machine)` / `MachineUp(machine)` (built in M1, used only in the D15 stretch): a down machine
  accepts no new deliveries and starts nothing; an order already processing is **paused** and resumes on
  `MachineUp` with its remaining time. Its input queue stays put. Robots see the machine as having zero
  input capacity, so no candidate routes there. A rush order is just an `OrderReleased` with a tight due
  time, generated by the scenario.
- **Known limitation (D16):** a robot commits to its task from assignment to drop-off; there is no preemption.
  The only way a task ends early is `RobotFailed`.
- `check_invariants(core)` must hold after every `apply`/`assign`: each order is in exactly one place,
  no capacity is exceeded, reservations are consistent, and time is monotone.

### 3.3 Decisions

A `DecisionRequest` carries the agent id, role, and its **candidate list**:

- **Robot**: one candidate per transportable order (status RELEASED or WAITING_PICKUP, unreserved,
  whose destination has input capacity, or is SINK). The destination is implied by the route, so the
  robot's only choice is *which order*. Plus `WAIT` (re-queried at the next event).
- **Machine**: one candidate per order in its input queue. Plus `WAIT` (can be disabled in config).

Agents that are busy (traveling, processing) are **not** in `pending_decisions`.

---

## 4. Simulation runtime (`som.sim`)

- `TravelModel.estimate(from_pose, to_pose) -> seconds`. Default `StraightLineTravel`:
  `rotate(Δθ)/ω_eff + dist/v_eff + dock_overhead`, with `v_eff≈0.15 m/s` and `ω_eff≈1.0 rad/s`, plus
  optional lognormal noise (σ configurable) so policies tolerate real-world timing variance.
  **Calibrate `v_eff` and the overhead from real controller runs in M4** and record them in the layout YAML.
- `SimRuntime` owns a heap of `(time, seq, Event)`. `advance_to_next_decision()` pops events, feeds
  `core.apply`, and schedules events for the returned commands, until `pending_decisions()` is non-empty
  or the episode ends (all delivered, or `t ≥ horizon`).
- **Time unit = seconds.** The legacy scenario used "steps" with values (due 146–376, process 10–47) that
  map naturally to seconds in a 4 m arena. `abstract_10` keeps them as seconds.

---

## 5. RL environment (`som.env`)

### 5.1 Semantics: semi-MDP over decision points

`SomParallelEnv` follows the **PettingZoo ParallelEnv** API, with these rules:

- `possible_agents` = all robots + all machines, fixed for an episode.
- Each `step(actions)` resolves decisions for **all agents in `pending_decisions`** at the current time,
  then calls `SimRuntime.advance_to_next_decision()`.
- Agents with no pending decision still receive obs and a mask, but the mask allows **only `NOOP`**.
  `infos[agent]["decision"] = False` tells the trainer to **exclude** those samples from the policy loss.
  This is the fix for legacy's "every agent acts every tick" noise.
- Robots deciding at the same instant are resolved **sequentially in a fixed order**, and candidates are
  re-validated after each one, so two robots cannot reserve the same order. A collided choice becomes
  WAIT and is logged.
- `infos[agent]["dt"]` = seconds elapsed until the next decision point. The trainer discounts with
  `γ^dt` (per-second γ, e.g. 0.995). Rewards are accumulated over the interval.
- **`terminations`** = all orders delivered. **`truncations`** = horizon reached. These are kept
  separate: the trainer bootstraps on truncation (legacy treated timeouts as terminal).

### 5.2 Action space

`Discrete(K + 1)` per role. Slot 0 = `WAIT`/`NOOP`, slots 1..K = candidates. Defaults: `K_robot=6`,
`K_machine=5`. Candidates are placed in slots in a **canonical, non-heuristic order** (time they became
available, then order id), so the policy has to learn prioritization rather than inherit EDD from slot
order. If there are more than K candidates, keep the K oldest and log a truncation count.

### 5.3 Observations [contract: versioned]

`OBS_SCHEMA_VERSION` is stored in the policy artifact; a mismatch must fail loudly at load.

**Robot obs** (fixed length, independent of robot count):
- self: last-station one-hot, carrying flag, time fraction
- per candidate slot k (zero-padded, plus a `valid` bit): pickup station one-hot, destination one-hot,
  ETA to pickup (normalized), pickup→dest travel time, slack (normalized, clipped), remaining ops count,
  time waiting
- factory summary: per-station queue length and input free capacity, per-machine busy flag and remaining
  time fraction
- other robots, aggregated per station: count of robots heading there (permutation- and N-invariant)

**Machine obs:**
- self: busy flag, remaining time fraction
- per queue slot k: process time, slack, remaining ops after this one, time waiting
- downstream pressure: input free capacity at each order's next station

### 5.4 Global state (MAPPO critic)

A fixed-size vector padded to `max_robots=4` and `max_orders=24`: every station queue, machine state,
robot state (status, carrying, ETA), and per-order (status one-hot, slack, step_idx). Includes a
role/agent one-hot so the critic can be shared across agents (agent-specific global state, as in MAPPO).

### 5.5 Reward (`RewardCfg`)

Default **team** reward (every agent gets the same scalar), computed from core deltas over the interval:

```
r = w_deliver · (#delivered)
  − w_tardy  · Σ_orders  (seconds overdue accrued during the interval)    # dense form of total tardiness
  + w_ontime · (#delivered on time)
  − w_time   · dt                                                          # optional, default 0
```

Defaults: `w_deliver=1.0, w_tardy=0.01, w_ontime=1.0, w_time=0`. `mode: local` gives the delivery and
on-time terms only to the delivering robot and the last machine; it exists for ablation only.
**No forced actions, no distance shaping, no per-pick rewards.** If learning stalls, fix observations
or the curriculum first, and add a shaping term only as a configured, documented ablation.

---

## 6. Learning (`som.rl`)

- **Algorithm: MAPPO** (centralized training, decentralized execution). The critic consumes the global state
  (§5.4) and exists only during training. Actors are always decentralized: each role's actor sees only its
  own obs, and only actors are exported. The trainer keeps a `critic_mode` flag: `central` (default, MAPPO)
  or `local` (IPPO, own obs). The `local` mode is for one optional ablation run; don't spend tuning time on it.
- **Parameter sharing by role:** `actors = {"robot": RoleActor, "machine": RoleActor}`. A critic per role
  (or shared with role one-hot; config).
- **Batched updates:** the buffer stores flat arrays `[T, E, A, ...]`. Advantage normalization is
  **per role**. Minibatches are tensors (no per-sample Python loops). Masked logits use `-1e9` fill.
  Value targets are normalized with running statistics.
- GAE uses `γ^dt` and separates `terminated` vs `truncated` (bootstrap the value on truncation).
- **Training distribution:** `random_10` (jittered arrivals ±20%, dues ±15%, job-type mix) with seeds
  disjoint from eval. `abstract_10` is evaluation only, so we never train on the test schedule.
- **Curriculum (optional):** start with loose due dates and tighten them.
- **Scale test:** train `mappo_2r`, evaluate the same artifact with 4 robots, then fine-tune `mappo_4r`.
- Logging: TensorBoard scalars for on-time rate, mean tardiness, delivered count, entropy per role,
  value loss, KL, and clip fraction. Eval every N updates on 10 fixed seeds.

### Policy artifact [contract]

```
runs/<id>/policy/
├── policy.pt          # {"robot": state_dict, "machine": state_dict}  (actors only)
└── policy_meta.json   # {"obs_schema_version": 1, "K_robot": 6, "K_machine": 5,
                       #  "net": {...}, "layout": "arena_13x6ft", "trained_on": "random_10",
                       #  "git_sha": "...", "created": "..."}
```

`LearnedPolicy.from_artifact(dir)` uses greedy (argmax) actions by default, with a `stochastic=True` option.

---

## 7. Evaluation (`som.eval`)

**Metrics per episode** (from `EpisodeLog`): on-time rate (of all orders, not only delivered ones), mean
and max tardiness, makespan, delivered count, robot utilization (loaded / empty travel / idle), machine
utilization, number of decisions, number of WAITs.

**Protocol:** policy × scenario × **20 seeds** (eval seeds fixed in config) → `eval/rows.csv` →
`compare.py` gives a markdown table with mean ± 95% CI. The headline table for SCCUR:

| Policy | on-time % | mean tardiness (s) | makespan (s) |
|---|---|---|---|
| Random | | | |
| FIFO / FIFO | | | |
| EDD / EDD | | | |
| MIN_SLACK / EDD (strongest heuristic) | | | |
| MAPPO robots + EDD machines | | | |
| **MAPPO (all learned)** | | | |
| IPPO (all learned), optional ablation | | | |
| Legacy abstract result | 38% (of delivered) | — | — |

---

## 8. Live runtime (`som.runtime`)

```python
class RobotBackend(Protocol):                                   # [contract]
    def start(self) -> None: ...
    def send_goal(self, robot: RobotId, pose: Pose2D, goal_id: str) -> None: ...
    def cancel(self, robot: RobotId) -> None: ...
    def poll(self) -> list[BackendEvent]: ...   # GoalSucceeded | GoalFailed(reason) | PoseUpdate
    def stop_all(self) -> None: ...             # e-stop: cancel + zero velocity
    def shutdown(self) -> None: ...
```

**Where policies run (D14):** in the November demo, `LiveRuntime` and every agent's actor run in one
process on the lab PC. Decentralization is preserved logically, because each actor is called with only its
own obs. A future `onboard` mode would broadcast factory state on a ROS topic and run each robot's actor on
its Pi (numpy forward pass, no torch). The artifact format and obs encoders already allow this; don't build it now.

`LiveRuntime` loop (about 10 Hz):
1. `poll()` the backend → map `GoalSucceeded` to `RobotArrived` and `GoalFailed` to `RobotFailed`, then `core.apply`.
2. Fire due virtual-machine timers → `ProcessDone`.
3. Release orders whose `release_time ≤ now − t0`.
4. For `pending_decisions()`, build obs exactly as `SomParallelEnv` does (shared encoder functions),
   call the policy, `core.assign`.
5. Execute the returned commands: `MoveRobot` → `send_goal(dock_pose)` (or `queue_pose` if the dock is locked).
6. Push the snapshot to the dashboard and the recorder.

- **Same obs encoders as training** (`som.env.observations`), called from the live loop. Never re-implement them.
- A **speed factor** (`time_scale`) is allowed only for virtual machine timers and due dates, so a long
  scenario fits a 3–5 min demo. Robot motion is real time.
- The dashboard has keyboard controls: `space` pause (cancel goals, freeze timers), `e` e-stop, `r` reset,
  and (D15 stretch) `1`–`4` toggle a machine down/up, which injects `MachineDown`/`MachineUp`.
- The recorder writes the same `EpisodeLog` format as the sim, so live runs can be replayed and scored
  with the same metrics. Report "lab run" metrics next to the sim ones.

Backends:
- `kinematic`: unicycle model at real time with the same caps as the controller. Use it for full demo
  rehearsal without robots, and for dashboard development.
- `ros2`: one `ActionClient(NavigateToPose, f"/{ns}/go_to_pose")` per robot, run in a background executor
  thread. Converts arena poses into each robot's frame (see HARDWARE_INTERFACE §3).

---

## 9. Key risks and mitigations

| Risk | Mitigation |
|---|---|
| Odometry drift over a 5 min run | Short runs; every robot starts at a known pose; the controller tolerance is looser than the dock spacing; the camera pose source is the upgrade |
| PC↔robot link drops | Local access point (D12); the action server keeps executing; the controller watchdog stops the robot if goals stop arriving; `RobotFailed` path in the core |
| Robots colliding near stations | Dock locks plus queue poses; LiDAR front-arc stop in `safety.py`; place stations ≥ 0.5 m apart |
| Policy fails on hardware timing | Travel-time noise in training; calibrate `v_eff` from real runs; the heuristic policy is the fallback for the live demo |
| Learned policy doesn't beat heuristics | Still a valid result. Present the comparison honestly, with the legacy 38% vs new setup as context; inspect per-role behavior (Gantt) for the explanation |
| Venue demo failure | Backup video by Nov 16; `desk_kinematic` live dashboard runs on a laptop alone |
