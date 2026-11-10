# cell_env recon: how the current repo fits together

Milestone 1 of `Context/CELL_ENV_SPEC.md` (the spec). Written Nov 9, 2026, against `main` at
`ae7dc72` (main was fast-forwarded to `daniel-testing`, so it includes the MAPPO and disruption
work).

**The spec was written for the legacy `MARL_new` layout** (`envs/`, `trainer/`,
`ippo_trainer.py`, `mappo_trainer.py`). This repo is the rebuild (`som-marl`): a `src/som/`
package with different names. The code is the source of truth, as the spec asks. The table
in §6 maps every spec path to its place here.

---

## 1. What exists today

```
configs/layouts/arena_13x6ft.yaml      arena 3.96 x 1.83 m, 6 point stations, 4 robot starts
configs/scenarios/abstract_10.yaml     the 10 abstract orders (source of truth for cell_10)
configs/scenarios/random_10.yaml       generated 10-order streams around abstract_10
configs/scenarios/*_disrupt.yaml       the same plus DisruptionCfg
configs/train/mappo_{2r,4r,4r_disrupt}.yaml

src/som/config/    pydantic schemas (StrictModel: extra=forbid, frozen) + loader.SCHEMA_BY_DIR
src/som/core/      FactoryCore: pure event-driven state machine, no clock, no I/O
src/som/sim/       SimRuntime (event heap), StraightLineTravel, scenario builder, disruptions
src/som/env/       SomParallelEnv (PettingZoo), observations v2, global state, rewards
src/som/policies/  Policy protocol, heuristics, random, LearnedPolicy, RoleMixPolicy
src/som/rl/        MAPPO trainer, networks, buffer + semi-MDP GAE, SyncVecEnv, artifact
src/som/eval/      metrics, evaluate (run_one/evaluate), compare (mean ± 95% CI), plots
src/som/viz/       replay.py (GIF animation)
scripts/           compare_baselines.py, train.py, evaluate.py, replay.py
reports/           committed tables and figures
```

### 1.1 The abstract env

- **`FactoryCore`** (`src/som/core/factory.py`) holds all factory logic. `apply(event, now)` and
  `assign(agent, choice, now)` return intent commands (`MoveRobot`, `HoldDock`, `StartProcess`).
  Time is passed in; the core never reads a clock.
  - Point stations, one dock per station.
  - A machine input queue with capacity 5 that counts inbound reservations; no output buffer.
  - The pickup/drop-off dock hold is `dock_overhead_s` = 2 s; there are no arms.
  - Robots: `IDLE`, `TO_PICKUP`, `TO_DROPOFF`, `WAITING_DOCK`, `DOCKING`, `OFFLINE`.
  - Events: `RobotFailed`, `RobotDown`/`RobotUp`, `MachineDown`/`MachineUp`.
- **`SimRuntime`** (`src/som/sim/runtime.py`) owns the event heap. It applies every event at
  the current time before calling `pending_decisions`, which is the decision-point
  (event-driven) loop.
  - Travel is straight-line: `rotate/w_eff + dist/v_eff` with lognormal noise. There is no
    roadmap and no traffic.
- **`make_runtime(layout, scenario, env, seed)`** (`src/som/sim/scenario.py`) builds a seeded
  episode. `seed_streams(seed)` spawns 4 independent streams: orders, travel, policy and
  disruptions.
- **`SomParallelEnv`** (`src/som/env/parallel_env.py`) is a PettingZoo `ParallelEnv`:
  - Agents are `robot_0..` plus `machine_<process>`; both robots and machines learn.
  - `step(actions) -> (obs, rewards, terminations, truncations, infos)`. Termination and
    truncation are kept separate on purpose: the legacy code treated timeouts as terminal.
  - Masks live in `obs[agent]["action_mask"]`. `infos[agent]` carries `decision` (False → NOOP
    only) and `dt`.
  - One step resolves every pending decision at the current time, in a fixed order with
    re-validation, then runs to the next decision point.
- **Disruptions** (`src/som/sim/disruptions.py`, `DisruptionCfg`):
  - blocked trips (the goal times out → `RobotFailed`)
  - delayed trips
  - machine MTBF breakdowns with repairs
  - robot outages, temporary or permanent, always leaving one robot up

  No rush orders, arm faults or aisle closures.

### 1.2 Heuristics (`src/som/policies/heuristics.py`)

- A pair is named `ROBOT/MACHINE`.
  - Robot rules: `FIFO`, `EDD`, `NEAREST`, `MIN_SLACK`.
  - Machine rules: `FIFO`, `EDD`, `SPT`.
  - `all_heuristics()` gives the 12 pairs and `make_policy("NEAREST/FIFO", layout, travel)`
    parses a name.
- Ties break on `(available_since, order number)`, the core's canonical candidate order.
- `remaining_work(...)` estimates the time to finish an order (travel + docks + process). It
  feeds the slack features and `MIN_SLACK`.
- `RandomPolicy` never waits by default.
- The spec's `MST` is the same rule as `MIN_SLACK` here. `SPT`, `BOTTLENECK` and `RANDOM` don't
  exist yet as robot rules.

### 1.3 Policy interface today

`som.policies.base.Policy` is **decision-level**:
`decide(request: DecisionRequest, snapshot: CoreSnapshot) -> Candidate | WAIT`.
- Heuristics, `RandomPolicy`, `LearnedPolicy` and `RoleMixPolicy` all implement it.
- `run_episode` (sim) and `run_one`/`evaluate` (eval) call it. Learned policies build their own
  observations from the snapshot with the shared encoders.

This is **not** the spec's `act(obs, masks, global_state)` interface (see decision D4).

### 1.4 Trainer (`src/som/rl/`)

- `MAPPOTrainer` keeps one actor and one critic per role (`robot`, `machine`), shared within the
  role. `critic_mode: central | local`, so MAPPO and IPPO are one switch, as the spec wants.
  - Policy loss uses decision samples only. GAE uses `gamma ** dt` with the termination and
    truncation split.
  - Advantages are normalized per role and value targets use running normalization.
- `SyncVecEnv` groups arrays by role, using the `robot_` / `machine_` id prefixes
  (`core.ids.role_of`).
- It evaluates every `eval_every` updates and keeps the best checkpoint on held-out seeds of the
  training scenario, writing `policy/`, `policy_final/`, `progress.csv`, `eval.csv` and
  TensorBoard logs.
- No LR or entropy annealing and no obs normalization yet. The spec asks for both (§1 lessons,
  §11.3); they belong on the method branch.
- Speed: about 600 decision points/s per run with 4 robots. A profile shows **55% of the time
  goes into Python observation and global-state encoding**, 18% into PPO updates and 9% into the
  simulator.

### 1.5 Evaluation, plots and replay

- **Evaluation:**
  - `som.eval.evaluate.run_one(layout, scenario, env, policy, seed)` returns an `EpisodeLog`.
  - `evaluate(...)` returns rows over policy × seed.
  - `compute_metrics(log)` gives on-time % (over **all** orders), mean/max tardiness, makespan,
    robot loaded/empty/dock/idle fractions, machine utilization, decisions, waits, failed trips
    and outages.
  - `compare.summarize` / `mean_ci95` (Student t) / `to_markdown` / `best_policy` build the
    tables.
- **`plots.gantt(log, path)`:**
  - Machine lanes, then one lane per robot. Each order has its own colour (tab10).
  - Solid = loaded, hatched = empty, grey cross-hatch = down, red ✕ = failed trip, dotted
    verticals = due times.
  - **The spec's Gantt style already exists here; reuse `order_colors` and the bar styling.**
- **`plots.learning_curves(curves, path, title, baseline)`** draws the dashed best-heuristic
  line.
- **`viz.replay.animate(log, layout, path)`:** floor plan, robots coloured by carried order, and
  an order table on the right, saved as a GIF with PillowWriter.
- **Scripts:**
  - `compare_baselines.py`: all heuristics on one scenario.
  - `evaluate.py`: MAPPO families vs baselines, writing `reports/results_*.md` and learning
    curves.
  - `train.py`: one training run.
  - `replay.py`: GIF of a saved episode.

### 1.6 Seeds today

- Eval seeds are 1000–1019, held-out evals use 2000–2019, and training episodes use seeds
  ≥ 1,000,000.
- The spec's cell seeds are different (train 0–9999, val 10000–10019, test 20000–20099), and
  `cell_env` will use the spec's lists.

---

## 2. Spec §1 numbers vs this repo's reports

The spec's §1 table matches `reports/results_*.md` and `Context/ROADMAP.md`, with three notes:
- **`abstract_10`, 4 robots:** "best heuristic 43.5, MAPPO swings 42–48" describes the learning
  curves (10 eval seeds). The 20-seed table has `mappo_4r` at 43.8 and `mappo_4r_disrupt` at 42.0.
- **`random_10_disrupt`, 4 robots:** the 32.0 for FIFO/SPT is on seeds 2000–2019.
- **`random_10`, 2 robots:** the 29.0 (NEAREST/SPT) vs 33.7 MAPPO is on seeds 2000–2019.

All of these use **placeholder** release and process times, which are flagged in the configs.

---

## 3. Reuse plan

| Need in cell_env | Reuse | Notes |
|---|---|---|
| YAML + validation | `som.config` StrictModel pattern, `load_yaml`, `SCHEMA_BY_DIR` | New schemas in `som/config/cell_schema.py`; register `cell` and `cell_layouts` so the "every config loads" test covers them |
| `cell_10` orders | `configs/scenarios/abstract_10.yaml` via `orders_from:` | Read, not copied, so the two cannot drift |
| Random orders | `GeneratorCfg` ideas | Spec's TWK due rule and ±25% times differ, so a new generator |
| Seeded streams | `seed_streams` pattern (SeedSequence.spawn) | One `SeedSequence(seed)` per episode, children per purpose |
| Metrics / CIs | `compare.mean_ci95`, `summarize`, `to_markdown` | cell metrics add arm/machine-blocked/blocked-trip columns |
| Gantt / replay | `plots.order_colors`, bar styles, `replay.animate` layout | New renderers for roadmap, arms and buffers (M4) |
| Heuristic names | `ROBOT/MACHINE` naming | Robot rules re-implemented on the task board (`NEAREST, FIFO, EDD, SPT, MST, BOTTLENECK, RANDOM`) |
| Trainer | `MAPPOTrainer`, `semi_mdp_gae`, networks | Needs a role-mapping generalization for `agv_*` ids (M5/method branch) |
| Abstract env | untouched | `som.core`, `som.sim`, `som.env` stay as they are; all tests keep passing |

**Not reused: `FactoryCore`.** The cell needs arms with FIFO handoffs, input and output
buffers, machine blocking, a roadmap with edge reservations, robot states such as
`BEING_LOADED`, and parking. Bolting those onto `FactoryCore` would change a contract the
hardware team builds against and risk the abstract env. `cell_env` gets its own DES, written
in the same style: pure, seeded, time passed in.

---

## 4. Design decisions for review (made for M2, easy to change)

**D1. Location.**
- The spec's `envs/cell_env/` becomes **`src/som/cell_env/`**, next to `src/som/env/`. The
  demo runs as `python -m som.cell_env.demo --config configs/cell/cell_10.yaml`.
- Scenario YAMLs go in `configs/cell/`, as the spec says. The floor layout goes in
  `configs/cell_layouts/sac_6x4.yaml`, so the seven scenario files share one layout.
- New docs go in `docs/` as the spec says. The spec itself stays in `Context/`.

**D2. API.**
- The spec says "same API as the existing env: ids `agv_1..agv_4`, 4-tuple with `dones`". The
  existing env is actually a PettingZoo `ParallelEnv` that returns a 5-tuple with separate
  `terminations`/`truncations` and puts masks in `obs["action_mask"]`.
- Plan: **PettingZoo 5-tuple, masks in `obs["action_mask"]`** (as the current trainer expects,
  and also copied to `infos[agent]["action_mask"]`), and **agent ids `agv_1..agv_4`** as the
  spec says.
- Keeping termination and truncation separate matters for bootstrapping. This is milestone 5
  work; M2 is the DES only.

**D3. Deadlocks.**
- In this model a robot waiting at a node holds nothing: nodes have unlimited capacity, a
  robot holds an edge only while it is traversing it, and a dock queue waits at the approach
  node. A cycle of waits therefore cannot form.
- Instead:
  - the DES checks the wait-for graph each time a robot blocks, and logs any cycle it finds
    (expected: never);
  - a blocked robot **reroutes** when the edge it waits for is closed (aisle blocked, or held
    by a robot that is down).
- The `deadlocks_resolved` metric exists and should stay 0.

**D4. Policy interface.** The spec's `act(obs, masks, global_state)` interface is new. The old
decision-level `Policy` protocol stays for the abstract env. The cell interface goes in
`som/policies/cell_base.py` + `registry.py` (M5), not in `policies/base.py`, so no old caller
changes.

**D5. Broken robot carrying a box.**
- Spec default: "stuck until repair". With a *permanent* breakdown (the `_disrupt` default)
  that box can never be delivered, so the episode always runs to `t_max`.
- Plan: keep the spec default in the engine (`rescue: false`), but set **`rescue: true` in the
  `_disrupt` scenarios**. That matches the current abstract runs, where the order goes back.
- The spec says a rescued box goes to "the nearest station's in-buffer". I put it in that
  station's **out-buffer** instead, as an open transport task. An in-buffer box would be
  processed by the wrong machine.

**D6. A down robot blocks its spot.** A robot that breaks on an edge or at a dock keeps
blocking it for `clear_delay_s` (default 20 s, "someone pushes it aside"), then releases it.
This is the realistic case, and it is what makes rerouting matter.

**D7. Docking.**
- Robots plan to the station's **approach node**, then wait FIFO for the dock.
- Docking takes spur travel + `dock_time_s`. The arm handoff (FIFO per arm) follows, then
  undocking takes `undock_time_s` + spur travel.
- A robot that frees up at a dock decides while still docked:
  - if it claims a box at the same station, the arm loads it without undocking;
  - for any other choice it undocks first, so idle robots never block a dock.

**D8. Battery** is off by default (spec), and **not implemented in M2** (Phase 2 in the spec).
The config key exists but is rejected if set.

**D9. Seeds.** "Every source of randomness goes through one seeded Generator": each episode
uses one `SeedSequence(seed)`, with child generators for orders, disruptions, travel noise and
arm noise. Disruptions are drawn at reset, so every policy sees the same disruptions for the
same seed, regardless of what the policy does.

---

## 5. Gaps that block later milestones

- **OR-Tools** (CP-SAT, `CENTRAL_*` baselines, M6) and **DEAP** (GP branch) aren't installed.
  M6 adds `ortools` to `requirements.txt`.
- **Placeholders:** the layout coordinates, arm 6 s, dock 3/1.5 s and robot 0.18 m/s are spec
  defaults, not measurements. Every one is marked `# PLACEHOLDER` in YAML.
- **The speed target** (≥ 2,000 decision points/s on one core) applies to the env. The current
  trainer's bottleneck is Python observation encoding, so `cell_env` observations (M5) should
  be built with NumPy arrays from the start.

---

## 6. Spec path → this repo

| Spec | Here |
|---|---|
| `envs/cell_env/*.py` | `src/som/cell_env/*.py` |
| `envs/cell_env/demo.py` | `python -m som.cell_env.demo` |
| `configs/cell/*.yaml` | `configs/cell/*.yaml` (+ `configs/cell_layouts/`) |
| `trainer/ippo_trainer.py`, `mappo_trainer.py` | `src/som/rl/mappo.py` (`critic_mode` switch) |
| `policies/base.py`, `policies/registry.py` | `src/som/policies/cell_base.py`, `src/som/policies/registry.py` |
| `baselines/*.py` | `src/som/baselines/*.py` |
| `scripts/evaluate.py` | exists (abstract MAPPO vs baselines); the spec's harness becomes `scripts/evaluate_cell.py` so the abstract script keeps working |
| `tests/test_cell_env.py` | `tests/cell_env/test_cell_env.py` (tests mirror `src/`) |
| `docs/*.md` | `docs/*.md` |
| `results/<policy>/` | `results/<policy>/` |

---

## 7. Milestone 2 results and questions for review

`python -m som.cell_env.demo --config configs/cell/cell_10.yaml` (NEAREST/FIFO, 4 robots):
10/10 delivered, **1/10 on time**, makespan 557 s, about 9,900 decision points/s on one core
(spec target ≥ 2,000; observations and masks for RL will cost some of that headroom).

**Stress check (not yet a pytest, that is M3):**
- **Setup:**
  - all 7 robot rules × 30 seeds × 2 and 4 robots
  - every fault on: permanent robot down with rescue, arm, machine, aisle block, rush order
  - MTBF robot failures without rescue
  - robot join
  - speed, arm and process noise
- **Results:** 0 exceptions, 0 deadlocks, and every released box in exactly one place at the
  end.
- With `t_max` raised to 5,000 s, every disrupted 4-robot run delivers all orders.

**Finding 1: the `abstract_10` due dates are infeasible in the cell.**
- Every order in `abstract_10` allows due − release = 135 s (the placeholder release rule).
- With the cell's overheads, one order with the whole cell to itself and an idle robot needs:

  | Route | Time alone |
  |---|---|
  | M | 134 s |
  | T→G, M→D | 175 s |
  | T→M→D | 234 s |

  The overheads per leg are dock 3 s, undock 1.5 s, arm 6 s + 6 s and conveyor 2 s, all spec
  defaults, plus 0.18 m/s on a 6 × 4 m floor.
- So **at most 3 of 10 orders (3, 6, 9) can ever be on time**, by under 1 s. `cell_10`
  on-time % will sit near 0–20% for every controller, which makes the comparison with the
  abstract env mostly noise.
- **Options:**
  - (a) keep `cell_10` as specified and compare tardiness rather than on-time %;
  - (b) add a `due_scale`/`due_offset` to `cell_10` (e.g. due = release + 2 × the cell work
    estimate) and keep an unscaled copy for reference;
  - (c) find the real release and process times (`ippo_delivery_10orders.yaml`), which may
    loosen things.
- **I recommend (b).** It is your call, because it changes what "same orders as `abstract_10`"
  means.

**Finding 2: the default `t_max` (2 × last due = 752 s) truncates normal runs.**
- 2 robots need about 905 s for `cell_10`.
- Disrupted 4-robot runs need 720–860 s.
- So 2-robot and disrupted episodes end with orders undelivered because of the clock, not the
  policy.
- **Recommendation:** default `t_max` = 3 × last due, or a fixed 1,500 s, so truncation means
  "something went really wrong".

**Finding 3: the bottleneck moves from milling to transport.**
- In the abstract env milling was 71% busy and robots were 60% idle.
- In the cell with 4 robots and NEAREST/FIFO, robot time splits into:

  | Docking, waiting for the arm, and handoffs | Idle | Empty travel | Loaded travel |
  |---|---|---|---|
  | 44% | 29% | 14% | 13% |

- Machines are about 27% busy and arms 11%. Who goes where, and when, matters much more
  here, which is the case the spec expects learning to help with.

**Choice made in M2: idle rule robots wait instead of parking.** A rule robot with nothing
to do now waits near its station (`idle="wait"`). Driving back to parking and returning cost
about 40 s per order. `idle="park"` remains available.

**Not done in M2 (by design):** observations, rewards, the PettingZoo wrapper, Gantt and
replay. Those are milestones 3–5.
