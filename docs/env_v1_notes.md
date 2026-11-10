# env-v1: Smart Assembly Cell environment, frozen

Spec: `Context/CELL_ENV_SPEC.md`. Recon and design decisions: `docs/cell_env_recon.md`.
Results: `results/summary.md` (all tables with 95% CIs and paired p-values),
`results/heatmap_on_time.png`, one CSV per controller and scenario in `results/<policy>/`.

Everything learned from here on is trained and evaluated against this tag. Any change to the
simulator, observation, action space, reward or scenarios means a new env version and a rerun
of the baselines. Treat the cell code as frozen at `env-v1`: `src/som/cell_env/`,
`src/som/config/cell_schema.py`, `configs/cell/` and `configs/cell_layouts/`.

**All station positions, arm/dock times and process times are still placeholders** until they
are measured in Isaac Sim or on the robots. The numbers below compare controllers on the same
placeholder cell. They are not predictions for the real cell.

## 1. The simulator

- **Discrete-event simulation** (`des.py`): a heap of timed events. Robots only decide at
  decision points: when they become free, when the task board changes, or when a WAIT runs out
  (`wait_cap_s` = 10 s).
- **Task board, first claim wins**: each robot sees the board's top K = 8 tasks, ordered by due
  time and then order number. Everything after a claim is automatic: A* path on the roadmap
  (with turn penalties), edge reservations with capacity, FIFO dock queue, arm handoff, undock.
- **Stations**: input and output buffers with capacities. A finished box that finds the output
  buffer full blocks the machine. One arm per station serves robots FIFO.
- **Disruptions** are drawn at reset from their own SeedSequence child streams. A controller's
  choices never change what disruptions happen. The disruption types are:
  - robot down (permanent, with rescue of the box it carried)
  - arm faults and machine faults
  - aisle blocks
  - rush orders
  - coordinator outages (these only affect the central baselines)
  - robots joining late (`cell_random_10_flex`)
- **Handling block** (`configs/cell_layouts/sac_6x4.yaml`, `HandlingCfg`): every arm, dock and
  conveyor time is in one place, and a scenario can override the whole block.

  | Time | Default |
  |---|---|
  | `arm_load_s`, `arm_unload_s` | 6 s |
  | `dock_time_s` | 3 s |
  | `undock_time_s` | 1.5 s |
  | `conveyor_transfer_s` | 2 s |

## 2. Agent interface (PettingZoo ParallelEnv, `cell_env.py`)

### Actions

`Discrete(K + 3)` with K = 8:

| Index | Meaning |
|---|---|
| `0..K-1` | claim board task i |
| `K` | go park |
| `K+1` | wait where it is |
| `K+2` | NOOP |

- The mask is in `infos[agent]["action_mask"]`.
- A robot with no decision to make (busy, down or absent) gets only NOOP.
- A robot that does have a decision never gets NOOP.
- A masked action raises an error.
- `terminations` (all orders delivered) and `truncations` (`t_max` reached) are kept separate.

### Observation v2 (`CELL_OBS_VERSION = 2`, env-v1.2; v1 had no `best_free_bid`)

The size is fixed and doesn't depend on the robot count (tested for 2, 3 and 4 robots):

| Mode | Observation size |
|---|---|
| `board` (default) and `local` | 292 (v1: 284) |
| `global` (an upper-reference ablation) | 652 (v1: 644) |

The critic state is 1528 dims in `board`/`local` mode and 2968 in `global` mode (v1: 1496 and 2936).

| Block | Contents |
|---|---|
| self | x, y, state one-hot, battery, agent-id one-hot (`max_robots` = 4), time, docked |
| tasks (K rows) | valid, travel_to_pickup, travel_pickup_to_dest, slack, remaining_ops, remaining_proc, dest_in_occupancy, dest_busy, dest_time_to_free, dest_arm_queue, waiting_time, due_in, dest_is_bottleneck, best_other_bid, pickup_here, best_free_bid (v2) |
| stations (one row each) | in_occupancy, out_occupancy, machine_busy, machine_blocked, machine_down, time_to_free, arm_busy, arm_down, robots_at_dock, inbound, queue_work |
| others (`max_robots` - 1 rows) | present, dx, dy, state one-hot, destination one-hot, eta_free, up |
| global | orders done, on time so far, past due, time to next release |
| orders (`global` mode only) | per order: present, stage, location one-hot, slack |

- **Scaling:** times are seconds / 200, clipped to ±10; counts are divided by 4.
- **What the features use:** only things a TurtleBot can know: its own state, the task board,
  and the station and robot broadcasts allowed by spec §1a.
- **`local` mode** hides robots outside a radius.
- **Decentralized policies never receive the global state.** The harness passes
  `global_state=None` to them, and a test checks this.

### Reward (team reward, named terms, `reward.py`)

| Term | Value |
|---|---|
| `on_time` | +10 per order delivered on time |
| `late` | -min(tardiness / 30, 5) per late order (also applied to undelivered orders at truncation, with tardiness up to `t_max`) |
| `past_due` | -0.01 per second per undelivered order past its due time |
| `feed_idle` (shaping) | +0.2 when a box lands in an idle machine's input buffer |
| `blocked` (shaping) | -0.05 per blocked trip |
| `empty_travel` (shaping) | -0.02 per second of empty travel |

With the shaping terms off, the summed team reward equals the episode objective exactly; this
is tested.

## 3. Scenarios (`configs/cell/`)

All scenarios use 4 robots unless the name says otherwise.

| Scenario | What it is | Purpose |
|---|---|---|
| `cell_10` | The abstract's 10 orders, deterministic, `due_k` = 1.50 | Main fixed benchmark |
| `cell_10_raw` | `cell_10` with the abstract's original due dates (release + 135 s) | Tardiness reference (see §4) |
| `cell_10_disrupt` | `cell_10` plus the spec §6 disruptions | Robustness |
| `cell_random_10` | 10 sampled orders, 25.6 s mean gap, speed noise σ = 0.05, process noise σ = 0.10 | Main random benchmark |
| `cell_random_10_disrupt` | `cell_random_10` plus disruptions | Robustness |
| `cell_random_10_flex` | Start with 2–3 robots; more join at U(60, 200) s | Claim C3: robots joining |
| `cell_random_20` / `_30` | 20 and 30 orders, 40 s mean gap | Scale |
| `cell_stream` | Poisson arrivals for 900 s, 40 s mean gap (about 23 orders) | Steady state |
| `cell_random_10` with `--robots 2` / `--robots 3` | Robot-count override (`_2r` and `_3r` in results) | Fleet size |

The cell sustains about 80 orders per hour, so the 40 s gap is about 90% load.

**Disruption rates** (all design choices, not measurements):
- One robot goes down permanently at U(80, 220) s; its box is rescued to the nearest station
  after 20 s.
- Each of these happens with p = 0.3 per episode: an arm fault of 20–60 s, a machine fault of
  30–90 s, an aisle block of 30–120 s.
- A rush order with p = 0.5.
- A coordinator outage of 30–90 s with p = 0.5.

**Seeds:** each split has fixed seeds. Every number in this file is from the 100 test seeds.

| Split | Seeds |
|---|---|
| train | 0–9999 |
| validation | 10000–10019 |
| test | 20000–20099 |

## 4. Due dates, k calibration and `t_max` (decision 1 and 2)

**Due date:** `due = release + k × min_flow_time(order)`.
- `min_flow_time` is the order's release-to-delivery time in an empty cell, with a robot
  already docked wherever the box is ready.
- Each leg costs load + undock + drive + dock + unload, and each machine step costs conveyor +
  process time.
- No controller can beat this lower bound.

**Calibration rule:** choose k so the best family-A rule gets about 40–50% of orders on time.
The script is `scripts/calibrate_cell_dues.py`:
1. Sweep a grid of k values.
2. At each k, run all 21 rule pairs.
3. Record the best pair's on-time %.

| Scenario | k | Best rule at calibration |
|---|---|---|
| `cell_10` (fixed orders, one seed) | **1.50** | FIFO/EDD, 50% (k = 1.48 gives 40%) |
| `cell_random_10` (generated, validation seeds) | **k ~ U(1.45, 1.95)**, i.e. 1.70 ± 0.25 per order | BOTTLENECK/SPT, 43% (49.7% on the test seeds) |
| `cell_random_20`, `_30`, `_stream`, `_disrupt`, `_flex` | the same U(1.45, 1.95) range, not re-tuned | — |

**`cell_10_raw`** keeps the abstract's original due dates, release + 135 s for every order.
In the cell, orders need 134–234 s on their own, so most are infeasible: every controller gets
10% on time. Use this scenario for **tardiness** only.

**`t_max` = 3 × the last due date.** An order not delivered by `t_max` counts as late, with its
tardiness measured up to `t_max`. Orders are never dropped from the metrics.

## 5. Arm handoff sensitivity (decision 3)

Arm load and unload times are set to 3, 6 or 9 s, with due dates still calibrated at 6 s.
Generated with `python scripts/arm_sensitivity_cell.py --policy <rule> --scenarios <scenario>`
(100 test seeds; `cell_10` is deterministic, so it uses one).
Robot time is split into dock / travel / idle.

| Scenario (best rule) | Arm s | On-time % | Robots busy | Dock / travel / idle | Busiest machine | Busiest arm | Bottleneck |
|---|---|---|---|---|---|---|---|
| `cell_10` (FIFO/EDD) | 3 | 20.0 | 66% | 38% / 29% / 34% | milling 63% | milling 9% | robot transport |
| `cell_10` (FIFO/EDD) | 6 | 50.0 | 71% | 43% / 28% / 29% | milling 61% | milling 17% | robot transport |
| `cell_10` (FIFO/EDD) | 9 | 20.0 | 73% | 48% / 24% / 27% | milling 57% | milling 24% | robot transport |
| `cell_random_10` (BOTTLENECK/SPT) | 3 | 50.2 | 64% | 37% / 27% / 36% | milling 62% | milling 9% | robot transport |
| `cell_random_10` (BOTTLENECK/SPT) | 6 | 49.7 | 70% | 44% / 25% / 30% | milling 59% | milling 17% | robot transport |
| `cell_random_10` (BOTTLENECK/SPT) | 9 | 46.4 | 74% | 50% / 24% / 26% | milling 55% | milling 24% | robot transport |

**Robot transport is the bottleneck at every arm time.** At 3 s it leads milling only narrowly
(64–66% against 62–63%). As handoffs get slower:
- docking takes a growing share of robot time (37% → 50%);
- milling utilisation falls, because boxes arrive at the mill later.

`cell_10` jumps from 20% to 50% to 20% because it is a single deterministic schedule, so a few
seconds can flip whether an order makes its due date. The 100-seed `cell_random_10` result
shows the smooth trend.

## 6. Baseline results (env-v1, 100 test seeds)

Each cell shows **on-time % / mean tardiness (s)**. Family A has 21 rule pairs (7 robot rules ×
3 machine rules); this table lists the best one per scenario. Family B is the contract-net
auction in three variants; the table lists the best. Family C is CP-SAT.
`results/summary.md` has confidence intervals, makespan, blocked trips, recovery time and
paired p-values.

| Scenario | Best rule (A) | Best auction (B) | CENTRAL_STATIC (C) | CENTRAL_REPLAN (C) |
|---|---|---|---|---|
| `cell_10` | FIFO/EDD 50.0 / 25 | AUCTION 20.0 / 36 | 80.0 / 25 | 80.0 / 25 |
| `cell_10_raw` | FIFO/EDD 10.0 / 107 | AUCTION 10.0 / 129 | 10.0 / 145 | 10.0 / 145 |
| `cell_10_disrupt` | SPT/EDD 15.9 / 92 | AUCTION/EDD 13.4 / 86 | 18.0 / 336 | 22.7 / 190 |
| `cell_random_10` | BOTTLENECK/SPT 49.7 / 29 | AUCTION/SPT 45.2 / 33 | 51.2 / 63 | 51.2 / 63 |
| `cell_random_10_2r` | NEAREST/SPT 21.0 / 130 | AUCTION/EDD 20.4 / 131 | 25.8 / 309 | 25.8 / 309 |
| `cell_random_10_3r` | NEAREST/SPT 34.7 / 51 | AUCTION/SPT 34.8 / 51 | 37.0 / 156 | 37.0 / 156 |
| `cell_random_10_disrupt` | SPT/SPT 26.8 / 74 | AUCTION/SPT 25.3 / 73 | 20.3 / 382 | 31.0 / 152 |
| `cell_random_10_flex` | NEAREST/SPT 40.3 / 45 | AUCTION/SPT 38.2 / 46 | 32.5 / 233 | 42.1 / 110 |
| `cell_random_20` | BOTTLENECK/SPT 47.8 / 36 | AUCTION/SPT 43.5 / 41 | 31.3 / 77 | 31.3 / 77 |
| `cell_random_30` | NEAREST/SPT 38.6 / 58 | AUCTION/SPT 34.9 / 65 | 22.3 / 162 | 22.3 / 162 |
| `cell_stream` | BOTTLENECK/SPT 43.2 / 51 | AUCTION/SPT 39.9 / 59 | 27.9 / 145 | 27.9 / 145 |

### What the table says

- **10-order scenarios:** the central scheduler gets the most orders on time.
  - `cell_10`: 80% against 50% for the best rule.
  - `cell_10_disrupt`: +6.8 points (p < 0.001).
  - `cell_random_10_disrupt`: +4.2 points (p = 0.02).
  - Nominal `cell_random_10`, `_3r` and `_flex`: the gain over the best rule is within noise
    (p > 0.2).
  - **Its mean tardiness is 2–5× higher.** The CP-SAT objective counts on-time orders first and
    total tardiness second, so once an order cannot make its due date the plan pushes it to
    the back.
- **20 orders and more:** the central scheduler falls back to its greedy list schedule (see
  §7), which is **worse than the rules** (−15 to −17 points).
- **Static versus replan:** the two match wherever nothing unplanned happens. Under
  disruptions, replanning is worth about +5 to +11 points (`_disrupt`, `_flex`).
- **The auction** is close to the best rule on random scenarios (+0.1 to −4.5 points). On
  `cell_10` it gets 20% against 50%.
- **Rule choice:** no single rule is best everywhere. BOTTLENECK, NEAREST, FIFO and SPT each
  win somewhere. This is the gap a learned policy should close, and there is clear headroom.
- **Disruptions** cost the rules, the auction and `CENTRAL_REPLAN` about 20–23 points on
  `cell_random_10`, and `CENTRAL_STATIC` 31 points.

## 7. Known limitations

- **Placeholders.** The layout, distances, process times, arm and dock times and disruption
  rates are guesses. Section 5 shows the qualitative picture (transport is the bottleneck)
  holds for arm times of 3–9 s.
- **Top-K board.** Every controller, central included, can only claim the board's top 8 tasks.
  This is an interface constraint shared by all methods, not a planning limit.
- **The central model ignores buffer capacities and aisle traffic.**
  - A plan can therefore deadlock the cell: every robot waiting on a box stuck behind a full
    output buffer.
  - Execution detects this (no robot and no machine busy, and no robot's next leg pending or
    claimable). Only then does the deciding robot take the claimable leg with the earliest
    planned start.
  - This is the only time execution leaves the plan. On `cell_10` it never fires.
- **Central falls back to a greedy schedule:**
  - When an order book has more than 35 legs (20+ orders), it uses its greedy list schedule
    (status `GREEDY`).
  - When CP-SAT returns no solution inside its time limit, it uses the same schedule (status
    `FALLBACK`).
  - Time limits: 60 s per fixed-scenario solve, 10 s per generated-scenario solve, 5 s per
    replan. Each solve uses deterministic time and 4 workers.
  - The solves are reproducible, but the plans are not proven optimal.
- **Central follows its plan strictly.**
  - A robot whose queue is empty waits even if tasks are on the board.
  - In 3 of the 100 `cell_random_10_2r` seeds (both central variants) the plan overloads one
    robot and the episode hits `t_max` with orders undelivered. Those orders count as late.
  - `CENTRAL_STATIC` never replans: 17 of 100 `cell_random_10_disrupt` seeds and 2 of 100
    `cell_random_10_flex` seeds end with undelivered orders.
- **Calibration** was tuned on `cell_10` and the `cell_random_10` validation seeds only. The
  larger and streaming scenarios reuse the same k range.

## 8. Later versions (additive; the simulator itself is unchanged)

**env-v1.1** adds four coordinator-outage scenarios for spec §12's "family C with and without
the outage":
- `cell_10_outage` and `cell_random_10_outage`: the nominal scenario plus one outage per
  episode (p = 1).
- `cell_10_disrupt_no_outage` and `cell_random_10_disrupt_no_outage`: `_disrupt` with the
  outage switched off. Every other disruption is identical seed by seed (tested).

Decentralized controllers never see the coordinator, so for them each pair is the same episode.
`results/summary.md` has a "Coordinator outage" table with paired differences.

**env-v1.2** (observation v2) adds one task feature, `best_free_bid`: the lowest bid among the
other *free* robots, i.e. their travel time to the pickup. It exists to fix the auction:
- `AUCTION` (and `/EDD`, `/SPT`) now lets only free robots bid.
- The env-v1 auction, where busy robots bid their time-to-free plus travel, stays registered as
  `AUCTION_v0`. Under it, a free robot often idled waiting for a busy robot that had out-bid
  it: robots were idle 36–40% of the time against about 30% for the rules.

Both auctions are tested to make exactly the decisions of a simulator-side reference.
- **Rules:** decide exactly as before (tested), so their results carry over. They were re-run on
  env-v1.2 anyway.
- **Central scheduler:** reads the simulator, not the observation, so its env-v1/v1.1 results
  stand.
- **Learned methods:** must train on observation v2. A v1 checkpoint is refused.

## 9. Provenance

- **Rule and auction results** record `env_version` `e621af1`.
- **Central results** record `9c41da7-dirty`. "Dirty" only means the old central CSVs had been
  deleted but not yet committed when the rerun started.
- **Code changes since `e621af1`:** `git diff e621af1 env-v1` touches no file under
  `src/som/cell_env/`, `src/som/config/` or `configs/cell*/`. The only change is
  `src/som/baselines/central_cpsat.py` in `9c41da7`:
  - The cached `cell_10` plan is now copied, so it isn't consumed by the first seed.
  - The deadlock break described in §7.
- **An earlier central run (before `9c41da7`) was wrong**, at 0.8% on time on `cell_10`. It was
  replaced.

**To reproduce:**
```
python scripts/evaluate_cell.py --policy baselines --workers 3
python scripts/evaluate_cell.py --policy baselines --scenarios cell_random_10 --robots 2
python scripts/evaluate_cell.py --policy baselines --scenarios cell_random_10 --robots 3
python scripts/compare_cell.py
```
