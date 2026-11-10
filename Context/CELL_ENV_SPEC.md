# Spec: `cell_env` — 4-TurtleBot Smart Assembly Cell for MARL + Baselines

**Audience:** Claude Code, working inside the `MARL_new` repo.
**Owner:** Design-is-ALL Lab (CSUF), Learning Team.
**Purpose:** Build a new, more realistic training environment that mirrors the Isaac Sim
Smart Assembly Cell (conveyors + Franka Panda arms + 4 TurtleBot3 AGVs). Then train one or
more learning methods (IPPO/MAPPO, evolved rules, …) and run the baselines on it, so we can compare it against the current
abstract environment (`abstract_10`, `random_10`, `*_disrupt`).

**Research goal (this outranks every other section):** show **self-organizing manufacturing**
with 4 TurtleBot3s. No central scheduler assigns the work. The robots choose their own tasks,
keep production on time, and adapt in real time when something changes. How the policies are
trained, and which algorithm is used, is flexible as long as §1a is met. If a detail in this
spec gets in the way of §1a, change the detail.

---

## 0. Ground rules for Claude Code

1. **Read before writing.** Before you write any code, read `README.md`, `envs/`, `trainer/`, `configs/`
   and whatever currently produces the `abstract_10` / `random_10` / `*_disrupt` scenarios,
   the dispatching-rule heuristics (`NEAREST/EDD`, `NEAREST/FIFO`, `FIFO/SPT`, `NEAREST/SPT`) and
   the Gantt and learning-curve plots. Write a short summary of what you find to
   `docs/cell_env_recon.md`. The README describes an older, low-level version of the env. The
   current runs use **decision points** (event-driven), so treat the code as the source of truth.
2. **Do not break the existing env.** Add the new env next to it (`envs/cell_env/`). Reuse
   the existing trainers, heuristics and plotting code wherever you can. If something has to be
   generalised, refactor it behind a shared interface and keep the old callers working.
3. **Same API as the existing multi-agent env.** Use the same dict-keyed API: agent ids
   `agv_1..agv_4` and `step(actions) -> (obs, rewards, dones, infos)`. Action masks go in
   `infos[agent]["action_mask"]`, or wherever the current trainers expect them.
4. **Reproducible:** every source of randomness goes through one seeded `np.random.Generator`.
   Each scenario has fixed train, validation and test seed lists.
5. **Fast:** the env is a discrete-event simulation with no physics. It should reach at least 2,000
   decision points/s on one CPU core, because we train for about 1.5M decision points per seed.
6. Work through the milestones in §13 in order. After each one, run the tests and commit.

---

## 1a. What "self-organizing" means here (the claims we must be able to show)

| # | Claim | How the env/experiments show it |
|---|---|---|
| C1 | **No central scheduler.** Nothing assigns tasks to robots at run time. Each robot decides for itself. | Robots pick work from a shared **task board** (§7.2). |
| C2 | **Adapts in real time.** Disruptions are handled without retraining, reprogramming or a human. | Disruption scenarios (§6). Measure the on-time drop vs nominal and the recovery time (§12). |
| C3 | **Flexible fleet.** The same policy works with 2, 3 or 4 robots, and when a robot drops out or joins mid-shift. | Robot-count-agnostic observations (§7.3); scaling and join/leave tests (§12). |
| C4 | **Better than the alternatives where it matters.** Beats dispatching rules and is competitive with a self-organizing auction. It is **more robust than a central scheduler**, which may win when nothing goes wrong but breaks under disruptions and stops entirely if the coordinator goes down. | Three families of baselines (§10). |

**Allowed shared infrastructure.** A self-organizing factory still has a network. These are
allowed at run time, because they share information but never make decisions:
- Stations broadcast their status (buffers, machine/arm up or down) and post open
  transport tasks to a **task board**.
- A robot posts its claim to the board. The first claim wins, like a job board, and a claim
  can be dropped.
- Robots can broadcast their own position, state and claim.

**Not allowed at run time:** any component that computes *who does what* for the whole fleet.
That is what the central-scheduler baselines in §10 are for.

**Training is unrestricted.** A global critic, centralized training, curriculum, reward shaping,
imitation warm-start from the best heuristic, population or self-play tricks are all fine. Use
whatever gets a robust policy that satisfies C1–C4.

---

## 1. Where we are now (the baseline to beat)

Current env ("abstract"): a 4.0 m × 1.75 m 2D map with `SRC`, `TURNING`, `GRINDING`,
`MILLING`, `DRILLING` and `SINK` as point stations. AGVs move single orders between stations
and there is no handoff mechanism, so a pickup or drop takes zero time. There are 10 orders.
Each order has a route through a subset of the machines, and a due time (dotted lines on the
Gantt charts).

Typical process times (read off the Gantt charts): turning ≈30 s, grinding ≈20 s,
**milling ≈40 s (the bottleneck: most routes include it)** and drilling ≈10 s.

Latest results (on-time %; MAPPO uses 10 eval seeds and the heuristics use 20 seeds):

| Scenario | Robots | Best heuristic | MAPPO (late training) | Note |
|---|---|---|---|---|
| `random_10` | 2 | 29 (NEAREST/SPT) | ~34–35 | clear, steady gain |
| `abstract_10` | 4 (disrupt-trained) | 43.5 (NEAREST/FIFO) | swings between 42 and 48 | unstable: the policy flips between two schedules |
| `abstract_10_disrupt` | 4 | 24.5 (NEAREST/FIFO) | ~33–36 | largest gain |
| `random_10_disrupt` | 4 | 32 (FIFO/SPT) | ~31–34 | roughly a tie |

Lessons to carry into the new env:
- Learning helps most **under disruptions** (a robot going down, blocked trips).
- On the deterministic nominal case the policy swings between ~42% and ~48%. Add
  validation-based **best-checkpoint selection**, LR/entropy annealing and more eval seeds.
- The Gantt charts show many empty (hatched) trips, and milling sets the pace. The
  observations need bottleneck-aware features (milling queue, time-to-free).
- With 10 orders × 10 seeds, every on-time % reading is in 1% steps and noisy. Report
  means with 95% CIs and add larger scenarios (20 and 30 orders).

---

## 2. The physical cell being modelled

This comes from the Isaac Sim 5.0 build ("Smart Assembly Cell", SAC). A walled cell holds:

- An **inbound conveyor and raw-material bins** (= `SRC`). A Franka arm picks a box/part
  off the conveyor and **places it on top of a TurtleBot3**.
- **Four production cells**: Turning, Grinding, Milling and Drilling. Each has a red work
  bin/buffer, a machine and a **Franka Panda arm** that unloads incoming boxes from the
  robot's top plate and loads finished boxes back onto a waiting robot.
- A **product shelf / outbound conveyor** (= `SINK`), where an arm unloads finished
  orders.
- An **AGV parking/charging zone**.
- Aisles between the cells, plus walls and obstacles that the robot LiDARs see. Some aisles are
  only wide enough for one TurtleBot.

**The core loop:** an arm places a box on a robot → the robot drives to the next station on that
order's route → it docks → that station's arm picks the box off → the machine processes it →
the arm places it on whichever robot is assigned to the next leg → … → `SINK`.

What the abstract env does **not** model, and `cell_env` must add:

| Feature | Why it matters |
|---|---|
| Arm handoff time at load and unload, with one arm serving one dock at a time | Robots queue at docks, so "nearest robot" is no longer free |
| A finite dock per station and finite input/output buffers | Blocking: if a machine's output is full, it stops |
| A roadmap with aisle capacity (single-lane segments) | Congestion and blocked trips, as on the real floor |
| Docking/alignment time and turning cost | Short trips are not free |
| Parking/charging zone, with optional battery | Idle robots have somewhere to go and charging can become a decision |
| Richer disruptions (robot, arm and machine faults, blocked aisle, rush orders) | This is where MARL is supposed to win |

---

## 3. Layout (default, fully configurable in YAML)

Default floor: **6.0 m × 4.0 m**. Every coordinate lives in `configs/cell/*.yaml` so we can
paste in real coordinates from the Isaac Sim USD stage, or from the physical testbed later.

```
 y(m)
 4.0 +---------------------------------------------------------------+
     |  [PK1] [PK2] [PK3] [PK4]   parking / charging                 |
 3.4 |    o-----o-----o-----o----------------------------------o     |
     |          |                                              |     |
 2.8 |   +--------------+   ║   +--------------+               |     |
     |   | TURNING  arm |   ║   | GRINDING arm |               |     |
     |   +-----[dT]-----+   ║   +-----[dG]-----+               |     |
 2.0 |=IN==o=====o==========o========o======== main aisle =====o==OUT|
 SRC |  [dS]     |    (2 lanes, 2-way)     |                   [dK]  | SINK
 conv|   +-----[dM]-----+   ║   +-----[dD]-----+               shelf |
     |   | MILLING  arm |   ║   | DRILLING arm |                     |
 1.2 |   +--------------+   ║   +--------------+                     |
     |                      ║  ← single-lane cross aisle (cap 1)     |
 0.0 +---------------------------------------------------------------+
     0                                                             6.0 x(m)
 [dX] = dock pose for station X     o = roadmap node     ║ = capacity-1 edge
```

The roadmap is a graph `G = (V, E)`:
- **Nodes:** one dock per station (`dS, dT, dG, dM, dD, dK`), aisle intersections, and 4 parking spots.
- **Edges:** each has a `length_m`, a `capacity` (1 = single lane, 2 = two-lane) and `directed: bool`.
- Robots follow the shortest time path, computed with A* on edge travel times and cached per (src, dst).

---

## 4. Entities and parameters

### 4.1 TurtleBot3 (AGV) × 4 (config: `n_robots`, which must also support 2 and 3)
| Param | Default | Note |
|---|---|---|
| `v_cruise` | 0.18 m/s | Burger max ≈ 0.22 m/s |
| `turn_penalty_s` | 1.5 s per heading change > 45° | |
| `dock_time_s` | 3.0 s | align under the arm |
| `undock_time_s` | 1.5 s | |
| `payload` | 1 box | |
| `speed_noise` | lognormal σ = 0.05 | 0 in deterministic scenarios |
| `battery` (optional, off by default) | 100%, −0.15 %/s moving, −0.03 %/s idle, +1.0 %/s at charger | force a charge below 15% |

Robot states: `IDLE_PARKED`, `TRAVEL_EMPTY`, `WAIT_DOCK`, `DOCKING`, `BEING_LOADED`,
`TRAVEL_LOADED`, `BEING_UNLOADED`, `CHARGING`, `DOWN`.
These states map onto the Gantt colours (hatched = empty travel, solid = loaded, grey = down).

### 4.2 Station (SRC, 4 machines, SINK)
| Param | Default |
|---|---|
| `dock_capacity` | 1 robot docked; others wait in a FIFO queue at the dock's approach node (also configurable as 2 docks: separate input and output) |
| `in_buffer_cap` | 3 (machines), ∞ (SINK) |
| `out_buffer_cap` | 2 (machines), ∞ (SRC; boxes appear there as orders are released) |
| `arm_load_s` / `arm_unload_s` | 6.0 / 6.0 s, with ±10% triangular noise in random scenarios |
| Arm | serves **one** handoff at a time; FIFO over requests |
| Machine | 1 job at a time; process time from the order; when it finishes the box moves to the out-buffer. **If the out-buffer is full, the machine is blocked** |
| `machine_seq_rule` | how the machine picks from its in-buffer: `FIFO` / `EDD` / `SPT` / `learned` (default `FIFO`) |

### 4.3 Orders / boxes
- Fields: `id, job_type, route[list of processes], proc_times[list], release_t, due_t, priority`.
- Routes are any ordered subset of `[Turning, Grinding, Milling, Drilling]`, always ending at `SINK`.
- An order (box) is released into the SRC out-buffer at `release_t` (the inbound conveyor).

### 4.4 Conveyors
Model these only as the buffers above, plus an optional `conveyor_transfer_s` (default 2 s)
between the arm and the machine. We do not simulate belt physics.

---

## 5. Dynamics (discrete-event simulation)

- Continuous time and a priority-queue event loop. Event types: `ROBOT_ARRIVE_NODE`,
  `DOCK_DONE`, `HANDOFF_DONE`, `MACHINE_DONE`, `ORDER_RELEASE`, `FAILURE`, `REPAIR`,
  `EDGE_FREE`.
- **Traffic:** a robot reserves the next edge (or edge segment) before entering it. If the edge is
  at capacity, the robot waits at the node and we log a `blocked_trip` (the red ✕ on the
  Gantt). Detect deadlocks (a cycle of waits), resolve them by having the lowest-id robot reroute,
  and log each one.
- **Decision points:** the env returns control to the policy whenever a robot
  becomes free: it has finished an unload, finished a repair, or finished charging, or it is idle
  and something new has happened (an order was released, or a box landed in an out-buffer). Only
  those robots act at that point. All others get a masked `NOOP`. A "training step"
  is **one decision point**, the same unit as the current learning curves.
- An episode ends when all orders reach SINK, or at `t_max` (default `2 × last due time`).

---

## 6. Disruptions (the `*_disrupt` scenarios)

All of these are configurable and seeded:

| Event | Default in `_disrupt` |
|---|---|
| Robot breakdown | one robot goes `DOWN` at t ~ U(80, 220) s for the rest of the episode (matches the current runs); also an MTBF mode: exp(600 s) with repair ~ U(30, 90) s |
| Robot carrying a box when it fails | the box stays on the broken robot until repair (or option `rescue: true`: the box is teleported to the nearest station's in-buffer after 20 s) |
| Arm fault | p = 0.3 per episode; one arm unavailable for U(20, 60) s |
| Machine fault | p = 0.3; one machine down for U(30, 90) s, and its in-progress job resumes afterwards |
| Aisle blocked | p = 0.3; one capacity-1 edge closed for U(30, 120) s |
| Rush order | p = 0.5; one extra order with tight slack inserted at a random time |
| Process-time noise | lognormal σ = 0.1 (only in random scenarios) |
| Robot joins mid-shift | (`_flex` scenarios) start with 2–3 robots and add one at t ~ U(60, 200) s |
| Coordinator outage | p = 0.5; the central scheduler (baseline family C only) is unreachable for U(30, 90) s. Its robots finish their current task, then wait. The task board is unaffected, because in this test it is spread across the stations |

The policy **must not** see future failures. It sees only the current up/down status.

---

## 7. Agents, actions, observations

### 7.1 Who learns
- **Default:** the 4 robots learn (shared-parameter actors with an agent-id input). Machines
  use `machine_seq_rule`.
- **Option:** `machine_as_agent: true` adds learned machine sequencers, which keeps the existing
  feature. Treat this as phase 2.

### 7.2 Robot action space (discrete, masked)
At a decision point, robot *i* chooses one of these. This is the C1 mechanism: the board lists
work and each robot decides for itself.
- `0..K-1`: **claim transport task k** from the top-K open tasks on the task board (K = 8). A task means
  "pick up box *b* waiting at station A's out-buffer (or SRC) and deliver it to its next station
  B". Candidates are sorted by a fixed, policy-independent key (earliest due) and padded and
  masked when there are fewer than K.
- `K`: **go park/charge**.
- `K+1`: **wait** (stay put until the next decision point; capped so it cannot loop forever).

Masks rule out: tasks already claimed by another robot, tasks whose destination in-buffer is
full (including boxes already claimed and heading there), every action while the robot is
`DOWN`, and `park` when the robot is already parked.

Once a task is claimed, everything else is automatic: path planning, traffic, docking, waiting
for the arm and delivery.

### 7.3 Observation per robot (fixed size, robot-count agnostic)
- **Self:** position (x, y normalised), state one-hot, battery, `agent_id` one-hot (size 4,
  padded), and the time since the episode started (normalised).
- **Per candidate task k (K rows):** travel time to pickup, travel time pickup→dest,
  slack = `due − now − est_remaining_time`, remaining ops count, remaining processing
  time, destination in-buffer occupancy, destination machine busy flag and time-to-free,
  destination arm queue length, and a valid flag.
- **Station summary (6 rows):** in/out buffer occupancy, machine busy/blocked/down,
  time-to-free, arm busy/down, number of robots waiting at the dock, and inbound claimed boxes.
  **Highlight the bottleneck:** include the milling queue's total remaining work.
- **Other robots (3 rows, padded):** relative position, state, the destination they
  have claimed, ETA, and up/down.
- **Global:** orders done, on-time so far, number past due, and the next release time.

Every item above is something a real TurtleBot could get from its own sensors, the task
board or the station and robot broadcasts in §1a. So the trained actor can run on the robot
with no central service. Add a config switch `obs_mode`:
- `board` (default): exactly the list above.
- `local`: other robots are visible only within `sense_radius_m` (default 1.5 m).
- `global`: adds the full order-status matrix. Use it as an upper-reference ablation only.

**Centralized critic state (MAPPO, training only):** the concatenation of all robots'
observations plus a global order-status matrix (orders × [stage, location, slack]).

---

## 8. Reward (team reward, with shaping that can be turned off)

The primary signal matches the metric we report:
- `+10` for each order delivered **on time**, and `−1 × min(tardiness_s/30, 5)` for each late order.
- A `−0.01` per-second cost for each order that is past due and not yet delivered (dense lateness signal).
- Shaping (each term has its own config key, so we can ablate it): `+0.2` when a box lands in a
  machine in-buffer whose machine is idle; `−0.05` per blocked trip; `−0.02 × empty_travel_s`.
- Team reward: everyone receives the sum. There is an option to also log the per-robot share,
  for IPPO comparisons.

Keep `reward.py`-style named terms, so every term is logged separately to TensorBoard.

---

## 9. Scenarios

Each scenario is one YAML file in `configs/cell/`:

| Name | Orders | Robots | Randomness | Purpose |
|---|---|---|---|---|
| `cell_10` | the same 10 orders, routes and due times as `abstract_10` | 4 | none | direct comparison with the old env |
| `cell_10_disrupt` | as above | 4 | robot down + §6 | comparison with `abstract_10_disrupt` |
| `cell_random_10` | 10 sampled | 4 (also 2) | routes, times, dues | comparison with `random_10` |
| `cell_random_10_disrupt` | 10 sampled | 4 | + §6 | comparison with `random_10_disrupt` |
| `cell_random_20`, `cell_random_30` | 20, 30 | 4 | + optional disrupt | lower-variance metric, more congestion |
| `cell_random_10_flex` | 10 sampled | starts at 2–3, then one joins | + robot join | fleet flexibility (C3) |
| `cell_stream` | Poisson arrivals for 900 s | 4 | full | steady-state throughput test |

**`cell_10` orders:** copy them from the existing `abstract_10` config, which is the source
of truth. As a cross-check, the routes read off the Gantt charts are:
1 T→G, 2 T→M→D, 3 M, 4 M→D, 5 T→G, 6 M, 7 T→M→D, 8 M→D, 9 M, 10 T→M→D, with
due times 146, 166, 196, 236, 266, 266, 296, 326, 346, 376 s.

**Random order generator:** route = a random job type, drawn from the same job-type pool
as `random_10`. Process times are drawn around these means: T 30, G 20, M 40, D 10 s
(±25% uniform). Releases follow Poisson arrivals at a configurable rate, or are evenly spaced.
Due = release + k × (Σ proc + est. travel + handoffs), with k ~ U(1.5, 3.0) (TWK rule).

**Seeds:** train seeds 0–9999, validation seeds 10000–10019 (used to select checkpoints),
test seeds 20000–20099 (used for the final numbers only).

---

## 10. Baselines (must run on every scenario)

There are three families. Together they show what the learned policy is better than (C4).

### A. Dispatching rules (decentralized, hand-designed; what we have today)
Each baseline is a pair: robot dispatching rule × machine sequencing rule, with the same
naming as today (`ROBOT/MACHINE`).

**Robot rules (choose a task when a robot frees up):**
- `NEAREST`: shortest travel time to pickup
- `FIFO`: the box that has waited longest
- `EDD`: earliest due
- `SPT`: shortest remaining processing time
- `MST`: minimum slack
- `BOTTLENECK`: prefer boxes whose next station is milling when milling would otherwise starve, otherwise `NEAREST`
- `RANDOM`

**Machine rules:** `FIFO`, `EDD`, `SPT`.

Run the full grid (7 × 3) on 100 test seeds and report the best combination per scenario.
This is the dashed "best heuristic" line on the learning curves.

### B. Self-organizing auction (decentralized, hand-designed: the classic non-learning alternative)
`AUCTION` (contract-net): when a task is posted, every free robot bids its estimated
completion time (travel + queue + handoff), and the lowest bid wins. Robots that are busy bid
their time-to-free plus that estimate. Rebid when a robot goes down. It runs on the same task
board, so the comparison with MARL is fair.

### C. Central scheduler (centralized: the thing the lab argues is brittle)
- `CENTRAL_STATIC`: CP-SAT (OR-Tools) computes a full plan at t = 0 from global information,
  with handoffs and dock capacity but without aisle traffic. It then executes the plan
  open-loop. On deterministic `cell_10` it is the near-optimal upper reference (60 s solve
  limit; report the best solution found).
- `CENTRAL_REPLAN`: the same model, re-solved after every disruption or new order (5 s limit,
  warm-started). It is subject to the **coordinator outage** in §6.

Expected story, which the experiments must test rather than assume: C wins or ties when
nothing goes wrong. MARL ≥ A and B overall, and MARL ≥ C under disruptions, rush orders and
coordinator outages.

---

## 11. Learning methods (method-agnostic)

**No particular algorithm is required.** Any method is acceptable if the deployed robots
choose their own tasks from local/board information (C1) and it is evaluated with the shared
harness below. Several methods can be developed in parallel on separate branches (§15).

### 11.1 The common policy interface (built on `main`, used by everything)
Every controller implements one interface: rules, auction, central scheduler and every
learned method.
```python
class Policy(Protocol):
    name: str
    decentralized: bool          # True → act() receives only obs + masks, never global state
    def reset(self, scenario_info: dict) -> None: ...
    def act(self, obs: dict[str, np.ndarray], masks: dict[str, np.ndarray],
            global_state: dict | None = None) -> dict[str, int]: ...
```
- The evaluation harness passes `global_state=None` whenever `decentralized=True`. A test
  enforces this, so C1 is checked by code rather than by promise.
- `scripts/evaluate.py --policy <registry-name> [--checkpoint PATH] --scenarios ... --split test`
  writes `results/<policy>/<scenario>.csv` with fixed columns (seed, on_time_pct, mean_tardiness,
  makespan, utilisations, blocked_trips, recovery_time, ...) and Gantt PNGs for seed 0.
- `scripts/compare.py` reads every `results/*/` folder and builds the report and plots in §12.
  A new method shows up automatically once its results exist.

### 11.2 Fair-comparison rules
- Every method runs on the same frozen env version (git tag, e.g. `env-v1`) and the same test seeds.
- **Same simulation budget:** at most 1.5M env decision points per training run, whatever
  the method (gradient steps, GP generations, etc.). Report wall-clock time as well.
- 3 independent training runs per method. Select on the 20 validation seeds and report on the
  100 test seeds.
- Train on the scenario mix you choose, but report every scenario in §9, including ones the
  method did not train on (generalisation is part of C2/C3).

### 11.3 Recommended methods (pick one or more; each can be its own branch)

**Track 1: PPO family (neural, multi-agent RL). Branch `method/ppo`.**
- **IPPO and MAPPO are the same code with one switch:** `critic: local | centralized`. Run
  both and let the results decide. IPPO is the purest self-organization story, because nothing
  is centralized even in training. MAPPO's global critic usually helps with credit assignment
  under a shared team reward. Reuse `trainer/ippo_trainer.py` / `mappo_trainer.py`.
- **Shared parameters** across robots. This is required for C3 (a different robot count
  with no retraining).
- **Actor architecture: candidate scoring.** One small shared network gives each of the K
  candidate tasks a score from (task features ⊕ self features ⊕ pooled station/other-robot
  context), plus separate logits for park and wait, then a masked softmax. It doesn't depend on
  the order of the candidates or on how many robots there are, and it generalises better than
  a flat MLP over a concatenated vector.
- **Behaviour-cloning warm-start** from the best family-A rule (about 50k decisions), then PPO
  fine-tuning. This addresses the unstable swings seen on `abstract_10`.
- Starting hyperparameters: lr 3e-4 with linear decay to 3e-5; entropy coefficient 0.01
  decaying to 0.001; γ 0.995; λ 0.95; clip 0.2; 4,096 decisions per rollout; 4 epochs; minibatch 512;
  obs normalisation on. Evaluate every 20k decisions and keep the best-on-validation checkpoint.
- Optional curriculum: `cell_10` → `cell_random_10` → `_disrupt` → `20`/`30`.

**Track 2: evolved dispatching rules (genetic programming). Branch `method/gp-rules`.**
- GP evolves a **priority formula** over the same per-task features the RL policy sees
  (slack, travel time, remaining work, destination queue, milling load, robots already
  heading there, …). Each robot scores the open tasks with the formula and claims the best
  one (or parks if every score is below an evolved threshold).
- Why it suits this project:
  - It is fully decentralized and **interpretable**: the result is a one-line formula that
    can run on a real TurtleBot.
  - It is cheap: CPU only and highly parallel.
  - It is an established, strong method in dynamic job-shop scheduling research, so reviewers
    will expect to see it.
  - It matches the lab's "self-evolving" framing.
- Setup: tree GP (`DEAP` is fine). Population 200–500, about 50 generations, tournament
  selection, parsimony pressure (a depth limit ≤ 8). Fitness = mean on-time % across a batch
  of training seeds that mixes nominal and disrupt, with mean tardiness as a tie-breaker.
  Re-sample the seed batch every generation. Count simulated decisions against the 1.5M budget.
- Deliverables: the best formula written out in `docs/method_gp-rules.md`, plus an ablation of
  which features the best rules use.

**Optional Track 3: learned auction. Branch `method/learned-auction`.**
The contract-net auction from family B, with the bid computed by a learned network (trained
with PPO or GP). It is the most literally "self-organizing" design (a market whose agents learn
to value tasks). Only start it if Tracks 1–2 are running smoothly.

**Not recommended first:** value-decomposition methods (QMIX/VDN). Only some robots act at
each decision point, which makes the mixing network awkward. The repo is also built on PPO, so
this is a lot of work for little expected gain.

**Transfer test (any track):** run a controller trained or evolved on the old abstract env
on `cell_*`, and the reverse. This answers "how much does realism change the policy?"

---

## 12. Metrics, plots and the comparison report

**Per episode (log to CSV and TensorBoard):**
- on-time %, mean and max tardiness, makespan, throughput
- robot utilisation split into loaded / empty / waiting-at-dock / blocked / idle / down
- arm utilisation, machine utilisation and blocked time
- number of blocked trips, number of deadlocks resolved

**Plots (reuse the current style):**
1. **Gantt:** machine rows (Turning, Grinding, Milling, Drilling) and one row per robot. Solid =
   loaded, hatched = empty travel, a dotted outline = waiting at dock/arm (new), grey cross-hatch =
   down, red ✕ = blocked trip. Coloured by order id, with dotted vertical lines at due times. The
   title format is `<policy> s<seed> · <scenario>`.
2. **Learning curves:** on-time % against decision points, one line per training seed, and a dashed
   line for the best heuristic. Add a shaded validation band.
3. **Utilisation stacked bars** per robot: the best heuristic vs each learned method.
4. **The new 2D replay animation**, like the current GIF: floor plan, roadmap, docks, arms
   (busy/idle), buffers and robots with the box colour on top, plus the order table on the right.
   Exported as a GIF/MP4.

**Self-organization experiments (C2, C3):**
- **Robustness:** on-time % on nominal vs `_disrupt` for every method, and the % drop.
  A smaller drop = more resilient.
- **Recovery time:** after each disruption event, the seconds until the rolling throughput is back within 10% of its
  pre-event level.
- **Fleet scaling:** train once, then evaluate with 2, 3 and 4 robots, and with a robot
  leaving (`_disrupt`) or joining (`_flex`). Plot on-time % against robot count for each method.
- **Coordinator outage:** family C with and without the outage, against the learned methods.
  This is the single-point-of-failure result.
- Summary figure: methods × {nominal, disrupt, flex, outage} heatmap of on-time %.

**Comparison report:** `results/cell_vs_abstract.md`, a table with mean ± 95% CI over the test
seeds:

| Scenario pair | Best heuristic (abstract) | Current MAPPO (abstract) | Best heuristic (cell) | AUCTION (cell) | CENTRAL_REPLAN (cell) | each learned method (cell) … |
|---|---|---|---|---|---|---|

Add a Welch t-test or bootstrap p-value for each learned method vs the best heuristic and vs
`AUCTION`. Then write 3–5 bullet findings: which method works best where, where learning
helps, and what the added realism changed.

---

## 13. Milestones (do them in this order)

### On `main` (one Claude Code session; shared by every method)

1. **Recon:** `docs/cell_env_recon.md` describes how the current env, heuristics, trainers and
   plots fit together, with the files and entry points to reuse.
2. **Core DES:** stations, arms, buffers, robots and the roadmap with A* and edge reservation, with no
   learning yet. `python -m envs.cell_env.demo --config configs/cell/cell_10.yaml` runs
   the `NEAREST/FIFO` heuristic end to end and prints a summary.
3. **Tests (pytest, `tests/test_cell_env.py`):**
   - box conservation: every order is in exactly one place at every time
   - never more than `capacity` robots on an edge, never more than 1 robot docked per dock
   - an arm performs ≤ 1 handoff at a time; buffer caps are never exceeded
   - the same seed gives an identical trajectory; heuristics are deterministic on `cell_10`
   - masks: a masked action is never executed; a `DOWN` robot only gets `NOOP`
   - speed benchmark ≥ 2,000 decision points/s
4. **Visualisation:** Gantt and 2D replay GIF, matching the current style.
5. **Policy interface and evaluation harness** (§11.1): `Policy`, a registry,
   `scripts/evaluate.py` and `scripts/compare.py`, and a test that decentralized policies never
   receive global state.
6. **Baselines:** family A (the rule grid), then B (`AUCTION`), then C (`CENTRAL_STATIC`,
   `CENTRAL_REPLAN`), on all scenarios, all through the harness. Results go to `results/<policy>/`.
7. **Freeze:** tag `env-v1`. Write `docs/env_v1_notes.md` listing the observation/action
   layout, the feature list and the baseline results table. Every method branch starts from
   this tag.

### On each method branch (one Claude Code session per branch; see §15)

8. **Implement the method** in `methods/<name>/` behind the `Policy` interface, with a
   `scripts/train_<name>.py`. Record design choices in `docs/method_<name>.md`.
9. **Train** 3 runs within the §11.2 budget, select on validation seeds, then evaluate on the test seeds
   through the harness, writing to `results/<name>/`.
10. **Self-organization experiments** (§12) for this method: robustness, recovery time, fleet
    scaling (`_flex`), coordinator-outage comparison.

### Back on `main` after merging branches

11. **Comparison report:** `results/cell_vs_abstract.md` and plots via `scripts/compare.py`,
    with one paragraph per claim C1–C4 saying whether the evidence supports it, per method.
12. **(Phase 2)** `machine_as_agent` / learned arms, battery/charging decisions, assembly
    scenarios, and the 20/30-order and streaming scenarios.
13. **(Phase 2) Sim-to-real hook:** an `Executor` interface with two implementations,
   `DESExecutor` (default) and a stub `ROS2Executor`. The stub turns the policy's
   "claim task" decisions into Nav2 goal poses (dock coordinates from the YAML) and listens
   for arm `handoff_done` topics. That lets the same policy drive TurtleBots in Isaac Sim, and
   later the real robots. Only define the interface and a mock; do not install ROS.

Suggested layout:
```
envs/cell_env/
  __init__.py  cell_env.py  des.py  roadmap.py  entities.py  disruptions.py
  obs.py  masks.py  reward.py  orders.py  render_gantt.py  render_replay.py  demo.py
policies/base.py  policies/registry.py           # Policy interface (main)
baselines/cell_rules.py  baselines/auction.py  baselines/central_cpsat.py
configs/cell/{cell_10,cell_10_disrupt,cell_random_10,cell_random_10_disrupt,cell_random_10_flex,cell_random_20,cell_stream}.yaml
scripts/evaluate.py  scripts/compare.py  scripts/run_baselines_cell.py
tests/test_cell_env.py  tests/test_policy_interface.py
docs/cell_env_recon.md  docs/env_v1_notes.md
results/<policy>/                                 # one folder per controller
methods/<name>/  scripts/train_<name>.py  docs/method_<name>.md   # method branches only
```

---

## 14. Assumptions and open questions (defaults are listed; change them in YAML)

- **Layout:** the 6 × 4 m default is a placeholder. Replace it with the Isaac Sim
  USD coordinates (dock poses in front of each Franka) and the real testbed dimensions.
- **Arm timing:** 6 s per handoff is a guess. Measure it from the Isaac Sim Franka pick/place
  cycle.
- **Docks:** we assume one dock per station. If the real cell has separate input and output docks
  (the video suggests an arm on each side of a conveyor), set `docks_per_station: 2`.
- **Machine sequencing and arms:** rule-based by default, because the focus is the 4
  TurtleBots. The SAC slide also mentions heterogeneous agents, so learned arms and machines
  are phase 2.
- **Assembly:** the main scenarios route one box through machining, as the current env does.
  An assembly scenario, where a station waits for several parts that different robots
  deliver, is phase 2. It is a strong showcase for coordination.
- **Failures:** whether a box on a failed robot is lost, stuck or rescued (default: stuck until repair).
- **Robot count:** policies must handle 2–4 robots, so we can study what an extra robot adds
  (2r vs 4r) and how well the system degrades when one goes down.

---

## 15. Working in parallel: one branch per method, one Claude Code per branch

**Setup (a human does this once, after milestone 7 is merged to `main` and tagged `env-v1`):**
```bash
git worktree add ../marl-ppo -b method/ppo      env-v1
git worktree add ../marl-gp  -b method/gp-rules env-v1
# open one Claude Code session in ../marl-ppo and another in ../marl-gp
```
Use separate worktrees (separate folders), never two sessions in the same folder. Each
session needs its own working copy, its own `results/` and its own virtualenv, or at least
pinned dependencies.

**Rules for method-branch sessions:**
1. **Do not edit shared code**: `envs/cell_env/`, `configs/cell/`, `policies/base.py`,
   `scripts/evaluate.py`, `scripts/compare.py`, the seed lists or the baselines. Only add files under
   `methods/<name>/`, `scripts/train_<name>.py`, `docs/method_<name>.md`, `results/<name>/`,
   and new tests under `tests/methods/<name>/`.
2. **If you find an env bug or need a new feature or observation:** stop and write it up in
   `docs/method_<name>.md` under "Requests to main", then tell the human. The fix lands on `main`
   as `env-v1.1`. Each branch then runs `git merge main` and **re-runs any results produced on
   the old version**. Every results CSV records the env tag it was produced with.
3. Each method branch registers its policy in `policies/registry.py` with a single line, which
   is the only shared file it may touch. That line will be a trivial merge conflict, so keep it
   in alphabetical order.
4. Stay within the §11.2 budget, and log wall-clock time and hardware.
5. Before the branch is called done: tests pass, `scripts/evaluate.py` has produced every
   scenario in §9 for the method, and `docs/method_<name>.md` reports results honestly,
   including where the method loses.

**Merging back:** merge each branch into `main`. Because the folders don't overlap, this should
be conflict-free apart from the registry line. Then run `scripts/compare.py` to produce the
combined report (milestone 11).

**Kickoff prompts:**
- `main`: "Read `docs/CELL_ENV_SPEC.md`. Do milestones 1–2 and stop so I can review
  `docs/cell_env_recon.md` and the demo output."
- `method/ppo`: "Read `docs/CELL_ENV_SPEC.md` (§11, §15) and `docs/env_v1_notes.md`. You
  are on `method/ppo`. Implement Track 1 (IPPO and MAPPO through one critic switch,
  candidate-scoring actor, BC warm-start) as milestones 8–10. Start with a short plan in
  `docs/method_ppo.md` and a smoke-test run on `cell_10`, then stop for review."
- `method/gp-rules`: the same, but for Track 2 and `docs/method_gp-rules.md`.
