# Legacy review (`MARL_new`)

The previous codebase was reviewed on Oct 19, 2026, from: `envs/factory_env.py`, `envs/reward.py`,
`trainer/ippo_trainer.py`, `trainer/mappo_trainer.py`, `configs/default.yaml`, and the abstract's
training log `10orders_results.txt`. If the human provides the old files, place them under
`legacy/reference/` (read-only, excluded from packaging and lint). **Nothing in `src/` may import them.**

## What it was

- A gym-style 2D continuous-space job shop. SRC → TURNING → GRINDING → MILLING → DRILLING → SINK on a 15×15 plane.
- AGV agents: `Discrete(9 + pickup_index_count + 1)`, i.e. **8 movement directions + stay**, then pick slots, then drop.
- Machine agents choose among the top-K queued eligible orders.
- Hand-written PyTorch IPPO and MAPPO.

## Results (verified)

The abstract run (`configs/ippo_delivery_10orders.yaml`, IPPO, 160k steps, horizon 1600 steps):
- Deliveries go from 0 to all 10 within about 35k steps and stay there.
- Last 20k steps, deduplicated: **267 delivered, 100 on time (37.5%)**. Matches the abstract's 266 / 38%.
- **The log is corrupted for plotting:** there is a gap at steps 63k→83k, and episodes 137–181 are
  duplicated (1,668 delivery lines, 1,227 unique). Deduplicate before reusing any number.

## Bugs and design problems (do not repeat)

| Problem | Where | Consequence | Fix in new repo |
|---|---|---|---|
| Low-level AGV actions (8-direction moves) | `factory_env.py`, agv module | RL learns steering; it can't transfer to real robots | Task-level actions (ARCH §3.3, §5.2) |
| Layout hard-coded (`self.processes`, `self.stations`, default positions) | `factory_env.py` | No other factory without code edits | `configs/layouts/*.yaml` |
| Timeouts treated as terminal | both trainers, `compute_gae*` | Biased values near the horizon | Separate `terminated`/`truncated`, bootstrap on truncation |
| MAPPO: one shared value per timestep, but per-agent rewards in GAE | `mappo_trainer.py` | Advantage is not a valid estimate for any agent | Agent-specific global state; per-agent or team reward used consistently |
| MAPPO: no action masking | `mappo_trainer.py` | Samples infeasible actions | Masks in rollout and update |
| MAPPO written for a different env (`mover_1`, `manuf_1` comments) | `mappo_trainer.py` | Never adapted or validated | One trainer, `critic_mode` flag |
| Advantage normalized across AGVs and machines together | both trainers | Role reward scales contaminate each other | Per-role normalization |
| Exploration-ε replaces the sampled action, but log-prob is taken under the policy rather than the ε-mixture that actually acted | `ippo_trainer.py` | PPO importance ratios are wrong for those samples | Remove ε; rely on entropy |
| Per-sample Python loops in the update; IPPO computes critic loss twice | both trainers | Very slow training | Batched tensor minibatches |
| Every agent acts every tick, including busy ones | env design | Mostly no-op samples dilute learning | Decision-point semi-MDP; exclude non-decision samples |
| ~25 reward terms + 4 "forced action" training aids | `reward.py`, env flags | Total reward (~3,500) is mostly shaping; forced actions make the key decisions for the policy | ≤4 terms, team reward, no forced actions |
| No baseline | — | Can't tell if RL beats a dispatch rule | Heuristic suite in `som.policies.heuristics` |
| `include_machine_timer` duplicated in YAML; many unused flags | `default.yaml` | Config is confusing | pydantic schemas reject unknown keys (`extra="forbid"`) |

## Ideas worth keeping

- **Event dict → reward calculator** separation. The new version computes reward from core deltas.
- **Action masking** for infeasible actions.
- **Order × process status matrix** (−1 not in route, 0 pending, 1 done). Reuse it in the dashboard's order table.
- **YAML job types and orders.**
- **Machine input buffer capacity = 5** for eligible orders.
- **Machine selection among top-K queued orders.**

## Recovered 10-order scenario (`configs/scenarios/abstract_10.yaml`)

Recovered from the log's order-status matrices and delivery lines. **Arrival and process times were not
in the log** and are placeholders until the original `ippo_delivery_10orders.yaml` is found.

Job types (from `default.yaml`):

| Job | Part | Route |
|---|---|---|
| P1 | Shaft | Turning → Grinding |
| P2 | Flange | Turning → Milling → Drilling |
| P3 | Plate | Milling |
| P4 | Frame | Milling → Drilling |

Orders:

| id | job | route (from the log) | due (s) | release | process times |
|---|---|---|---|---|---|
| 1 | P1 | T, G | 146 | PLACEHOLDER | PLACEHOLDER |
| 2 | P2 | T, M, D | 166 | PLACEHOLDER | PLACEHOLDER |
| 3 | P3 | M | 196 | PLACEHOLDER | PLACEHOLDER |
| 4 | P4 | M, D | 236 | PLACEHOLDER | PLACEHOLDER |
| 5 | P1 | T, G | 266 | PLACEHOLDER | PLACEHOLDER |
| 6 | P3 | M | 266 | PLACEHOLDER | PLACEHOLDER |
| 7 | P2 | T, M, D | 296 | PLACEHOLDER | PLACEHOLDER |
| 8 | P4 | M, D | 326 | PLACEHOLDER | PLACEHOLDER |
| 9 | P3 | M | 346 | PLACEHOLDER | PLACEHOLDER |
| 10 | P2 | T, M, D | 376 | PLACEHOLDER | PLACEHOLDER |

Placeholder rule until real values arrive (mirrors the 5-order `default.yaml`, where release ≈ due − 130
to 145, and process times are near the defaults Turning 30, Grinding 20, Milling 40, Drilling 10 with ±20%):
`release = max(0, due − 135)`; process times = defaults. Mark every such value `# PLACEHOLDER` in YAML.
The legacy horizon was 1600 steps; use `horizon: 1600` seconds for sim eval, and a `time_scale` for live demos.
