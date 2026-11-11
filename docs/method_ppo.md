# Method: PPO family (IPPO and MAPPO)

Spec §11.3, track 1. Branch `PPO-training` (the spec suggests `method/ppo`), started from
`env-v1` and merged with `env-v1.1` and `env-v1.2`:
- `env-v1.1` adds the coordinator-outage scenarios.
- `env-v1.2` adds observation v2 with the task feature `best_free_bid`, and fixes the auction.

**Everything from the full runs on is trained and evaluated on observation v2.** Code is in
`src/som/methods/ppo/`, the training script is `scripts/train_ppo.py` and tests are in
`tests/methods/ppo/`.

## Design

- **One code path, one switch.** `--critic local` gives IPPO: the critic sees the robot's own
  observation, so nothing is centralized even in training. `--critic central` gives MAPPO: the
  critic sees that observation plus the env's critic state (every robot's observation and the
  order matrix, 1528 dims in obs v2). **The actor is identical in both and sees only its own
  observation**, so execution is decentralized either way. The evaluation harness passes
  `global_state=None`, which is the C1 guarantee.
- **Shared parameters.** One actor serves every robot. The observation has a fixed size for 2,
  3 or 4 robots, so the same network runs any fleet size without retraining (claim C3).
- **Candidate-scoring actor** (`model.py`).
  - One shared MLP scores each of the K = 8 board tasks from (task row ⊕ context).
  - The context is built from self, station, global and pooled other-robot features. The pool is
    a masked mean and max over an MLP embedding of each robot present.
  - Park and wait have their own logits. NOOP has a constant logit and is masked out whenever a
    robot decides.
  - Scores do not depend on the order of the candidates or of the other robots; a test checks
    this.
- **Machines** use the SPT sequencing rule, the machine half of the best family-A pair on
  `cell_random_10` (BOTTLENECK/SPT). The learned policy controls the robots only. Machine
  agents are Phase 2 in the spec.
- **Training follows the harness's decision protocol.** Robots at a decision point act one at
  a time, each with a fresh observation, so a claim is on the board before the next robot looks.
  `CellEnv.step` instead applies all pending actions from observations taken beforehand. Using
  the harness order means training and evaluation see the same thing.
- **Semi-MDP credit.** A robot's transition runs from one of its decisions to its next one.
  - The team reward earned in between is credited to every open transition, discounted by
    γ^(t − t_decision).
  - GAE uses γ^Δt per transition, with γ = 0.995 per second.
  - Rollouts end on episode boundaries, so no chain is cut.
  - A terminated episode bootstraps 0; a truncated one (`t_max`) bootstraps V of the final state.
- **Behaviour-cloning warm-start.** 50,000 decisions of BOTTLENECK/SPT on the training mix,
  then 30 epochs of cross-entropy (lr 1e-3). The same demonstrations fit the frozen observation
  normalization (per-feature mean and std, std floor 0.1, clip ±10). The critic's input
  normalization is fitted on the first PPO rollout.
- **PPO** (spec §11.3 starting values):

  | Setting | Value |
  |---|---|
  | Learning rate | 3e-4, decaying linearly to 3e-5 |
  | Entropy coefficient | 0.01, decaying linearly to 0.001 |
  | λ | 0.95 |
  | Clip | 0.2 |
  | Decisions per rollout | about 4,096 (rounded up to whole episodes) |
  | Epochs per rollout | 4 |
  | Minibatch | 512 |
  | Value targets | normalized with running statistics |
  | Gradient-norm clip | 0.5 |

  Two additions:
  - Epochs stop early when the approximate KL divergence goes above 1.5 × 0.03.
  - The first 2 rollouts train only the critic, so an untrained value function cannot pull the
    cloned actor around.
- **Training mix:** `cell_random_10`, `cell_random_10_disrupt` and `cell_10_disrupt`, chosen
  uniformly per episode, on training seeds 0–9999. The policy is evaluated on every scenario,
  including ones it never trained on.
- **`cell_10` is held out on purpose** (decision 2026-11-10).
  - It is a single fixed schedule: the abstract's 10 orders, with no noise and no disruptions.
    Every seed is the same episode.
  - Training on it would teach the policy one timetable rather than a dispatching skill.
  - It is also a fragile yardstick. Its due dates were calibrated so that the best of 21 rules
    gets 50%, and only FIFO/EDD does. The other rules get 20–40%, and several orders miss by a
    few seconds.
  - It stays in the evaluation as a generalisation check, reported as held out.
- **Selection.** Every 20,000 decisions the greedy actor runs on the 20 validation seeds
  (10000–10019) of every training scenario. The best mean on-time % (ties broken by lower
  tardiness) is saved as `best/actor.pt`, and that checkpoint is the one evaluated on the test
  seeds.
- **Budget.** The spec (§11.2) caps a run at 1.5M env decisions. The team raised it to
  **10M per run for both PPO and GP** (decision 2026-11-10), so the comparison stays equal.
  Demonstrations count toward the budget; validation runs do not.
  - For 10M runs, validation runs every 100k decisions instead of 20k: 100 checkpoints, not 500.
  - The learning-rate and entropy decay stretch over the whole budget.

## Smoke checks (this PC: Ryzen 5 3600, one torch thread per run)

Full tables, including the superseded observation-v1 checks: `results/ppo/smoke_test_seeds.md`.
Learning curves: `results/ppo/smoke_learning_curves.png`. All numbers are on the 100 test seeds,
observation v2 (`env-v1.2`).

1. **Behaviour cloning reaches the teacher.** The clone matches 100% of the 50k demonstrated
   decisions. On `cell_random_10` it gets 49.5% on time against BOTTLENECK/SPT's 49.7%. (On
   observation v1, 93 of 100 episodes were identical to the teacher's.)
2. **A short PPO run improves on its start.** The run used 250k decisions: 50k of behaviour
   cloning plus 200k of PPO. MAPPO took 6.8 minutes and IPPO 5.3.

   | Scenario | Best rule | Best auction | BC start | MAPPO smoke | IPPO smoke | MAPPO − BC | IPPO − BC |
   |---|---|---|---|---|---|---|---|
   | `cell_random_10` | 49.7 | 51.9 | 49.5 | 50.0 | 51.4 | +0.5 (p = 0.49) | +1.9 (p = 0.065) |
   | `cell_random_10_disrupt` | 26.8 | 26.4 | 26.0 | 28.5 | 28.4 | +2.5 (p = 0.005) | +2.4 (p = 0.012) |
   | `cell_10_disrupt` | 15.9 | 11.5 | 10.2 | 17.1 | 14.9 | +6.9 (p < 0.001) | +4.7 (p < 0.001) |

   - **Both methods improve significantly** on the two disruption scenarios.
   - On nominal `cell_random_10` the gain is not yet significant after 200k PPO decisions.
   - The fixed auction (`env-v1.2`) is now the strongest baseline on `cell_random_10`, at
     51.9%. The full runs have to beat it there, not just the best rule.

**Speed.** A single run alone does about 1,100 decisions/s. With six running at once, see the
full-runs section below.

## Full runs on Daniel's PC (decision 2026-11-10)

IPPO and MAPPO, 3 seeds each, 10M decisions per run, all 6 at once (one core per run). The
Xeon runs the GP track. Windows PowerShell, from the repo root on branch `PPO-training`:

```powershell
# Start all 6 runs. They are hidden background processes, so closing the terminal does not
# stop them. Logs go to runs\ppo\<RUN>.log; each run also writes runs\ppo\<RUN>\train.log.
New-Item -ItemType Directory -Force runs\ppo | Out-Null
foreach ($s in 0, 1, 2) {
  foreach ($c in "central", "local") {
    $m = if ($c -eq "central") { "MAPPO" } else { "IPPO" }
    Start-Process -WindowStyle Hidden -FilePath .venv\Scripts\python.exe `
      -ArgumentList "scripts/train_ppo.py --critic $c --seed $s --budget 10000000 --eval-every 100000" `
      -RedirectStandardOutput "runs\ppo\${m}_s$s.log" -RedirectStandardError "runs\ppo\${m}_s$s.err"
  }
}

# Watch progress: one validation line every 100k decisions per run
Get-Content runs\ppo\MAPPO_s0.log -Tail 5 -Wait

# Stop everything if needed (only the training processes)
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like "*train_ppo.py*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId }

# When all 6 have a summary.json: pick the best seed per method on validation, evaluate on
# every scenario's test seeds plus 2 and 3 robots, and refresh results/summary.md
.venv\Scripts\python.exe -m som.methods.ppo.finalize --workers 6
git add results reports docs; git commit -m "docs(results): add IPPO and MAPPO results (10M decisions, 3 seeds each)"
git push origin PPO-training
```

**Measured speed** with all 6 running at once on the Ryzen 5 3600 (6 cores / 12 threads),
while the PC was also running central-scheduler jobs:

| Method | Speed per run | 10M decisions |
|---|---|---|
| MAPPO | 610–780 decisions/s | about 3.5–4.5 h |
| IPPO | 1,000–1,300 decisions/s | about 2–3 h |

- Validation every 100k decisions adds about 10–15 min per run.
- **All six finish in about 4–5 h** (MAPPO is the slowest); finalize adds about 10 min.
- The PC is fully busy throughout.

**On another machine** (for example the Xeon): set up as in the README's Linux section and
check the baselines reproduce first:
- Run `scripts/evaluate_cell.py` on the rules and auctions with `--out runs/repro`.
- Then run `python -m som.methods.ppo.check_repro runs/repro`, which must end "0 differ, 0 missing".
- Launch with the same `train_ppo.py` arguments, using `nohup ... &` in place of `Start-Process`.

## Results (full runs, 2026-11-10): not valid as a win, the policies abandon orders

Six runs of 10M decisions each, observation v2, on Daniel's PC (MAPPO 214–217 min, IPPO 131–200
min). Best seed on validation: MAPPO s0 (53.4%; s1 52.7, s2 52.4) and IPPO s2 (52.0%; s0 44.4,
s1 44.7). Per-seed test results are in `results/ppo/seeds/`, the selection in
`results/ppo/selection.json` and every controller in `results/summary.md`.

**The on-time % looks like a large win, but it is reward hacking.** Both learned policies
deliver the orders they can get there on time and then leave the rest undelivered until the
episode is cut off at `t_max`. Robots sit idle 60–84% of the time, against 11–31% for the
auction. Test seeds:

| Scenario | Fixed auction on-time % | MAPPO on-time % | MAPPO on-time orders (auction) | MAPPO orders delivered | MAPPO robots idle |
|---|---|---|---|---|---|
| `cell_random_10` | 51.9 | 64.2 | 6.4 (5.2) | 71% | 76% |
| `cell_random_10_disrupt` | 26.4 | 47.4 | 5.0 (2.8) | 51% | 61% |
| `cell_10_disrupt` | 11.5 | 47.0 | 4.9 (1.2) | 49% | 62% |
| `cell_10` (held out) | 20.0 (best rule 50.0) | 60.0 | 6.0 (2.0) | 70% | 79% |
| `cell_random_30` | 40.4 | 45.1 | 13.5 (12.1) | 49% | 84% |
| `cell_random_10_2r` | 20.9 | 35.3 | 3.5 (2.1) | 38% | 76% |

IPPO behaves the same: it delivers 44–69% of orders. Every rule, auction and the central
scheduler deliver 100%.

- **The extra on-time orders are real.** The on-time % divides by all orders, so MAPPO really
  does get more orders there on time. It does that by giving up on the rest, which no real cell
  can accept. The huge tardiness and makespan columns (makespan = `t_max`) in `summary.md` show
  it.
- **Why the reward allows it.** The spec §8 reward pays +10 per on-time delivery. A late order
  costs at most −5 (capped at 150 s of tardiness), plus −0.01 per second while it is overdue.
  With γ = 0.995 per second, a penalty 500 s away is worth 8% of its face value, and the
  truncation penalty at `t_max` (about 1,000 s later) about 0.7%. So abandoning a late order is
  almost free, while delivering it costs robot time that could serve an order still on time.
- **This is a reward-design problem, not a trainer bug.** The rules and the auction cannot
  abandon an order, because they always claim a task when one is available, so they never
  exposed it.
- **Not yet presentable as a result.** Fix options (decision needed):
  1. *Method side (no env change):* mask park and wait while a task is claimable, so robots are
     work-conserving like the rules. Every order then gets delivered and PPO only chooses the
     order. Needs a retrain (about 3.6 h for all six).
  2. *Env side (on `main`, `env-v1.3`):* make abandoning expensive, for example an uncapped late
     penalty or a per-undelivered-order penalty applied as it happens. This changes the reward
     for the GP track too.
  3. In every case, report "orders delivered %" next to on-time %.

What does hold: MAPPO was more reliable than IPPO across seeds (all three MAPPO seeds 52–53% on
validation, against two of three IPPO seeds stuck at 44%). The outage result is unchanged:
learned robots have no coordinator, so an outage costs them nothing.

## Requests to main

None so far. The coordinator-outage scenarios this method is evaluated on came from `main`
(`env-v1.1`).
