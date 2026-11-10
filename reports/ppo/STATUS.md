# PPO track: status

*Last updated: 2026-11-10. Branch `PPO-training`. Details: [docs/method_ppo.md](../../docs/method_ppo.md).*

**What this track is.** We train the robots' dispatching policy with reinforcement learning (RL):
the robots learn which order to pick up next by trial and error in the simulated cell.
- **PPO** is the RL algorithm.
- **IPPO** and **MAPPO** are two versions of it. In MAPPO, a shared "coach" network sees the
  whole cell while training. In IPPO, nothing is shared even in training. In both, each robot
  decides from its own view only when it runs, which is the self-organizing part.
- **Behaviour cloning (BC)** means first copying the best hand-written rule, so learning starts
  from a sensible policy instead of random moves.

## Decisions (2026-11-10)
1. **`cell_10` stays out of the training mix** and is reported as a held-out scenario.
   - It is a single fixed schedule: the same 10 orders, with no randomness and no breakdowns, so
     every run is the identical episode.
   - Training on it would teach one timetable, not a dispatching skill.
   - Its 50% target comes from the single best of 21 rules (FIFO/EDD). The other rules get
     20–40%, and several orders miss their due time by only a few seconds.
2. **The auction now lets only free robots bid** (on `main`, tag `env-v1.2`).
   - The old auction is kept as `AUCTION_v0`. In it, busy robots also bid, so a free robot
     often sat idle waiting for a busy one: robots were idle 36–40% of the time against about
     30% for the rules.
   - The fix needed one new piece of information in each robot's view, the best bid among
     *free* robots, so the observation went from version 1 to version 2.
3. **`main` and the `env-v1.1` and `env-v1.2` tags** are pushed once the outage baselines finish.
4. **Full runs: 10M decisions per run, 3 seeds each, for both PPO and GP.** The spec's limit was
   1.5M; the team raised it for both tracks so the comparison stays fair. PPO runs on Daniel's
   PC (IPPO and MAPPO, 6 runs) and GP on the Xeon.
5. **GP must also move to `env-v1.2`** (observation v2) before its full runs. Anything trained on
   observation v1 can no longer be compared: a v1 PPO checkpoint is refused by the code.

## Done
- The PPO trainer works on the cell simulation. IPPO and MAPPO share one code path; one switch
  picks between them.
- On `main`:
  - Coordinator-outage scenarios (`env-v1.1`).
  - Auction fix and observation v2 (`env-v1.2`). Both auctions match a reference implementation
    decision for decision.
  - Every rule and both auctions re-run on `env-v1.2`. The rules reproduce env-v1 exactly
    (24,300 episodes), and so does `AUCTION_v0` (4,500).
- **Both smoke checks passed again on observation v2:**
  - BC copies the best rule: 49.5% vs 49.7%.
  - A short PPO run improves on the BC start, significantly on both disruption scenarios.
- Run time measured: **about 4–5 hours** for the six 10M runs on Daniel's PC.

## Running now
- Outage-scenario baselines (central scheduler) on `env-v1.1`; about 4 jobs left.
- Next: the six 10M PPO runs start once that finishes, so the two jobs don't compete for CPU.

## Latest results
On-time % on 100 test scenarios, **observation v2, smoke run only** (250k decisions).
- The "best auction" is the fixed one. It is now the strongest baseline on `cell_random_10`.
- Full table, including the old v1 numbers:
  [smoke_test_seeds.md](../../results/ppo/smoke_test_seeds.md).

| Scenario | Best rule | Best auction | MAPPO (smoke) | IPPO (smoke) |
|---|---|---|---|---|
| cell_random_10 | 49.7 | 51.9 | 50.0 | 51.4 |
| cell_random_10_disrupt | 26.8 | 26.4 | 28.5 | 28.4 |
| cell_10_disrupt | 15.9 | 11.5 | 17.1 | 14.9 |

Plots: [learning curves](../../results/ppo/smoke_learning_curves.png).

## Blockers / decisions needed
- None blocking.
- **GP track:** merge `main` (`env-v1.2`) before starting its 10M runs; see decision 5.

## Next steps
1. Pass both smoke checks on observation v2, then estimate the run time and start the 6 full
   PPO runs on Daniel's PC.
2. When they finish: pick the best seed per method on validation, evaluate on every scenario,
   and compare with the rules, both auctions and the central scheduler.
3. Fleet-size and outage comparisons for the presentation (Nov 21).
