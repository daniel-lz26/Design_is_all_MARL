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
3. **`main` and the `env-v1.1` and `env-v1.2` tags are pushed** (done 2026-11-10).
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
- **Coordinator outage result** (central scheduler, on `main`).
  - One 30–90 s outage of its coordinator computer drops it from 80% to 24% of orders on time on
    `cell_10`, and from 51% to 29% on `cell_random_10`.
  - Rules, the auction and the learned robots don't depend on a coordinator, so they are
    unaffected.
  - Table: "Coordinator outage" in [results/summary.md](../../results/summary.md).

- **The six full PPO runs finished** (10M decisions each, 2.2–3.6 h). Best seed on validation:
  MAPPO s0 and IPPO s2, both evaluated on every scenario.

## Running now
- Nothing.

## Latest results: the learned robots abandon orders
On 100 test scenarios, full runs. "Delivered" is the share of orders that reach shipping at all.
Every rule, the auction and the central scheduler deliver 100%.

| Scenario | Best rule | Best auction | MAPPO on-time % | MAPPO delivered | IPPO on-time % | IPPO delivered |
|---|---|---|---|---|---|---|
| cell_random_10 | 49.7 | 51.9 | 64.2 | 71% | 62.9 | 69% |
| cell_random_10_disrupt | 26.8 | 26.4 | 47.4 | 51% | 47.4 | 53% |
| cell_10_disrupt | 15.9 | 11.5 | 47.0 | 49% | 41.4 | 44% |
| cell_10 (held out) | 50.0 | 20.0 | 60.0 | 70% | 50.0 | 60% |

- **The on-time % looks like a big win but isn't one.** The robots deliver the orders that can
  still be on time, then sit idle (60–84% of the time) and never deliver the rest.
- **Cause: the reward has a loophole** ("reward hacking"). A late order costs at most a small,
  capped penalty, and the penalty for never delivering it comes so far in the future that the
  learner barely counts it. The rules and the auction always pick up a waiting order, so they
  never showed this.
- **What still holds:** MAPPO was more reliable than IPPO (all three MAPPO seeds trained well;
  two of three IPPO seeds got stuck), and learned robots don't depend on a coordinator.
- Details and fix options: [docs/method_ppo.md](../../docs/method_ppo.md). All controllers:
  [results/summary.md](../../results/summary.md). Smoke-run plots:
  [learning curves](../../results/ppo/smoke_learning_curves.png).

## Blockers / decisions needed
- **Pick a fix for order abandonment before retraining:**
  1. PPO side: a robot may not park or wait while an order is available to pick up, so every
     order gets delivered (no env change, about 3.6 h to retrain).
  2. Env side (`main`, `env-v1.3`): change the reward so undelivered orders keep costing a lot.
     This also changes the GP track's reward.
- **GP track:** merge `main` (`env-v1.2`) before its 10M runs (decision 5), and report
  "delivered %" too; the same loophole is open to it.

## Next steps
1. Apply the chosen fix, retrain the six runs and re-evaluate, reporting delivered % next to
   on-time %.
2. Fleet-size and outage comparisons for the presentation (Nov 21).
