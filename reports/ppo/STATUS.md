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

## Done
- The PPO trainer works on the new Smart Assembly Cell simulation. IPPO and MAPPO share one code
  path; a single switch picks between them.
- **Smoke check 1 passed.** BC copies the best rule (BOTTLENECK/SPT): 49.5% on time against the
  rule's 49.7% on `cell_random_10`.
- **Smoke check 2 passed.** A short PPO run (250k decisions, about 6 minutes) improves on the BC
  start on all three training scenarios.
- **On `main`:**
  - Added the coordinator-outage scenarios (tag `env-v1.1`). These show what happens to the
    central scheduler when its computer goes down.
  - Checked the auction baseline. It has no bug; it is a real weakness: free robots sit idle
    waiting for busy robots that out-bid them.

## Running now
- Baselines (rules, auction, central scheduler) on the 4 outage scenarios. This runs on `main`
  on Daniel's PC.

## Latest results
On-time % on 100 test scenarios. Smoke-run numbers only: 250k decisions, against a 1.5M budget
for the full runs.

| Scenario | Best rule | MAPPO (smoke) | vs best rule | IPPO (smoke) | vs best rule |
|---|---|---|---|---|---|
| cell_random_10 | 49.7 | 51.3 | +1.6 | 50.3 | +0.6 |
| cell_random_10_disrupt | 26.8 | 28.4 | +1.6 | 26.6 | −0.2 |
| cell_10_disrupt | 15.9 | 16.6 | +0.7 | 15.6 | −0.3 |

Plots: [learning curves](../../results/ppo/smoke_learning_curves.png). Table with p-values:
[smoke_test_seeds.md](../../results/ppo/smoke_test_seeds.md).

## Blockers / decisions needed
- None blocking.
- **Decision for the team:** the full runs train on a mix of random 10-order orders, random 10
  orders with disruptions, and the fixed 10 orders with disruptions. Say so if `cell_10` itself
  should be in the mix: the smoke model gets 30% there against FIFO/EDD's 50%.

## Next steps
1. On the Xeon PC: reproduce the baseline numbers, then train IPPO and MAPPO with 3 seeds each
   (commands in [docs/method_ppo.md](../../docs/method_ppo.md#running-on-the-xeon-ubuntu-2204-w3-2423-6-cores)).
2. Pick the best seed per method on the validation scenarios, evaluate on every scenario, and
   compare with the rules, the auction and the central scheduler.
3. Fleet-size and outage comparisons for the presentation (Nov 21).
