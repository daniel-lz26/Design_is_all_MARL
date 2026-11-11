# PPO track: status

*Last updated: 2026-11-11. Branch `PPO-training`. Details: [docs/method_ppo.md](../../docs/method_ppo.md).*

**What this track is.** We train the robots' dispatching policy with reinforcement learning (RL):
the robots learn which order to pick up next by trial and error in the simulated cell.
- **PPO** is the RL algorithm.
- **IPPO** and **MAPPO** are two versions of it. In MAPPO, a shared "coach" network sees the
  whole cell while training. In IPPO, nothing is shared even in training. In both, each robot
  decides from its own view only when it runs, which is the self-organizing part.
- **Behaviour cloning (BC)** means first copying the best hand-written rule, so learning starts
  from a sensible policy instead of random moves.

## Decisions (2026-11-10, plus 6 on 2026-11-11)
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

6. **(2026-11-11) Order abandonment is fixed on the PPO side.** A robot may not park or wait
   while there is an order it could pick up, the same rule the hand-written rules and the
   auction follow. No env change, so the GP track is unaffected.

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

- **The six full PPO runs finished** (2026-11-10, 10M decisions each, 2.2–3.6 h). Best seed on
  validation: MAPPO s0 and IPPO s2. *Superseded:* those policies abandoned orders (see the
  2026-11-10 report), so they were retrained with the fix.

- **Retrained with the fix on the new Ubuntu PC** (2026-11-11): IPPO and MAPPO, seeds 0–2,
  10M decisions each, then seed selection on validation and evaluation on test. Baselines
  reproduce exactly on that PC (1,600 episodes, 0 differences).

## Running now
- Nothing. (2026-11-11 extras finished: seeds 3–5 of both methods, and a no-behaviour-cloning
  ablation, which showed that training without first copying the rule ends at the same
  level. See the [2026-11-11 report](2026-11-11.md).)

## Latest results (2026-11-11): the learned robots now win fairly
On 100 test scenarios each. "Delivered" is the share of orders that reach shipping at all.
**Every controller, including both learned ones, now delivers 100%.** On-time %:

| Scenario | Best rule | Best auction | Central scheduler | IPPO | MAPPO |
|---|---|---|---|---|---|
| cell_random_10 | 49.7 | 51.9 | 51.2 | 53.7 | 53.3 |
| cell_random_10_disrupt (breakdowns) | 26.8 | 26.4 | 31.0 | 33.3 | 32.7 |
| cell_10_disrupt (breakdowns) | 15.9 | 13.4 | 22.7 | 24.2 | 25.8 |
| cell_10 (held out) | 50.0 | 20.0 | 80.0 | 30.0 | 50.0 |
| cell_random_30 (3x the orders, never trained on) | 38.6 | 40.4 | 22.3 | 43.7 | 45.1 |
| cell_10 with a coordinator outage | 50.0 | 20.0 | 24.0 | 30.0 | 50.0 |

- **A fair win over the best rule** on every random and disrupted scenario, and with 2, 3 or 4
  robots (MAPPO +3.6 to +9.9 points, all statistically significant).
- **Over the auction** the gain is clear with breakdowns (+6 to +12.5) but small on the
  nominal scenarios (+1.4 on cell_random_10, not significant).
- **Held-out cell_10:** the chosen MAPPO seed ties the best rule (50%), the chosen IPPO seed
  gets 30%. But each trained seed lands on either 30% or 50% on this one fixed schedule, so
  that difference is mostly luck of the seed. The central scheduler wins (80%) only while its
  coordinator is up.
- **IPPO and MAPPO are about equal**, and the result doesn't depend on a lucky seed: six seeds
  per method all land within about 1 point of each other, and even the worst one beats the best
  rule on every random and disrupted scenario. The earlier "MAPPO is more reliable" observation came from the broken runs and no
  longer holds.
- **The honest number is much smaller than the old one:** 53.3% on cell_random_10, not the
  64% the order-abandoning policy showed.
- Details: [2026-11-11 report](2026-11-11.md), [docs/method_ppo.md](../../docs/method_ppo.md),
  every controller: [results/summary.md](../../results/summary.md).

## Blockers / decisions needed
- **Commit and push:** all of 2026-11-11's work is uncommitted on `PPO-training`, waiting for
  Daniel's review.
- **Poster:** the scripts and `.pptx` live only on the old PC (uncommitted), and rendering needs
  PowerPoint, so the poster update has to happen there (or the scripts get committed).
- **GP track:** merge `main` (`env-v1.2`) before its 10M runs, report "delivered %" too, and
  either adopt the same rule (robots can't idle while an order is waiting) or show GP doesn't
  abandon orders. Otherwise the two tracks aren't compared on the same terms.

## Next steps
1. Review and commit the fix and the new results; update the poster (remove the
   "not yet a valid win" flag; the results now justify a qualified win, see above).
2. Fleet-size and outage comparisons for the presentation (Nov 21): the numbers are already in
   results/summary.md.
