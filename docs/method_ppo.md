# Method: PPO family (IPPO and MAPPO)

Spec §11.3, track 1. Branch `PPO-training` (the spec suggests `method/ppo`), started from
`env-v1` and merged with `env-v1.1`, which adds the coordinator-outage scenarios. Code is in
`src/som/methods/ppo/`, the training script is `scripts/train_ppo.py` and tests are in
`tests/methods/ppo/`.

## Design

- **One code path, one switch.** `--critic local` gives IPPO: the critic sees the robot's own
  observation, so nothing is centralized even in training. `--critic central` gives MAPPO: the
  critic sees that observation plus the env's critic state (every robot's observation and the
  order matrix, 1496 dims). **The actor is identical in both and sees only its own
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
- **Selection.** Every 20,000 decisions the greedy actor runs on the 20 validation seeds
  (10000–10019) of every training scenario. The best mean on-time % (ties broken by lower
  tardiness) is saved as `best/actor.pt`, and that checkpoint is the one evaluated on the test
  seeds.
- **Budget (spec §11.2).** 1.5M env decisions per run. Demonstrations count toward the budget;
  validation runs do not.

## Smoke checks (this PC: Ryzen 5 3600, one torch thread per run)

Full table: `results/ppo/smoke_test_seeds.md`. Learning curves:
`results/ppo/smoke_learning_curves.png`. All numbers are on the 100 test seeds.

1. **Behaviour cloning reaches the teacher.**
   - The clone matches 100% of the 50k demonstrated decisions.
   - On `cell_random_10` it gets 49.5% on time against BOTTLENECK/SPT's 49.7%, and 93 of 100
     episodes are identical.
   - The remaining differences come from states the teacher never visited in the
     demonstrations.
2. **A short PPO run improves on its start.** The run used 250k decisions: 50k of behaviour
   cloning plus 200k of PPO. MAPPO took 6.4 minutes and IPPO 4.9.

   | Scenario | Best rule | BC start | MAPPO smoke | IPPO smoke | MAPPO − BC | IPPO − BC |
   |---|---|---|---|---|---|---|
   | `cell_random_10` | 49.7 | 49.5 | 51.3 | 50.3 | +1.8 (p = 0.016) | +0.8 (p = 0.29) |
   | `cell_random_10_disrupt` | 26.8 | 25.9 | 28.4 | 26.6 | +2.5 (p = 0.004) | +0.7 (p = 0.40) |
   | `cell_10_disrupt` | 15.9 | 10.4 | 16.6 | 15.6 | +6.2 (p < 0.001) | +5.1 (p < 0.001) |

   The smoke MAPPO was also run on the scenarios it did not train on. It was at or above the
   best rule on every one except `cell_10`:

   | Scenario | Smoke MAPPO | Best rule |
   |---|---|---|
   | `cell_random_20` | 48.5 | 47.8 |
   | `cell_random_30` | 41.4 | 38.6 |
   | `cell_stream` | 45.1 | 43.2 |
   | `cell_random_10_flex` | 41.9 | 40.3 |
   | `cell_random_10` with 3 robots | 35.9 | 34.7 |
   | `cell_random_10` with 2 robots | 20.9 | 21.0 |
   | `cell_10` | 30.0 | 50.0 (FIFO/EDD) |

   `cell_10` is one fixed schedule on which only FIFO/EDD reaches 50%; other rules get 20–40%.

**Speed.** One run collects and trains at about 1,100 decisions/s when the machine is idle, and
about 650/s when it shares the CPU. A full 1.5M-decision run takes about 25–45 minutes.

## Running on the Xeon (Ubuntu 22.04, W3-2423, 6 cores)

```bash
# 1. Set up once (about 10 min)
sudo apt install python3.10-venv tmux
git clone git@github.com:daniel-lz26/Design_is_all_MARL.git && cd Design_is_all_MARL
git checkout PPO-training
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt && pip install -e .
python -m pytest -q                       # 195 tests, about 2-3 min

# 2. Reproduce the env-v1 baselines first (about 10-15 min): every rule and auction on every
#    env-v1 scenario, and the central scheduler on cell_10. Then compare seed by seed with
#    the committed results/.
python scripts/evaluate_cell.py --workers 6 --out runs/repro \
  --scenarios cell_10 cell_10_raw cell_10_disrupt cell_random_10 cell_random_10_disrupt \
              cell_random_10_flex cell_random_20 cell_random_30 cell_stream \
  --policy $(python -c "from som.policies.registry import REGISTRY, family; print(' '.join(n for n in REGISTRY if family(n) in 'AB'))")
python scripts/evaluate_cell.py --policy CENTRAL_STATIC CENTRAL_REPLAN --scenarios cell_10 \
  --out runs/repro --workers 2
python -m som.methods.ppo.check_repro runs/repro    # must end "0 differ, 0 missing"

# 3. Train IPPO and MAPPO, 3 seeds each, all 6 at once (about 45-60 min; one core per run)
tmux new -s ppo                           # detach with Ctrl-b d, come back with: tmux attach -t ppo
mkdir -p runs/ppo
for s in 0 1 2; do
  nohup python scripts/train_ppo.py --critic central --seed $s > runs/ppo/MAPPO_s$s.log 2>&1 &
  nohup python scripts/train_ppo.py --critic local   --seed $s > runs/ppo/IPPO_s$s.log 2>&1 &
done
tail -f runs/ppo/*.log                    # one validation line every 20k decisions

# 4. Pick the best seed per method on validation, then evaluate on every scenario's test seeds
#    plus 2 and 3 robots (about 15-20 min). Writes results/IPPO, results/MAPPO,
#    results/ppo/seeds/ and results/ppo/selection.json, and refreshes results/summary.md.
python -m som.methods.ppo.finalize --workers 6

# 5. Commit the results and push the branch
git add results && git commit -m "docs(results): add IPPO and MAPPO results (3 seeds each)"
git push origin PPO-training
```

Set git's name and email on the Xeon before the first commit (`git config user.name`,
`git config user.email`). Model files stay in `runs/`, which git ignores. Copy
`runs/ppo/*/best/` somewhere safe, or a teammate cannot re-evaluate them.

## Results

(after the full runs)

## Requests to main

None so far. The coordinator-outage scenarios this method is evaluated on came from `main`
(`env-v1.1`).
