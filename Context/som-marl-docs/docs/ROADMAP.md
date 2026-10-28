# Roadmap: Oct 28 → Nov 21, 2026

*Re-baselined Oct 27; the repo starts Oct 28. Algorithm: MAPPO.*

Work top to bottom. A milestone is done only when its **acceptance check** passes and is committed.
Dates are targets; the **Nov 16 video** and the **Nov 21 presentation** are fixed. The schedule is
compressed, so M0 and M1 run back-to-back and M3 starts in parallel with M2 training runs.

Owners: **L** = learning/sim team (+ Claude Code), **H** = hardware team. Hardware tasks H1–H8 are
detailed in `HARDWARE_INTERFACE.md §5`, and the hardware team can start them now, independent of L.

---

## M0 · Scaffold (Oct 28) · L

- [x] `pyproject.toml` (src layout, package `som`, Python ≥3.10), Makefile, ruff, pytest, pre-commit, `.gitignore`
- [x] Empty module tree exactly as in `ARCHITECTURE.md §2`, with module docstrings stating each responsibility
- [x] `som.config.schema` with pydantic models (`extra="forbid"`) for Layout, Scenario, Env, Reward, Train, Deploy
- [x] `configs/layouts/arena_13x6ft.yaml` (placeholders from HARDWARE_INTERFACE §3)
- [x] `configs/scenarios/abstract_10.yaml` (from LEGACY_REVIEW; placeholders marked)

**Accept:** `make install && make lint && make test` pass on a clean checkout; every config file loads
through its schema in a test.

## M1 · Core, sim, baselines (Oct 29 – Nov 2) · L

- [x] `som.core`: ids, layout, order state machine, events (**including `MachineDown`/`MachineUp`**),
      commands, decisions, `FactoryCore`, invariants
- [x] `som.sim`: `StraightLineTravel` (with travel-time noise), `SimRuntime`, scenario builder (fixed + generated)
- [x] `som.policies`: Policy protocol, random, heuristics (robot FIFO/EDD/NEAREST/MIN_SLACK × machine FIFO/EDD/SPT)
- [x] `som.eval`: metrics, evaluate, compare (markdown + csv), Gantt plot
- [x] `som.viz.replay`: animate an episode log over the arena layout
- [x] Tests: order conservation, capacity, reservations, dock locks, `RobotFailed` recovery,
      machine down/up pause-and-resume, determinism (same seed → identical log), heuristics complete `abstract_10`

**Accept:** `make baselines` prints the baseline table for `abstract_10` over 20 seeds;
`make replay` produces a gif of the best heuristic. **This is the first showable result.** Commit it to `reports/`.

## M2 · RL env + MAPPO (Nov 1–8) · L

- [ ] `SomParallelEnv` (decision-point semi-MDP, NOOP masking, `dt`, terminated vs truncated)
- [ ] PettingZoo `parallel_api_test` passes
- [ ] Obs encoders (versioned) + global state padded to `max_robots=4`
- [ ] Rewards (team default, ≤4 terms)
- [ ] `som.rl`: networks, buffer, **MAPPO trainer** (`critic_mode: central` default), `γ^dt` GAE, per-role
      advantage normalization, vec env, value normalization, artifact export
- [ ] Sanity tests: the trainer overfits a 3-order toy scenario; a masked action is never sampled
- [ ] Train `mappo_2r` on `random_10` (3 seeds); evaluate on `abstract_10` + held-out `random_10`
- [ ] Evaluate the `mappo_2r` artifact with 4 robots; start the `mappo_4r` fine-tune
- [ ] *(Optional, only if a training slot is free)* one `ippo_2r` run for the legacy comparison row

**Accept:** results table with MAPPO rows next to the baselines. Target: MAPPO on-time rate ≥ best
heuristic on `abstract_10`. If it isn't met, record the gap and hypotheses, then continue to M3; don't block on it.

> **Nov 8 checkpoint with the human:** review results; pick the policy that goes live; apply the
> **disruption gate (D15)**: if MAPPO beats the best heuristic, schedule the disruption stretch in M3 week.

## M3 · Live runtime + robot controller (Nov 4–11) · L + H

Starts while M2 runs are training.

- [ ] L: `RobotBackend` protocol, `kinematic` backend, `LiveRuntime` (policies hosted on the PC, D14),
      recorder, dashboard
- [ ] L: `configs/deploy/desk_kinematic.yaml`. A full 10-order demo runs on a laptop with the MAPPO policy
- [ ] L: `ros2_ws/src/som_robot` (`goto_controller`, `safety`, `pose_sources`, launch, params) with unit
      tests for the control law (no ROS needed for the math)
- [ ] L: `ros2` backend (action clients, arena→odom conversion)
- [ ] H: H1–H5 (shared domain + namespaces, access point, arena taped and measured, controller on tb1, then tb2)
- [ ] L: update `arena_13x6ft.yaml` with measured coordinates
- [ ] *(Stretch, only if the Nov 8 gate passed)* train `mappo_2r_disrupt` with random machine down/up and
      rush orders; add a disruption-recovery row to the results; add machine toggles to the dashboard

**Accept:** `make lab CFG=configs/deploy/desk_kinematic.yaml` runs end to end and the recorded log scores
with `som.eval`. On hardware, `ros2 action send_goal /tb1/go_to_pose ...` reaches a station within 5 cm.

## M4 · Lab integration, 2 robots (Nov 10–16) · L + H

- [ ] H6 drift test (by Nov 12) → decide odom-only vs camera for the demo
- [ ] Calibrate `travel_calibration` (v_eff, dock overhead) from logged real runs; re-evaluate policies with it
- [ ] Full `abstract_10` (time-scaled) runs with 2 real robots: MAPPO and the best heuristic
- [ ] Score live runs with the same metrics; add a "lab" row to the results table
- [ ] *(Stretch, only if the normal 2-robot run is reliable by Nov 14)* a live machine-down demo
- [ ] **Record the backup demo video** (overhead phone shot + dashboard screen capture, side by side)

**Accept:** at least 3 clean live runs recorded with metrics; video saved in `reports/`. **Hard date: Nov 16.**

## M5 · Freeze + presentation (Nov 17–21) · L

- [ ] Code freeze on `main` (tag `sccur-2026`). Bug fixes only
- [ ] Figures: learning curves (on-time %), baseline vs MAPPO bars with CIs, 2→4 robot scaling, Gantt of
      one episode, sim vs lab metrics (+ IPPO row and disruption recovery if they were done)
- [ ] Story slide: abstract 38% → new result; what changed (task-level actions, semi-MDP, reward, MAPPO with
      centralized training / decentralized execution)
- [ ] Demo runbook in `README.md`: power-on order, bringup, start poses, `make lab`, e-stop, fallbacks
- [ ] Rehearse the live demo twice from a cold start

## Stretch (only after M4 is accepted, unless gated earlier)

- Factory disruptions in sim and live (gated: Nov 8 sim, Nov 14 live, see D15)
- Camera pose source in the live demo (H7)
- 4 robots live (H8)
- IPPO ablation, robot-only learning ablation, local vs team reward ablation
- Onboard policy execution on each robot's Pi (D14), post-SCCUR
- Isaac Sim visualization backend (`RobotBackend` implementation), post-SCCUR

## Fallback ladder for the live demo

1. MAPPO policy, 2 real robots
2. Best heuristic, 2 real robots (proves the hardware pipeline)
3. MAPPO policy on the `kinematic` backend, dashboard on a laptop
4. Backup video
