# Project context

Background, constraints, and a decision log, captured from the planning conversation (Oct 19–20, 2026).
If you want to know *why* the repo is shaped the way it is, it is here.

## The lab and project

- **Lab:** Design-is-ALL Lab, California State University, Fullerton. Faculty mentor: Prof. Bingling Huang
  (Mechanical Engineering).
- **Funded project:** "Smart Manufacturing System Empowered by RL" (EG-RSCA 2026), in collaboration with USC.
- **Research framing:** self-organizing manufacturing (SOM). Transport robots and machines make their own
  decisions, with no central scheduler.
- **Presentation:** SCCUR 2026, **Nov 21, 2026**. Presenters on the abstract: Daniel Lopez, Jim Alvarez,
  Paige Saychareun, Jocelyn Mendoza.

### Teams (as of Oct 2026)

- **Learning + Sim:** Daniel Lopez plus two teammates. They have no access to earlier sim/learning plans,
  so this repo starts clean.
- **Hardware:** Matthew Nassif and team (the hardware deck also lists Katherine Martinez, Claudia G.L.,
  Shaan Singh, Isaiah Portillo and Haley Barrios Gomez). They own the robots, the network and the arena.

## The submitted abstract (SCCUR 2026)

**Title:** *Multi-Agent Reinforcement Learning for Decentralized Scheduling in Self-Organizing Manufacturing*

Claims, from the legacy repo:
- Simulated factory: 2 transport robots + 4 machining stations (turning, grinding, milling, drilling).
  Each is a learning agent with partial observations, trained with IPPO and action masking.
- A 160k-step run on a 10-order schedule. Reward rose to about 3,500, with 9.6 of 10 orders delivered per
  episode over the final 20 episodes.
- Only **38% of delivered orders (100 of 266) were on time** in the last 20k steps.
- Next steps: a rule-based dispatching baseline, then transfer to Isaac Sim and TurtleBot3 hardware.

**The numbers were checked against the log** (after removing a duplicated block): the last 20k steps have
267 unique deliveries, 100 of them on time. The abstract is accurate.

**We may change the research if results improve.** The human confirmed this. Presenting "since the
abstract we redesigned X, and on-time delivery went from 38% to Y%" is the intended story. The title
(decentralized MARL scheduling in SOM) must still describe the work.

## Hardware facts (from the hardware team's decks, Oct 2026)

| Item | Value |
|---|---|
| Robots | 4× TurtleBot3 **Burger** built |
| Compute | Raspberry Pi 4 + OpenCR 1.0 |
| OS / ROS | Ubuntu 22.04, **ROS2 Humble** |
| Sensor | LDS-02 2D LiDAR (360°) |
| Motion | Max 0.22 m/s linear, 2.84 rad/s angular |
| Footprint | 138 × 178 × 192 mm |
| Battery | 11.1 V LiPo, about 2 h |
| Middleware | CycloneDDS; Tailscale VPN between the PC and the robots (campus WiFi issues) |
| Isolation | Per-robot `ROS_DOMAIN_ID` **and** namespaces (we recommend changing this; see HARDWARE_INTERFACE) |
| Working today | Keyboard teleop; hand-set `/cmd_vel`; two robots move in coordination |
| Not working | Some PC↔robot links are flaky; no autonomous go-to-goal; no Nav2; no map |
| Their roadmap | P1 build → P2 multi-robot motion → P3 obstacle stop → P4 Nav2 → P5 coordination → P6 RL |

**Planned testbed (parts being ordered):**
- 13 × 6 ft arena (≈ 3.96 × 1.83 m) with an 8 ft 80/20 aluminum frame overhead.
- 2× FLIR Blackfly S USB3 global-shutter cameras plus AprilTags on each robot, giving live arena-frame
  poses on the Ubuntu PC.
- A dedicated access point / PoE switch wired near the arena.
- Reflective line tape and magnetic boundary tape (we recommend deferring these).

**Main PC:** Ubuntu 22.04.5 LTS. This is where the coordinator runs.

## Decision log

| # | Decision | Why |
|---|---|---|
| D1 | **Fresh repo.** The legacy code is reference only | It is hard-coded to one layout, its AGV actions are 8-direction moves (they can't transfer to real robots), and its trainers have correctness bugs. See LEGACY_REVIEW |
| D2 | **Task-level actions.** Robots choose a transport job; machines choose a queued order | A real robot can only execute "go to station X". This also makes the research about scheduling and deadlines, which is the open problem |
| D3 | **Keep the abstract's factory** (SRC → turning/grinding/milling/drilling → SINK, 2 robots, 10-order schedule) as the headline scenario | Lets the old 38% sit beside the new number and the baselines on one chart |
| D4 | **Event-driven core, shared by training and the live lab** | No sim/real logic drift. The lab demo is the same state machine with wall-clock events |
| D5 | **MAPPO is the algorithm (decided Oct 27).** Centralized critic during training; decentralized actors at execution. IPPO stays available as a config flag (`critic_mode: local`) for an *optional* ablation only | MAPPO gives a clearer coordination signal, and execution is still decentralized (centralized training, decentralized execution). The abstract reported IPPO, so an IPPO run is a cheap "before" comparison if time allows |
| D6 | **Shared parameters per role**, with obs and state padded to `max_robots=4` | Two robots now, four later, without retraining from scratch |
| D7 | **Machines remain learning agents.** "Machines on a fixed rule" becomes a baseline | Matches the "heterogeneous agents" claim; the comparison shows whether learning machines help |
| D8 | **Few reward terms, team reward by default** | Legacy had ~25 shaped terms and forced actions, so reward stopped meaning anything |
| D9 | **No Isaac Sim, Nav2, or SLAM for Nov 21** | Two people, 4.5 weeks. In an open 13×6 ft arena, LiDAR localization has little to see. Isaac is post-November future work |
| D10 | **Motion = a simple go-to-pose controller.** Pose comes from **odometry now**, and the **overhead camera later** (swap by parameter) | Needs no purchases to start. The camera rig may not be ready by Nov 21 |
| D11 | **Controller exposes `nav2_msgs/action/NavigateToPose`** | It is a standard interface. If the team adopts Nav2 later, the coordinator doesn't change |
| D12 | **One shared `ROS_DOMAIN_ID` with namespaces `/tb1..tb4`**, on a dedicated local access point. Tailscale only for remote access | One PC process must reach every robot. Separate domain IDs prevent that |
| D13 | **Record a backup video** of the live demo by Nov 16 | Live robots at a conference venue are risky |
| D14 | **Policies are hosted on the lab PC for the demo.** Each agent's actor still sees only its own local observations. Onboard execution on each robot's Pi is a future option | The machines are virtual and live on the PC, so the factory state is there. Talk wording: "decentralized policies, hosted on one machine for the demo" |
| D15 | **Factory disruptions (machine down/up, rush orders) are a stretch goal, gated.** The core supports `MachineDown`/`MachineUp` from M1. Train with disruptions only if MAPPO beats the baselines by **Nov 8**. Live disruptions only if the normal 2-robot run is reliable by **Nov 14** | Strongest "self-organizing" demo moment, but the core result must not be put at risk |
| D16 | **No mid-trip job switching (preemption).** A robot commits to its job until drop-off. Known limitation, not a bug | Keeps training simple for November. Possible extension later |
| D17 | **Physical reactivity is the controller's job, not RL's.** LiDAR stop-and-wait only for November; no learned obstacle avoidance | Learned motion needs physics sim + sim-to-real; out of scope |

## Open questions (waiting on humans)

1. **10-order config.** The abstract run used `configs/ippo_delivery_10orders.yaml` (not provided). Order IDs,
   job types (routes) and due times are recovered from the log; **arrival times and process times are
   unknown.** The scenario file marks them `# PLACEHOLDER`.
2. **Arena availability.** Is the 13×6 ft space taped out now? We need real station coordinates.
3. **Camera rig ETA.** When will the cameras and frame be ordered and arrive? This decides whether the
   camera pose source is in or out for Nov 21.
4. **Domain ID setup.** Will the hardware team switch to a shared domain + namespaces? (D12)
5. **Demo format.** Live robots at SCCUR, or recorded video plus a live sim dashboard? Plan for both.

## Glossary

- **SOM:** self-organizing manufacturing.
- **Transport job:** move order *o* from the station where it waits to the next station on its route
  (or to SINK when its route is finished).
- **Decision point:** a moment where some agent is idle and has at least one feasible non-wait action.
- **Slack:** `due − now − estimated remaining work`. Negative slack means the order will be late.
- **Tardiness:** `max(0, delivered_time − due)`.
