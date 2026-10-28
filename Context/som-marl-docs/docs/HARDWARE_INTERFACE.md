# Hardware interface (learning ↔ hardware contract)

The hardware team builds against this document. Keep it short and stable. Changes need agreement
from both teams. It doubles as the brief for the hardware team.

## 1. What the robots must do

**One capability:** *drive to a pose in the arena when told, report success or failure, and stop safely.*
Nothing RL-specific runs on the robots. No torch, no `som` package.

That capability is the `goto_controller` node in `ros2_ws/src/som_robot`. The learning team writes the
first version. The hardware team runs, tunes, and owns it on the robots.

## 2. Network and ROS setup

| Setting | Value | Note |
|---|---|---|
| ROS distro | Humble | already installed |
| RMW | `rmw_cyclonedds_cpp` | already in use |
| `ROS_DOMAIN_ID` | **one shared value for all robots and the PC** (e.g. `30`) | Replaces per-robot domain IDs: one PC process must reach every robot |
| Namespaces | `tb1`, `tb2`, `tb3`, `tb4` | Robot topics become `/tb1/odom`, `/tb1/cmd_vel`, `/tb1/scan`, … |
| Network | dedicated local access point / switch near the arena, PC wired | Campus WiFi and venue WiFi often block device-to-device traffic |
| Tailscale | remote access only | If ROS must run over Tailscale, use the CycloneDDS peer config below (no multicast on Tailscale) |
| Time sync | `chrony` on robots, synced to the PC (recommended) | Not required by the controller, but it helps log analysis |

CycloneDDS unicast peers (only needed over Tailscale or networks without multicast), via
`CYCLONEDDS_URI=file:///path/cyclonedds.xml`:

```xml
<CycloneDDS><Domain><General><AllowMulticast>false</AllowMulticast></General>
  <Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>50</MaxAutoParticipantIndex>
    <Peers><Peer address="PC_IP"/><Peer address="TB1_IP"/><Peer address="TB2_IP"/>
           <Peer address="TB3_IP"/><Peer address="TB4_IP"/></Peers></Discovery></Domain></CycloneDDS>
```

**Acceptance check:** from the PC, `ros2 topic hz /tb1/odom` and `/tb2/odom` both report about 30 Hz at
the same time, for 5 minutes, with no dropouts.

## 3. Frames and poses

- **Arena frame `arena`:** origin at the arena corner marked "O" on the floor tape. +x runs along the
  13 ft side, +y along the 6 ft side, z up, θ counter-clockwise from +x. Units: meters and radians.
- All station and start poses live in `configs/layouts/arena_13x6ft.yaml` (the learning repo).
  The hardware team measures them on the floor; the learning team updates the YAML.
- **Odometry mode (now):** each robot is placed at its **start pose** and bringup is (re)started, so its
  `odom` frame begins at that pose. The PC converts arena goals into that robot's odom frame using the
  start pose, then sends the goal with `header.frame_id = "odom"`.
- **Camera mode (later):** the tracker publishes `/tbN/arena_pose` (`geometry_msgs/PoseStamped`,
  `frame_id="arena"`, ≥ 10 Hz). Goals are sent with `frame_id = "arena"`. The controller rejects goals
  whose frame doesn't match its pose source.

Draft layout (all values **PLACEHOLDER** until measured):

```yaml
arena: {frame: arena, size_m: [3.96, 1.83]}
stations:
  SRC:      {kind: source,  dock: [0.35, 0.92, 0.0],  queue: [0.35, 0.60, 0.0]}
  TURNING:  {kind: machine, process: Turning,  dock: [1.30, 1.45, 0.0], queue: [1.00, 1.45, 0.0], input_capacity: 5}
  GRINDING: {kind: machine, process: Grinding, dock: [2.30, 1.45, 0.0], queue: [2.00, 1.45, 0.0], input_capacity: 5}
  MILLING:  {kind: machine, process: Milling,  dock: [1.30, 0.38, 0.0], queue: [1.00, 0.38, 0.0], input_capacity: 5}
  DRILLING: {kind: machine, process: Drilling, dock: [2.30, 0.38, 0.0], queue: [2.00, 0.38, 0.0], input_capacity: 5}
  SINK:     {kind: sink,    dock: [3.60, 0.92, 0.0],  queue: [3.60, 1.24, 0.0]}
robots:
  tb1: {start: [0.35, 1.45, 0.0]}
  tb2: {start: [0.35, 0.38, 0.0]}
  tb3: {start: [3.60, 1.45, 3.14159]}
  tb4: {start: [3.60, 0.38, 3.14159]}
travel_calibration: {v_eff: 0.15, w_eff: 1.0, dock_overhead_s: 2.0}   # updated after M4 runs
```

Rule of thumb: dock poses at least **0.5 m apart**, and queue poses at least 0.3 m from any dock.

## 4. `goto_controller` node (runs on each robot)

**Interface [contract]:**

| Direction | Name | Type |
|---|---|---|
| Action server | `/tbN/go_to_pose` | `nav2_msgs/action/NavigateToPose` (apt: `ros-humble-nav2-msgs`; only the message package, not Nav2) |
| Subscribes | `/tbN/odom` | `nav_msgs/Odometry` (pose_source = odom) |
| Subscribes | `/tbN/arena_pose` | `geometry_msgs/PoseStamped` (pose_source = arena) |
| Subscribes | `/tbN/scan` | `sensor_msgs/LaserScan` (safety) |
| Publishes | `/tbN/cmd_vel` | `geometry_msgs/Twist` |

Using the Nav2 action type means that if the team later runs real Nav2, it exposes the same action and
the PC side doesn't change.

**Behavior:**
1. On goal: validate the frame, then **ROTATE** toward the target, **DRIVE** (proportional control on
   distance and heading, re-aiming while driving), and **ALIGN** to the goal yaw (optional per goal).
2. **Succeed** when within `pos_tol` (and `yaw_tol` if aligning). Publish zero velocity.
3. **Abort** on `timeout_s`, stale pose (> 0.5 s), or blocked longer than `blocked_abort_s`.
4. **Cancel** → zero velocity immediately. A new goal preempts the current one.
5. **Safety (always on):** if any LiDAR return in the front ±25° arc is closer than `stop_dist`, publish
   zero velocity and pause; resume after it has been clear for 0.5 s. Velocities are always clamped.
6. Feedback at 5 Hz: `current_pose`, `distance_remaining`, `navigation_time`.

**Parameters (`config/goto_params.yaml`, starting values):**

```yaml
goto_controller:
  ros__parameters:
    pose_source: odom          # odom | arena
    rate_hz: 20.0
    v_max: 0.15                # m/s  (hardware max 0.22; leave margin)
    w_max: 1.2                 # rad/s
    k_v: 0.8
    k_w: 2.0
    rotate_in_place_above: 0.5 # rad heading error → rotate before driving
    pos_tol: 0.04              # m
    yaw_tol: 0.10              # rad
    timeout_s: 60.0
    stop_dist: 0.18            # m (LDS-02 min range ≈ 0.16 m)
    blocked_abort_s: 15.0
```

Launch: `ros2 launch som_robot robot.launch.py namespace:=tb1` (alongside the robot's normal TurtleBot3 bringup).

## 5. Hardware team checklist (in order)

| # | Task | Done when |
|---|---|---|
| H1 | Switch to one shared `ROS_DOMAIN_ID` + namespaces `tb1..tb4` | The §2 acceptance check passes for 2 robots |
| H2 | Set up the dedicated access point near the arena; PC wired in | No link drops in a 10 min two-robot test |
| H3 | Tape out the arena, origin "O", 6 station docks, and 4 start spots; measure coordinates | Measured values sent to the learning team for `arena_13x6ft.yaml` |
| H4 | Build and run `goto_controller` on tb1; send goals from the PC with `ros2 action send_goal` | 10 goals to each station; arrival error < 5 cm (tape measure) |
| H5 | Same on tb2; run both at once | Two robots reach separate stations simultaneously with no interference |
| H6 | Drift test: a 5 min loop SRC→T→M→D→SINK→SRC with odometry | Record final error at SRC. This decides whether odom is good enough for the demo |
| H7 | (Stretch) camera tracker publishing `/tbN/arena_pose` | `pose_source: arena` passes the H4 test |
| H8 | (Stretch) tb3, tb4 | H5 with 4 robots |

H1–H5 are needed by **Nov 9**. H6 by **Nov 12**.

## 6. Out of scope for Nov 21 (by agreement)

Nav2, SLAM/mapping, line-following tape, magnetic boundary tape, Jetson boards, Isaac Sim.
These remain on the hardware roadmap as future work.
