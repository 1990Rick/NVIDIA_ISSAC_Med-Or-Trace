# medortrace_ros

ROS 2 (Humble / Jazzy) interface of the MED-OR-TRACE object-chain verifier. The package runs the
**unmodified** `medortrace` autonomy stack against the lite simulator, the Isaac Sim digital twin or the
physical prototype. Only the middleware edge differs between them. The topic graph, QoS, TF tree and
the sim-to-real remapping table are in [`docs/ros2_message_graph.md`](../../../docs/ros2_message_graph.md).

| node | role |
|---|---|
| `autonomy_node` | Subscribes sensor and workflow topics. At 10 Hz it assembles a `SensorBundle`, runs `AutonomyStack.step` and publishes `/medortrace/cmd_vel` (already safety-gated), the acoustic probe target, item beliefs, verdicts, provenance events, safety state, scene graph, uncertainty and occupancy grids, NBV goal and path, pose, map->odom TF and diagnostics. It serves operator ack, verify-claim and explain. |
| `sim_bridge_node` | Drives a `SimBackend` (`--backend lite` without Isaac Sim, or `--backend isaac` inside Isaac Sim's python) in lockstep with the autonomy node. Publishes all sensor topics, `/clock`, odom->base_link, the latched mission and simulation ground truth. |
| `workflow_gateway_node` | Replays or tails an OR log / voice transcript (JSONL) onto `/medortrace/workflow/events`. |
| `operator_console_node` | Answers HANDOVER requests through `/medortrace/operator/ack`: `auto` (simulated operator), `rviz` (2D Pose Estimate) or `interactive` (terminal). It also prints ABSTAIN verdicts for human confirmation. |

These modules import no ROS and are unit-tested with plain pytest: `assembler.py`, `records.py`,
`mission.py`, `frames.py`, the parsing half of `workflow_gateway_node.py`, and `OperatorPolicy`.
The node shells import `rclpy` lazily, so every module imports and byte-compiles without ROS.

## Build

```bash
# ROS 2 Humble (Python 3.10) or Jazzy (Python 3.12), plus the medortrace package itself
source /opt/ros/$ROS_DISTRO/setup.bash
pip install -e /path/to/NVIDIA_ISSAC_Med-Or-Trace          # or: export MEDORTRACE_ROOT=/path/to/repo
cd /path/to/NVIDIA_ISSAC_Med-Or-Trace/ros2_ws
rosdep install --from-paths src -y --ignore-src
colcon build --symlink-install
source install/setup.bash
```

`medortrace_ros/__init__.py` finds `medortrace` in this order: `sys.path`, `$MEDORTRACE_ROOT`, then a
parent directory of the (symlinked) source file.

## Run

**Lite simulator: the full graph on a laptop**

```bash
ros2 launch medortrace_ros sim_lite.launch.py scenario:=scenarios/cf_a.yaml seed:=3 rviz:=true
#   policy:=active|fixed_route|passive  rate_factor:=0 (as fast as possible)  audit_dir:=/tmp/audit
#   workflow_log:=/path/events.jsonl    (replay an OR log instead of the simulator's own)
#   operator_mode:=auto|rviz|interactive
```

**Isaac Sim digital twin**

```bash
ros2 launch medortrace_ros isaac_sim.launch.py isaac_python:=$HOME/isaacsim/python.sh \
    scenario:=scenarios/reflective.yaml seed:=7 headless:=true isaac_graph:=true
```

The bridge runs inside Isaac Sim's `python.sh`, so that interpreter must be able to import `rclpy` and the
generated `medortrace_msgs` Python package.

* Isaac Sim 4.5 uses Python 3.10, which matches Humble: source the Humble workspace before launching.
* Isaac Sim 5.x uses Python 3.11: build `medortrace_msgs` with that interpreter
  (`colcon build --cmake-args -DPython3_EXECUTABLE=...`), or use Isaac Sim's bundled ROS 2 libraries
  (`isaacsim.ros2.bridge`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, `LD_LIBRARY_PATH` pointing at the
  extension's `humble|jazzy/lib`).

To drive the twist subscriber of the OmniGraph instead of the lockstep bridge:
`./python.sh scripts/isaac/ros2_sim.py --bridge medortrace_ros.sim_bridge_node:create`, plus
`isaac_sim.launch.py isaac_python:=none` for the other nodes.

**Physical prototype**

```bash
# once: export the survey (prior map, slots, zones, fiducials, items, staff roster) as a mission file
python -c "from medortrace_ros.mission import mission_for_scenario, save_mission_file; \
save_mission_file(mission_for_scenario('scenarios/nominal.yaml', 0), '/tmp/or_survey.json')"
ros2 launch medortrace_ros robot.launch.py mission_file:=/tmp/or_survey.json \
    lidar_topic:=/ouster/points odom_topic:=/odom cmd_vel_topic:=/cmd_vel radar_input:=pointcloud2 \
    workflow_log:=/var/log/or/case.jsonl operator_mode:=rviz
```

A real OR supplies the survey from its facility model. The schema is documented in `mission.py`.

**Services**

```bash
ros2 service call /medortrace/verification/verify_claim medortrace_msgs/srv/VerifyClaim \
    "{item_id: sponge_2, slot_id: kick_bucket_1:inside, deadline_s: 20.0}"
ros2 service call /medortrace/verification/explain medortrace_msgs/srv/ExplainVerdict "{claim_id: count2_sponge_2}"
ros2 service call /medortrace/operator/ack medortrace_msgs/srv/OperatorAck \
    "{operator_id: rn_smith, relocalize: true, pose_hint: {x: 4.1, y: 2.0, theta: 1.57}}"
```

## Workflow gateway input (JSONL)

```json
{"t": 42.0, "type": "handoff", "item_id": "sponge_2", "src": "hand:scrub_nurse", "dst": "field:top", "event_id": "or_17", "reported_t": 43.1}
{"kind": "workflow_log", "t": 42.0, "type": "handoff", "item": "sponge_2", "src": "hand:scrub_nurse", "dst": "field:top", "id": "wf_003"}
{"kind": "voice", "t": 61.5, "text": "sponge two into kick bucket one", "speaker": "surgeon", "asr_confidence": 0.8}
```

The second form is the `events.jsonl` written by `medortrace.data.writer`. Its `truth_move`, `verdict`
and similar lines are ignored, because the robot never sees the truth. Voice lines are parsed with the
vocabulary in `config/voice_vocabulary.yaml`.

## Configuration

* `config/params.yaml`: parameters of all four nodes. The inline comments explain each one; staleness
  limits are under `max_age.*`.
* `config/qos.yaml`: QoS profiles and per-topic overrides. The topic -> profile assignment is in
  `medortrace_ros/topics.py`.
* `config/voice_vocabulary.yaml`: spoken phrases mapped to item and slot ids.
* `config/medortrace.rviz`: occupancy belief, lidar, EKF pose, ground truth, NBV path, and the
  2D Pose Estimate tool for the operator console.

## Tests (no ROS needed)

```bash
cd /path/to/NVIDIA_ISSAC_Med-Or-Trace
python -m pytest ros2_ws/src/medortrace_ros/test -q
```

* `test_assembler.py` covers the delivery semantics per channel, staleness, timing, resets and
  bookkeeping. It also has a loopback test that replays a 3 s lite episode message by message through the
  assembler into `AutonomyRuntime`, after a mission JSON round trip, and requires the commands to match
  the in-process episode loop exactly. The streamed provenance events must re-verify as a hash chain.
* `test_ros_free_modules.py` checks:
  * interface files: rosidl naming rules, CMake registration, coverage of the dataclasses, constants;
  * the topic registry against `medortrace.isaac.ros2_bridge.TOPICS`, and the QoS classes;
  * a lossless lidar layout, and scan-pattern refill for driver clouds;
  * conversions on duck-typed messages;
  * mission JSON (strict and lossless);
  * the verify / explain / operator-ack services and the audit export;
  * the voice grammar, the gateway record formats and the operator policy.

## Known limitations

* The node shells, launch files and `*_to_ros` converters could not be run against a real ROS 2
  installation in the development container. They were exercised end to end on a strict fake-rclpy
  harness only, which is not part of the repository.
* In `scripts/isaac/ros2_sim.py` mode, the loop steps the backend with an idle `VelocityCommand`. The
  acoustic probe target published by the autonomy node therefore never reaches `IsaacBackend` there. Use
  the lockstep bridge (`isaac_sim.launch.py`) when acoustic probing matters.
* The stack models each sensor by its rig mount (`configs/robot/rig.yaml`), not by a TF lookup. Driver
  frame ids and extrinsics must match the rig.
