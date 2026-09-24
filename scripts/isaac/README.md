# MED-OR-TRACE in NVIDIA Isaac Sim

Entry points for the Isaac Sim digital twin. The autonomy stack, scene specification, workflow,
fault model and metrics are shared with the lite simulator, so a registry `scenario_id` is the
same experiment on both backends: same scene, workflow, faults and RNG streams. The two backends
still produce different measurements, because their physics and sensor models differ.

| Script | Needs Isaac Sim | Purpose |
|---|---|---|
| `run_episode_isaac.py` | yes | one episode through `eval.runner.run_episode(backend="isaac")`, optional ROS 2 monitoring graph and dataset export |
| `generate_synthetic_data.py` | yes (`--dry-run`: no) | Replicator dataset with causal labels and matched counterfactual frame pairs |
| `validate_sensor_configs.py` | yes (`--static-only`: no) | profile/rig/material checks, then creates every RTX/physics sensor and prints resolved names and output shapes |
| `ros2_sim.py` | yes (+ ROS 2) | twin driven by an external `/medortrace/cmd_vel` |
| `check_offline.py` | no | causal lock / nuisance randomiser / pure-helper checks with `usd-core` |

## Running

Isaac Sim ships its own Python. Run the scripts with `python.sh` (Linux) or `python.bat`
(Windows) from the Isaac Sim installation. Each script puts the repository root on `sys.path`
through `scripts/isaac/_bootstrap.py`, so no install into Isaac's interpreter is needed. The
project's pure-Python dependencies (numpy, scipy, pyyaml) are already in Isaac's Python.

```bash
ISAAC=~/isaacsim            # 4.5: ~/.local/share/ov/pkg/isaac-sim-4.5.0 ; 5.x: pip or workstation install
REPO=/path/to/NVIDIA_ISSAC_Med-Or-Trace
cd $REPO

# 1. configuration sanity (fast, no Isaac Sim)
python scripts/isaac/validate_sensor_configs.py --static-only
PYTHONPATH=. python scripts/isaac/check_offline.py

# 2. sensors inside Isaac Sim (prints command/annotator names + output shapes)
$ISAAC/python.sh scripts/isaac/validate_sensor_configs.py --frames 10

# 3. an episode (same registry entry as the lite benchmark)
$ISAAC/python.sh scripts/isaac/run_episode_isaac.py --scenario-id cf_b__p0000__real_obstacle \
    --duration 60 --export runs/isaac/cf_b_p0000
$ISAAC/python.sh scripts/isaac/run_episode_isaac.py --config scenarios/nominal.yaml --seed 7 \
    --no-headless --ros2 --staff-mode people --detector model:checkpoints/det.pt

# 4. synthetic data (both arms of every selected pair are rendered)
$ISAAC/python.sh scripts/isaac/generate_synthetic_data.py --factor CF-B --split train --limit 10 \
    --frames 40 --out datasets/cfb_train
python scripts/isaac/generate_synthetic_data.py --factor CF-B --limit 1 --frames 4 --t-max 30 \
    --dry-run --out /tmp/cfb_plan          # plan + labels only, no Isaac Sim

# 5. ROS 2 (source ROS 2 Humble/Jazzy first, or use Isaac's bundled ROS 2 libraries)
$ISAAC/python.sh scripts/isaac/ros2_sim.py --scenario-id nominal__0000 --realtime
ros2 topic pub -r 10 /medortrace/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}"
```

Common options: `--headless/--no-headless`, `--prims-api stable|experimental`
(`MEDORTRACE_ISAAC_PRIMS`) and `--physics-hz`. To select a scenario, pass `--scenario-id` for a
registry entry, or `--config` with `--seed`. `--override '<json>'` is deep-merged into the
config.

### What `run_episode_isaac.py` writes (`--export DIR`)
* `DIR/episodes/<scenario>__s<seed>__<policy>/`: the dataset schema of `medortrace.data.writer`,
  with `meta.json:backend = "isaac"`.
* `DIR/usd/`: the authored stage and robot rig.
* `DIR/isaac_run.json`: enabled extensions, resolved sensor commands, annotators and render
  products, the non-visual token rewrites that were applied, and the metrics.

Backend options (`staff_mode`, `detector`, `drive`, ...) are set with
`IsaacBackend.configure()` and are not part of the scenario config, so `cfg_hash` matches the
lite run.

### Synthetic data layout (`generate_synthetic_data.py`)
```
OUT/<scenario_id>/{rgb,semantic,instance,depth,bbox2d,camera,labels}/<frame>.*
OUT/usd/<scenario_id>.usda
OUT/index.json      # frames[] with the registry split, pairs{pair_frame_key: [frame_uid, ...]},
                    # nuisance_mismatched_pairs (must be empty)
```
* The render unit is a counterfactual pair, or a single scenario. The frame times, camera
  viewpoints and nuisance seed are drawn once per unit. Frame `k` of each arm is therefore the
  same view with the same nuisance, and the arms differ only through the hidden cause.
* Splits come from the registry and are assigned per pair. The index builder refuses to write
  an index where a matched frame pair spans two splits.
* `labels/<frame>.json` holds:
  * scenario, family, seed, pair id, split and `cfg_hash`
  * the hidden factor and value, including the CF-B `aisle_point`
  * each item's truth slot and position at the frame time
  * staff positions and the fault state
  * causal and nuisance signatures, and the fog settings
  * camera pose (position, yaw/pitch, quaternion, 4x4 matrix) and intrinsics
* Causal safety: every item or staff move goes through `CausalLock.causal_edit()`, and
  `NuisanceRandomizer.apply()` verifies the lock before and after every randomiser. See
  `medortrace/isaac/replicator_randomizers.py`.

## Version-dependent API names

Every name that differs between releases is resolved in `medortrace/isaac/compat.py`, or in the
adapter that uses it. The newest name is tried first.

| Concern | Isaac Sim 4.5 / 5.x | Legacy (≤ 4.2) | Where |
|---|---|---|---|
| App | `isaacsim.SimulationApp` | `omni.isaac.kit.SimulationApp` | `compat.simulation_app_cls` |
| World | `isaacsim.core.api.World` | `omni.isaac.core.World` | `compat.world_cls` |
| Articulation | `isaacsim.core.prims.SingleArticulation`; 5.x experimental `isaacsim.core.experimental.prims.Articulation` (`--prims-api experimental`) | `omni.isaac.core.articulations.Articulation` | `compat.articulation_cls` |
| Xform prims | `isaacsim.core.prims.SingleXFormPrim`; 5.x `isaacsim.core.experimental.prims.XformPrim` (batched, warp arrays) | `omni.isaac.core.prims.XFormPrim` | `compat.XformGroup` |
| ArticulationAction | `isaacsim.core.utils.types` | `omni.isaac.core.utils.types` | `compat.articulation_action_cls` |
| IMU / contact | `isaacsim.sensors.physics.{IMUSensor,ContactSensor}` | `omni.isaac.sensor` | `compat` |
| Stage open | `isaacsim.core.utils.stage.open_stage/is_stage_loading` | `omni.isaac.core.utils.stage` | `compat.open_stage` |
| Assets root | `isaacsim.storage.native.get_assets_root_path` | `omni.isaac.nucleus` | `compat.assets_root_path` |
| Extensions | `isaacsim.sensors.rtx`, `isaacsim.sensors.physics`, `isaacsim.robot.wheeled_robots`, `isaacsim.ros2.bridge` | `omni.isaac.sensor`, `omni.isaac.wheeled_robots`, `omni.isaac.ros2_bridge` | `compat.EXTENSION_ALIASES` |
| RTX lidar annotator | `IsaacCreateRTXLidarScanBuffer` (5.x) | `RtxSensorCpuIsaacCreateRTXLidarScanBuffer` (4.x) | `sensors.LIDAR_ANNOTATORS` |
| RTX radar annotator | `IsaacComputeRTXRadarPointCloud` (5.x) | `RtxSensorCpuIsaacComputeRTXRadarPointCloud` (4.x) | `sensors.RADAR_ANNOTATORS` |
| RTX acoustic | `IsaacSensorCreateRtxAcoustic` / `IsaacSensorCreateAcoustic` (experimental) | none | `sensors.ACOUSTIC_COMMANDS` |
| OmniGraph ROS 2 nodes | `isaacsim.ros2.bridge.*`, `isaacsim.core.nodes.*`, `isaacsim.robot.wheeled_robots.DifferentialController` | `omni.isaac.ros2_bridge.*`, `omni.isaac.core_nodes.*`, `omni.isaac.wheeled_robots.*` | `compat.ros2_node_namespaces` |
| camera_info publisher | `ROS2CameraInfoHelper` | `ROS2CameraHelper(type="camera_info")` | `ros2_bridge.build_ros2_graph` (automatic fallback) |
| Replicator step | `rep.orchestrator.step(rt_subframes, delta_time=0.0, pause_timeline=True)` | `rep.orchestrator.step(rt_subframes)` | `generate_synthetic_data.ReplicatorCapture` |
| RTX fog carb settings | `/rtx/fog/{enabled,fogColorIntensity,fogStartDist,fogEndDist}` | same | `NuisanceRandomizer.fog_settings` |
| People characters | `omni.anim.people` + `omni.anim.graph.core`, assets `Isaac/People/Characters/{M,F}_Medical_01` | `isaacsim.anim.people` alias | `staff.CHARACTERS` |

## Design notes
* **Timing.** `World(physics_dt=1/120, rendering_dt=dt)`. One `World.step(render=True)` per
  control tick runs the 12 PhysX substeps internally, and RTX sensors see exactly one control
  period per rendered frame.
* **Lidar contract.** RTX returns are re-binned onto the lite simulator's fixed ray grid:
  `ring × azimuth`, with `inf` where a ray had no return. This keeps free-space carving and
  ghost reasoning unchanged. The rings come from the profile's 16 emitters, and the azimuth
  resolution from `cfg.sensors.lidar.az_res_deg`.
* **Camera contract.** Rays are exact pinhole rays rotated by the mast pitch. Range is Euclidean
  (depth × ray norm). The `gt_surrogate` detector applies the lite detection, confusion and tag
  model to Replicator tight boxes. Visibility comes from RTX `occlusionRatio`, and the item
  identity from `primPaths`.
* **Custody truth.** Items are kinematic and are teleported with the lite placement rules
  (`medortrace/isaac/truth.py`, with the same per-item offsets). Visibility is toggled when an
  item enters or leaves the room.
* **RNG parity.** The battery start charge, the workflow-log latencies and the item offsets use
  the same streams and forks as `LiteBackend`.
* **LOS rays.** Landmark and acoustic line-of-sight rays start 0.35 m outside the robot and
  ignore `/World/Robot` colliders. A ray that starts inside the mast would report a
  zero-distance hit.
* **Non-visual materials.** `validate_sensor_configs.py` checks
  `omni:simready:nonvisual:{base,coating,attributes}` against the RTX vocabulary in
  `medortrace/isaac/sensors.py`. The backend rewrites project tokens that are outside the
  vocabulary in the opened stage (`steel_stainless` → `steel`, `glass` → `clear_glass`). The
  authored USD is left unchanged.

## Known limitations
* None of the Isaac-side code has run inside Isaac Sim in this repository's CI; it has only been
  compiled. What was checked here (`check_offline.py`, `--static-only`, `--dry-run`) covers the
  numpy/pxr logic only. Annotator output field names (`data`, `distance`, `intensity`,
  `objectId`, `radialVelocities`, `rcs`, `occlusionRatio`, `primPaths`) are read defensively.
  Confirm them with `validate_sensor_configs.py` on your release.
* **Custom lidar/radar profiles.** These are registered through the carb settings
  `/app/sensors/nv/{lidar,radar}/profileBaseFolder` and created with
  `IsaacSensorCreateRtx{Lidar,Radar}(config=<profile>)`. Isaac Sim 5.x moves RTX sensors to USD
  `OmniLidar`/`OmniRadar` prims and may ignore JSON profiles. If the resolved profile is not
  `rtx_lidar_or16`, author the sensor attributes on the prim instead.
* **Acoustic.** The measurement always comes from the calibrated PhysX echo model. The RTX
  acoustic prim is created only when the experimental extension exists, and only for
  visualisation.
* **Landmarks** are a ground-truth fiducial surrogate with PhysX line-of-sight. No AprilTag
  detection is run on RGB.
* **People mode** is experimental. It needs the Isaac People assets and, for walk cycles,
  `omni.anim.graph.core`. On any failure it falls back to capsules. The dataset generator always
  uses capsules.
* **Items.** Items are kinematic, so they cannot be knocked over or grasped physically. The arm's
  contact gate sees only joint efforts.
* **Lidar ground truth.** `gt_is_ghost` is not available from RTX, so ghost precision and recall
  metrics are lite-only.
* **ROS 2.** The custom-message topics (radar, acoustic, workflow, verification) come from
  `medortrace_ros`'s sim bridge node, which is not part of these scripts. Hook it in with
  `ros2_sim.py --bridge module:factory`.
* **Non-visual vocabulary.** The token list follows the SimReady non-visual material spec as
  documented for Isaac Sim 4.5/5.x. Verify it against your release's documentation.
