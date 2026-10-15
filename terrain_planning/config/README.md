<div align="center">
  <h1 align="center">TRG-planner<br></h1>
  <a href="https://github.com/url-kaist/TRG-planner"><img src="https://img.shields.io/badge/-C++-blue?logo=cplusplus" /></a>
  <a href="https://github.com/url-kaist/TRG-planner"><img src="https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54" /></a>
  <a href="https://github.com/url-kaist/TRG-planner"><img src="https://img.shields.io/badge/ROS1-Noetic-blue" /></a>
  <a href="https://github.com/url-kaist/TRG-planner"><img src="https://img.shields.io/badge/ROS2-Humble-blue" /></a>
  <a href="https://github.com/url-kaist/TRG-planner"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
  <br/>
<a href="https://ieeexplore.ieee.org/document/10819646"><img src="https://img.shields.io/badge/RA--L-10819646-004088.svg"/></a>
<a href="https://arxiv.org/abs/2501.01806"><img src="https://img.shields.io/badge/arXiv-2501.01806-b33737.svg"/></a>
  <br/>
  <br/>
  <p align="center">
      <img src="https://github.com/user-attachments/assets/7d1b97c0-ed94-47c3-ac20-a92d7039fea4" alt="TRG-planner banner" width=80%></a>
      <br>
      <p><strong><em>Most versatile safe-aware path planner.</em></strong></p>
  </p>
</div>

______________________________________________________________________

# RoboCup TRG Configuration

Use one scene-independent hyperparameter file:

```text
config/robocup_default.yaml
```

Do not add one YAML per map or scene. The prior-map PCD path is injected at
runtime by ROS launch (`map_path:=...`) or by the RoboCup bringup scripts, which
write a run-specific snapshot under `run_output/<run_name>/`.

### Baseline Configuration

```yaml
isVerbose: false
timer:
  graphRate: 5.0
  planningRate: 10.0
map:
  isPrebuiltMap: true
  prebuiltMapPath: ""
  waitPoseBeforeInit: true
  isVoxelize: true
  voxelSize: 0.08
trg:
  isPrebuiltTRG: false
  prebuiltTRGPath: ""
  isUpdate: false
  expandDist: 0.65
  robotSize: 0.32
  sampleNum: 16
  randomSeed: 7
  deterministicSampling: true
  heightThreshold: 0.30
  collisionThreshold: 0.12
  updateCollisionThreshold: 0.5
  safetyFactor: 3.0
  goalTolerance: 0.6
  pathSearchMode: native
trgAStar:
  fallbackToNative: true
  headingBins: 8
  lengthWeight: 1.0
  riskWeight: 3.0
  climbWeight: 0.35
  slopeWeight: 0.35
  turnWeight: 0.2
  heuristicWeight: 1.0
  lengthScaleM: 0.65
  climbScaleM: 0.30
  slopeScaleTan: 0.7
  turnScale: 0.25
  maxEdgeClimbM: 0.0
  maxEdgeSlopeTan: 0.0
  footprintCostEnabled: true
  footprintRejectInvalid: false
  robotLengthM: 0.70
  robotWidthM: 0.43
  maxBodyHeightDiffM: 0.70
  maxBodyTiltDeg: 35.0
  maxInteriorPenetrationM: 0.30
  bodyHeightWeight: 0.35
  bodyTiltWeight: 0.35
  bodyPenetrationWeight: 0.25
  bodyInvalidWeight: 4.0
  footprintSampleStepM: 0.05
  footprintEdgeBandM: 0.06
boundary:
  enabled: false
  allowedAreaPath: ""
  keepoutMargin: 0.0
```

### Parameter Descriptions

| Parameter                 | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| `isVerbose`               | Flag to enable or disable verbose logging                    |
| `timer.graphRate`         | Rate of Graph finite state machine (Hz)                      |
| `timer.planningRate`      | Rate of Planning finite state machine (Hz)                   |
| `map.isPrebuiltMap`       | Flag to indicate whether a prebuilt map is used              |
| `map.prebuiltMapPath`     | Runtime prior-map PCD path; keep empty in the source template |
| `map.waitPoseBeforeInit`  | Wait for odometry/pose before initializing a prebuilt-map graph |
| `map.isVoxelize`          | Flag to indicate whether voxelization is applied to the map  |
| `map.voxelSize`           | Size of each voxel in the voxelized map (if applicable)      |
| `trg.isPrebuiltTRG`       | Flag to indicate whether a prebuilt TRG is used (TBU)        |
| `trg.prebuiltTRGPath`     | Path to the prebuilt TRG file (TBU)                          |
| `trg.isUpdate`            | Flag to indicate whether the TRG is updated during operation |
| `trg.expandDist`          | Distance used to expand the TRG                              |
| `trg.robotSize`           | Size of the robot used in the TRG                            |
| `trg.sampleNum`           | Number of samples to generate for the TRG                    |
| `trg.randomSeed`          | Fixed seed for repeatable graph sampling; use `-1` for random startup seed |
| `trg.deterministicSampling` | Use fixed angular samples per node instead of random angular samples |
| `trg.heightThreshold`     | Threshold value for height in the TRG                        |
| `trg.collisionThreshold`  | Threshold for collision detection in the TRG                 |
| `trg.updateCollisionThreshold` | Threshold for updating collision in the TRG             |
| `trg.safetyFactor`        | Safety factor applied during the planning process            |
| `trg.goalTolerance`       | Tolerance for goal reaching in the planning process          |
| `trg.pathSearchMode`      | `native` keeps original TRG A*; `trg_astar` enables the migrated Path_Planing_QRC A* cost |
| `trgAStar.fallbackToNative` | Fall back to native A* when TRG-AStar cannot find a path   |
| `trgAStar.headingBins`    | Direction bins for heading-aware state expansion             |
| `trgAStar.*Weight`        | Length, risk, climb, slope, turn, and heuristic cost weights |
| `trgAStar.*Scale*`        | Normalization scales for length, climb, slope, and turn costs |
| `trgAStar.maxEdgeClimbM`  | Optional hard per-edge climb limit; `0.0` disables it        |
| `trgAStar.maxEdgeSlopeTan`| Optional hard per-edge slope tangent limit; `0.0` disables it |
| `trgAStar.footprintCostEnabled` | Add rectangular body feasibility costs from `astar_go2w_rect.py` |
| `trgAStar.footprintRejectInvalid` | Reject edges with invalid footprint samples instead of only penalizing them |
| `trgAStar.robotLengthM` / `robotWidthM` | Rectangular body footprint dimensions used by the migrated cost |
| `trgAStar.maxBodyHeightDiffM` | Front/rear corner-pair height-difference limit for footprint cost |
| `trgAStar.maxBodyTiltDeg` | Left/right body tilt limit for footprint cost |
| `trgAStar.maxInteriorPenetrationM` | Solid-body interior penetration limit for footprint cost |
| `trgAStar.body*Weight`    | Penalty weights for body height, tilt, penetration, and invalid footprint ratio |
| `trgAStar.footprintSampleStepM` | Along-edge sampling step for footprint metrics |
| `trgAStar.footprintEdgeBandM` | Edge/corner band width used when sampling footprint support from the PCD |
| `boundary.enabled`        | Enable allowed-area polygon constraint                       |
| `boundary.allowedAreaPath`| YAML file containing `allowed_area` points                   |
| `boundary.keepoutMargin`  | Optional extra inside-boundary margin in meters              |

Allowed-area YAML:

```yaml
frame_id: map
keepout_margin: 0.0
allowed_area:
  - {x: 0.0, y: 0.0}
  - {x: 4.0, y: 0.0}
  - {x: 4.0, y: 2.0}
  - {x: 0.0, y: 2.0}
```

______________________________________________________________________

## 📝 Citation

If you use this package for any academic work, please cite our original [paper](https://ieeexplore.ieee.org/document/10819646), or [arxiv](https://arxiv.org/abs/2501.01806)

```bibtex
@article{lee2025trg,
      title     = {{TRG-planner: Traversal risk graph-based path planning in unstructured environments for safe and efficient navigation}},
      author    = {Lee, Dongkyu and Nahrendra, I Made Aswin and Oh, Minho and Yu, Byeongho and Myung, Hyun},
      journal   = {IEEE Robotics and Automation Letters},
      volume    = {10},
      number    = {2},
      pages     = {1736--1743},
      year      = {2025},
      publisher = {IEEE}
    }
```

______________________________________________________________________

## 📜 License

The TRG-planner code provided in this repository is released under the [Apache-2.0 with Commons Clause license](../LICENSE).
