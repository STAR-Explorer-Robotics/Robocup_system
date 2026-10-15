# Software contributions and reuse

The contribution of this collection is the implementation and adaptation of navigation and sensing components for rescue-robot tasks. It reuses established estimation and graph-planning methods.

| Area | Reused foundation | Work represented in this snapshot |
| --- | --- | --- |
| State estimation | SuperOdom/SuperLoc and Lightning-LM | JT128 point layout/timing adaptation, frame conversion, configuration, per-frame and dense-cloud output interfaces |
| Terrain representation | Point-cloud and elevation-grid primitives | Vertical-layer handling, grounded obstacles, virtual no-go regions, reachable boundaries and frontier information |
| Terrain planning | TRG-planner | Allowed-area integration, configurable graph construction, ROS 2 interfaces, optional heading/footprint costs |
| Path execution | Standard path-tracking concepts | Vector-field and pure-pursuit implementations, velocity regulation and stale-input/deviation handling |
| Robot interface | Official Unitree SDK2 | ROS command conversion, rate/velocity limits, command timeout and execution services |
| Task coordination | ROS 2 services/topics | Candidate goal/path validation, stage transitions, retry behavior and recorded/planned path selection |
| Perception | Existing detection/decoding and ONNX inference components | Combined sensing runtime, temporal confirmation, evidence capture and ROS publication |

## Boundaries of the claims

- This collection does not introduce a new LiDAR–inertial estimator or a new vendor locomotion policy.
- Current C++ TRG native edges use symmetric edge weights. Optional climb/slope terms use absolute height differences. The snapshot does **not** establish a separate uphill-versus-downhill cost model.
- Optional planner features must be distinguished from the mode enabled in a particular configuration.
- Mission coordination includes operator participation. Specify which stages are autonomous when describing a deployment.
- A software implementation, a successful module example, and a competition performance measurement are different forms of evidence.
- Release preparation in October does not establish that an algorithm was added after RoboCup. Dated implementation/deployment evidence is required for that distinction.

## Paper-facing description

A description consistent with the intended release scope is:

> A modular ROS 2 navigation implementation for rescue robots, combining established LiDAR–inertial estimation and terrain-graph planning with platform-specific sensor interfaces, terrain preprocessing, path execution and mission coordination.

Use present-tense public-release wording only when the corresponding source collection is actually available. Competition results should identify the evaluated platform/configuration and should not be attributed to individual modifications without an appropriate comparison.

