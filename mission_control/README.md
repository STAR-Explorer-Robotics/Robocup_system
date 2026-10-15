# Mission control

The navigation task layer, organized as a standard ROS 2 Python package.

## Included functionality

- Operator target and frontier candidate selection with planner-path validation.
- Navigation-stage state machine, retries and leg completion.
- End-of-lane/frontier-based completion information.
- Recorded-path management and selection between planned and recorded paths.

## Interfaces

**Inputs:** Estimator/body odometry, TRG graph/path outputs, terrain frontiers, tracker state and operator stage requests.

**Outputs:** Planner goals, selected follower paths, mission state and control-service requests.

See the shared [interface reference](../docs/interfaces.md) before connecting modules.

## Source map

| File | Role |
| --- | --- |
| [mission_control/goal_selector.py](mission_control/goal_selector.py) | Candidate ranking and path validation |
| [mission_control/mission_fsm.py](mission_control/mission_fsm.py) | Task state transitions |
| [mission_control/end_detector.py](mission_control/end_detector.py) | Completion information |
| [mission_control/recorded_path_manager.py](mission_control/recorded_path_manager.py) | Recorded-route processing |
| [mission_control/path_source_mux.py](mission_control/path_source_mux.py) | Path selection |
| [config/mission.yaml](config/mission.yaml) | Task and interface parameters |
| [launch/mission.launch.py](launch/mission.launch.py) | Five-node module launch |

## Dependencies

Python 3.10+, ROS 2 Humble, rclpy, standard geometry/navigation/sensor/visualization messages and standard services.

## Scope and attribution

Operator targets, stage confirmation and path-source selection are part of the interface. This is not an unattended end-to-end mission claim. The module-level defaults are retained; competition scripts and their additional overrides are deliberately not included.

The reorganized source is not claimed to be a validated deployment. See [source origins](../docs/source_origins.md) and [license information](../LICENSES.md).

