# Terrain-risk graph planning

An adapted TRG-planner C++ core with a ROS 2 interface and a representative RoboCup configuration.

## Included functionality

- Terrain-risk graph construction, path search and allowed-area constraints.
- Configurable deterministic sampling and graph construction options.
- Optional heading-aware TRG-AStar and rectangular-footprint costs.
- The default search mode remains native.

## Interfaces

**Inputs:** A prior map or observation cloud, robot pose/odometry, a goal pose, and optional allowed-area polygons.

**Outputs:** A nav_msgs/Path on /trg/output/path plus map/graph debug outputs.

See the shared [interface reference](../docs/interfaces.md) before connecting modules.

## Source map

| File | Role |
| --- | --- |
| [cpp/trg_planner/core/trg_planner/src/graph/trg.cpp](cpp/trg_planner/core/trg_planner/src/graph/trg.cpp) | Graph construction and search costs |
| [cpp/trg_planner/core/trg_planner/src/planner/trg_planner.cpp](cpp/trg_planner/core/trg_planner/src/planner/trg_planner.cpp) | Planner lifecycle and configuration |
| [pipelines/ros2/src/ros2_node.cpp](pipelines/ros2/src/ros2_node.cpp) | ROS 2 inputs, outputs and boundary updates |
| [config/robocup_default.yaml](config/robocup_default.yaml) | Representative planner parameters |
| [pipelines/ros2/launch/trg_planner_wMap.py](pipelines/ros2/launch/trg_planner_wMap.py) | Module-level launch description |
| [LICENSE](LICENSE) | Original core license |

## Dependencies

C++17, CMake, PCL, Eigen, OpenCV and yaml-cpp. The ROS interface additionally requires ROS 2 Humble. The core library and ROS package have separate CMake projects.

## Scope and attribution

The original TRG method is attributed to its upstream authors. Current native edges are symmetric and optional climb/slope terms use absolute height differences; a distinct uphill/downhill cost model is not claimed. ROS 1, Python bindings, upstream map downloads and bulky demos are excluded.

The reorganized source is not claimed to be a validated deployment. See [source origins](../docs/source_origins.md) and [license information](../LICENSES.md).

