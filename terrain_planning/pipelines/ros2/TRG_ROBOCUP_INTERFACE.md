# TRG ROS 2 interface

ROS package: trg_planner_ros. Main executable: trg_ros2_node.

The core C++ library is built from the sibling cpp/trg_planner project and supplied through CMake's package search path. See the [module overview](../../README.md).

## Inputs

| Default topic | Type | Role |
| --- | --- | --- |
| /laser_odometry | nav_msgs/Odometry | Robot pose in the map frame |
| /trg/input/pose | geometry_msgs/PoseStamped | Alternative pose input |
| /goal_pose | geometry_msgs/PoseStamped | Requested destination |
| /laser_cloud_map | sensor_msgs/PointCloud2 | Optional online terrain observations |
| /navigation/detected_allowed_area | geometry_msgs/PolygonStamped | Optional dynamic allowed area |

Input names are configurable. Terrain processing produces /trg/input/terrain_cloud; connect it explicitly when using the online pipeline.

## Outputs

| Default topic | Type | Role |
| --- | --- | --- |
| /trg/output/path | nav_msgs/Path | Planned reference path |
| /trg/output/prebuilt_map | sensor_msgs/PointCloud2 | Prior-map visualization |
| /trg/debug/global_trg | visualization_msgs/MarkerArray | Graph diagnostics |
| /trg/debug/path_info | std_msgs/Float32MultiArray | Distance, path-length, timing and risk diagnostics |

## Configuration

[robocup_params.yaml](config/robocup_params.yaml) contains ROS defaults. [robocup_default.yaml](../../config/robocup_default.yaml) contains planner parameters.

The module launch exposes map_config_path, map_path, boundary_file, allowed_area_topic and rviz_config. Maps and field-specific boundaries are external inputs.

Native search is the default. The optional trg_astar mode adds heading and footprint terms; it does not create a distinct uphill/downhill model. Full competition bringup scripts are outside this collection.
