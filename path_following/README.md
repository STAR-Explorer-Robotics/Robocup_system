# Path following

The trg_path_follower ROS 2 package, providing planar path tracking with a lightweight offline simulator.

## Included functionality

- Vector-field path-tube tracking and an optional pure-pursuit mode.
- Curvature, approach and deviation-based speed regulation.
- Stale-odometry and excessive-deviation stopping; optional final-yaw alignment.
- A kinematic offline simulator and the small existing geometry unit test are retained.

## Interfaces

**Inputs:** nav_msgs/Path and nav_msgs/Odometry, with an enable/disable service.

**Outputs:** geometry_msgs/Twist plus tracker state, lookahead and diagnostic messages.

See the shared [interface reference](../docs/interfaces.md) before connecting modules.

## Source map

| File | Role |
| --- | --- |
| [src/rpp_follower_node.cpp](src/rpp_follower_node.cpp) | Tracking and regulation |
| [include/trg_path_follower/rpp_geometry.hpp](include/trg_path_follower/rpp_geometry.hpp) | Path geometry helpers |
| [config/rpp_follower.yaml](config/rpp_follower.yaml) | Tracker defaults |
| [src/rpp_offline_sim_node.cpp](src/rpp_offline_sim_node.cpp) | Kinematic simulation helper |
| [launch/rpp_follower.launch.py](launch/rpp_follower.launch.py) | Follower launch |
| [test/test_rpp_geometry.cpp](test/test_rpp_geometry.cpp) | Existing component test |

## Dependencies

ROS 2 Humble, rclcpp, geometry_msgs, nav_msgs, std_msgs, std_srvs, tf2 and tf2_geometry_msgs.

## Scope and attribution

The ROS package name remains trg_path_follower. Default inputs are /trg/output/path and /laser_odometry; connect the mission mux and body-reference odometry explicitly when used. The simulator does not model complete quadruped dynamics or rescue terrain contact.

The reorganized source is not claimed to be a validated deployment. See [source origins](../docs/source_origins.md) and [license information](../LICENSES.md).

