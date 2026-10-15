# Interface reference

Topic names below describe the imported implementations. This edition preserves their local defaults; it does not silently remap all modules into a single launch configuration.

## Navigation data

| Producer / consumer | Data | Representative interface |
| --- | --- | --- |
| JT128 adapter | Decoded cloud with point time and channel | PointCloud2: x, y, z, intensity, time, ring |
| SuperOdom body transform | Body-frame pose in the map | /body_odometry, nav_msgs/Odometry |
| Lightning-LM | Per-frame LIO state | /lightning/lio_odom, nav_msgs/Odometry |
| Lightning-LM | Dense undistorted cloud registered in the map | /lightning/dense_registered_scan, sensor_msgs/PointCloud2 |
| Body-reference bridge | Configurable robot reference pose | /robot_center/odom, nav_msgs/Odometry |
| Terrain processing | Walking surface and obstacle representation | /trg/input/terrain_cloud, sensor_msgs/PointCloud2 |
| Terrain processing | Reachable allowed-area boundary | /navigation/detected_allowed_area, geometry_msgs/PolygonStamped |
| Terrain processing | Exploration candidates | /navigation/soft_frontiers, sensor_msgs/PointCloud2 |
| Mission / operator | Planner target | /goal_pose, geometry_msgs/PoseStamped |
| TRG | Planned path | /trg/output/path, nav_msgs/Path |
| Path-source mux | Selected planned or recorded path | /mission/follower_path, nav_msgs/Path |
| Follower | Planar body command | geometry_msgs/Twist |
| Unitree adapter | Vendor command execution | SDK2 Move(vx, vy, vyaw) |

## Configuration choices to make explicit

The standalone follower defaults to /trg/output/path and /laser_odometry. The task path-source mux publishes /mission/follower_path; current online operation may use /robot_center/odom. Connecting these modules requires explicit parameters/remapping.

The mission launch/config retains its original require_start_signal default; the competition-oriented development launcher used additional overrides. Choose operator-start semantics explicitly when assembling a deployment.

TRG defaults to native search. Heading and rectangular-footprint terms belong to the optional trg_astar mode. They are not automatically active because their YAML parameters exist.

## Coordinates and time

- Point clouds supplied to the terrain library must already be in a common map frame.
- Estimator reference points, LiDAR frames and body reference points are distinct. Apply the corresponding transform once.
- JT128 adaptation preserves the interpretation of per-point time and physical channel fields; unit conversion and gravity-aligned frames are configured in the localization module.
- A configured transform is not proof of a calibrated transform.
- Lightning-LM currently stamps its exported LIO odometry and dense registered cloud with publication time. Do not infer acquisition-time synchronization guarantees from a shared frame name.

## Control and supervision

The follower exposes /rpp/set_enabled and publishes /rpp/state. The mission layer uses these alongside goal selection and path-source services.

The Unitree adapter exposes enable, stop and body-height services. Body height uses the dedicated service, not Twist. SDK2 support is optional at build time; the mock backend does not execute hardware commands.

No deployment is implied by these interface descriptions. Verify message frames, topic remapping, start conditions and the selected hardware backend for a particular robot.

