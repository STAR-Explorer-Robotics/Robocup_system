# SuperOdom platform adaptation

A source snapshot of the SuperOdom/SuperLoc-based LiDAR–inertial estimation and prior-map localization fork, including the JT128 interfaces used by the rescue navigation work.

## Included functionality

- JT128 decoded-point conversion with per-point timing and channel checks.
- Configurable IMU unit scaling and gravity-aligned sensor frames.
- Map/localization modes and a planner-facing body odometry transform.

## Interfaces

**Inputs:** LiDAR PointCloud2, IMU messages and, for localization, a prior map and initial pose.

**Outputs:** Estimator odometry, registered/map clouds, and optional body-frame odometry/TF.

See the shared [interface reference](../../docs/interfaces.md) before connecting modules.

## Source map

| File | Role |
| --- | --- |
| [super_odometry/src/hesaiToSuperOdom_node.cpp](super_odometry/src/hesaiToSuperOdom_node.cpp) | JT128 point layout and time handling |
| [super_odometry/src/odomFrameTransform_node.cpp](super_odometry/src/odomFrameTransform_node.cpp) | Estimator-to-body pose conversion |
| [super_odometry/src/LaserMapping/laserMapping.cpp](super_odometry/src/LaserMapping/laserMapping.cpp) | Mapping and prior-map localization |
| [super_odometry/config/hesai_jt128.yaml](super_odometry/config/hesai_jt128.yaml) | JT128 parameters and frame/unit conventions |
| [super_odometry_msgs/package.xml](super_odometry_msgs/package.xml) | Local message definitions |

## Dependencies

ROS 2 Humble, Eigen, PCL, Ceres, GTSAM, Sophus, TBB and livox_ros_driver2. The imported build currently requires the Livox message dependency even for a JT128 configuration.

## Scope and attribution

The estimation foundation is reused. The publishable software contribution is its platform adaptation and integration. This edition does not certify a new estimator, a calibration result or every sensor combination.

The reorganized source is not claimed to be a validated deployment. See [source origins](../../docs/source_origins.md) and [license information](../../LICENSES.md).

