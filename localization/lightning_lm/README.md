# Lightning-LM navigation backend

The current Lightning-LM fork retained as an alternative LiDAR–inertial SLAM/localization backend for the online terrain-processing path.

## Included functionality

- Per-frame LIO odometry publication in addition to keyframe outputs.
- Dense undistorted registered clouds for terrain processing.
- JT128 and Livox configuration files; optional visualization and offline tools remain source-level build options.

## Interfaces

**Inputs:** The LiDAR and IMU streams selected by a configuration in config/.

**Outputs:** Representative outputs include /lightning/lio_odom and /lightning/dense_registered_scan, plus the upstream map/localization outputs.

See the shared [interface reference](../../docs/interfaces.md) before connecting modules.

## Source map

| File | Role |
| --- | --- |
| [src/core/system/slam.cc](src/core/system/slam.cc) | SLAM coordination and added output interfaces |
| [src/core/lio/laser_mapping.cc](src/core/lio/laser_mapping.cc) | LIO frontend |
| [src/app/run_slam_online.cc](src/app/run_slam_online.cc) | Online SLAM entry |
| [config/default_jt128.yaml](config/default_jt128.yaml) | JT128 configuration |
| [config/default_livox.yaml](config/default_livox.yaml) | Livox configuration |
| [cmake/packages.cmake](cmake/packages.cmake) | External build dependencies |

## Dependencies

ROS 2 Humble, PCL, Eigen, OpenCV, yaml-cpp, glog/gflags, TBB and optional Pangolin/OpenGL. Sophus headers and Livox message definitions are retained in thirdparty/; downloaded archives and compiled executables are omitted.

## Scope and attribution

This is an adapted existing estimator, not a newly proposed SLAM algorithm. Dense-cloud frame handling and publication-time stamps should be understood before connecting downstream consumers. Module license declaration remains unresolved in the imported source.

The reorganized source is not claimed to be a validated deployment. See [source origins](../../docs/source_origins.md) and [license information](../../LICENSES.md).

