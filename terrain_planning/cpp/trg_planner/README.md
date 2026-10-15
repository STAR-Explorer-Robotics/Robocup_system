# TRG C++ core

This CMake project builds the terrain-risk graph planning library.

- Graph construction and search: [trg.cpp](core/trg_planner/src/graph/trg.cpp).
- Planner lifecycle and parameters: [trg_planner.cpp](core/trg_planner/src/planner/trg_planner.cpp).
- Public declarations: [trg_planner.h](core/trg_planner/include/planner/trg_planner.h).
- Dependencies: PCL, Eigen, OpenCV and yaml-cpp.

The ROS 2 wrapper is a separate project under pipelines/ros2. See the [module overview](../../README.md) and [original core license](../../LICENSE).
