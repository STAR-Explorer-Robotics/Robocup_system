# License and attribution inventory

This collection preserves the imported copyright notices and available license files. It does not apply a new umbrella license to third-party modules.

| Component | Declaration found in the source checkout | Status in this edition |
| --- | --- | --- |
| SuperOdom fork | Root README states GPLv3; super_odometry/package.xml states BSD | Conflicting declarations and no root license file in the provided checkout; requires confirmation |
| Lightning-LM fork | package.xml contains TODO: License declaration | No root license file in the provided checkout; requires confirmation |
| Terrain-processing Python package | No package-level license declaration/file found | No new license assigned; requires an owner declaration |
| TRG planner core | Apache License 2.0 with Commons Clause condition | [Original license retained](terrain_planning/LICENSE) |
| TRG ROS pipeline | Separate original license file | [Pipeline license retained](terrain_planning/pipelines/LICENSE) |
| Path following | package.xml and source headers declare Apache-2.0 | Declaration and source notices retained |
| Unitree interface | Apache-2.0 | [Original license retained](robot_interface/LICENSE) |
| Mission control | package metadata declares Apache-2.0 | Existing declaration retained |
| Inspection and ROS sensing bridge | Package metadata declares MIT | Existing declaration retained; no root license file supplied |
| Sensing weights and prototypes | Files provided by the source checkout | Distribution terms/provenance are not inferred from code-package metadata |

The Lightning-LM subtree also includes Sophus headers and Livox message definitions. Their embedded notices are retained; the root module declaration should not be treated as replacing those notices.

Source repository URLs and commit anchors are listed in [source origins](docs/source_origins.md). This inventory records observed declarations and unresolved items; it does not resolve conflicts by relabeling code.

