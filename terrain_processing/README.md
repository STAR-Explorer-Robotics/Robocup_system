# Terrain processing

Point-cloud-to-terrain processing used between localization and the C++ graph planner. The original Python namespace trg_planner is retained to keep the geometry implementation and imports recognizable.

## Included functionality

- Full-field elevation accumulation, vertical layering and grounded-obstacle handling.
- Reachable allowed-area boundaries, frontier information and virtual no-go overlays.
- ROS 2 terrain-cloud bridge and a configurable body-reference odometry bridge.
- The Python graph/planner implementation is retained as a reference dependency, rather than presented as the active C++ planner.

## Interfaces

**Inputs:** Point clouds in a common map frame and a robot reference pose; optional mapping-stage and floor-calibration requests.

**Outputs:** ElevationMap data, /trg/input/terrain_cloud, /navigation/detected_allowed_area and /navigation/soft_frontiers.

See the shared [interface reference](../docs/interfaces.md) before connecting modules.

## Source map

| File | Role |
| --- | --- |
| [trg_planner/online/elevation_builder.py](trg_planner/online/elevation_builder.py) | Elevation and obstacle algorithms |
| [trg_planner/online/field_extent.py](trg_planner/online/field_extent.py) | Geometric extent utilities used by the bridge |
| [trg_planner/elevation_map.py](trg_planner/elevation_map.py) | Grid representation and sampling |
| [ros/elevation_cloud_bridge.py](ros/elevation_cloud_bridge.py) | ROS 2 integration and stage-aware mapping |
| [ros/robot_center_odom.py](ros/robot_center_odom.py) | Body-reference conversion |
| [pyproject.toml](pyproject.toml) | Python package metadata |

## Dependencies

Python 3.10+ and NumPy; optional plotting/reference utilities use Matplotlib. ROS wrappers additionally require ROS 2 Humble and standard message packages.

## Scope and attribution

The supplied cloud must have the intended frame/time semantics. Geometry-derived walking/obstacle/frontier classes do not imply a learned semantic segmentation model or a guarantee of collision-free traversal. Experimental bag runners, visualizers' output files and tests from the development work are not exported.

The reorganized source is not claimed to be a validated deployment. See [source origins](../docs/source_origins.md) and [license information](../LICENSES.md).

