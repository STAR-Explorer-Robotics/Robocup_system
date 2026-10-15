#!/usr/bin/env python3
# Copyright 2026 bichy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bridge a LiDAR cloud into a semantic elevation cloud for TRG.

The node keeps the full-field elevation pipeline (layer split, hole fill,
range gating) from the sibling trg_planner package, then converts the map into PointCloud2
samples that C++ TRG can consume: walkable surface cells plus grounded hard
obstacle columns. Raw ceilings/overhead clutter stay out of the TRG input.

Two input modes (``input_cloud_frame``):

* ``map``    — the cloud is already in the map frame (e.g. a SLAM
  ``registered_scan``); used as-is. Default, matches the offline bag-replay
  test.
* ``sensor`` — the cloud is a **raw LiDAR scan in the sensor frame**
  (e.g. ``/lidar_points``); it is transformed into the map frame with the
  latest odometry pose (optionally composed with a static sensor→base
  extrinsic). This is the production path: raw scans are denser than the
  SLAM-registered cloud and multiple LiDARs can be fused by giving each its
  own extrinsic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import Point32, PointStamped, PolygonStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from std_srvs.srv import Trigger


def _append_trg_python_path() -> None:
    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent
    configured_root = os.environ.get("TRG_PYTHON_ROOT")
    candidates = ([Path(configured_root)] if configured_root else []) + [package_root]
    for candidate in candidates:
        if not candidate:
            continue
        if (candidate / "trg_planner").is_dir():
            sys.path.insert(0, str(candidate))
            return
    raise RuntimeError(
        "Cannot locate the terrain_processing Python library. "
        "Set TRG_PYTHON_ROOT to the directory containing trg_planner, "
        f"or keep the library at {package_root}."
    )


_append_trg_python_path()

from trg_planner.online.elevation_builder import ElevationConfig, ElevationGridBuilder
from trg_planner.online.field_extent import (  # noqa: E402
    FieldExtent,
    FieldExtentAccumulator,
    extract_field_extent,
)


def _pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    if msg.width == 0 or msg.point_step == 0:
        return np.empty((0, 3), dtype=np.float32)

    fields = {field.name: field for field in msg.fields}
    required = ["x", "y", "z"]
    if not all(name in fields for name in required):
        return np.empty((0, 3), dtype=np.float32)

    if any(fields[name].datatype != PointField.FLOAT32 for name in required):
        return np.empty((0, 3), dtype=np.float32)

    point_step = int(msg.point_step)
    offsets = [int(fields[name].offset) for name in required]
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    total_bytes = int(msg.width) * int(msg.height) * point_step
    if raw.size < total_bytes:
        return np.empty((0, 3), dtype=np.float32)

    raw = raw[:total_bytes].reshape((-1, point_step))
    xyz = np.empty((raw.shape[0], 3), dtype=np.float32)
    for index, offset in enumerate(offsets):
        # A column slice of ``raw`` is not C-contiguous, so ``.view(float32)``
        # (which reinterprets bytes) would raise. Copy to a contiguous buffer
        # before the dtype reinterpretation.
        column = np.ascontiguousarray(raw[:, offset : offset + 4])
        xyz[:, index] = column.view(np.float32).reshape(-1)

    if not msg.is_dense:
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
    return xyz


def _pose_to_matrix(position, orientation) -> np.ndarray:
    """4x4 homogeneous transform from a geometry_msgs position + quaternion."""
    x, y, z, w = (
        float(orientation.x),
        float(orientation.y),
        float(orientation.z),
        float(orientation.w),
    )
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm > 0.0:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = 1.0 - 2.0 * (y * y + z * z)
    transform[0, 1] = 2.0 * (x * y - z * w)
    transform[0, 2] = 2.0 * (x * z + y * w)
    transform[1, 0] = 2.0 * (x * y + z * w)
    transform[1, 1] = 1.0 - 2.0 * (x * x + z * z)
    transform[1, 2] = 2.0 * (y * z - x * w)
    transform[2, 0] = 2.0 * (x * z - y * w)
    transform[2, 1] = 2.0 * (y * z + x * w)
    transform[2, 2] = 1.0 - 2.0 * (x * x + y * y)
    transform[0, 3] = float(position.x)
    transform[1, 3] = float(position.y)
    transform[2, 3] = float(position.z)
    return transform


def _rpy_to_matrix(xyz, rpy) -> np.ndarray:
    """4x4 homogeneous transform from a translation + roll/pitch/yaw (rad)."""
    tx, ty, tz = (float(v) for v in xyz)
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    transform[:3, 3] = (tx, ty, tz)
    return transform


def _xyz_to_pointcloud2(points_xyz: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        points_xyz = np.empty((0, 3), dtype=np.float32)

    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = int(points_xyz.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = points_xyz.astype(np.float32, copy=False).tobytes()
    return msg


def _terrain_points_from_map(builder: ElevationGridBuilder) -> np.ndarray:
    # Use the live filled surface rather than build_map()'s cached ElevationMap.
    # Online publishing must reflect small newly observed patches immediately.
    heights, _ = builder.filled_surface()
    heights = np.asarray(heights, dtype=np.float32)
    mask = np.isfinite(heights)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    resolution = float(builder.resolution)
    origin_x, origin_y = builder.origin
    xs = origin_x + (cols.astype(np.float32) + 0.5) * resolution
    ys = origin_y + (rows.astype(np.float32) + 0.5) * resolution
    zs = heights[rows, cols].astype(np.float32) + float(builder.ground_height)
    return np.column_stack((xs, ys, zs)).astype(np.float32)


def _relative_points_to_map_frame(
    points_xyz: np.ndarray,
    builder: ElevationGridBuilder,
) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3 or points_xyz.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    out = points_xyz.astype(np.float32, copy=True)
    out[:, 2] += float(builder.ground_height)
    return out


def _polygon_area(poly: np.ndarray) -> float:
    poly = _open_polygon(poly)
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _points_in_polygon(points_xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    poly = _open_polygon(polygon)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(poly) < 3:
        return np.zeros(len(pts), dtype=bool)
    x, y = pts[:, 0], pts[:, 1]
    px, py = poly[:, 0], poly[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        yi, yj = py[i], py[j]
        if yi == yj:
            continue
        xi, xj = px[i], px[j]
        cond = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        )
        inside ^= cond
    return inside


def _convex_hull(points_xy: np.ndarray) -> np.ndarray:
    pts = np.unique(np.asarray(points_xy, dtype=float), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _open_polygon(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    if len(poly) >= 2 and np.linalg.norm(poly[0] - poly[-1]) < 1e-6:
        return poly[:-1]
    return poly


def _simplify_open_polygon(poly: np.ndarray, tol: float) -> np.ndarray:
    poly = _open_polygon(poly)
    if len(poly) < 4 or tol <= 0.0:
        return poly

    def dp(points: np.ndarray) -> np.ndarray:
        if len(points) < 3:
            return points
        a, b = points[0], points[-1]
        ab = b - a
        nrm = np.hypot(*ab)
        if nrm < 1e-9:
            d = np.hypot(*(points - a).T)
        else:
            d = np.abs((points - a) @ np.array([-ab[1], ab[0]]) / nrm)
        idx = int(np.argmax(d))
        if d[idx] > tol:
            left = dp(points[: idx + 1])
            right = dp(points[idx:])
            return np.vstack((left[:-1], right))
        return np.vstack((a, b))

    simplified = dp(np.vstack((poly, poly[:1])))[:-1]
    return simplified if len(simplified) >= 3 else poly


def _expand_polygon_from_anchor(
    poly: np.ndarray, anchor_xy: Optional[np.ndarray], padding_m: float
) -> np.ndarray:
    poly = _open_polygon(poly)
    if len(poly) < 3 or anchor_xy is None or padding_m <= 0.0:
        return poly
    anchor = np.asarray(anchor_xy, dtype=float)[:2]
    vec = poly - anchor
    norm = np.linalg.norm(vec, axis=1)
    out = poly.copy()
    valid = norm > 1e-6
    out[valid] += vec[valid] / norm[valid, None] * float(padding_m)
    return out


def _elevation_columns(
    terrain_points: np.ndarray,
    resolution: float,
    max_height: float,
) -> np.ndarray:
    """Expand each top-surface cell into a vertical column of stacked voxels.

    Rendered as RViz ``Boxes`` this yields a height-proportional cuboid per
    cell (a 2.5D bar field) instead of a single square at the surface vertex,
    which makes terrain relief far easier to read. The column base is a robust
    floor (2nd percentile of observed heights) so outliers do not inflate every
    bar; each column is clamped to ``max_height`` to bound the point count.
    """
    if terrain_points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    resolution = max(1e-3, float(resolution))
    xs = terrain_points[:, 0]
    ys = terrain_points[:, 1]
    zs = terrain_points[:, 2]

    base_z = float(np.percentile(zs, 2.0))
    tops = np.minimum(zs, base_z + float(max_height))
    col_heights = np.maximum(tops - base_z, 0.0)
    # +1 so even a flat (base-level) cell renders at least one voxel.
    nsteps = np.floor(col_heights / resolution).astype(np.int64) + 1
    total = int(nsteps.sum())
    if total == 0:
        return np.empty((0, 3), dtype=np.float32)

    cols_x = np.repeat(xs, nsteps)
    cols_y = np.repeat(ys, nsteps)
    # Per-column step index 0..nsteps-1 via a ramp minus each segment's start.
    starts = np.repeat(np.cumsum(nsteps) - nsteps, nsteps)
    step_idx = np.arange(total, dtype=np.int64) - starts
    cols_z = base_z + step_idx.astype(np.float32) * resolution
    return np.column_stack((cols_x, cols_y, cols_z)).astype(np.float32)


class ElevationCloudBridge(Node):
    """ROS2 bridge from raw map-frame cloud to cleaned terrain cloud."""

    def __init__(self) -> None:
        super().__init__("elevation_cloud_bridge")

        self.declare_parameter("input_cloud_topic", "/registered_scan")
        self.declare_parameter("input_odom_topic", "/laser_odometry")
        self.declare_parameter("output_cloud_topic", "/trg/input/terrain_cloud")
        self.declare_parameter("output_frame_id", "map")
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("min_publish_interval_sec", 0.5)
        self.declare_parameter("full_publish_every", 1)
        self.declare_parameter("min_points_to_publish", 50)
        self.declare_parameter("diagnostics.enabled", True)
        self.declare_parameter("diagnostics.log_interval_sec", 5.0)
        self.declare_parameter("mapping.start_active", True)
        self.declare_parameter("mapping.start_service", "/elevation/start_mapping")
        self.declare_parameter("mapping.reset_service", "/elevation/reset_mapping")
        self.declare_parameter("mapping.bootstrap_cache_frames", 600)
        self.declare_parameter("mapping.bootstrap_voxel_size", 0.05)
        self.declare_parameter("mapping.bootstrap_max_voxels", 1000000)
        self.declare_parameter("mapping.bootstrap_max_points_per_frame", 4000)
        self.declare_parameter("mapping.bootstrap_min_points", 5000)
        self.declare_parameter("mapping.bootstrap_ground_bin", 0.03)
        self.declare_parameter("mapping.bootstrap_ground_band", 0.06)
        self.declare_parameter("mapping.bootstrap_min_ground_ratio", 0.05)
        self.declare_parameter("mapping.bootstrap_output_json", "")

        # Input cloud frame handling. ``map`` consumes a map-frame cloud as-is
        # (SLAM registered_scan); ``sensor`` transforms a raw sensor-frame scan
        # into map using the latest odom pose composed with a static
        # sensor->base extrinsic (for raw /lidar_points and multi-LiDAR fusion).
        self.declare_parameter("input_cloud_frame", "map")
        self.declare_parameter("sensor_to_base_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("sensor_to_base_rpy", [0.0, 0.0, 0.0])

        # Height-proportional cuboid (column) visualization of the elevation map.
        self.declare_parameter("publish_elevation_boxes", True)
        self.declare_parameter("elevation_boxes_topic", "/trg/debug/elevation_boxes")
        self.declare_parameter("box_max_height", 1.5)
        self.declare_parameter("publish_hard_obstacles", True)
        self.declare_parameter("hard_obstacle_cloud_topic", "/trg/debug/hard_obstacles")
        self.declare_parameter("soft_frontier.enabled", True)
        self.declare_parameter("soft_frontier.topic", "/navigation/soft_frontiers")
        self.declare_parameter("soft_frontier.exclude_hard_neighbors", True)

        self.declare_parameter("elevation.resolution", 0.10)
        self.declare_parameter("elevation.window_size", 40.0)
        self.declare_parameter("elevation.ground_percentile", 3.0)
        self.declare_parameter("elevation.ceiling_clip", 3.0)
        self.declare_parameter("elevation.max_height_above_lidar", 0.10)
        self.declare_parameter("elevation.floor_offset", 0.0)
        self.declare_parameter("elevation.range_max", 6.0)
        self.declare_parameter("elevation.range_min", 0.0)
        self.declare_parameter("elevation.obstacle_height", 0.0)
        self.declare_parameter("elevation.obstacle_floor_band", 0.12)
        self.declare_parameter("elevation.obstacle_min_span", 0.25)
        self.declare_parameter("elevation.include_obstacles_in_trg_cloud", True)
        self.declare_parameter("elevation.obstacle_column_step", 0.05)
        self.declare_parameter("elevation.seed_robot_footprint", False)
        self.declare_parameter("elevation.robot_seed_radius", 0.45)
        # Seed is a blind-spot fallback. Height is relative to the elevation
        # builder's current ground estimate unless robot_seed_height_mode is
        # set to "lidar_height".
        self.declare_parameter("elevation.robot_seed_height_mode", "fixed")
        self.declare_parameter("elevation.robot_seed_height", 0.0)
        self.declare_parameter("elevation.robot_seed_lidar_height", 0.560425)
        self.declare_parameter(
            "elevation.robot_seed_body_to_lidar_xyz",
            [-0.011, -0.02329, 0.04412],
        )
        self.declare_parameter("elevation.robot_seed_center_mode", "lidar_projection")
        self.declare_parameter("elevation.robot_seed_offset_x", 0.0)
        self.declare_parameter("elevation.robot_seed_offset_y", 0.0)
        self.declare_parameter("elevation.robot_seed_fallback_only", True)
        self.declare_parameter("elevation.robot_seed_max_relative_height", 0.25)
        self.declare_parameter("elevation.min_points_per_cell", 1)
        self.declare_parameter("elevation.persistence_frames", 1)
        self.declare_parameter("elevation.vertical_layering", True)
        self.declare_parameter("elevation.layer_bin", 0.10)
        self.declare_parameter("elevation.layer_gap", 0.75)
        self.declare_parameter("elevation.min_points_per_bin", 2)
        self.declare_parameter("elevation.fill_holes_iters", 3)
        self.declare_parameter("elevation.fill_min_neighbors", 5)

        self.declare_parameter("field_extent.enabled", False)
        self.declare_parameter("field_extent.output_json", "")
        self.declare_parameter(
            "field_extent.polygon_topic", "/navigation/detected_allowed_area"
        )
        self.declare_parameter("field_extent.extract_every", 50)
        self.declare_parameter("field_extent.voxel", 0.10)
        self.declare_parameter("field_extent.fence_band", [0.2, 0.8])
        self.declare_parameter("field_extent.headroom_band", [0.9, 1.5])
        self.declare_parameter("field_extent.alpha", 0.8)
        self.declare_parameter("field_extent.simplify_eps", 0.3)
        self.declare_parameter("field_extent.close_radius", 1.8)
        self.declare_parameter("field_extent.grid_voxel", 0.15)
        self.declare_parameter("field_extent.min_count", 2)
        self.declare_parameter("field_extent.require_robot_inside", True)
        self.declare_parameter("field_extent.publish_polygon", False)

        self.declare_parameter("avoid_holes.enabled", False)
        self.declare_parameter(
            "avoid_holes.calibrate_service",
            "/elevation/avoid_holes/calibrate_floor",
        )
        self.declare_parameter("avoid_holes.output_json", "")
        self.declare_parameter("avoid_holes.floor_band", 0.05)
        self.declare_parameter("avoid_holes.min_height_separation", 0.10)
        self.declare_parameter("avoid_holes.virtual_wall_height", 0.90)
        self.declare_parameter("avoid_holes.calibration_min_points", 2000)
        self.declare_parameter("avoid_holes.calibration_ground_bin", 0.03)
        self.declare_parameter("avoid_holes.calibration_ground_band", 0.05)
        self.declare_parameter("avoid_holes.calibration_min_ground_ratio", 0.05)
        self.declare_parameter("avoid_holes.calibrate_point_topic", "/rviz/avoid_holes_calibrate_click")
        self.declare_parameter("avoid_holes.calibration_seed_source", "robot")
        self.declare_parameter("avoid_holes.calibration_floor_scope", "full_cache")
        self.declare_parameter("avoid_holes.calibration_seed_radius", 1.20)
        self.declare_parameter("avoid_holes.calibration_min_seed_cells", 20)
        self.declare_parameter("avoid_holes.max_candidates_per_frame", 200000)
        self.declare_parameter("avoid_holes.clear_bootstrap_cache_on_calibrate", True)
        self.declare_parameter("avoid_holes.connected_only", False)
        self.declare_parameter("avoid_holes.connectivity_resolution", 0.05)
        self.declare_parameter("avoid_holes.publish_debug_clouds", True)
        self.declare_parameter("avoid_holes.field_surface_protection", True)
        self.declare_parameter("avoid_holes.field_surface_min_height_above_floor", 0.08)
        self.declare_parameter("avoid_holes.field_surface_max_height_above_ground", 0.35)
        self.declare_parameter("avoid_holes.field_surface_support_radius", 0.25)
        self.declare_parameter("avoid_holes.field_surface_min_support_cells", 8)
        self.declare_parameter("avoid_holes.field_surface_min_support_sectors", 4)
        self.declare_parameter("avoid_holes.field_surface_min_opposing_pairs", 2)
        self.declare_parameter(
            "avoid_holes.low_floor_candidates_topic",
            "/navigation/avoid_holes_low_floor_candidates",
        )
        self.declare_parameter(
            "avoid_holes.outside_low_floor_topic",
            "/navigation/avoid_holes_outside_low_floor",
        )
        self.declare_parameter(
            "avoid_holes.safe_low_floor_topic",
            "/navigation/avoid_holes_safe_low_floor",
        )
        self.declare_parameter(
            "avoid_holes.field_surface_topic",
            "/navigation/avoid_holes_field_surface_support",
        )
        self.declare_parameter(
            "avoid_holes.protected_low_floor_topic",
            "/navigation/avoid_holes_protected_low_floor",
        )

        self.declare_parameter("boundary.enabled", True)
        self.declare_parameter(
            "boundary.polygon_topic", "/navigation/detected_allowed_area"
        )
        self.declare_parameter("boundary.output_json", "")
        self.declare_parameter("boundary.extract_every", 5)
        self.declare_parameter("boundary.step_tolerance", 0.30)
        self.declare_parameter("boundary.simplify_eps", 0.20)
        self.declare_parameter("boundary.padding_m", 0.15)
        self.declare_parameter("boundary.anchor_max_distance", 1.0)
        self.declare_parameter("boundary.min_area", 0.5)

        cfg = ElevationConfig(
            resolution=float(self.get_parameter("elevation.resolution").value),
            window_size=float(self.get_parameter("elevation.window_size").value),
            ground_percentile=float(self.get_parameter("elevation.ground_percentile").value),
            ceiling_clip=float(self.get_parameter("elevation.ceiling_clip").value),
            floor_offset=float(self.get_parameter("elevation.floor_offset").value),
            range_max=float(self.get_parameter("elevation.range_max").value),
            range_min=float(self.get_parameter("elevation.range_min").value),
            obstacle_height=float(self.get_parameter("elevation.obstacle_height").value),
            obstacle_floor_band=float(self.get_parameter("elevation.obstacle_floor_band").value),
            obstacle_min_span=float(self.get_parameter("elevation.obstacle_min_span").value),
            min_points_per_cell=int(self.get_parameter("elevation.min_points_per_cell").value),
            persistence_frames=int(self.get_parameter("elevation.persistence_frames").value),
            vertical_layering=bool(self.get_parameter("elevation.vertical_layering").value),
            layer_bin=float(self.get_parameter("elevation.layer_bin").value),
            layer_gap=float(self.get_parameter("elevation.layer_gap").value),
            min_points_per_bin=int(self.get_parameter("elevation.min_points_per_bin").value),
            fill_holes_iters=int(self.get_parameter("elevation.fill_holes_iters").value),
            fill_min_neighbors=int(self.get_parameter("elevation.fill_min_neighbors").value),
        )
        self.builder = ElevationGridBuilder(cfg)
        self.max_height_above_lidar = float(
            self.get_parameter("elevation.max_height_above_lidar").value
        )

        self.output_frame_id = str(self.get_parameter("output_frame_id").value)
        self.input_cloud_frame = str(
            self.get_parameter("input_cloud_frame").value
        ).strip().lower()
        self.transform_to_map = self.input_cloud_frame == "sensor"
        s2b_xyz = self.get_parameter("sensor_to_base_xyz").value
        s2b_rpy = self.get_parameter("sensor_to_base_rpy").value
        self.T_base_sensor = _rpy_to_matrix(s2b_xyz, s2b_rpy)
        self.T_map_base: Optional[np.ndarray] = None
        self.mapping_active = bool(self.get_parameter("mapping.start_active").value)
        self.bootstrap_cache_frames = max(
            1, int(self.get_parameter("mapping.bootstrap_cache_frames").value)
        )
        self.bootstrap_voxel_size = max(
            1e-3, float(self.get_parameter("mapping.bootstrap_voxel_size").value)
        )
        self.bootstrap_max_voxels = max(
            0, int(self.get_parameter("mapping.bootstrap_max_voxels").value)
        )
        self.bootstrap_max_points_per_frame = max(
            1, int(self.get_parameter("mapping.bootstrap_max_points_per_frame").value)
        )
        self.bootstrap_min_points = max(
            1, int(self.get_parameter("mapping.bootstrap_min_points").value)
        )
        self.bootstrap_ground_bin = max(
            1e-3, float(self.get_parameter("mapping.bootstrap_ground_bin").value)
        )
        self.bootstrap_ground_band = max(
            1e-3, float(self.get_parameter("mapping.bootstrap_ground_band").value)
        )
        self.bootstrap_min_ground_ratio = max(
            0.0, float(self.get_parameter("mapping.bootstrap_min_ground_ratio").value)
        )
        self.bootstrap_output_json = str(
            self.get_parameter("mapping.bootstrap_output_json").value
        ).strip()
        self.rng = np.random.default_rng(7)
        self.bootstrap_cache: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=self.bootstrap_cache_frames
        )
        self.bootstrap_voxels: dict[tuple[int, int, int], tuple[float, float, float]] = {}
        self.bootstrap_seen_frames = 0
        self.bootstrap_seen_points = 0
        self.bootstrap_voxel_evictions = 0
        self.full_publish_every = max(1, int(self.get_parameter("full_publish_every").value))
        self.min_points_to_publish = max(
            1, int(self.get_parameter("min_points_to_publish").value)
        )
        self.min_publish_interval_sec = float(
            self.get_parameter("min_publish_interval_sec").value
        )
        self.diagnostics_enabled = bool(
            self.get_parameter("diagnostics.enabled").value
        )
        self.diagnostics_log_interval_sec = max(
            0.0, float(self.get_parameter("diagnostics.log_interval_sec").value)
        )
        self.include_obstacles_in_trg_cloud = bool(
            self.get_parameter("elevation.include_obstacles_in_trg_cloud").value
        )
        self.obstacle_column_step = max(
            1e-3, float(self.get_parameter("elevation.obstacle_column_step").value)
        )
        self.publish_hard_obstacles = bool(
            self.get_parameter("publish_hard_obstacles").value
        )
        self.soft_frontier_enabled = bool(
            self.get_parameter("soft_frontier.enabled").value
        )
        self.soft_frontier_topic = str(
            self.get_parameter("soft_frontier.topic").value
        ).strip()
        self.soft_frontier_exclude_hard_neighbors = bool(
            self.get_parameter("soft_frontier.exclude_hard_neighbors").value
        )
        self.seed_robot_footprint = bool(
            self.get_parameter("elevation.seed_robot_footprint").value
        )
        self.robot_seed_radius = float(
            self.get_parameter("elevation.robot_seed_radius").value
        )
        self.robot_seed_height = float(
            self.get_parameter("elevation.robot_seed_height").value
        )
        self.robot_seed_height_mode = str(
            self.get_parameter("elevation.robot_seed_height_mode").value
        ).strip().lower()
        self.robot_seed_lidar_height = float(
            self.get_parameter("elevation.robot_seed_lidar_height").value
        )
        body_to_lidar_xyz = self.get_parameter(
            "elevation.robot_seed_body_to_lidar_xyz"
        ).value
        self.robot_seed_body_to_lidar_xyz = np.array(
            [float(v) for v in body_to_lidar_xyz], dtype=np.float64
        )
        if self.robot_seed_body_to_lidar_xyz.shape != (3,):
            self.robot_seed_body_to_lidar_xyz = np.array(
                [-0.011, -0.02329, 0.04412], dtype=np.float64
            )
        self.robot_seed_center_mode = str(
            self.get_parameter("elevation.robot_seed_center_mode").value
        ).strip().lower()
        self.robot_seed_offset_xy = np.array(
            [
                float(self.get_parameter("elevation.robot_seed_offset_x").value),
                float(self.get_parameter("elevation.robot_seed_offset_y").value),
            ],
            dtype=np.float64,
        )
        self.robot_seed_fallback_only = bool(
            self.get_parameter("elevation.robot_seed_fallback_only").value
        )
        self.robot_seed_max_relative_height = float(
            self.get_parameter("elevation.robot_seed_max_relative_height").value
        )

        self.field_extent_enabled = bool(self.get_parameter("field_extent.enabled").value)
        self.field_extent_output_json = str(
            self.get_parameter("field_extent.output_json").value
        ).strip()
        self.field_extent_polygon_topic = str(
            self.get_parameter("field_extent.polygon_topic").value
        ).strip()
        self.field_extent_extract_every = max(
            1, int(self.get_parameter("field_extent.extract_every").value)
        )
        self.field_extent_require_robot_inside = bool(
            self.get_parameter("field_extent.require_robot_inside").value
        )
        self.field_extent_publish_polygon = bool(
            self.get_parameter("field_extent.publish_polygon").value
        )
        self.field_extent_acc: Optional[FieldExtentAccumulator] = None
        if self.field_extent_enabled:
            fence_band = self.get_parameter("field_extent.fence_band").value
            headroom_band = self.get_parameter("field_extent.headroom_band").value
            self.field_extent_acc = FieldExtentAccumulator(
                voxel=float(self.get_parameter("field_extent.voxel").value),
                fence_band=(float(fence_band[0]), float(fence_band[1])),
                headroom_band=(float(headroom_band[0]), float(headroom_band[1])),
            )

        self.avoid_holes_enabled = bool(
            self.get_parameter("avoid_holes.enabled").value
        )
        self.avoid_holes_calibrate_service = str(
            self.get_parameter("avoid_holes.calibrate_service").value
        ).strip()
        self.avoid_holes_output_json = str(
            self.get_parameter("avoid_holes.output_json").value
        ).strip()
        self.avoid_holes_floor_band = max(
            1e-3, float(self.get_parameter("avoid_holes.floor_band").value)
        )
        self.avoid_holes_min_height_separation = max(
            0.0, float(self.get_parameter("avoid_holes.min_height_separation").value)
        )
        self.avoid_holes_virtual_wall_height = max(
            0.01, float(self.get_parameter("avoid_holes.virtual_wall_height").value)
        )
        self.avoid_holes_calibration_min_points = max(
            1, int(self.get_parameter("avoid_holes.calibration_min_points").value)
        )
        self.avoid_holes_calibration_ground_bin = max(
            1e-3, float(self.get_parameter("avoid_holes.calibration_ground_bin").value)
        )
        self.avoid_holes_calibration_ground_band = max(
            1e-3, float(self.get_parameter("avoid_holes.calibration_ground_band").value)
        )
        self.avoid_holes_calibration_min_ground_ratio = max(
            0.0,
            float(self.get_parameter("avoid_holes.calibration_min_ground_ratio").value),
        )
        self.avoid_holes_calibrate_point_topic = str(
            self.get_parameter("avoid_holes.calibrate_point_topic").value
        ).strip()
        self.avoid_holes_calibration_seed_source = str(
            self.get_parameter("avoid_holes.calibration_seed_source").value
        ).strip().lower()
        if self.avoid_holes_calibration_seed_source not in ("robot", "click"):
            self.get_logger().warn(
                "avoid_holes.calibration_seed_source must be 'robot' or 'click'; "
                f"got {self.avoid_holes_calibration_seed_source!r}, using 'robot'"
            )
            self.avoid_holes_calibration_seed_source = "robot"
        self.avoid_holes_calibration_floor_scope = str(
            self.get_parameter("avoid_holes.calibration_floor_scope").value
        ).strip().lower()
        if self.avoid_holes_calibration_floor_scope not in ("full_cache", "local_seed"):
            self.get_logger().warn(
                "avoid_holes.calibration_floor_scope must be 'full_cache' or 'local_seed'; "
                f"got {self.avoid_holes_calibration_floor_scope!r}, using 'full_cache'"
            )
            self.avoid_holes_calibration_floor_scope = "full_cache"
        self.avoid_holes_calibration_seed_radius = max(
            0.0, float(self.get_parameter("avoid_holes.calibration_seed_radius").value)
        )
        self.avoid_holes_calibration_min_seed_cells = max(
            1, int(self.get_parameter("avoid_holes.calibration_min_seed_cells").value)
        )
        self.avoid_holes_max_candidates_per_frame = max(
            1000, int(self.get_parameter("avoid_holes.max_candidates_per_frame").value)
        )
        self.avoid_holes_clear_bootstrap_cache_on_calibrate = bool(
            self.get_parameter("avoid_holes.clear_bootstrap_cache_on_calibrate").value
        )
        self.avoid_holes_connected_only = bool(
            self.get_parameter("avoid_holes.connected_only").value
        )
        self.avoid_holes_connectivity_resolution = max(
            1e-3,
            float(self.get_parameter("avoid_holes.connectivity_resolution").value),
        )
        self.avoid_holes_publish_debug_clouds = bool(
            self.get_parameter("avoid_holes.publish_debug_clouds").value
        )
        self.avoid_holes_field_surface_protection = bool(
            self.get_parameter("avoid_holes.field_surface_protection").value
        )
        self.avoid_holes_field_surface_min_height_above_floor = max(
            0.0,
            float(
                self.get_parameter(
                    "avoid_holes.field_surface_min_height_above_floor"
                ).value
            ),
        )
        self.avoid_holes_field_surface_max_height_above_ground = max(
            0.0,
            float(
                self.get_parameter(
                    "avoid_holes.field_surface_max_height_above_ground"
                ).value
            ),
        )
        self.avoid_holes_field_surface_support_radius = max(
            0.0,
            float(self.get_parameter("avoid_holes.field_surface_support_radius").value),
        )
        self.avoid_holes_field_surface_min_support_cells = max(
            1,
            int(self.get_parameter("avoid_holes.field_surface_min_support_cells").value),
        )
        self.avoid_holes_field_surface_min_support_sectors = min(
            4,
            max(
                1,
                int(
                    self.get_parameter(
                        "avoid_holes.field_surface_min_support_sectors"
                    ).value
                ),
            ),
        )
        self.avoid_holes_field_surface_min_opposing_pairs = min(
            2,
            max(
                0,
                int(
                    self.get_parameter(
                        "avoid_holes.field_surface_min_opposing_pairs"
                    ).value
                ),
            ),
        )
        self.avoid_holes_low_floor_candidates_topic = str(
            self.get_parameter("avoid_holes.low_floor_candidates_topic").value
        ).strip()
        self.avoid_holes_outside_low_floor_topic = str(
            self.get_parameter("avoid_holes.outside_low_floor_topic").value
        ).strip()
        self.avoid_holes_safe_low_floor_topic = str(
            self.get_parameter("avoid_holes.safe_low_floor_topic").value
        ).strip()
        self.avoid_holes_field_surface_topic = str(
            self.get_parameter("avoid_holes.field_surface_topic").value
        ).strip()
        self.avoid_holes_protected_low_floor_topic = str(
            self.get_parameter("avoid_holes.protected_low_floor_topic").value
        ).strip()
        self.avoid_holes_floor_z: Optional[float] = None
        self.avoid_holes_low_floor_cells: set[tuple[int, int]] = set()
        self.avoid_holes_outside_low_floor_cells: set[tuple[int, int]] = set()
        self.avoid_holes_seed_cells: set[tuple[int, int]] = set()
        self.avoid_holes_field_surface_cells: set[tuple[int, int]] = set()
        self.avoid_holes_protected_low_floor_cells: set[tuple[int, int]] = set()
        self.avoid_holes_last_virtual_hard_cells: set[tuple[int, int]] = set()
        if self.avoid_holes_enabled and not self.include_obstacles_in_trg_cloud:
            self.get_logger().warn(
                "avoid_holes requires hard-obstacle columns in /trg/input/terrain_cloud; "
                "forcing elevation.include_obstacles_in_trg_cloud=true"
            )
            self.include_obstacles_in_trg_cloud = True

        self.boundary_enabled = bool(self.get_parameter("boundary.enabled").value)
        self.boundary_polygon_topic = str(
            self.get_parameter("boundary.polygon_topic").value
        ).strip()
        self.boundary_output_json = str(
            self.get_parameter("boundary.output_json").value
        ).strip()
        self.boundary_extract_every = max(
            1, int(self.get_parameter("boundary.extract_every").value)
        )
        self.boundary_step_tolerance = float(
            self.get_parameter("boundary.step_tolerance").value
        )
        self.boundary_simplify_eps = float(
            self.get_parameter("boundary.simplify_eps").value
        )
        self.boundary_padding_m = float(self.get_parameter("boundary.padding_m").value)
        self.boundary_anchor_max_distance = float(
            self.get_parameter("boundary.anchor_max_distance").value
        )
        self.boundary_min_area = float(self.get_parameter("boundary.min_area").value)

        qos = QoSProfile(depth=5)
        self.pub_cloud = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("output_cloud_topic").value),
            qos,
        )
        self.sub_cloud = self.create_subscription(
            PointCloud2,
            str(self.get_parameter("input_cloud_topic").value),
            self._on_cloud,
            qos,
        )
        self.sub_odom = self.create_subscription(
            Odometry,
            str(self.get_parameter("input_odom_topic").value),
            self._on_odom,
            qos,
        )

        # Height-proportional cuboid visualization of the elevation surface.
        self.publish_elevation_boxes = bool(
            self.get_parameter("publish_elevation_boxes").value
        )
        self.box_max_height = float(self.get_parameter("box_max_height").value)
        self.pub_boxes = None
        if self.publish_elevation_boxes:
            self.pub_boxes = self.create_publisher(
                PointCloud2,
                str(self.get_parameter("elevation_boxes_topic").value),
                qos,
            )
        self.pub_hard_obstacles = None
        if self.publish_hard_obstacles:
            self.pub_hard_obstacles = self.create_publisher(
                PointCloud2,
                str(self.get_parameter("hard_obstacle_cloud_topic").value),
                qos,
            )
        self.pub_soft_frontiers = None
        if self.soft_frontier_enabled and self.soft_frontier_topic:
            frontier_qos = QoSProfile(depth=1)
            frontier_qos.reliability = QoSReliabilityPolicy.RELIABLE
            frontier_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
            self.pub_soft_frontiers = self.create_publisher(
                PointCloud2,
                self.soft_frontier_topic,
                frontier_qos,
            )
        self.pub_avoid_holes_low_floor_candidates = None
        self.pub_avoid_holes_outside_low_floor = None
        self.pub_avoid_holes_safe_low_floor = None
        self.pub_avoid_holes_field_surface = None
        self.pub_avoid_holes_protected_low_floor = None
        if self.avoid_holes_enabled and self.avoid_holes_publish_debug_clouds:
            debug_qos = QoSProfile(depth=1)
            debug_qos.reliability = QoSReliabilityPolicy.RELIABLE
            debug_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
            if self.avoid_holes_low_floor_candidates_topic:
                self.pub_avoid_holes_low_floor_candidates = self.create_publisher(
                    PointCloud2,
                    self.avoid_holes_low_floor_candidates_topic,
                    debug_qos,
                )
            if self.avoid_holes_outside_low_floor_topic:
                self.pub_avoid_holes_outside_low_floor = self.create_publisher(
                    PointCloud2,
                    self.avoid_holes_outside_low_floor_topic,
                    debug_qos,
                )
            if self.avoid_holes_safe_low_floor_topic:
                self.pub_avoid_holes_safe_low_floor = self.create_publisher(
                    PointCloud2,
                    self.avoid_holes_safe_low_floor_topic,
                    debug_qos,
                )
            if self.avoid_holes_field_surface_topic:
                self.pub_avoid_holes_field_surface = self.create_publisher(
                    PointCloud2,
                    self.avoid_holes_field_surface_topic,
                    debug_qos,
                )
            if self.avoid_holes_protected_low_floor_topic:
                self.pub_avoid_holes_protected_low_floor = self.create_publisher(
                    PointCloud2,
                    self.avoid_holes_protected_low_floor_topic,
                    debug_qos,
                )
        self.pub_field_extent = None
        if (
            self.field_extent_enabled
            and self.field_extent_publish_polygon
            and self.field_extent_polygon_topic
        ):
            boundary_qos = QoSProfile(depth=1)
            boundary_qos.reliability = QoSReliabilityPolicy.RELIABLE
            boundary_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
            self.pub_field_extent = self.create_publisher(
                PolygonStamped,
                self.field_extent_polygon_topic,
                boundary_qos,
            )
        self.pub_boundary = None
        if self.boundary_enabled and self.boundary_polygon_topic:
            boundary_qos = QoSProfile(depth=1)
            boundary_qos.reliability = QoSReliabilityPolicy.RELIABLE
            boundary_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
            self.pub_boundary = self.create_publisher(
                PolygonStamped,
                self.boundary_polygon_topic,
                boundary_qos,
            )
        self.start_mapping_srv = self.create_service(
            Trigger,
            str(self.get_parameter("mapping.start_service").value),
            self._on_start_mapping,
        )
        self.reset_mapping_srv = self.create_service(
            Trigger,
            str(self.get_parameter("mapping.reset_service").value),
            self._on_reset_mapping,
        )
        self.avoid_holes_calibrate_srv = None
        if self.avoid_holes_enabled and self.avoid_holes_calibrate_service:
            self.avoid_holes_calibrate_srv = self.create_service(
                Trigger,
                self.avoid_holes_calibrate_service,
                self._on_calibrate_avoid_holes_floor,
            )
        self.avoid_holes_calibrate_sub = None
        if self.avoid_holes_enabled and self.avoid_holes_calibrate_point_topic:
            self.avoid_holes_calibrate_sub = self.create_subscription(
                PointStamped,
                self.avoid_holes_calibrate_point_topic,
                self._on_avoid_holes_calibrate_point,
                10,
            )

        self.odom_xy: Optional[np.ndarray] = None
        self.robot_xy: Optional[np.ndarray] = None
        self.seed_xy: Optional[np.ndarray] = None
        self.lidar_position_map: Optional[np.ndarray] = None
        self.pending_cloud: Optional[np.ndarray] = None
        self.frame_index = 0
        self.last_publish_time = self.get_clock().now()
        self.last_diagnostics_time = self.get_clock().now()
        self.clouds_received = 0
        self.clouds_overwritten = 0
        self.clouds_processed = 0
        self.clouds_published = 0
        self.diag_prev_received = 0
        self.diag_prev_processed = 0
        self.diag_prev_published = 0
        self.diag_prev_overwritten = 0
        self.last_walkable_points = 0
        self.last_hard_points = 0
        self.last_soft_points = 0
        self.last_trg_points = 0
        self.last_height_clip_removed = 0
        self.height_clip_removed_total = 0
        self.last_avoid_holes_candidates = 0
        self.last_avoid_holes_new_cells = 0
        self.last_avoid_holes_safe_cells = 0
        self.last_avoid_holes_protected_cells = 0
        self.last_process_ms = 0.0
        self.last_publish_ms = 0.0
        self.last_extent: Optional[FieldExtent] = None
        self.last_boundary: Optional[np.ndarray] = None

        publish_rate_hz = max(0.2, float(self.get_parameter("publish_rate_hz").value))
        self.timer = self.create_timer(1.0 / publish_rate_hz, self._on_timer)

        self.get_logger().info(
            "ElevationCloudBridge started: "
            f"input={self.get_parameter('input_cloud_topic').value}, "
            f"odom={self.get_parameter('input_odom_topic').value}, "
            f"output={self.get_parameter('output_cloud_topic').value}, "
            f"res={cfg.resolution:.2f}m, "
            f"max_above_lidar={self.max_height_above_lidar:.2f}m, "
            f"mapping_active={self.mapping_active}, "
            f"trg_hard_obstacles={self.include_obstacles_in_trg_cloud}, "
            f"soft_frontiers={self.soft_frontier_enabled}, "
            f"seed_enabled={self.seed_robot_footprint}, "
            f"seed_mode={self.robot_seed_height_mode}, "
            f"seed_fallback_only={self.robot_seed_fallback_only}, "
            f"avoid_holes={self.avoid_holes_enabled}"
        )
        if self.avoid_holes_enabled:
            self.get_logger().warn(
                "Avoid-holes mode enabled: calibrate outside floor before field "
                f"mapping via {self.avoid_holes_calibrate_service}; "
                f"floor_band={self.avoid_holes_floor_band:.3f}m, "
                f"min_separation={self.avoid_holes_min_height_separation:.3f}m, "
                f"virtual_wall_height={self.avoid_holes_virtual_wall_height:.3f}m, "
                f"connected_only={self.avoid_holes_connected_only}, "
                f"seed_radius={self.avoid_holes_calibration_seed_radius:.2f}m, "
                f"field_surface_protection={self.avoid_holes_field_surface_protection}, "
                f"support_radius={self.avoid_holes_field_surface_support_radius:.2f}m, "
                f"support_sectors={self.avoid_holes_field_surface_min_support_sectors}, "
                f"opposing_pairs={self.avoid_holes_field_surface_min_opposing_pairs}"
            )
        if not self.mapping_active:
            self.get_logger().warn(
                "Elevation mapping is waiting in SLAM_EXPLORE cache mode; "
                f"call {self.get_parameter('mapping.start_service').value} to bootstrap ground "
                "and start terrain publication. "
                f"bootstrap_voxel={self.bootstrap_voxel_size:.3f}m, "
                f"max_voxels={self.bootstrap_max_voxels or 'unlimited'}"
            )

    def _range_filtered_points(
        self, cloud: np.ndarray, robot_xy: np.ndarray
    ) -> np.ndarray:
        points = np.asarray(cloud, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)
        self.last_height_clip_removed = 0
        finite = np.isfinite(points).all(axis=1)
        if not finite.all():
            points = points[finite]
        if points.shape[0] == 0:
            return points
        cfg = self.builder.config
        if self.max_height_above_lidar >= 0.0 and self.lidar_position_map is not None:
            max_z = float(self.lidar_position_map[2]) + float(self.max_height_above_lidar)
            before_clip = int(points.shape[0])
            points = points[points[:, 2] <= max_z]
            removed = before_clip - int(points.shape[0])
            self.last_height_clip_removed = removed
            self.height_clip_removed_total += removed
            if points.shape[0] == 0:
                return points
        if cfg.range_max > 0.0 or cfg.range_min > 0.0:
            dxy = points[:, :2] - np.asarray(robot_xy, dtype=np.float32)[:2]
            dist = np.linalg.norm(dxy, axis=1)
            keep = np.ones(points.shape[0], dtype=bool)
            if cfg.range_max > 0.0:
                keep &= dist <= cfg.range_max
            if cfg.range_min > 0.0:
                keep &= dist >= cfg.range_min
            points = points[keep]
        return points

    def _cache_bootstrap_cloud(self, cloud: np.ndarray) -> None:
        if self.robot_xy is None:
            return
        points = self._range_filtered_points(cloud, self.robot_xy)
        if points.shape[0] == 0:
            return
        self.bootstrap_seen_frames += 1
        self.bootstrap_seen_points += int(points.shape[0])
        if points.shape[0] > self.bootstrap_max_points_per_frame:
            idx = self.rng.choice(
                points.shape[0], self.bootstrap_max_points_per_frame, replace=False
            )
            points = points[idx]
        self.bootstrap_cache.append(
            (points.astype(np.float32, copy=False), self.robot_xy.astype(np.float32).copy())
        )
        self._add_bootstrap_voxels(points)

    def _add_bootstrap_voxels(self, points: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            return

        keys = np.floor(points / self.bootstrap_voxel_size).astype(np.int64)
        if keys.shape[0] > 1:
            _, unique_idx = np.unique(keys, axis=0, return_index=True)
            keys = keys[unique_idx]
            points = points[unique_idx]

        for key_values, point in zip(keys, points):
            key = (int(key_values[0]), int(key_values[1]), int(key_values[2]))
            if key in self.bootstrap_voxels:
                del self.bootstrap_voxels[key]
            self.bootstrap_voxels[key] = (
                float(point[0]),
                float(point[1]),
                float(point[2]),
            )

        if self.bootstrap_max_voxels > 0:
            overflow = len(self.bootstrap_voxels) - self.bootstrap_max_voxels
            if overflow > 0:
                for key in list(self.bootstrap_voxels.keys())[:overflow]:
                    del self.bootstrap_voxels[key]
                self.bootstrap_voxel_evictions += int(overflow)
                self.get_logger().warn(
                    "Bootstrap voxel cache reached cap; evicting oldest voxels "
                    f"(cap={self.bootstrap_max_voxels}, total_evicted={self.bootstrap_voxel_evictions})",
                    throttle_duration_sec=5.0,
                )

    def _bootstrap_points(self) -> np.ndarray:
        if self.bootstrap_voxels:
            return np.asarray(list(self.bootstrap_voxels.values()), dtype=np.float32)
        if self.bootstrap_cache:
            return np.concatenate([entry[0] for entry in self.bootstrap_cache], axis=0)
        return np.empty((0, 3), dtype=np.float32)

    def _bootstrap_cache_stats(self) -> dict:
        source = "voxel" if self.bootstrap_voxels else "recent_frames"
        return {
            "cache_source": source,
            "cache_frames": int(len(self.bootstrap_cache)),
            "explore_frames": int(self.bootstrap_seen_frames),
            "explore_range_filtered_points": int(self.bootstrap_seen_points),
            "voxel_size": float(self.bootstrap_voxel_size),
            "voxel_count": int(len(self.bootstrap_voxels)),
            "voxel_max": int(self.bootstrap_max_voxels),
            "voxel_evictions": int(self.bootstrap_voxel_evictions),
            "max_points_per_frame": int(self.bootstrap_max_points_per_frame),
        }

    def _estimate_ground_from_points(
        self,
        points: np.ndarray,
        *,
        min_points: int,
        ground_bin: float,
        ground_band: float,
        min_ground_ratio: float,
    ) -> tuple[bool, dict]:
        points = np.asarray(points, dtype=np.float32)
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] < min_points:
            return False, {
                "reject_reason": "not_enough_points",
                "points": int(points.shape[0]),
                "required_points": int(min_points),
            }

        z = points[:, 2].astype(np.float64)
        z_low = float(np.percentile(z, 0.5))
        z_high = float(np.percentile(z, 60.0))
        if z_high <= z_low:
            ground_z = float(np.median(z))
            inliers = np.abs(z - ground_z) <= ground_band
        else:
            bins = np.arange(
                z_low,
                z_high + ground_bin,
                ground_bin,
                dtype=np.float64,
            )
            if bins.size < 2:
                ground_z = float(np.median(z))
                inliers = np.abs(z - ground_z) <= ground_band
            else:
                hist, edges = np.histogram(z, bins=bins)
                support = max(
                    20,
                    int(np.ceil(float(hist.max()) * 0.10)),
                    int(np.ceil(float(z.shape[0]) * 0.005)),
                )
                supported = np.flatnonzero(hist >= support)
                peak = int(supported[0]) if supported.size else int(np.argmax(hist))
                center = 0.5 * (float(edges[peak]) + float(edges[peak + 1]))
                rough = np.abs(z - center) <= max(
                    ground_band,
                    2.0 * ground_bin,
                )
                ground_z = float(np.median(z[rough])) if np.any(rough) else center
                inliers = np.abs(z - ground_z) <= ground_band

        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = float(inlier_count / max(1, z.shape[0]))
        if inlier_ratio < min_ground_ratio:
            return False, {
                "reject_reason": "low_ground_inlier_ratio",
                "ground_z": ground_z,
                "points": int(z.shape[0]),
                "ground_inliers": inlier_count,
                "ground_inlier_ratio": inlier_ratio,
                "required_ground_inlier_ratio": min_ground_ratio,
            }
        rmse = 0.0
        if inlier_count > 0:
            rmse = float(np.sqrt(np.mean((z[inliers] - ground_z) ** 2)))
        return True, {
            "reject_reason": "",
            "ground_z": ground_z,
            "points": int(z.shape[0]),
            "ground_inliers": inlier_count,
            "ground_inlier_ratio": inlier_ratio,
            "ground_rmse": rmse,
            "z_percentiles": {
                "p00_5": z_low,
                "p03": float(np.percentile(z, 3.0)),
                "p50": float(np.percentile(z, 50.0)),
                "p60": z_high,
            },
            "cache_frames": int(len(self.bootstrap_cache)),
        }

    def _estimate_bootstrap_ground(self, points: np.ndarray) -> tuple[bool, dict]:
        return self._estimate_ground_from_points(
            points,
            min_points=self.bootstrap_min_points,
            ground_bin=self.bootstrap_ground_bin,
            ground_band=self.bootstrap_ground_band,
            min_ground_ratio=self.bootstrap_min_ground_ratio,
        )

    def _save_bootstrap_json(self, payload: dict) -> None:
        if not self.bootstrap_output_json:
            return
        out_path = Path(self.bootstrap_output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_avoid_holes_json(self, payload: dict) -> None:
        if not self.avoid_holes_output_json:
            return
        out_path = Path(self.avoid_holes_output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _clear_bootstrap_cache(self) -> None:
        self.bootstrap_cache.clear()
        self.bootstrap_voxels.clear()
        self.bootstrap_seen_frames = 0
        self.bootstrap_seen_points = 0
        self.bootstrap_voxel_evictions = 0

    def _avoid_holes_floor_mask(
        self,
        points: np.ndarray,
        *,
        require_below_mapping_ground: bool,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if (
            not self.avoid_holes_enabled
            or self.avoid_holes_floor_z is None
            or points.ndim != 2
            or points.shape[1] != 3
            or points.shape[0] == 0
        ):
            return np.zeros(points.shape[0] if points.ndim == 2 else 0, dtype=bool)
        finite = np.isfinite(points).all(axis=1)
        dz = np.abs(points[:, 2].astype(np.float64) - float(self.avoid_holes_floor_z))
        mask = finite & (dz <= self.avoid_holes_floor_band)
        if require_below_mapping_ground and self.builder.ground_height_locked:
            mask &= points[:, 2] <= (
                float(self.builder.ground_height) - self.avoid_holes_min_height_separation
            )
        return mask

    def _avoid_holes_cells_from_xy(self, xy: np.ndarray) -> set[tuple[int, int]]:
        xy = np.asarray(xy, dtype=np.float32)
        if xy.ndim != 2 or xy.shape[1] < 2 or xy.shape[0] == 0:
            return set()
        finite = np.isfinite(xy[:, :2]).all(axis=1)
        xy = xy[finite, :2]
        if xy.shape[0] == 0:
            return set()
        keys = np.floor(xy / self.avoid_holes_connectivity_resolution).astype(np.int64)
        if keys.shape[0] > 1:
            keys = np.unique(keys, axis=0)
        return {(int(k[0]), int(k[1])) for k in keys}

    def _avoid_holes_points_from_cells(
        self, cells: set[tuple[int, int]], *, z: Optional[float] = None
    ) -> np.ndarray:
        if not cells:
            return np.empty((0, 3), dtype=np.float32)
        if z is None:
            if self.avoid_holes_floor_z is None:
                return np.empty((0, 3), dtype=np.float32)
            z = float(self.avoid_holes_floor_z)
        res = self.avoid_holes_connectivity_resolution
        arr = np.asarray(sorted(cells), dtype=np.float32)
        xs = (arr[:, 0] + 0.5) * res
        ys = (arr[:, 1] + 0.5) * res
        zs = np.full(xs.shape, float(z), dtype=np.float32)
        return np.column_stack((xs, ys, zs)).astype(np.float32)

    @staticmethod
    def _avoid_holes_neighbor_cells(cell: tuple[int, int]):
        x, y = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                yield (x + dx, y + dy)

    def _avoid_holes_expand_outside_component(
        self, new_cells: set[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        if not new_cells:
            return set()
        if not self.avoid_holes_connected_only:
            newly = new_cells - self.avoid_holes_outside_low_floor_cells
            self.avoid_holes_outside_low_floor_cells.update(new_cells)
            return newly

        seeds: list[tuple[int, int]] = []
        for cell in new_cells:
            if cell in self.avoid_holes_outside_low_floor_cells:
                continue
            if any(
                nb in self.avoid_holes_outside_low_floor_cells
                for nb in self._avoid_holes_neighbor_cells(cell)
            ):
                self.avoid_holes_outside_low_floor_cells.add(cell)
                seeds.append(cell)

        newly: set[tuple[int, int]] = set(seeds)
        queue: deque[tuple[int, int]] = deque(seeds)
        while queue:
            cell = queue.popleft()
            for nb in self._avoid_holes_neighbor_cells(cell):
                if nb in self.avoid_holes_outside_low_floor_cells:
                    continue
                if nb not in self.avoid_holes_low_floor_cells:
                    continue
                self.avoid_holes_outside_low_floor_cells.add(nb)
                newly.add(nb)
                queue.append(nb)
        return newly

    def _filter_points_near_xy(
        self,
        points: np.ndarray,
        center_xy: Optional[np.ndarray],
        radius: float,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if (
            center_xy is None
            or radius <= 0.0
            or points.ndim != 2
            or points.shape[1] != 3
            or points.shape[0] == 0
        ):
            return points
        center = np.asarray(center_xy, dtype=np.float32)[:2]
        dist = np.linalg.norm(points[:, :2] - center, axis=1)
        return points[dist <= float(radius)]

    def _avoid_holes_field_surface_mask(self, points: np.ndarray) -> np.ndarray:
        if (
            not self.avoid_holes_enabled
            or self.avoid_holes_floor_z is None
            or not self.builder.ground_height_locked
        ):
            return np.zeros(points.shape[0] if points.ndim == 2 else 0, dtype=bool)
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        finite = np.isfinite(points).all(axis=1)
        z = points[:, 2].astype(np.float64)
        min_z = float(self.avoid_holes_floor_z) + float(
            self.avoid_holes_field_surface_min_height_above_floor
        )
        max_z = float(self.builder.ground_height) + float(
            self.avoid_holes_field_surface_max_height_above_ground
        )
        return finite & (z >= min_z) & (z <= max_z)

    def _avoid_holes_supported_low_cells(
        self, low_cells: set[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        if (
            not self.avoid_holes_field_surface_protection
            or not low_cells
            or not self.avoid_holes_field_surface_cells
            or self.avoid_holes_field_surface_support_radius <= 0.0
        ):
            return set()

        radius_cells = max(
            1,
            int(
                np.ceil(
                    self.avoid_holes_field_surface_support_radius
                    / self.avoid_holes_connectivity_resolution
                )
            ),
        )
        radius_sq = radius_cells * radius_cells
        protected: set[tuple[int, int]] = set()
        min_cells = self.avoid_holes_field_surface_min_support_cells
        min_sectors = self.avoid_holes_field_surface_min_support_sectors
        min_opposing_pairs = self.avoid_holes_field_surface_min_opposing_pairs
        support = self.avoid_holes_field_surface_cells

        for cell in low_cells:
            x, y = cell
            support_count = 0
            sectors: set[int] = set()
            pos_x = False
            neg_x = False
            pos_y = False
            neg_y = False
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy > radius_sq:
                        continue
                    if (x + dx, y + dy) not in support:
                        continue
                    support_count += 1
                    if dx == 0 and dy == 0:
                        continue
                    pos_x |= dx > 0
                    neg_x |= dx < 0
                    pos_y |= dy > 0
                    neg_y |= dy < 0
                    if abs(dx) >= abs(dy):
                        sectors.add(0 if dx > 0 else 1)
                    else:
                        sectors.add(2 if dy > 0 else 3)
            opposing_pairs = int(pos_x and neg_x) + int(pos_y and neg_y)
            if (
                support_count >= min_cells
                and len(sectors) >= min_sectors
                and opposing_pairs >= min_opposing_pairs
            ):
                protected.add(cell)

        # The operator-selected outside seed is always outside, even if it sits
        # near the field edge and sees the high deck on one side.
        return protected - self.avoid_holes_seed_cells

    def _avoid_holes_recompute_components(self) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        protected = self._avoid_holes_supported_low_cells(self.avoid_holes_low_floor_cells)
        self.avoid_holes_protected_low_floor_cells = protected
        eligible = self.avoid_holes_low_floor_cells - protected
        if not self.avoid_holes_connected_only:
            outside = set(eligible)
        else:
            seeds = list(self.avoid_holes_seed_cells & eligible)
            outside: set[tuple[int, int]] = set(seeds)
            queue: deque[tuple[int, int]] = deque(seeds)
            while queue:
                cell = queue.popleft()
                for nb in self._avoid_holes_neighbor_cells(cell):
                    if nb in outside or nb not in eligible:
                        continue
                    outside.add(nb)
                    queue.append(nb)
        self.avoid_holes_outside_low_floor_cells = outside
        safe = self.avoid_holes_low_floor_cells - outside
        self.last_avoid_holes_safe_cells = len(safe)
        self.last_avoid_holes_protected_cells = len(protected)
        return outside, safe

    def _avoid_holes_sync_virtual_hard(self, hard_cells: set[tuple[int, int]]) -> int:
        if hard_cells == self.avoid_holes_last_virtual_hard_cells:
            return 0
        changed = len(hard_cells ^ self.avoid_holes_last_virtual_hard_cells)
        if hard_cells:
            hard_points = self._avoid_holes_points_from_cells(hard_cells)
            self.builder.set_virtual_hard_obstacles(
                hard_points[:, :2],
                base_height=0.0,
                top_height=self.avoid_holes_virtual_wall_height,
                anchor_xy=self.robot_xy,
            )
        else:
            self.builder.clear_virtual_hard_obstacles()
        self.avoid_holes_last_virtual_hard_cells = set(hard_cells)
        return int(changed)

    def _exclude_avoid_holes_floor_for_ground(self, points: np.ndarray) -> tuple[np.ndarray, int]:
        if not self.avoid_holes_enabled or self.avoid_holes_floor_z is None:
            return points, 0
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            return points, 0
        finite = np.isfinite(points).all(axis=1)
        # During field-ground bootstrap, reject every point that is still within
        # the unsafe low-floor band, not only points tightly around the measured
        # outside floor. The field's lowest valid surface is expected to sit at
        # least min_height_separation above outside_floor_z.
        low_cutoff = float(self.avoid_holes_floor_z) + max(
            self.avoid_holes_floor_band,
            self.avoid_holes_min_height_separation,
        )
        mask = finite & (points[:, 2] <= low_cutoff)
        removed = int(np.count_nonzero(mask))
        if removed == 0:
            return points, 0
        return points[~mask], removed

    def _apply_avoid_holes_virtual_walls(
        self,
        points: np.ndarray,
        *,
        apply_range_filter: bool = True,
    ) -> tuple[int, int]:
        if not self.avoid_holes_enabled or self.avoid_holes_floor_z is None:
            self.last_avoid_holes_candidates = 0
            self.last_avoid_holes_new_cells = 0
            self.last_avoid_holes_protected_cells = 0
            return 0, 0
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            self.last_avoid_holes_candidates = 0
            self.last_avoid_holes_new_cells = 0
            self.last_avoid_holes_protected_cells = 0
            return 0, 0
        if apply_range_filter and self.robot_xy is not None:
            points = self._range_filtered_points(points, self.robot_xy)
        surface_mask = self._avoid_holes_field_surface_mask(points)
        if np.any(surface_mask):
            surface_cells = self._avoid_holes_cells_from_xy(points[surface_mask, :2])
            self.avoid_holes_field_surface_cells.update(surface_cells)
        mask = self._avoid_holes_floor_mask(
            points, require_below_mapping_ground=True
        )
        candidates = points[mask]
        candidate_count = int(candidates.shape[0])
        if candidate_count == 0:
            self.last_avoid_holes_candidates = 0
            outside_cells, _ = self._avoid_holes_recompute_components()
            changed_cells = self._avoid_holes_sync_virtual_hard(outside_cells)
            self.last_avoid_holes_new_cells = changed_cells
            return 0, int(changed_cells)
        if candidate_count > self.avoid_holes_max_candidates_per_frame:
            idx = self.rng.choice(
                candidate_count,
                self.avoid_holes_max_candidates_per_frame,
                replace=False,
            )
            candidates = candidates[idx]
        candidate_cells = self._avoid_holes_cells_from_xy(candidates[:, :2])
        self.avoid_holes_low_floor_cells.update(candidate_cells)
        outside_cells, _ = self._avoid_holes_recompute_components()
        changed_cells = self._avoid_holes_sync_virtual_hard(outside_cells)
        self.last_avoid_holes_candidates = candidate_count
        self.last_avoid_holes_new_cells = int(changed_cells)
        if changed_cells > 0:
            self.get_logger().info(
                "Avoid-holes virtual hard boundary updated: "
                f"floor_z={self.avoid_holes_floor_z:.3f}m, "
                f"candidates={candidate_count}, outside_cells={len(outside_cells)}, "
                f"changed_cells={changed_cells}, "
                f"protected_cells={self.last_avoid_holes_protected_cells}, "
                f"safe_cells={self.last_avoid_holes_safe_cells}, "
                f"total_cells={self.builder.virtual_hard_obstacle_count()}"
            )
        return candidate_count, int(changed_cells)

    def _on_start_mapping(
        self, req: Trigger.Request, resp: Trigger.Response
    ) -> Trigger.Response:
        del req
        if self.mapping_active:
            resp.success = True
            resp.message = "mapping is already active"
            return resp
        if self.avoid_holes_enabled and self.avoid_holes_floor_z is None:
            resp.success = False
            resp.message = (
                "avoid_holes floor is not calibrated; call "
                "/elevation/avoid_holes/calibrate_floor before /elevation/start_mapping"
            )
            return resp

        points = self._bootstrap_points()
        ground_points, excluded_floor_points = self._exclude_avoid_holes_floor_for_ground(points)
        ok, result = self._estimate_bootstrap_ground(ground_points)
        result.update(self._bootstrap_cache_stats())
        result["source"] = "elevation_ground_bootstrap"
        result["mapping_active"] = bool(ok)
        if self.avoid_holes_enabled:
            result["avoid_holes"] = {
                "enabled": True,
                "outside_floor_z": self.avoid_holes_floor_z,
                "excluded_floor_points": int(excluded_floor_points),
                "floor_band": float(self.avoid_holes_floor_band),
                "min_height_separation": float(self.avoid_holes_min_height_separation),
                "bootstrap_low_cutoff": float(
                    self.avoid_holes_floor_z
                    + max(
                        self.avoid_holes_floor_band,
                        self.avoid_holes_min_height_separation,
                    )
                )
                if self.avoid_holes_floor_z is not None
                else None,
                "connected_only": bool(self.avoid_holes_connected_only),
                "connectivity_resolution": float(
                    self.avoid_holes_connectivity_resolution
                ),
                "low_floor_candidate_cells": int(
                    len(self.avoid_holes_low_floor_cells)
                ),
                "outside_low_floor_cells": int(
                    len(self.avoid_holes_outside_low_floor_cells)
                ),
                "safe_low_floor_cells": int(
                    len(
                        self.avoid_holes_low_floor_cells
                        - self.avoid_holes_outside_low_floor_cells
                    )
                ),
                "protected_low_floor_cells": int(
                    len(self.avoid_holes_protected_low_floor_cells)
                ),
                "field_surface_cells": int(len(self.avoid_holes_field_surface_cells)),
                "field_surface_protection": bool(
                    self.avoid_holes_field_surface_protection
                ),
                "field_surface_min_opposing_pairs": int(
                    self.avoid_holes_field_surface_min_opposing_pairs
                ),
            }
        self._save_bootstrap_json(result)
        if not ok:
            resp.success = False
            resp.message = (
                f"ground bootstrap rejected: {result.get('reject_reason', 'unknown')}; "
                f"points={result.get('points', 0)}"
            )
            return resp

        ground_z = float(result["ground_z"])
        self.builder.reset()
        self.builder.set_ground_height(ground_z, locked=True)
        self.last_extent = None
        self.last_boundary = None
        self.frame_index = max(self.frame_index, self.bootstrap_seen_frames)
        replay_robot_xy = self.robot_xy
        if replay_robot_xy is None:
            replay_robot_xy = points[:, :2].mean(axis=0)
        self.builder.add_cloud(points, replay_robot_xy, apply_range_filter=False)
        avoid_candidates, avoid_new_cells = self._apply_avoid_holes_virtual_walls(
            points, apply_range_filter=False
        )
        self.mapping_active = True
        published = self._publish_current_outputs(force=True)
        self.get_logger().warn(
            "Elevation mapping activated: "
            f"ground_z={ground_z:.3f}m, explore_frames={self.bootstrap_seen_frames}, "
            f"voxel_count={len(self.bootstrap_voxels)}, "
            f"points={result['points']}, ground_ratio={result['ground_inlier_ratio']:.2%}, "
            f"avoid_holes_candidates={avoid_candidates}, "
            f"avoid_holes_new_cells={avoid_new_cells}"
        )
        resp.success = bool(published)
        resp.message = (
            "mapping active; "
            f"ground_z={ground_z:.3f}m; "
            f"terrain_published={str(bool(published)).lower()}"
        )
        return resp

    def _calibrate_avoid_holes_floor(
        self,
        *,
        center_xy: Optional[np.ndarray],
        source: str,
    ) -> tuple[bool, dict, str]:
        if not self.avoid_holes_enabled:
            return False, {}, "avoid_holes mode is disabled"
        if self.mapping_active:
            return False, {}, (
                "avoid_holes floor calibration must run before /elevation/start_mapping"
            )

        points = self._bootstrap_points()
        if center_xy is None:
            result = self._bootstrap_cache_stats()
            result.update(
                {
                    "source": source,
                    "mapping_active": self.mapping_active,
                    "reject_reason": "no_seed_center",
                    "seed_source": self.avoid_holes_calibration_seed_source,
                    "floor_scope": self.avoid_holes_calibration_floor_scope,
                }
            )
            self._save_avoid_holes_json(result)
            return False, result, (
                "avoid_holes floor calibration rejected: no_seed_center; "
                "wait for /lightning/lio_odom or use RViz click seed_source=click"
            )
        local_points = self._filter_points_near_xy(
            points,
            center_xy,
            self.avoid_holes_calibration_seed_radius,
        )
        if (
            self.avoid_holes_calibration_floor_scope == "local_seed"
            and local_points.shape[0] >= self.avoid_holes_calibration_min_points
        ):
            estimate_points = local_points
            estimate_source = "local_seed_radius"
        else:
            estimate_points = points
            estimate_source = "full_bootstrap_cache_lowest_plane"
        ok, result = self._estimate_ground_from_points(
            estimate_points,
            min_points=self.avoid_holes_calibration_min_points,
            ground_bin=self.avoid_holes_calibration_ground_bin,
            ground_band=self.avoid_holes_calibration_ground_band,
            min_ground_ratio=self.avoid_holes_calibration_min_ground_ratio,
        )
        result.update(self._bootstrap_cache_stats())
        result["source"] = source
        result["mapping_active"] = self.mapping_active
        result["floor_band"] = float(self.avoid_holes_floor_band)
        result["min_height_separation"] = float(self.avoid_holes_min_height_separation)
        result["virtual_wall_height"] = float(self.avoid_holes_virtual_wall_height)
        result["bootstrap_cache_cleared"] = False
        result["seed_radius"] = float(self.avoid_holes_calibration_seed_radius)
        result["seed_center_xy"] = (
            [float(center_xy[0]), float(center_xy[1])] if center_xy is not None else None
        )
        result["local_seed_points"] = int(local_points.shape[0])
        result["seed_source"] = self.avoid_holes_calibration_seed_source
        result["floor_scope"] = self.avoid_holes_calibration_floor_scope
        result["estimate_points_source"] = estimate_source
        if not ok:
            self._save_avoid_holes_json(result)
            return False, result, (
                "avoid_holes floor calibration rejected: "
                f"{result.get('reject_reason', 'unknown')}; "
                f"points={result.get('points', 0)}"
            )

        self.avoid_holes_floor_z = float(result["ground_z"])
        self.builder.clear_virtual_hard_obstacles()
        self.avoid_holes_last_virtual_hard_cells.clear()
        seed_source = local_points if center_xy is not None else estimate_points
        seed_mask = self._avoid_holes_floor_mask(
            seed_source, require_below_mapping_ground=False
        )
        seed_cells = self._avoid_holes_cells_from_xy(seed_source[seed_mask, :2])
        if len(seed_cells) < self.avoid_holes_calibration_min_seed_cells:
            result["outside_floor_z"] = self.avoid_holes_floor_z
            result["outside_seed_cells"] = int(len(seed_cells))
            result["required_seed_cells"] = int(
                self.avoid_holes_calibration_min_seed_cells
            )
            result["reject_reason"] = "not_enough_seed_cells"
            self.avoid_holes_floor_z = None
            self._save_avoid_holes_json(result)
            return False, result, (
                "avoid_holes floor calibration rejected: not_enough_seed_cells; "
                f"seed_cells={len(seed_cells)}"
            )
        self.avoid_holes_low_floor_cells.clear()
        self.avoid_holes_outside_low_floor_cells.clear()
        self.avoid_holes_field_surface_cells.clear()
        self.avoid_holes_protected_low_floor_cells.clear()
        self.avoid_holes_seed_cells = set(seed_cells)
        self.avoid_holes_low_floor_cells.update(seed_cells)
        self.avoid_holes_outside_low_floor_cells.update(seed_cells)
        if self.avoid_holes_clear_bootstrap_cache_on_calibrate:
            self._clear_bootstrap_cache()
            result["bootstrap_cache_cleared"] = True
        result["outside_floor_z"] = self.avoid_holes_floor_z
        result["outside_seed_cells"] = int(len(seed_cells))
        result["connected_only"] = bool(self.avoid_holes_connected_only)
        result["connectivity_resolution"] = float(
            self.avoid_holes_connectivity_resolution
        )
        result["field_surface_protection"] = bool(
            self.avoid_holes_field_surface_protection
        )
        result["field_surface_min_opposing_pairs"] = int(
            self.avoid_holes_field_surface_min_opposing_pairs
        )
        self._save_avoid_holes_json(result)
        message = (
            "avoid_holes outside floor calibrated; "
            f"outside_floor_z={self.avoid_holes_floor_z:.3f}m; "
            f"seed_cells={len(seed_cells)}; "
            f"cache_cleared={str(result['bootstrap_cache_cleared']).lower()}"
        )
        self.get_logger().warn(message)
        return True, result, message

    def _on_calibrate_avoid_holes_floor(
        self, req: Trigger.Request, resp: Trigger.Response
    ) -> Trigger.Response:
        del req
        center_xy = self.robot_xy.copy() if self.robot_xy is not None else None
        ok, _result, message = self._calibrate_avoid_holes_floor(
            center_xy=center_xy,
            source="avoid_holes_outside_floor_calibration_service",
        )
        resp.success = bool(ok)
        resp.message = message
        return resp

    def _on_avoid_holes_calibrate_point(self, msg: PointStamped) -> None:
        clicked_xy = np.array([msg.point.x, msg.point.y], dtype=np.float32)
        if (
            self.avoid_holes_calibration_seed_source == "robot"
            and self.robot_xy is not None
        ):
            center_xy = self.robot_xy.copy()
            seed_note = "robot"
        else:
            center_xy = clicked_xy
            seed_note = "click"
        ok, _result, message = self._calibrate_avoid_holes_floor(
            center_xy=center_xy,
            source="avoid_holes_outside_floor_calibration_rviz_point",
        )
        level = self.get_logger().warn if ok else self.get_logger().error
        level(
            "RViz avoid-holes calibration click handled by elevation bridge: "
            f"success={ok} seed_source={seed_note} "
            f"clicked=({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f}) "
            f"seed_center=({center_xy[0]:.2f}, {center_xy[1]:.2f}) "
            f"message={message}"
        )

    def _on_reset_mapping(
        self, req: Trigger.Request, resp: Trigger.Response
    ) -> Trigger.Response:
        del req
        self.builder.reset()
        self._clear_bootstrap_cache()
        self.mapping_active = False
        self.last_extent = None
        self.last_boundary = None
        self.last_avoid_holes_candidates = 0
        self.last_avoid_holes_new_cells = 0
        self.last_avoid_holes_safe_cells = 0
        self.last_avoid_holes_protected_cells = 0
        self.avoid_holes_floor_z = None
        self.avoid_holes_low_floor_cells.clear()
        self.avoid_holes_outside_low_floor_cells.clear()
        self.avoid_holes_seed_cells.clear()
        self.avoid_holes_field_surface_cells.clear()
        self.avoid_holes_protected_low_floor_cells.clear()
        self.avoid_holes_last_virtual_hard_cells.clear()
        self.frame_index = 0
        resp.success = True
        resp.message = "mapping reset; back to SLAM_EXPLORE cache mode"
        self.get_logger().warn(resp.message)
        return resp

    def _on_odom(self, msg: Odometry) -> None:
        self.odom_xy = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=np.float32
        )
        pose_matrix = _pose_to_matrix(msg.pose.pose.position, msg.pose.pose.orientation)
        self.T_map_base = pose_matrix
        self.robot_xy = self.odom_xy.astype(np.float32)

        lidar_pos = pose_matrix[:3, :3] @ self.robot_seed_body_to_lidar_xyz + pose_matrix[:3, 3]
        self.lidar_position_map = lidar_pos.astype(np.float64)
        offset_xy = pose_matrix[:2, :2] @ self.robot_seed_offset_xy
        if self.robot_seed_center_mode in ("lidar", "lidar_projection", "lidar_ground_projection"):
            seed_xy = lidar_pos[:2] + offset_xy
        else:
            seed_xy = self.odom_xy.astype(np.float64) + offset_xy
        self.seed_xy = seed_xy.astype(np.float32)
        if self.transform_to_map:
            self.T_map_base = pose_matrix

    def _on_cloud(self, msg: PointCloud2) -> None:
        xyz = _pointcloud2_to_xyz(msg)
        if xyz.shape[0] == 0:
            return
        self.clouds_received += 1
        if self.pending_cloud is not None:
            self.clouds_overwritten += 1
        self.pending_cloud = xyz

    def _maybe_log_diagnostics(self) -> None:
        if not self.diagnostics_enabled or self.diagnostics_log_interval_sec <= 0.0:
            return
        now = self.get_clock().now()
        elapsed = (now - self.last_diagnostics_time).nanoseconds * 1e-9
        if elapsed < self.diagnostics_log_interval_sec:
            return
        elapsed = max(elapsed, 1e-6)
        recv_delta = self.clouds_received - self.diag_prev_received
        proc_delta = self.clouds_processed - self.diag_prev_processed
        pub_delta = self.clouds_published - self.diag_prev_published
        overwrite_delta = self.clouds_overwritten - self.diag_prev_overwritten
        self.get_logger().info(
            "Elevation diagnostics: "
            f"mapping_active={self.mapping_active}, "
            f"recv={recv_delta / elapsed:.1f}Hz, "
            f"processed={proc_delta / elapsed:.1f}Hz, "
            f"published={pub_delta / elapsed:.1f}Hz, "
            f"overwritten={overwrite_delta}, "
            f"last_points walkable={self.last_walkable_points} "
            f"hard={self.last_hard_points} soft={self.last_soft_points} "
            f"trg={self.last_trg_points}, "
            f"height_clip_removed={self.last_height_clip_removed} "
            f"height_clip_total={self.height_clip_removed_total} "
            f"avoid_holes_candidates={self.last_avoid_holes_candidates} "
            f"avoid_holes_new_cells={self.last_avoid_holes_new_cells} "
            f"avoid_holes_protected_cells={self.last_avoid_holes_protected_cells} "
            f"avoid_holes_safe_cells={self.last_avoid_holes_safe_cells} "
            f"avoid_holes_outside_cells={len(self.avoid_holes_outside_low_floor_cells)} "
            f"avoid_holes_field_surface_cells={len(self.avoid_holes_field_surface_cells)} "
            f"avoid_holes_total_cells={self.builder.virtual_hard_obstacle_count()}, "
            f"last_process={self.last_process_ms:.1f}ms, "
            f"last_publish={self.last_publish_ms:.1f}ms"
        )
        self.last_diagnostics_time = now
        self.diag_prev_received = self.clouds_received
        self.diag_prev_processed = self.clouds_processed
        self.diag_prev_published = self.clouds_published
        self.diag_prev_overwritten = self.clouds_overwritten

    def _save_extent_json(
        self,
        extent: FieldExtent,
        *,
        published: bool,
        robot_inside: Optional[bool] = None,
        reject_reason: str = "",
    ) -> None:
        if not self.field_extent_output_json:
            return
        out_path = Path(self.field_extent_output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        robot_xy = None
        if self.robot_xy is not None:
            robot_xy = self.robot_xy.astype(float).tolist()
        odom_xy = None
        if self.odom_xy is not None:
            odom_xy = self.odom_xy.astype(float).tolist()
        payload = {
            "source": "field_extent_experiment",
            "area": float(extent.area),
            "polygon": extent.polygon.astype(float).tolist(),
            "vertex_count": int(len(extent.polygon)),
            "is_valid": bool(extent.is_valid()),
            "published": bool(published),
            "robot_xy": robot_xy,
            "odom_xy": odom_xy,
            "robot_inside": robot_inside,
            "reject_reason": reject_reason,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_boundary_json(
        self,
        polygon: np.ndarray,
        *,
        published: bool,
        reject_reason: str = "",
    ) -> None:
        if not self.boundary_output_json:
            return
        out_path = Path(self.boundary_output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        robot_xy = None
        if self.robot_xy is not None:
            robot_xy = self.robot_xy.astype(float).tolist()
        odom_xy = None
        if self.odom_xy is not None:
            odom_xy = self.odom_xy.astype(float).tolist()
        payload = {
            "source": "traversable_reachable_boundary",
            "area": _polygon_area(polygon),
            "polygon": _open_polygon(polygon).astype(float).tolist(),
            "vertex_count": int(len(_open_polygon(polygon))),
            "published": bool(published),
            "robot_xy": robot_xy,
            "odom_xy": odom_xy,
            "robot_seed_offset_xy": self.robot_seed_offset_xy.astype(float).tolist(),
            "robot_seed_center_mode": self.robot_seed_center_mode,
            "robot_seed_height_mode": self.robot_seed_height_mode,
            "robot_seed_lidar_height": self.robot_seed_lidar_height,
            "robot_seed_body_to_lidar_xyz": self.robot_seed_body_to_lidar_xyz.astype(float).tolist(),
            "step_tolerance": self.boundary_step_tolerance,
            "padding_m": self.boundary_padding_m,
            "reject_reason": reject_reason,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _seed_relative_height(self) -> Optional[float]:
        if self.robot_seed_height_mode in ("fixed", "relative", "relative_ground"):
            return self.robot_seed_height
        if self.robot_seed_height_mode not in ("lidar_height", "lidar_ground_height"):
            self.get_logger().warn(
                f"Unknown robot_seed_height_mode={self.robot_seed_height_mode!r}; "
                "using fixed relative height",
                throttle_duration_sec=5.0,
            )
            return self.robot_seed_height
        if self.lidar_position_map is None:
            return None

        floor_z_map = float(self.lidar_position_map[2] - self.robot_seed_lidar_height)
        relative_height = float(floor_z_map - self.builder.ground_height)
        max_abs = self.robot_seed_max_relative_height
        if max_abs > 0.0 and abs(relative_height) > max_abs:
            self.get_logger().warn(
                "Skipping robot seed: lidar-height floor estimate differs from "
                f"builder ground by {relative_height:.3f}m "
                f"(limit {max_abs:.3f}m). Check odom z / LiDAR extrinsic.",
                throttle_duration_sec=5.0,
            )
            return None
        return relative_height

    def _publish_polygon(self, pub, polygon: np.ndarray) -> None:
        if pub is None:
            return
        polygon = _open_polygon(polygon)
        if len(polygon) < 3:
            return
        msg = PolygonStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.output_frame_id
        for x, y in polygon:
            point = Point32()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.0
            msg.polygon.points.append(point)
        pub.publish(msg)

    def _nearest_observed_anchor(self) -> Optional[np.ndarray]:
        if self.robot_xy is None:
            return None
        observed = np.asarray(self.builder.walkable_mask(), dtype=bool)
        rows, cols = np.nonzero(observed)
        if rows.size == 0:
            return None
        resolution = float(self.builder.resolution)
        origin_x, origin_y = self.builder.origin
        xs = origin_x + (cols.astype(np.float32) + 0.5) * resolution
        ys = origin_y + (rows.astype(np.float32) + 0.5) * resolution
        pts = np.column_stack((xs, ys)).astype(np.float32)
        diff = pts - self.robot_xy.astype(np.float32)
        d2 = np.einsum("ij,ij->i", diff, diff)
        best = int(np.argmin(d2))
        dist = float(np.sqrt(d2[best]))
        if dist > self.boundary_anchor_max_distance:
            return None
        return pts[best]

    def _reachable_hull_boundary(self, anchor_xy: np.ndarray) -> Optional[np.ndarray]:
        reachable = self.builder.reachable_mask(
            anchor_xy, step_tolerance=self.boundary_step_tolerance
        )
        if not reachable.any():
            return None
        points = self.builder.cell_centres_of(reachable)
        if points.shape[0] < 3:
            return None
        return _convex_hull(points)

    def _prepare_boundary_polygon(self, polygon: np.ndarray) -> np.ndarray:
        polygon = _open_polygon(polygon)
        polygon = _simplify_open_polygon(polygon, self.boundary_simplify_eps)
        return _expand_polygon_from_anchor(
            polygon, self.robot_xy, self.boundary_padding_m
        )

    def _extract_traversable_boundary(self) -> None:
        if not self.boundary_enabled or self.robot_xy is None:
            return
        if (
            self.last_boundary is not None
            and self.frame_index % self.boundary_extract_every != 0
        ):
            return
        anchor_xy = self.robot_xy
        polygon = self.builder.traversable_boundary(
            anchor_xy, step_tolerance=self.boundary_step_tolerance
        )
        if polygon is None:
            anchor_xy = self._nearest_observed_anchor()
            if anchor_xy is None:
                return
            polygon = self.builder.traversable_boundary(
                anchor_xy, step_tolerance=self.boundary_step_tolerance
            )
        if polygon is None:
            return
        polygon = self._prepare_boundary_polygon(polygon)
        if not bool(_points_in_polygon(np.asarray([self.robot_xy]), polygon)[0]):
            fallback = self._reachable_hull_boundary(anchor_xy)
            if fallback is not None:
                polygon = self._prepare_boundary_polygon(fallback)
            if not bool(_points_in_polygon(np.asarray([self.robot_xy]), polygon)[0]):
                self._save_boundary_json(
                    polygon,
                    published=False,
                    reject_reason="robot_outside_reachable_boundary",
                )
                self.get_logger().warn(
                    "Traversable boundary rejected: robot is outside reachable "
                    "polygon; waiting for a valid walkable component",
                    throttle_duration_sec=5.0,
                )
                return
        area = _polygon_area(polygon)
        if len(polygon) < 3 or area < self.boundary_min_area:
            fallback = self._reachable_hull_boundary(anchor_xy)
            if fallback is not None:
                fallback = self._prepare_boundary_polygon(fallback)
                fallback_area = _polygon_area(fallback)
                if len(fallback) >= 3 and fallback_area >= self.boundary_min_area:
                    polygon = fallback
                    area = fallback_area
            if len(polygon) < 3 or area < self.boundary_min_area:
                self._save_boundary_json(
                    polygon, published=False, reject_reason="too_small"
                )
                return
        self.last_boundary = polygon
        self._save_boundary_json(polygon, published=True)
        self._publish_polygon(self.pub_boundary, polygon)
        self.get_logger().info(
            "Traversable boundary updated: "
            f"area={area:.2f} m^2, vertices={len(polygon)}"
        )

    def _extract_field_extent(self) -> None:
        if not self.field_extent_enabled or self.field_extent_acc is None:
            return
        if self.frame_index % self.field_extent_extract_every != 0:
            return
        pts = self.field_extent_acc.short_points(
            min_count=int(self.get_parameter("field_extent.min_count").value)
        )
        if len(pts) < 16:
            return
        extent = extract_field_extent(
            pts,
            simplify_tol=float(self.get_parameter("field_extent.simplify_eps").value),
            close_radius=float(self.get_parameter("field_extent.close_radius").value),
            grid_res=float(self.get_parameter("field_extent.grid_voxel").value),
        )
        if extent.is_valid():
            robot_inside = None
            if self.robot_xy is not None:
                robot_inside = bool(
                    extent.contains(np.asarray([self.robot_xy], dtype=float))[0]
                )
            if self.field_extent_require_robot_inside and robot_inside is False:
                self._save_extent_json(
                    extent,
                    published=False,
                    robot_inside=robot_inside,
                    reject_reason="robot_outside",
                )
                self.get_logger().warn(
                    "Field extent rejected: robot is outside detected polygon "
                    f"area={extent.area:.2f} m^2 vertices={len(extent.polygon)}",
                    throttle_duration_sec=5.0,
                )
                return
            self.last_extent = extent
            self._save_extent_json(
                extent, published=True, robot_inside=robot_inside
            )
            self._publish_polygon(self.pub_field_extent, extent.polygon)
            self.get_logger().info(
                f"Field extent updated: area={extent.area:.2f} m^2, vertices={len(extent.polygon)}"
            )

    def _publish_current_outputs(
        self,
        *,
        cloud_for_field_extent: Optional[np.ndarray] = None,
        force: bool = False,
    ) -> bool:
        """Publish terrain/debug outputs from the current builder state."""
        publish_start = time.perf_counter()

        self._extract_traversable_boundary()

        if (
            cloud_for_field_extent is not None
            and self.field_extent_enabled
            and self.field_extent_acc is not None
        ):
            self.field_extent_acc.add_cloud(cloud_for_field_extent)
            self._extract_field_extent()

        dirty = self.builder.peek_dirty()
        if not dirty and not force:
            self.last_publish_ms = (time.perf_counter() - publish_start) * 1000.0
            return False

        now = self.get_clock().now()
        elapsed = (now - self.last_publish_time).nanoseconds * 1e-9
        if not force and elapsed < self.min_publish_interval_sec:
            self.last_publish_ms = (time.perf_counter() - publish_start) * 1000.0
            return False

        # Full semantic publish keeps C++ TRG map tree coherent with cleaned map.
        terrain_points = _terrain_points_from_map(self.builder)
        if terrain_points.shape[0] < self.min_points_to_publish:
            self.last_publish_ms = (time.perf_counter() - publish_start) * 1000.0
            return False
        dirty = self.builder.consume_dirty()

        hard_points_relative = self.builder.hard_obstacle_points(
            column_step=self.obstacle_column_step,
            max_height=self.box_max_height,
        )
        hard_points = _relative_points_to_map_frame(hard_points_relative, self.builder)
        if self.include_obstacles_in_trg_cloud and hard_points.shape[0] > 0:
            trg_points = np.vstack((terrain_points, hard_points)).astype(np.float32)
        else:
            trg_points = terrain_points

        msg = _xyz_to_pointcloud2(trg_points, self.output_frame_id, now.to_msg())
        self.pub_cloud.publish(msg)
        self.last_publish_time = now
        self.clouds_published += 1
        self.last_walkable_points = int(terrain_points.shape[0])
        self.last_hard_points = int(hard_points.shape[0])
        self.last_trg_points = int(trg_points.shape[0])
        self.get_logger().debug(
            "Published semantic terrain cloud: "
            f"walkable={terrain_points.shape[0]}, hard={hard_points.shape[0]}, "
            f"total={trg_points.shape[0]}, dirty_cells={len(dirty)}"
        )

        # Height-proportional cuboid view (separate debug topic, never fed to TRG).
        if self.pub_boxes is not None:
            columns = _elevation_columns(
                terrain_points, float(self.builder.resolution), self.box_max_height
            )
            if columns.shape[0] > 0:
                box_msg = _xyz_to_pointcloud2(
                    columns, self.output_frame_id, now.to_msg()
                )
                self.pub_boxes.publish(box_msg)
        if self.pub_hard_obstacles is not None:
            hard_msg = _xyz_to_pointcloud2(
                hard_points, self.output_frame_id, now.to_msg()
            )
            self.pub_hard_obstacles.publish(hard_msg)
        if self.pub_soft_frontiers is not None and self.robot_xy is not None:
            soft_points_relative = self.builder.soft_frontier_points(
                self.robot_xy,
                step_tolerance=self.boundary_step_tolerance,
                exclude_hard_neighbors=self.soft_frontier_exclude_hard_neighbors,
            )
            soft_points = _relative_points_to_map_frame(
                soft_points_relative, self.builder
            )
            soft_msg = _xyz_to_pointcloud2(
                soft_points, self.output_frame_id, now.to_msg()
            )
            self.pub_soft_frontiers.publish(soft_msg)
            self.last_soft_points = int(soft_points.shape[0])
        if self.avoid_holes_enabled and self.avoid_holes_floor_z is not None:
            if self.pub_avoid_holes_low_floor_candidates is not None:
                pts = self._avoid_holes_points_from_cells(
                    self.avoid_holes_low_floor_cells
                )
                self.pub_avoid_holes_low_floor_candidates.publish(
                    _xyz_to_pointcloud2(pts, self.output_frame_id, now.to_msg())
                )
            if self.pub_avoid_holes_outside_low_floor is not None:
                pts = self._avoid_holes_points_from_cells(
                    self.avoid_holes_outside_low_floor_cells
                )
                self.pub_avoid_holes_outside_low_floor.publish(
                    _xyz_to_pointcloud2(pts, self.output_frame_id, now.to_msg())
                )
            if self.pub_avoid_holes_safe_low_floor is not None:
                safe_cells = (
                    self.avoid_holes_low_floor_cells
                    - self.avoid_holes_outside_low_floor_cells
                )
                pts = self._avoid_holes_points_from_cells(safe_cells)
                self.pub_avoid_holes_safe_low_floor.publish(
                    _xyz_to_pointcloud2(pts, self.output_frame_id, now.to_msg())
                )
            if self.pub_avoid_holes_field_surface is not None:
                pts = self._avoid_holes_points_from_cells(
                    self.avoid_holes_field_surface_cells,
                    z=float(self.builder.ground_height),
                )
                self.pub_avoid_holes_field_surface.publish(
                    _xyz_to_pointcloud2(pts, self.output_frame_id, now.to_msg())
                )
            if self.pub_avoid_holes_protected_low_floor is not None:
                pts = self._avoid_holes_points_from_cells(
                    self.avoid_holes_protected_low_floor_cells
                )
                self.pub_avoid_holes_protected_low_floor.publish(
                    _xyz_to_pointcloud2(pts, self.output_frame_id, now.to_msg())
                )
        self.last_publish_ms = (time.perf_counter() - publish_start) * 1000.0
        return True

    def _on_timer(self) -> None:
        if self.pending_cloud is None or self.robot_xy is None:
            return
        process_start = time.perf_counter()

        cloud = self.pending_cloud
        self.pending_cloud = None

        if self.transform_to_map:
            # Raw sensor-frame scan: transform into map with the latest pose.
            if self.T_map_base is None:
                return
            transform = self.T_map_base @ self.T_base_sensor
            pts = cloud.astype(np.float64, copy=False)
            cloud = (pts @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)

        if not self.mapping_active:
            self._cache_bootstrap_cloud(cloud)
            self.clouds_processed += 1
            self.last_process_ms = (time.perf_counter() - process_start) * 1000.0
            self._maybe_log_diagnostics()
            return

        filtered_cloud = self._range_filtered_points(cloud, self.robot_xy)
        self.builder.add_cloud(filtered_cloud, self.robot_xy, apply_range_filter=False)
        self._apply_avoid_holes_virtual_walls(filtered_cloud, apply_range_filter=False)
        if self.seed_robot_footprint and self.seed_xy is not None:
            seed_height = self._seed_relative_height()
            if seed_height is not None:
                self.builder.seed_robot_footprint(
                    self.seed_xy,
                    radius=self.robot_seed_radius,
                    height=seed_height,
                    only_unobserved=self.robot_seed_fallback_only,
                )
        self.frame_index += 1
        self._publish_current_outputs(cloud_for_field_extent=filtered_cloud)
        self.clouds_processed += 1
        self.last_process_ms = (time.perf_counter() - process_start) * 1000.0
        self._maybe_log_diagnostics()



class BridgeArgParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__(
            description=(
                "Elevation bridge from raw map-frame cloud to cleaned TRG terrain cloud. "
                "All runtime options are ROS2 parameters."
            )
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    # Keep argparse so --help is descriptive before ROS args parsing.
    parser = BridgeArgParser()
    parser.parse_known_args(argv)

    rclpy.init(args=argv)
    node = ElevationCloudBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        if "publisher's context is invalid" not in str(exc):
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
