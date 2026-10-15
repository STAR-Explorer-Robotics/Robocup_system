"""Read a recorded rosbag2 and yield paired (cloud, pose) frames.

The online mapping pipeline consumes a map-frame point cloud
(``/lightning/registered_scan``) together with the robot pose
(``/lightning/slam_odom``). This module reads those two topics straight from a
``rosbag2`` (sqlite3) directory and yields them paired in playback order, so the
incremental pipeline can be validated offline without a running ROS graph.

ROS message classes are imported lazily inside :func:`read_frames` so importing
this module never forces a sourced ROS environment unless replay is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class BagFrame:
    """One synchronised cloud + pose sample."""

    stamp: float  # bag receive time, seconds
    points_world: np.ndarray  # (N, 3) map-frame points
    robot_xy: tuple[float, float]
    robot_z: float


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Yaw (rad) from a quaternion (used only for optional logging)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def read_frames(
    bag_dir: str | Path,
    scan_topic: str = "/lightning/registered_scan",
    odom_topic: str = "/lightning/slam_odom",
    *,
    stride: int = 1,
) -> Iterator[BagFrame]:
    """Yield :class:`BagFrame` samples from ``bag_dir`` in playback order.

    Each scan is paired with the most recent pose seen before it. ``stride``
    lets the caller subsample scans (e.g. ``stride=2`` processes every other
    cloud) to trade fidelity for speed.
    """
    import rosbag2_py  # noqa: PLC0415
    from rclpy.serialization import deserialize_message  # noqa: PLC0415
    from sensor_msgs.msg import PointCloud2  # noqa: PLC0415
    from nav_msgs.msg import Odometry  # noqa: PLC0415
    import sensor_msgs_py.point_cloud2 as pc2  # noqa: PLC0415

    bag_dir = Path(bag_dir)
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {
        scan_topic: PointCloud2,
        odom_topic: Odometry,
    }

    latest_pose: tuple[float, float, float] | None = None
    scan_index = 0

    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        msg_type = type_map.get(topic)
        if msg_type is None:
            continue
        msg = deserialize_message(data, msg_type)

        if topic == odom_topic:
            p = msg.pose.pose.position
            latest_pose = (float(p.x), float(p.y), float(p.z))
            continue

        # PointCloud2
        if latest_pose is None:
            continue  # wait until we have a pose to anchor the first scan
        scan_index += 1
        if (scan_index - 1) % max(1, stride) != 0:
            continue

        pts = pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        arr = np.array([(p[0], p[1], p[2]) for p in pts], dtype=float)
        if arr.size == 0:
            continue

        yield BagFrame(
            stamp=stamp_ns * 1e-9,
            points_world=arr,
            robot_xy=(latest_pose[0], latest_pose[1]),
            robot_z=latest_pose[2],
        )


def final_pose(
    bag_dir: str | Path, odom_topic: str = "/lightning/slam_odom"
) -> tuple[float, float, float] | None:
    """Return the last odometry position in the bag, or ``None`` if absent.

    Useful as a default navigation goal (the far end of the recorded run).
    """
    import rosbag2_py  # noqa: PLC0415
    from rclpy.serialization import deserialize_message  # noqa: PLC0415
    from nav_msgs.msg import Odometry  # noqa: PLC0415

    bag_dir = Path(bag_dir)
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    last: tuple[float, float, float] | None = None
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != odom_topic:
            continue
        msg = deserialize_message(data, Odometry)
        p = msg.pose.pose.position
        last = (float(p.x), float(p.y), float(p.z))
    return last
