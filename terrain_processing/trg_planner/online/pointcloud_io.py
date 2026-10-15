"""Point cloud loading and pose transform helpers (ROS-free).

Used by the offline pipeline to read a saved ``global.pcd`` and to transform
sensor-frame clouds into the map frame using a pose. At the real-robot stage
the same transform logic applies to live ``/lidar_points`` messages using the
TF from lightning.
"""

from __future__ import annotations

import numpy as np


def load_pcd(path: str) -> np.ndarray:
    """Load an ``(N, 3)`` XYZ array from a PCD file using Open3D.

    Open3D is imported lazily so the rest of the package stays dependency-light.
    """
    import open3d as o3d  # noqa: PLC0415 (lazy import by design)

    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points, dtype=float)
    if points.size == 0:
        raise ValueError(f"no points loaded from {path}")
    return points


def pose_to_matrix(
    xyz: tuple[float, float, float] | np.ndarray,
    rpy: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build a 4x4 homogeneous transform from translation + roll/pitch/yaw."""
    x, y, z = (float(v) for v in xyz)
    roll, pitch, yaw = (float(v) for v in rpy)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rot = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rot
    transform[:3, 3] = (x, y, z)
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an ``(N, 3)`` point array."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")
    rotated = points @ transform[:3, :3].T
    return rotated + transform[:3, 3]
