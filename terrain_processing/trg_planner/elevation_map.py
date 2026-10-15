from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class ElevationMap:
    heights: np.ndarray
    resolution: float
    origin: tuple[float, float] = (0.0, 0.0)
    _grid_x: np.ndarray = field(init=False, repr=False)
    _grid_y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.heights = np.asarray(self.heights, dtype=float)
        if self.heights.ndim != 2:
            raise ValueError("heights must be a 2D array")
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")

        rows, cols = self.heights.shape
        x_coords = self.origin[0] + (np.arange(cols) + 0.5) * self.resolution
        y_coords = self.origin[1] + (np.arange(rows) + 0.5) * self.resolution
        self._grid_x, self._grid_y = np.meshgrid(x_coords, y_coords)

    @property
    def shape(self) -> tuple[int, int]:
        return self.heights.shape

    @property
    def width(self) -> float:
        return self.shape[1] * self.resolution

    @property
    def height(self) -> float:
        return self.shape[0] * self.resolution

    @property
    def extent(self) -> tuple[float, float, float, float]:
        xmin = self.origin[0]
        xmax = self.origin[0] + self.width
        ymin = self.origin[1]
        ymax = self.origin[1] + self.height
        return xmin, xmax, ymin, ymax

    def in_bounds_xy(self, xy: tuple[float, float] | np.ndarray) -> bool:
        x, y = float(xy[0]), float(xy[1])
        xmin, xmax, ymin, ymax = self.extent
        return xmin <= x < xmax and ymin <= y < ymax

    def world_to_grid(self, xy: tuple[float, float] | np.ndarray) -> tuple[int, int] | None:
        if not self.in_bounds_xy(xy):
            return None
        x, y = float(xy[0]), float(xy[1])
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        row = min(max(row, 0), self.shape[0] - 1)
        col = min(max(col, 0), self.shape[1] - 1)
        return row, col

    def height_at(self, xy: tuple[float, float] | np.ndarray) -> float:
        index = self.world_to_grid(xy)
        if index is None:
            return float("nan")
        value = self.heights[index]
        return float(value) if np.isfinite(value) else float("nan")

    def point_at(self, xy: tuple[float, float] | np.ndarray) -> np.ndarray | None:
        z = self.height_at(xy)
        if not np.isfinite(z):
            return None
        return np.array([float(xy[0]), float(xy[1]), z], dtype=float)

    def disk_points(self, center_xy: tuple[float, float] | np.ndarray, radius: float) -> np.ndarray:
        center_xy = np.asarray(center_xy, dtype=float)
        if radius <= 0.0:
            raise ValueError("radius must be positive")

        # Only evaluate the local bounding-box subgrid instead of the full grid;
        # this keeps results identical while making per-query cost O(window^2)
        # rather than O(grid^2), which is essential for the online runner.
        sub = self._bbox_slice(center_xy[0], center_xy[1], radius)
        if sub is None:
            return np.empty((0, 3), dtype=float)
        rows, cols, gx, gy, gz = sub
        distances_sq = (gx - center_xy[0]) ** 2 + (gy - center_xy[1]) ** 2
        mask = distances_sq <= radius**2
        return self._masked_points_sub(mask, gx, gy, gz)

    def ellipse_points(
        self,
        focus_a_xy: tuple[float, float] | np.ndarray,
        focus_b_xy: tuple[float, float] | np.ndarray,
        lateral_radius: float,
    ) -> np.ndarray:
        focus_a_xy = np.asarray(focus_a_xy, dtype=float)
        focus_b_xy = np.asarray(focus_b_xy, dtype=float)
        if lateral_radius <= 0.0:
            raise ValueError("lateral_radius must be positive")

        focal_distance = float(np.linalg.norm(focus_b_xy - focus_a_xy))
        semi_minor = lateral_radius
        semi_major = float(np.sqrt((0.5 * focal_distance) ** 2 + semi_minor**2))
        margin = 0.5 * self.resolution

        # Bounding box covering both foci, expanded by the semi-major axis.
        cx = 0.5 * (focus_a_xy[0] + focus_b_xy[0])
        cy = 0.5 * (focus_a_xy[1] + focus_b_xy[1])
        half_extent = 0.5 * focal_distance + semi_major + margin
        sub = self._bbox_slice(cx, cy, half_extent)
        if sub is None:
            return np.empty((0, 3), dtype=float)
        rows, cols, gx, gy, gz = sub
        dist_a = np.sqrt((gx - focus_a_xy[0]) ** 2 + (gy - focus_a_xy[1]) ** 2)
        dist_b = np.sqrt((gx - focus_b_xy[0]) ** 2 + (gy - focus_b_xy[1]) ** 2)
        mask = dist_a + dist_b <= (2.0 * semi_major + margin)
        return self._masked_points_sub(mask, gx, gy, gz)

    def _bbox_slice(self, cx: float, cy: float, half_extent: float):
        """Return (row_slice, col_slice, grid_x, grid_y, heights) for the
        axis-aligned box [cx±half_extent, cy±half_extent], or None if empty."""
        rows, cols = self.heights.shape
        c0 = int(np.floor((cx - half_extent - self.origin[0]) / self.resolution))
        c1 = int(np.ceil((cx + half_extent - self.origin[0]) / self.resolution))
        r0 = int(np.floor((cy - half_extent - self.origin[1]) / self.resolution))
        r1 = int(np.ceil((cy + half_extent - self.origin[1]) / self.resolution))
        c0 = max(0, c0)
        r0 = max(0, r0)
        c1 = min(cols, c1 + 1)
        r1 = min(rows, r1 + 1)
        if c1 <= c0 or r1 <= r0:
            return None
        row_slice = slice(r0, r1)
        col_slice = slice(c0, c1)
        return (
            row_slice,
            col_slice,
            self._grid_x[row_slice, col_slice],
            self._grid_y[row_slice, col_slice],
            self.heights[row_slice, col_slice],
        )

    @staticmethod
    def _masked_points_sub(
        mask: np.ndarray, gx: np.ndarray, gy: np.ndarray, gz: np.ndarray
    ) -> np.ndarray:
        finite_mask = mask & np.isfinite(gz)
        if not np.any(finite_mask):
            return np.empty((0, 3), dtype=float)
        return np.column_stack(
            (gx[finite_mask], gy[finite_mask], gz[finite_mask])
        ).astype(float, copy=False)

    def _masked_points(self, mask: np.ndarray) -> np.ndarray:
        finite_mask = mask & np.isfinite(self.heights)
        if not np.any(finite_mask):
            return np.empty((0, 3), dtype=float)
        xs = self._grid_x[finite_mask]
        ys = self._grid_y[finite_mask]
        zs = self.heights[finite_mask]
        return np.column_stack((xs, ys, zs)).astype(float, copy=False)
