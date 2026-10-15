"""Detect the competition field's outer wall boundary.

The RoboCup arena is enclosed by a **solid outer wall that is clearly taller
than any internal obstacle**, and its corners are right angles. We exploit both
facts:

- *Tall + solid* gating (``ElevationGridBuilder.wall_mask``) isolates the real
  wall cells and rejects internal clutter / complex terrain and overhead
  structures (e.g. a height-limit bar, which has free space below it).
- The right-angle assumption lets us fit a **minimum-area enclosing rectangle**
  to the detected wall cells, which is robust to missing wall segments (the
  rectangle simply reflects the currently observed extent) and to internal
  obstacles (they are never wall cells, so they cannot pull the fit inward).

The result is intentionally lightweight (pure numpy) and recomputed each frame
so the boundary estimate sharpens as more of the wall is observed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FieldBoundary:
    """Detected field-wall boundary for one frame."""

    wall_xy: np.ndarray
    """``(N, 2)`` world XY of the cells classified as solid outer wall."""

    corners: np.ndarray | None
    """``(4, 2)`` rectilinear boundary corners (world XY), or ``None`` when too
    few wall cells have been seen to fit one yet."""

    angle: float = 0.0
    """Orientation of the fitted rectangle (radians, 0 when not fitted)."""

    def is_valid(self) -> bool:
        return self.corners is not None


def fit_min_area_rectangle(points_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Minimum-area enclosing rectangle of ``points_xy`` (``(4, 2)``, angle).

    Because the field is rectilinear, sweeping the rectangle orientation and
    keeping the smallest-area axis-aligned bounding box recovers the wall
    alignment even when the field is rotated relative to the map axes.
    """
    pts = np.asarray(points_xy, dtype=float)
    best_area = np.inf
    best_corners = None
    best_angle = 0.0

    # The rectangle is symmetric under 90 deg, so only sweep [0, pi/2).
    for angle in np.linspace(0.0, np.pi / 2.0, 90, endpoint=False):
        c, s = np.cos(angle), np.sin(angle)
        u = np.array([c, s])           # rectangle axis 1
        v = np.array([-s, c])          # rectangle axis 2 (perpendicular)
        pu = pts @ u
        pv = pts @ v
        umin, umax = float(pu.min()), float(pu.max())
        vmin, vmax = float(pv.min()), float(pv.max())
        area = (umax - umin) * (vmax - vmin)
        if area < best_area:
            best_area = area
            best_angle = float(angle)
            best_corners = np.array(
                [
                    umin * u + vmin * v,
                    umax * u + vmin * v,
                    umax * u + vmax * v,
                    umin * u + vmax * v,
                ],
                dtype=float,
            )

    assert best_corners is not None
    return best_corners, best_angle


def detect_field_boundary(
    builder,
    *,
    wall_height: float = 1.0,
    floor_band: float = 0.3,
    min_wall_cells: int = 30,
) -> FieldBoundary:
    """Classify wall cells and fit the rectilinear field boundary.

    Parameters
    ----------
    builder:
        An :class:`~trg_planner.online.elevation_builder.ElevationGridBuilder`.
    wall_height:
        Minimum height-above-ground for a cell to count as wall. Set this
        clearly above the tallest expected internal obstacle.
    floor_band:
        A wall cell must also have its lowest observed point within this band of
        the ground (solid floor-to-top), which excludes overhead structures.
    min_wall_cells:
        Below this many wall cells the boundary is left unfitted (too little
        evidence early in the run).
    """
    mask = builder.wall_mask(wall_height, floor_band)
    wall_xy = builder.cell_centres_of(mask)
    if len(wall_xy) < min_wall_cells:
        return FieldBoundary(wall_xy=wall_xy, corners=None, angle=0.0)
    corners, angle = fit_min_area_rectangle(wall_xy)
    return FieldBoundary(wall_xy=wall_xy, corners=corners, angle=angle)
