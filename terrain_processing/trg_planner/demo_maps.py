from __future__ import annotations

import numpy as np

from .elevation_map import ElevationMap


def make_ridge_pass_map(
    shape: tuple[int, int] = (90, 90),
    resolution: float = 0.15,
) -> ElevationMap:
    rows, cols = shape
    xs = (np.arange(cols) + 0.5) * resolution
    ys = (np.arange(rows) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys)

    base = 0.015 * np.sin(0.8 * xx) + 0.015 * np.cos(0.6 * yy)
    rolling = 0.02 * np.sin(0.25 * xx + 0.3 * yy)

    wall_center = 0.52 * rows * resolution
    wall_sigma = 0.20
    wall = 0.30 * np.exp(-((yy - wall_center) ** 2) / (2.0 * wall_sigma**2))

    pass_center = 0.50 * cols * resolution
    pass_sigma = 0.55
    pass_relief = np.exp(-((xx - pass_center) ** 2) / (2.0 * pass_sigma**2))
    wall *= 1.0 - 0.98 * pass_relief

    start_basin = -0.03 * np.exp(-((xx - 1.5) ** 2 + (yy - 1.5) ** 2) / 2.5)
    goal_basin = -0.03 * np.exp(-((xx - 11.5) ** 2 + (yy - 11.0) ** 2) / 2.5)

    heights = base + rolling + wall + start_basin + goal_basin
    return ElevationMap(heights=heights, resolution=resolution)
