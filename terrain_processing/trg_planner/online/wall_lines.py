"""Extract field wall lines from the raw point cloud (strong geometry).

The competition arena is bounded by long, mutually-perpendicular walls. Rather
than fitting a single enclosing rectangle to grid cells (which over-extends an
L-shaped field and is polluted by tall internal clutter), this module works on
the *raw* point cloud:

1. :class:`WallBandAccumulator` keeps only points in a height band above the
   ground (where walls/fences live, above the floor and below overhead rigs),
   voxel-downsamples their 2D projection, and accumulates hit counts. Real walls
   are hit from many frames -> high count; transient clutter stays sparse.
2. :func:`extract_wall_lines` runs an iterative RANSAC to pull out the dominant
   long straight segments, snaps them to the field's two perpendicular
   orientations (right-angle prior), and merges collinear pieces.

The result is a small set of clean wall segments that follow the true (possibly
L-shaped) outline, ignoring internal obstacles that never form long lines.
Everything is pure numpy and runs on the already-downsampled voxel centres, so
it is cheap enough to call periodically rather than every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LineSegment:
    """A fitted wall segment in world XY."""

    p0: np.ndarray  # (2,) one endpoint
    p1: np.ndarray  # (2,) other endpoint
    angle: float    # orientation in radians, in [0, pi)
    n_inliers: int

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))


@dataclass
class WallLines:
    """Detected wall geometry for one extraction."""

    segments: list[LineSegment] = field(default_factory=list)
    points_xy: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    orientation: float = 0.0  # dominant field axis (radians, in [0, pi/2))


class WallBandAccumulator:
    """Accumulate raw wall-band points into a downsampled 2D hit grid.

    Parameters
    ----------
    voxel:
        2D voxel size (m) for downsampling; also the spatial resolution of the
        wall-point set handed to the line extractor.
    band:
        ``(z_lo, z_hi)`` height-above-ground window (m) kept as wall candidates.
        Set above the floor/low clutter and below overhead rigs.
    ground_percentile:
        Percentile of per-frame z used as the ground reference.
    """

    def __init__(
        self,
        voxel: float = 0.10,
        band: tuple[float, float] = (0.5, 1.5),
        ground_percentile: float = 3.0,
    ) -> None:
        if voxel <= 0.0:
            raise ValueError("voxel must be positive")
        self.voxel = float(voxel)
        self.band = (float(band[0]), float(band[1]))
        self.ground_percentile = float(ground_percentile)
        # Sparse hit grid: (ix, iy) -> count.
        self._hits: dict[tuple[int, int], int] = {}

    def reset(self) -> None:
        self._hits.clear()

    def add_cloud(
        self,
        points_world: np.ndarray,
        ground_z: float | None = None,
    ) -> None:
        """Integrate a map-frame cloud, keeping only wall-band points.

        ``ground_z`` overrides the per-frame ground estimate (e.g. the builder's
        shared ground height) when given.
        """
        pts = np.asarray(points_world, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            return
        z = pts[:, 2]
        ground = ground_z if ground_z is not None else float(
            np.percentile(z, self.ground_percentile)
        )
        ha = z - ground
        keep = (ha >= self.band[0]) & (ha <= self.band[1])
        if not np.any(keep):
            return
        xy = pts[keep, :2]
        ix = np.floor(xy[:, 0] / self.voxel).astype(np.int64)
        iy = np.floor(xy[:, 1] / self.voxel).astype(np.int64)
        # Count unique voxels touched this frame (one vote per frame per voxel).
        keys = np.unique(np.stack((ix, iy), axis=1), axis=0)
        for kx, ky in keys:
            k = (int(kx), int(ky))
            self._hits[k] = self._hits.get(k, 0) + 1

    def points(self, min_count: int = 1) -> np.ndarray:
        """World XY centres of voxels hit in at least ``min_count`` frames."""
        if not self._hits:
            return np.empty((0, 2), dtype=float)
        items = [(k, c) for k, c in self._hits.items() if c >= min_count]
        if not items:
            return np.empty((0, 2), dtype=float)
        ks = np.array([k for k, _ in items], dtype=float)
        return (ks + 0.5) * self.voxel

    def counts(self, min_count: int = 1) -> np.ndarray:
        """Hit counts aligned with :meth:`points` (same ordering)."""
        items = [c for _, c in self._hits.items() if c >= min_count]
        return np.array(items, dtype=np.int64)


def _ransac_line(
    pts: np.ndarray,
    threshold: float,
    iters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float] | None:
    """One RANSAC line over ``pts``; returns ``(inlier_mask, angle)`` or None.

    A candidate is scored by inlier count; ties favour the model seen first.
    """
    n = len(pts)
    if n < 2:
        return None
    best_mask = None
    best_count = 0
    best_angle = 0.0
    for _ in range(iters):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        p, q = pts[i], pts[j]
        d = q - p
        norm = np.hypot(d[0], d[1])
        if norm < 1e-6:
            continue
        # Normal of the line; perpendicular distance of every point.
        nx, ny = -d[1] / norm, d[0] / norm
        dist = np.abs((pts[:, 0] - p[0]) * nx + (pts[:, 1] - p[1]) * ny)
        mask = dist <= threshold
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
            best_angle = float(np.arctan2(d[1], d[0]) % np.pi)
    if best_mask is None:
        return None
    return best_mask, best_angle


def _fit_angle(pts: np.ndarray) -> float:
    """Total-least-squares orientation of ``pts`` in [0, pi)."""
    c = pts - pts.mean(axis=0)
    # Principal direction = eigenvector of the largest eigenvalue of cov.
    cov = c.T @ c
    w, v = np.linalg.eigh(cov)
    d = v[:, np.argmax(w)]
    return float(np.arctan2(d[1], d[0]) % np.pi)


def _segment_from_inliers(pts: np.ndarray, angle: float) -> LineSegment:
    """Build a segment by projecting inliers onto the (snapped) direction."""
    u = np.array([np.cos(angle), np.sin(angle)])
    c = pts.mean(axis=0)
    t = (pts - c) @ u
    p0 = c + t.min() * u
    p1 = c + t.max() * u
    return LineSegment(p0=p0, p1=p1, angle=angle, n_inliers=len(pts))


def _largest_contiguous_run(
    pts: np.ndarray, angle: float, max_gap: float
) -> np.ndarray:
    """Return the inlier subset forming the longest gap-free run along the line.

    A real wall is a *continuous* strip of points; phantom lines fitted through
    field-wide scattered clutter have large gaps between their inliers. Project
    every inlier onto the line direction, sort, and keep the longest contiguous
    block whose consecutive spacing never exceeds ``max_gap``.
    """
    if len(pts) <= 1:
        return pts
    u = np.array([np.cos(angle), np.sin(angle)])
    t = pts @ u
    order = np.argsort(t)
    ts = t[order]
    # Split into runs wherever a gap exceeds max_gap.
    gaps = np.diff(ts)
    breaks = np.where(gaps > max_gap)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(ts)]))
    best_i = int(np.argmax([ts[e - 1] - ts[s] for s, e in zip(starts, ends)]))
    s, e = starts[best_i], ends[best_i]
    return pts[order[s:e]]


def extract_wall_lines(
    points_xy: np.ndarray,
    *,
    threshold: float = 0.12,
    min_inliers: int = 40,
    min_length: float = 1.5,
    max_gap: float = 0.6,
    max_lines: int = 12,
    iters: int = 200,
    snap_right_angle: bool = True,
    random_seed: int = 7,
) -> WallLines:
    """Extract dominant, right-angle-snapped wall segments from 2D points.

    Parameters
    ----------
    threshold:
        Max perpendicular distance (m) for a point to count as a line inlier.
    min_inliers / min_length:
        A segment is kept only if it has enough inliers AND spans this length,
        which rejects short clutter blobs that happen to be collinear.
    max_gap:
        Max spacing (m) allowed between consecutive inliers along the line. The
        fitted line is trimmed to its longest gap-free run, so phantom lines
        threaded through scattered field-wide clutter collapse and are rejected.
    max_lines:
        Stop after this many segments (a field boundary needs only a handful).
    snap_right_angle:
        Snap every segment to the field's two perpendicular orientations,
        derived from the longest segment.
    """
    pts = np.asarray(points_xy, dtype=float)
    rng = np.random.default_rng(random_seed)
    remaining = pts.copy()
    raw: list[tuple[np.ndarray, float]] = []  # (inlier_pts, angle)

    while len(remaining) >= min_inliers and len(raw) < max_lines:
        res = _ransac_line(remaining, threshold, iters, rng)
        if res is None:
            break
        mask, _ = res
        if int(mask.sum()) < min_inliers:
            break
        inliers = remaining[mask]
        angle = _fit_angle(inliers)
        # Trim to the longest continuous run -> drops scattered clutter lines.
        run = _largest_contiguous_run(inliers, angle, max_gap)
        if len(run) >= min_inliers:
            angle = _fit_angle(run)
            if _segment_from_inliers(run, angle).length >= min_length:
                raw.append((run, angle))
        remaining = remaining[~mask]

    if not raw:
        return WallLines(segments=[], points_xy=pts, orientation=0.0)

    # Dominant orientation = angle of the segment with the most inliers.
    dom_idx = int(np.argmax([len(p) for p, _ in raw]))
    orientation = raw[dom_idx][1] % (np.pi / 2.0)

    segments: list[LineSegment] = []
    for inliers, angle in raw:
        if snap_right_angle:
            # Snap to {orientation, orientation + pi/2}.
            cand = np.array([orientation, orientation + np.pi / 2.0]) % np.pi
            diff = np.abs(((angle - cand + np.pi / 2.0) % np.pi) - np.pi / 2.0)
            angle = float(cand[int(np.argmin(diff))])
        segments.append(_segment_from_inliers(inliers, angle))

    segments = _merge_collinear(segments, threshold)
    segments.sort(key=lambda s: s.length, reverse=True)
    return WallLines(segments=segments, points_xy=pts, orientation=orientation)


def _merge_collinear(segments: list[LineSegment], threshold: float) -> list[LineSegment]:
    """Merge segments with the same orientation and offset that overlap/abut."""
    if len(segments) <= 1:
        return segments
    used = [False] * len(segments)
    out: list[LineSegment] = []
    for i, si in enumerate(segments):
        if used[i]:
            continue
        ui = np.array([np.cos(si.angle), np.sin(si.angle)])
        ni = np.array([-ui[1], ui[0]])
        group_pts = [si.p0, si.p1]
        off_i = si.p0 @ ni
        used[i] = True
        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            sj = segments[j]
            # Same orientation?
            if abs(((si.angle - sj.angle + np.pi / 2.0) % np.pi) - np.pi / 2.0) > 0.05:
                continue
            # Same line (offset close)?
            if abs(sj.p0 @ ni - off_i) > 2.0 * threshold:
                continue
            group_pts.extend([sj.p0, sj.p1])
            used[j] = True
        g = np.array(group_pts)
        merged = _segment_from_inliers(g, si.angle)
        merged.n_inliers = sum(
            s.n_inliers for k, s in enumerate(segments) if used[k] and k >= i
        ) or si.n_inliers
        out.append(merged)
    return out
