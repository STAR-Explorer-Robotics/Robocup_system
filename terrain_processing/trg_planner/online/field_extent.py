"""Field extent from short-structure points (concave hull).

The competition field is enclosed by a *short* fence (< 1 m) that is
geometrically indistinguishable from internal fences/obstacles, so it cannot be
recovered by picking "the longest straight line". What *is* robust is the field
*extent*: the outermost short-structure points trace the perimeter, while every
internal fence/obstacle falls inside and is ignored.

Pipeline
--------
1. :class:`FieldExtentAccumulator` integrates raw clouds, keeping voxels that sit
   in the fence height band **and** have empty headroom just above (the
   "short-structure" test). This drops hall walls / rope rigs / ceiling, which
   span from the floor up past the headroom band.
2. :func:`extract_field_extent` density-filters the short voxels, builds an
   alpha-shape (concave hull), keeps the single largest boundary loop (filling
   interior holes and dropping stray outliers), and simplifies it to a compact
   rectilinear-ish polygon used to confine elevation/TRG and for visualisation.

The accumulator is cheap (voxel-downsampled, one vote per frame per voxel) and
the extraction is meant to run periodically, not every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _point_in_polygon(points_xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Ray-casting test (crossing-number): ``bool[N]`` mask for points inside a
    closed polygon (boundary points counted as inside)."""
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must be (N,2)")
    poly = np.asarray(polygon, dtype=float)
    if poly.shape[0] < 3:
        return np.zeros(len(pts), dtype=bool)
    x, y = pts[:, 0], pts[:, 1]
    px, py = poly[:, 0], poly[:, 1]
    n = len(poly)
    inside = np.zeros(len(pts), dtype=bool)
    for i in range(n):
        j = (i + 1) % n
        yi, yj = py[i], py[j]
        # skip horizontal edges (they do not change the crossing count)
        if yi == yj:
            continue
        xi, xj = px[i], px[j]
        # does the horizontal ray at y intersect this edge?
        cond = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        )
        inside ^= cond
    return inside


@dataclass
class FieldExtent:
    """Detected field extent for one extraction."""

    polygon: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    points_xy: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    area: float = 0.0

    def is_valid(self) -> bool:
        return len(self.polygon) >= 3 and self.area > 0.0

    def contains(self, points_xy: np.ndarray) -> np.ndarray:
        """``bool[N]`` mask: which of ``points_xy`` (world XY) lie inside the polygon.

        Uses ray-casting (crossing-number).  Points exactly on the boundary are
        counted as inside (the polygon is treated as a closed set).
        """
        return _point_in_polygon(points_xy, self.polygon)


class FieldExtentAccumulator:
    """Accumulate short-structure (fence-height) voxels across frames.

    Parameters
    ----------
    voxel:
        2D voxel size (m) for downsampling.
    fence_band:
        ``(lo, hi)`` height-above-ground window (m) where the short fence lives.
    headroom_band:
        ``(lo, hi)`` window just above the fence; a column occupied here is a
        *tall* structure (hall wall, rig) and is excluded. Keep this below the
        ceiling/overhead rigs so they do not blanket the whole field.
    """

    def __init__(
        self,
        voxel: float = 0.10,
        fence_band: tuple[float, float] = (0.2, 0.8),
        headroom_band: tuple[float, float] = (0.9, 1.5),
    ) -> None:
        if voxel <= 0.0:
            raise ValueError("voxel must be positive")
        self.voxel = float(voxel)
        self.fence_band = (float(fence_band[0]), float(fence_band[1]))
        self.headroom_band = (float(headroom_band[0]), float(headroom_band[1]))
        self._fence_hits: dict[tuple[int, int], int] = {}
        self._tall: set[tuple[int, int]] = set()

    def reset(self) -> None:
        self._fence_hits.clear()
        self._tall.clear()

    def add_cloud(
        self, points_world: np.ndarray, ground_z: float | None = None
    ) -> None:
        """Integrate one map-frame cloud (``ground_z`` overrides ground est.)."""
        pts = np.asarray(points_world, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            return
        z = pts[:, 2]
        ground = ground_z if ground_z is not None else float(np.percentile(z, 3.0))
        ha = z - ground
        ix = np.floor(pts[:, 0] / self.voxel).astype(np.int64)
        iy = np.floor(pts[:, 1] / self.voxel).astype(np.int64)
        fm = (ha >= self.fence_band[0]) & (ha <= self.fence_band[1])
        if np.any(fm):
            keys = np.unique(np.stack((ix[fm], iy[fm]), axis=1), axis=0)
            for kx, ky in keys:
                k = (int(kx), int(ky))
                self._fence_hits[k] = self._fence_hits.get(k, 0) + 1
        tm = (ha >= self.headroom_band[0]) & (ha <= self.headroom_band[1])
        if np.any(tm):
            keys = np.unique(np.stack((ix[tm], iy[tm]), axis=1), axis=0)
            for kx, ky in keys:
                self._tall.add((int(kx), int(ky)))

    def short_points(self, min_count: int = 2, dilate: int = 1) -> np.ndarray:
        """World XY of fence voxels seen >= ``min_count`` and not tall.

        ``dilate`` grows the tall set by this many voxels so fence cells abutting
        a hall wall (within the dilation radius) are also rejected.
        """
        if not self._fence_hits:
            return np.empty((0, 2), dtype=float)
        offs = [
            (dx, dy)
            for dx in range(-dilate, dilate + 1)
            for dy in range(-dilate, dilate + 1)
        ]
        kept = []
        for (kx, ky), c in self._fence_hits.items():
            if c < min_count:
                continue
            if any((kx + dx, ky + dy) in self._tall for dx, dy in offs):
                continue
            kept.append((kx, ky))
        if not kept:
            return np.empty((0, 2), dtype=float)
        return (np.array(kept, dtype=float) + 0.5) * self.voxel


class FloorAccumulator:
    """Accumulate floor-height voxels (absolute world Z) across frames.

    Parameters
    ----------
    voxel:
        2D voxel size (m) for downsampling.
    z_min, z_max:
        Absolute world-Z range of the field floor (e.g. [-0.50, -0.10]).
    """

    def __init__(self, voxel: float = 0.10, z_min: float = -0.50, z_max: float = -0.10) -> None:
        self.voxel = float(voxel)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self._hits: dict[tuple[int, int], int] = {}

    def reset(self) -> None:
        self._hits.clear()

    def add_cloud(self, points_world: np.ndarray) -> None:
        pts = np.asarray(points_world, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            return
        z = pts[:, 2]
        m = (z >= self.z_min) & (z <= self.z_max)
        if not np.any(m):
            return
        ix = np.floor(pts[m, 0] / self.voxel).astype(np.int64)
        iy = np.floor(pts[m, 1] / self.voxel).astype(np.int64)
        keys = np.unique(np.stack((ix, iy), axis=1), axis=0)
        for kx, ky in keys:
            k = (int(kx), int(ky))
            self._hits[k] = self._hits.get(k, 0) + 1

    def floor_points(self, min_count: int = 1) -> np.ndarray:
        if not self._hits:
            return np.empty((0, 2), dtype=float)
        kept = [(kx, ky) for (kx, ky), c in self._hits.items() if c >= min_count]
        if not kept:
            return np.empty((0, 2), dtype=float)
        return (np.array(kept, dtype=float) + 0.5) * self.voxel


def _density_filter(pts: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    """Drop isolated points with fewer than ``min_neighbors`` within ``radius``."""
    if len(pts) == 0:
        return pts
    from collections import defaultdict

    keys = np.floor(pts / radius).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (kx, ky) in enumerate(keys):
        buckets[(int(kx), int(ky))].append(i)
    keep = np.zeros(len(pts), dtype=bool)
    r2 = radius * radius
    for i, p in enumerate(pts):
        kx, ky = int(keys[i, 0]), int(keys[i, 1])
        cnt = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((kx + dx, ky + dy), ()):
                    if j != i:
                        d = pts[j] - p
                        if d[0] * d[0] + d[1] * d[1] <= r2:
                            cnt += 1
            # cheap early exit
            if cnt >= min_neighbors:
                break
        keep[i] = cnt >= min_neighbors
    return pts[keep]


def _alpha_shape_boundary_edges(pts: np.ndarray, alpha: float) -> list[tuple[int, int]]:
    """Boundary edges (kept-triangle circumradius < ``alpha``).

    Optional helper kept for diagnostics; the runtime extractor uses the
    grid-morphology path below (no scipy dependency).
    """
    from collections import Counter

    from scipy.spatial import Delaunay

    if len(pts) < 4:
        return []
    tri = Delaunay(pts)
    edge_count: Counter = Counter()
    for ia, ib, ic in tri.simplices:
        a, b, c = pts[ia], pts[ib], pts[ic]
        la = np.hypot(*(b - c))
        lb = np.hypot(*(a - c))
        lc = np.hypot(*(a - b))
        s = (la + lb + lc) / 2.0
        area = max(s * (s - la) * (s - lb) * (s - lc), 1e-12) ** 0.5
        R = (la * lb * lc) / (4.0 * area)
        if R < alpha:
            for u, v in ((ia, ib), (ib, ic), (ic, ia)):
                edge_count[(min(u, v), max(u, v))] += 1
    return [e for e, cnt in edge_count.items() if cnt == 1]


def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _dilate(mask: np.ndarray, iters: int) -> np.ndarray:
    """Binary dilation by ``iters`` 8-connected steps (square growth)."""
    out = mask
    for _ in range(max(iters, 0)):
        p = np.pad(out, 1, mode="constant")
        out = (
            p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
            | p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
        )
    return out


def _erode(mask: np.ndarray, iters: int) -> np.ndarray:
    """Binary erosion by ``iters`` 8-connected steps."""
    out = mask
    for _ in range(max(iters, 0)):
        p = np.pad(out, 1, mode="constant", constant_values=True)
        out = (
            p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
            & p[:-2, :-2] & p[:-2, 2:] & p[2:, :-2] & p[2:, 2:]
        )
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes: flood background from the border, holes are the rest."""
    h, w = mask.shape
    free = ~mask
    reach = np.zeros_like(mask)
    from collections import deque

    dq: deque = deque()
    for x in range(w):
        for y in (0, h - 1):
            if free[y, x] and not reach[y, x]:
                reach[y, x] = True
                dq.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if free[y, x] and not reach[y, x]:
                reach[y, x] = True
                dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and free[ny, nx] and not reach[ny, nx]:
                reach[ny, nx] = True
                dq.append((ny, nx))
    # Holes = free cells not reachable from the border.
    return mask | (free & ~reach)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected component of ``mask``."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    from collections import deque

    cur = 0
    sizes: list[int] = [0]
    for sy in range(h):
        for sx in range(w):
            if mask[sy, sx] and labels[sy, sx] == 0:
                cur += 1
                size = 0
                dq: deque = deque([(sy, sx)])
                labels[sy, sx] = cur
                while dq:
                    y, x = dq.popleft()
                    size += 1
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if (
                                0 <= ny < h and 0 <= nx < w
                                and mask[ny, nx] and labels[ny, nx] == 0
                            ):
                                labels[ny, nx] = cur
                                dq.append((ny, nx))
                sizes.append(size)
    if cur == 0:
        return mask
    best = int(np.argmax(sizes))
    return labels == best


def _trace_outer_contour(mask: np.ndarray) -> np.ndarray:
    """Moore-neighbour boundary trace of a filled binary blob -> (N,2) [col,row]."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=float)
    # Start at the topmost-leftmost filled cell.
    s = int(np.argmin(ys * w + xs))
    start = (int(ys[s]), int(xs[s]))
    # 8-neighbour offsets clockwise starting from "left".
    nbrs = [(0, -1), (-1, -1), (-1, 0), (-1, 1),
            (0, 1), (1, 1), (1, 0), (1, -1)]

    def is_set(y: int, x: int) -> bool:
        return 0 <= y < h and 0 <= x < w and mask[y, x]

    contour = [start]
    cur = start
    b_idx = 0  # came-from direction index
    max_steps = 8 * int(mask.sum()) + 10
    for _ in range(max_steps):
        found = False
        for k in range(8):
            idx = (b_idx + k) % 8
            dy, dx = nbrs[idx]
            ny, nx = cur[0] + dy, cur[1] + dx
            if is_set(ny, nx):
                cur = (ny, nx)
                contour.append(cur)
                # Next search starts "behind" the direction we entered from.
                b_idx = (idx + 5) % 8
                found = True
                break
        if not found:
            break
        if cur == start and len(contour) > 2:
            break
    poly = np.array([[c, r] for r, c in contour], dtype=float)
    return poly


def _simplify_polygon(poly: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker simplification of a closed polygon."""
    if len(poly) < 4:
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

    simplified = dp(np.vstack((poly, poly[:1])))
    return simplified[:-1]


def extract_field_extent(
    points_xy: np.ndarray,
    *,
    grid_res: float = 0.15,
    close_radius: float = 1.8,
    density_radius: float = 0.40,
    density_min_neighbors: int = 6,
    simplify_tol: float = 0.30,
) -> FieldExtent:
    """Field extent from short-structure points via grid morphology (pure numpy).

    The short points are rasterised, morphologically closed to bridge the sparse
    fence gaps, hole-filled to solidify the enclosed field, reduced to the
    largest connected component (dropping stray outliers), and the outer contour
    is traced and simplified.

    Parameters
    ----------
    grid_res:
        Rasterisation cell size (m).
    close_radius:
        Morphological-close radius (m); must exceed half the largest fence gap
        so the perimeter ring closes. Larger smooths more.
    density_radius / density_min_neighbors:
        Isolated-point rejection before rasterising, removes stray spikes.
    simplify_tol:
        Douglas-Peucker tolerance (m) for the final polygon.
    """
    pts = np.asarray(points_xy, dtype=float)
    if len(pts) < 4:
        return FieldExtent(polygon=np.empty((0, 2)), points_xy=pts, area=0.0)
    dpts = _density_filter(pts, density_radius, density_min_neighbors)
    if len(dpts) < 4:
        dpts = pts

    pad = close_radius + grid_res * 2.0
    mn = dpts.min(axis=0) - pad
    mx = dpts.max(axis=0) + pad
    cols = int(np.ceil((mx[0] - mn[0]) / grid_res)) + 1
    rows = int(np.ceil((mx[1] - mn[1]) / grid_res)) + 1
    if rows < 3 or cols < 3 or rows * cols > 4_000_000:
        return FieldExtent(polygon=np.empty((0, 2)), points_xy=dpts, area=0.0)

    ci = np.floor((dpts[:, 0] - mn[0]) / grid_res).astype(int)
    ri = np.floor((dpts[:, 1] - mn[1]) / grid_res).astype(int)
    mask = np.zeros((rows, cols), dtype=bool)
    mask[ri, ci] = True

    rd = max(int(round(close_radius / grid_res)), 1)
    mask = _erode(_dilate(mask, rd), rd)        # morphological close
    mask = _fill_holes(mask)                      # solidify interior
    mask = _largest_component(mask)               # drop outliers
    if not mask.any():
        return FieldExtent(polygon=np.empty((0, 2)), points_xy=dpts, area=0.0)

    contour = _trace_outer_contour(mask)
    if len(contour) < 3:
        return FieldExtent(polygon=np.empty((0, 2)), points_xy=dpts, area=0.0)
    poly = mn + (contour + 0.5) * grid_res        # grid -> world (col=x, row=y)
    poly = _simplify_polygon(poly, simplify_tol)
    area = _polygon_area(poly) if len(poly) >= 3 else 0.0
    return FieldExtent(polygon=poly, points_xy=dpts, area=area)
