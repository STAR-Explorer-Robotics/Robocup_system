"""Full-field 2.5D elevation and obstacle processing.

Consumes point clouds already expressed in a common map frame and produces an
ElevationMap for downstream planning. The fixed-anchor grid grows in whole-cell
steps while preserving world coordinates. Dirty-cell tracking supports
incremental graph updates.

The implementation includes vertical layering, grounded-obstacle masks, virtual
no-go overlays, optional hole filling, reference-ground locking and reachable
boundary extraction. These are geometric processing rules; they do not establish
sensor calibration or a complete collision-avoidance guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..elevation_map import ElevationMap


@dataclass
class ElevationConfig:
    """Parameters for :class:`ElevationGridBuilder`.

    Mirrors the ``elevation:`` block documented in the plan so values can be
    loaded from YAML at the ROS integration stage.
    """

    resolution: float = 0.10
    """Grid cell size in metres."""

    window_size: float = 20.0
    """Side length (metres) of the initial allocation, centred on the first
    robot pose. The grid grows beyond this as the robot explores; cells are
    never dropped."""

    ground_percentile: float = 3.0
    """Percentile used to estimate the local ground height (robust to noise)."""

    ceiling_clip: float = 3.0
    """Points whose height-above-ground exceeds this are treated as ceiling
    and excluded from the traversability layer."""

    min_points_per_cell: int = 1
    """Minimum hits for a cell to be considered observed."""

    decay: float = 0.0
    """Reserved: time decay for stale observations (0 = no decay)."""

    clearance_threshold: float = 0.0
    """Reserved: if > 0, cells whose clearance (ceiling - ground) is below this
    are marked no-go (too low to pass under)."""

    obstacle_height: float = 0.0
    """If > 0, cells with a solid vertical span rising above this height are
    treated as non-traversable obstacles instead of walking surface."""

    obstacle_floor_band: float = 0.12
    """A solid obstacle must include returns within this height above the local
    floor estimate. Suspended bars therefore do not look like floor obstacles."""

    obstacle_min_span: float = 0.25
    """Minimum vertical span required for the obstacle filter."""

    floor_offset: float = 0.0
    """Minimum height above estimated ground to keep a point (filters the
    ground plane itself out of the obstacle layer when > 0)."""

    range_max: float = 0.0
    """If > 0, drop points farther than this (metres) from the robot before
    integrating them. Rejects far-field noise / ghost returns."""

    range_min: float = 0.0
    """If > 0, drop points closer than this (metres) from the robot (e.g. the
    robot's own body / very-near clutter)."""

    persistence_frames: int = 1
    """A cell only counts as observed once it has been hit in at least this
    many distinct frames. Values > 1 reject transient dynamic objects (people /
    passing robots) that appear in only a few frames."""

    vertical_layering: bool = True
    """If True, keep a per-cell vertical height histogram and report the top of
    the *lowest contiguous layer* as the walking surface, instead of the global
    ``height_max``. This is what lets the builder ignore overhead structures
    (height-limit bars, fence rails) that hang above passable ground: such a
    structure forms a *separate* upper layer and no longer raises the terrain."""

    layer_bin: float = 0.10
    """Vertical bin size (m) of the per-cell height histogram used for layer
    separation. Finer bins localise the surface better at more memory cost."""

    layer_gap: float = 0.75
    """A vertical empty span (m) larger than this splits the column into a new
    layer. Set above the tallest real step/kerb but below the clearance under an
    overhead structure, so terrain stays one layer while a suspended bar splits
    off into its own upper layer."""

    min_points_per_bin: int = 2
    """A height bin needs at least this many hits to count as occupied, so a
    lone noise point cannot fabricate a spurious layer."""

    fill_holes_iters: int = 3
    """Number of interpolation passes used to fill *interior* holes in the
    observed surface (radial-sparsity / occlusion pinholes). 0 disables it.
    Each pass fills only unobserved cells that are well surrounded by known
    cells, so the field is never grown outward past its real boundary."""

    fill_min_neighbors: int = 5
    """A hole cell is only filled when at least this many of its 8 neighbours
    are already known. Higher = fills only deep interior holes (conservative);
    must stay >4 so convex boundaries are not extended outward."""



class ElevationGridBuilder:
    """Accumulate map-frame point clouds into a full-field 2.5D elevation grid.

    The grid is anchored to a fixed world origin (set on the first cloud) and
    grows outward as the robot explores; observed cells are never dropped. Each
    cell stores the maximum and minimum height observed plus a hit count, which
    is enough for the current planner and leaves room for the documented
    extensions. Cells touched by each update are recorded in a dirty set so
    consumers can perform incremental graph updates.
    """

    def __init__(self, config: ElevationConfig | None = None) -> None:
        self.config = config or ElevationConfig()
        if self.config.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.config.window_size <= 0.0:
            raise ValueError("window_size must be positive")

        self._init_cells = int(round(self.config.window_size / self.config.resolution))
        if self._init_cells < 1:
            raise ValueError("window_size too small for the chosen resolution")
        # Current grid dimensions; grow as the robot explores (rows == cols at
        # init, may differ after asymmetric growth).
        self._rows = self._init_cells
        self._cols = self._init_cells

        # Layers are (rows, cols) = (y, x). Observation is tracked by _count;
        # height_max/_min use -inf/+inf sentinels so np.maximum.at / np.minimum.at
        # accumulate correctly (NaN would poison those reductions).
        self._height_max = np.full((self._rows, self._cols), -np.inf, dtype=float)
        self._height_min = np.full((self._rows, self._cols), np.inf, dtype=float)
        self._count = np.zeros((self._rows, self._cols), dtype=np.int64)
        # Number of distinct frames each cell has been hit in (temporal
        # persistence gate for observed_mask).
        self._frames_seen = np.zeros((self._rows, self._cols), dtype=np.int64)
        # Known no-go cells injected by task semantics rather than physical
        # grounded obstacles. For the "avoid holes" task these are low outside
        # floor cells rendered as virtual walls: not traversable and not soft
        # frontiers to explore.
        self._virtual_hard = np.zeros((self._rows, self._cols), dtype=bool)
        self._virtual_hard_base = np.full((self._rows, self._cols), np.nan, dtype=float)
        self._virtual_hard_top = np.full((self._rows, self._cols), np.nan, dtype=float)

        # Per-cell vertical height histogram for layer separation (task 1).
        self._n_bins = self._compute_n_bins()
        self._zhist = np.zeros((self._rows, self._cols, self._n_bins), dtype=np.int32)

        # World coordinate of the (0, 0) cell corner.
        self._origin = np.array([0.0, 0.0], dtype=float)
        self._origin_initialised = False
        self._ground_height = 0.0
        self._ground_height_locked = False

        # Cells (row, col) touched since the last consume_dirty().
        self._dirty: set[tuple[int, int]] = set()
        self._map_dirty = True
        self._cached_map: ElevationMap | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def resolution(self) -> float:
        return self.config.resolution

    @property
    def origin(self) -> tuple[float, float]:
        return float(self._origin[0]), float(self._origin[1])

    @property
    def ground_height(self) -> float:
        """Latest robust map-frame ground estimate used for relative heights."""
        return float(self._ground_height)

    @property
    def ground_height_locked(self) -> bool:
        """Whether the builder is using an externally bootstrapped ground."""
        return bool(self._ground_height_locked)

    def set_ground_height(self, ground_height: float, *, locked: bool = True) -> None:
        """Set the map-frame ground reference used for relative heights."""
        self._ground_height = float(ground_height)
        self._ground_height_locked = bool(locked)
        self._map_dirty = True

    def clear_ground_height_lock(self) -> None:
        """Return to per-cloud percentile ground estimation."""
        self._ground_height_locked = False

    def reset(self) -> None:
        """Clear all accumulated data (e.g. after a loop-closure rebuild)."""
        self._rows = self._init_cells
        self._cols = self._init_cells
        self._height_max = np.full((self._rows, self._cols), -np.inf, dtype=float)
        self._height_min = np.full((self._rows, self._cols), np.inf, dtype=float)
        self._count = np.zeros((self._rows, self._cols), dtype=np.int64)
        self._frames_seen = np.zeros((self._rows, self._cols), dtype=np.int64)
        self._virtual_hard = np.zeros((self._rows, self._cols), dtype=bool)
        self._virtual_hard_base = np.full((self._rows, self._cols), np.nan, dtype=float)
        self._virtual_hard_top = np.full((self._rows, self._cols), np.nan, dtype=float)
        self._n_bins = self._compute_n_bins()
        self._zhist = np.zeros((self._rows, self._cols, self._n_bins), dtype=np.int32)
        self._origin_initialised = False
        self._ground_height = 0.0
        self._ground_height_locked = False
        self._dirty.clear()
        self._map_dirty = True
        self._cached_map = None

    def add_cloud(
        self,
        points_world: np.ndarray,
        robot_xy: tuple[float, float] | np.ndarray | None = None,
        *,
        apply_range_filter: bool = True,
    ) -> None:
        """Integrate a point cloud expressed in the map frame.

        Parameters
        ----------
        points_world:
            ``(N, 3)`` array of XYZ points already transformed into the map
            frame. Sensor-agnostic: fuse multiple LiDARs by concatenating their
            transformed clouds or calling this repeatedly.
        robot_xy:
            Current robot XY in the map frame; used to (re)centre the rolling
            window. If ``None``, the cloud centroid is used.
        apply_range_filter:
            When false, skip ``range_min/range_max`` gating. This is used for
            replaying clouds that were already range-filtered at collection time.
        """
        points_world = np.asarray(points_world, dtype=float)
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError("points_world must be an (N, 3) array")
        if len(points_world) == 0:
            return

        if robot_xy is None:
            robot_xy = points_world[:, :2].mean(axis=0)
        robot_xy = np.asarray(robot_xy, dtype=float)[:2]

        # Ray-distance gating: drop points outside the trusted range ring before
        # they ever touch the grid (far-field noise / very-near self clutter).
        if apply_range_filter and (
            self.config.range_max > 0.0 or self.config.range_min > 0.0
        ):
            dist = np.linalg.norm(points_world[:, :2] - robot_xy, axis=1)
            keep = np.ones(len(points_world), dtype=bool)
            if self.config.range_max > 0.0:
                keep &= dist <= self.config.range_max
            if self.config.range_min > 0.0:
                keep &= dist >= self.config.range_min
            points_world = points_world[keep]
            if len(points_world) == 0:
                return

        self._init_origin(robot_xy)

        z = points_world[:, 2]
        if not self._ground_height_locked:
            # Estimate ground height from the local cloud (robust percentile).
            self._ground_height = float(np.percentile(z, self.config.ground_percentile))

        height_above = z - self._ground_height

        # Keep points in the band [floor_offset, ceiling_clip] above ground.
        band = (height_above >= self.config.floor_offset) & (
            height_above <= self.config.ceiling_clip
        )
        if not np.any(band):
            return

        xy = points_world[band, :2]
        h = height_above[band]

        # Grow the grid so every in-band point falls inside it (full-field:
        # cells are never dropped, the grid expands instead).
        self._ensure_bounds(xy)

        col = ((xy[:, 0] - self._origin[0]) / self.config.resolution).astype(np.int64)
        row = ((xy[:, 1] - self._origin[1]) / self.config.resolution).astype(np.int64)
        inside = (
            (row >= 0) & (row < self._rows) & (col >= 0) & (col < self._cols)
        )
        if not np.any(inside):
            return
        row, col, h = row[inside], col[inside], h[inside]

        # Apply optional decay to existing observations before accumulating.
        if self.config.decay > 0.0:
            self._count = np.maximum(
                0, (self._count.astype(float) * (1.0 - self.config.decay)).astype(np.int64)
            )

        np.maximum.at(self._height_max, (row, col), h)
        np.minimum.at(self._height_min, (row, col), h)
        np.add.at(self._count, (row, col), 1)

        # Accumulate the per-cell vertical height histogram (layer separation).
        if self.config.vertical_layering:
            bins = ((h - self.config.floor_offset) / self.config.layer_bin).astype(np.int64)
            np.clip(bins, 0, self._n_bins - 1, out=bins)
            np.add.at(self._zhist, (row, col, bins), 1)

        # Each cell touched this frame counts once towards temporal persistence.
        touched = np.unique(np.stack((row, col), axis=1), axis=0)
        self._frames_seen[touched[:, 0], touched[:, 1]] += 1

        # Record touched cells so consumers can update incrementally.
        self._dirty.update(zip(row.tolist(), col.tolist()))
        if not self._map_dirty and self._cached_map is not None:
            cur = int(np.isfinite(self._cached_map.heights).sum())
            new_total = int(self.observed_mask().sum())
            if cur == 0 or abs(new_total - cur) / max(new_total, 1) >= 0.10:
                self._map_dirty = True

    def seed_robot_footprint(
        self,
        robot_xy: tuple[float, float] | np.ndarray,
        *,
        radius: float,
        height: float = 0.0,
        only_unobserved: bool = False,
    ) -> int:
        """Mark a small disk around the robot as known flat traversable floor.

        A top-mounted LiDAR often cannot see the ground immediately below the
        robot. The seed provides a local start island for reachable-area growth
        without inventing distant terrain. ``height`` is the builder's relative
        terrain height, i.e. metres above the latest robust ground estimate, not
        an absolute map-frame z. When ``only_unobserved`` is true, cells already
        containing real observations are left untouched so this remains a blind
        spot fallback rather than an override.
        """
        radius = float(radius)
        if radius <= 0.0:
            return 0
        robot_xy = np.asarray(robot_xy, dtype=float)[:2]
        self._init_origin(robot_xy)

        res = self.config.resolution
        half = int(np.ceil(radius / res))
        offsets = np.arange(-half, half + 1, dtype=float) * res
        gx = robot_xy[0] + offsets
        gy = robot_xy[1] + offsets
        xx, yy = np.meshgrid(gx, gy)
        mask = (xx - robot_xy[0]) ** 2 + (yy - robot_xy[1]) ** 2 <= radius**2
        if not mask.any():
            return 0

        xy = np.column_stack((xx[mask], yy[mask]))
        self._ensure_bounds(xy)
        col = ((xy[:, 0] - self._origin[0]) / res).astype(np.int64)
        row = ((xy[:, 1] - self._origin[1]) / res).astype(np.int64)
        inside = (row >= 0) & (row < self._rows) & (col >= 0) & (col < self._cols)
        if not np.any(inside):
            return 0
        row, col = row[inside], col[inside]
        if only_unobserved:
            unobserved = self._count[row, col] <= 0
            if not np.any(unobserved):
                return 0
            row, col = row[unobserved], col[unobserved]
        h = np.full(row.shape, float(height), dtype=float)

        np.maximum.at(self._height_max, (row, col), h)
        np.minimum.at(self._height_min, (row, col), h)
        np.add.at(self._count, (row, col), 1)
        if self.config.vertical_layering:
            bins = ((h - self.config.floor_offset) / self.config.layer_bin).astype(np.int64)
            np.clip(bins, 0, self._n_bins - 1, out=bins)
            np.add.at(self._zhist, (row, col, bins), 1)

        touched = np.unique(np.stack((row, col), axis=1), axis=0)
        self._frames_seen[touched[:, 0], touched[:, 1]] += 1
        self._dirty.update(zip(row.tolist(), col.tolist()))
        self._map_dirty = True
        return int(len(touched))

    def mark_virtual_hard_obstacles(
        self,
        points_xy: np.ndarray,
        *,
        base_height: float = 0.0,
        top_height: float = 0.9,
        anchor_xy: tuple[float, float] | np.ndarray | None = None,
    ) -> int:
        """Mark world-XY samples as known hard no-go cells.

        This is used for task-level virtual walls, not raw perception. The cells
        are treated as hard obstacles by reachable-area flood-fill and soft
        frontier generation, and exported by :meth:`hard_obstacle_points`, but
        they do not become walkable observations.
        """
        points_xy = np.asarray(points_xy, dtype=float)
        if points_xy.ndim != 2 or points_xy.shape[0] == 0:
            return 0
        if points_xy.shape[1] < 2:
            raise ValueError("points_xy must have at least two columns")
        xy = points_xy[:, :2]
        finite = np.isfinite(xy).all(axis=1)
        xy = xy[finite]
        if xy.shape[0] == 0:
            return 0

        if not self._origin_initialised:
            if anchor_xy is None:
                anchor_xy = xy.mean(axis=0)
            self._init_origin(np.asarray(anchor_xy, dtype=float)[:2])

        self._ensure_bounds(xy)
        res = self.config.resolution
        col = np.floor((xy[:, 0] - self._origin[0]) / res).astype(np.int64)
        row = np.floor((xy[:, 1] - self._origin[1]) / res).astype(np.int64)
        inside = (row >= 0) & (row < self._rows) & (col >= 0) & (col < self._cols)
        if not np.any(inside):
            return 0

        cells = np.unique(np.stack((row[inside], col[inside]), axis=1), axis=0)
        row = cells[:, 0]
        col = cells[:, 1]

        changed = ~self._virtual_hard[row, col]
        self._virtual_hard[row, col] = True

        base = float(base_height)
        top = max(base, float(top_height))
        self._virtual_hard_base[row, col] = base
        self._virtual_hard_top[row, col] = top

        self._dirty.update(zip(row.tolist(), col.tolist()))
        self._map_dirty = True
        return int(np.count_nonzero(changed))

    def set_virtual_hard_obstacles(
        self,
        points_xy: np.ndarray,
        *,
        base_height: float = 0.0,
        top_height: float = 0.9,
        anchor_xy: tuple[float, float] | np.ndarray | None = None,
    ) -> int:
        """Replace task-level virtual hard cells with the provided world XY set."""
        self.clear_virtual_hard_obstacles()
        return self.mark_virtual_hard_obstacles(
            points_xy,
            base_height=base_height,
            top_height=top_height,
            anchor_xy=anchor_xy,
        )

    def clear_virtual_hard_obstacles(self) -> int:
        """Clear all task-level virtual hard obstacle cells."""
        rows, cols = np.nonzero(self._virtual_hard)
        if rows.size == 0:
            return 0
        self._virtual_hard[rows, cols] = False
        self._virtual_hard_base[rows, cols] = np.nan
        self._virtual_hard_top[rows, cols] = np.nan
        self._dirty.update(zip(rows.tolist(), cols.tolist()))
        self._map_dirty = True
        return int(rows.size)

    def virtual_hard_obstacle_count(self) -> int:
        """Number of cells currently marked as task-level hard obstacles."""
        return int(np.count_nonzero(self._virtual_hard))

    def build_map(self) -> ElevationMap:
        """Produce an :class:`ElevationMap` from the current grid.

        Unknown cells remain NaN so the planner naturally cannot place nodes
        there — only genuinely observed terrain is traversable.
        Use :meth:`effective_observed_mask` for the set of cells that carry
        real or interpolated data.

        The result is cached and only rebuilt when ``add_cloud`` or ``clean``
        has been called since the last ``build_map``.
        """
        if not self._map_dirty and self._cached_map is not None:
            return self._cached_map
        heights, _ = self.filled_surface()
        self._cached_map = ElevationMap(
            heights=heights,
            resolution=self.config.resolution,
            origin=(float(self._origin[0]), float(self._origin[1])),
        )
        self._map_dirty = False
        return self._cached_map

    def observed_mask(self) -> np.ndarray:
        """Boolean ``(rows, cols)`` mask of cells with enough observations.

        A cell qualifies when it has both enough point hits
        (``min_points_per_cell``) and has been seen across enough distinct
        frames (``persistence_frames``); the latter rejects transient dynamics.
        """
        mask = self._count >= self.config.min_points_per_cell
        if self.config.persistence_frames > 1:
            mask &= self._frames_seen >= self.config.persistence_frames
        return mask

    def filled_surface(self) -> tuple[np.ndarray, np.ndarray]:
        """Walking surface with interior holes interpolated.

        Returns ``(heights, known)`` where ``heights`` is a ``(rows, cols)``
        float array (NaN where still unknown) and ``known`` is the boolean mask
        of cells that carry real *or* interpolated data.

        Only unobserved cells with at least ``fill_min_neighbors`` known
        neighbours are filled, repeated for ``fill_holes_iters`` passes. Because
        a cell needs a majority of its neighbourhood already known, the filled
        region creeps into concave holes but never balloons past the real
        boundary, so the field is not invented outside its walls.
        """
        surf = self._walking_surface()
        blocked = self.no_go_mask()
        heights = np.where(self.observed_mask() & ~blocked, surf, np.nan)
        known = np.isfinite(heights)

        iters = max(0, int(self.config.fill_holes_iters))
        min_nb = max(5, int(self.config.fill_min_neighbors))
        for _ in range(iters):
            holes = ~known & ~blocked
            if not holes.any():
                break
            filled_vals = np.where(known, heights, 0.0)
            nb_sum = self._neighbour_sum(filled_vals)
            nb_cnt = self._neighbour_sum(known.astype(float))
            ready = holes & (nb_cnt >= min_nb)
            if not ready.any():
                break
            heights = heights.copy()
            heights[ready] = nb_sum[ready] / nb_cnt[ready]
            known = known | ready
        return heights, known

    def effective_observed_mask(self) -> np.ndarray:
        """Cells that carry real or hole-filled data (for display / export)."""
        _, known = self.filled_surface()
        return known

    def walkable_mask(self) -> np.ndarray:
        """Cells that currently represent reachable walking-surface candidates.

        This is the semantic surface exported to TRG as terrain. Solid no-go
        structures are deliberately excluded here and exported separately as
        hard-obstacle evidence.
        """
        heights, _ = self.filled_surface()
        return np.isfinite(heights)

    def hard_obstacle_mask(self) -> np.ndarray:
        """Cells that contain grounded, non-traversable vertical structure."""
        return self.no_go_mask()

    def hard_obstacle_points(
        self,
        *,
        column_step: float | None = None,
        max_height: float | None = None,
    ) -> np.ndarray:
        """Point samples for grounded hard obstacles.

        The C++ TRG ROS interface consumes a PointCloud2 sampled from the
        elevation grid. For wall/fence semantics we export no-go cells as short
        vertical columns, preserving enough local z-spread for TRG collision and
        frontier logic while avoiding raw ceiling/ghost points.
        """
        mask = self.hard_obstacle_mask()
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        bases = self._height_min[rows, cols].astype(float)
        tops_layer = self._hard_obstacle_top()
        tops = tops_layer[rows, cols].astype(float)
        virtual = self._virtual_hard[rows, cols]
        if virtual.any():
            vbases = self._virtual_hard_base[rows, cols]
            vtops = self._virtual_hard_top[rows, cols]
            bases = np.where(virtual, vbases, bases)
            tops = np.where(virtual, vtops, tops)
        finite = np.isfinite(bases) & np.isfinite(tops)
        if not finite.any():
            return np.empty((0, 3), dtype=np.float32)

        rows = rows[finite]
        cols = cols[finite]
        bases = bases[finite]
        tops = tops[finite]

        height_cap = self.config.ceiling_clip if max_height is None else float(max_height)
        if height_cap > 0.0:
            tops = np.minimum(tops, height_cap)
        bases = np.maximum(bases, 0.0)
        tops = np.maximum(tops, bases)

        step = self.config.resolution if column_step is None else float(column_step)
        step = max(1e-3, step)
        nsteps = np.floor((tops - bases) / step).astype(np.int64) + 1
        total = int(nsteps.sum())
        if total <= 0:
            return np.empty((0, 3), dtype=np.float32)

        res = self.config.resolution
        xs = self._origin[0] + (cols.astype(np.float32) + 0.5) * res
        ys = self._origin[1] + (rows.astype(np.float32) + 0.5) * res
        xs_rep = np.repeat(xs, nsteps)
        ys_rep = np.repeat(ys, nsteps)
        bases_rep = np.repeat(bases.astype(np.float32), nsteps)
        tops_rep = np.repeat(tops.astype(np.float32), nsteps)
        starts = np.repeat(np.cumsum(nsteps) - nsteps, nsteps)
        step_idx = np.arange(total, dtype=np.int64) - starts
        zs = np.minimum(bases_rep + step_idx.astype(np.float32) * step, tops_rep)
        return np.column_stack((xs_rep, ys_rep, zs)).astype(np.float32)

    def reachable_mask(
        self,
        robot_xy: tuple[float, float] | np.ndarray,
        *,
        step_tolerance: float = 0.30,
    ) -> np.ndarray:
        """Walkable component reachable from ``robot_xy`` under step limits."""
        out = np.zeros((self._rows, self._cols), dtype=bool)
        robot_xy = np.asarray(robot_xy, dtype=float)[:2]
        res = self.config.resolution
        col = int(np.floor((robot_xy[0] - self._origin[0]) / res))
        row = int(np.floor((robot_xy[1] - self._origin[1]) / res))
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            return out

        surf, _ = self.filled_surface()
        walkable = np.isfinite(surf)
        if not walkable[row, col]:
            return out

        stack: list[tuple[int, int]] = [(row, col)]
        out[row, col] = True
        while stack:
            r, c = stack.pop()
            ref_z = surf[r, c]
            if not np.isfinite(ref_z):
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < self._rows and 0 <= nc < self._cols):
                        continue
                    if out[nr, nc] or not walkable[nr, nc]:
                        continue
                    nz = surf[nr, nc]
                    if np.isfinite(nz) and abs(nz - ref_z) <= step_tolerance:
                        out[nr, nc] = True
                        stack.append((nr, nc))
        return out

    def soft_frontier_points(
        self,
        robot_xy: tuple[float, float] | np.ndarray,
        *,
        step_tolerance: float = 0.30,
        exclude_hard_neighbors: bool = True,
    ) -> np.ndarray:
        """Reachable cells adjacent to unknown space but not hard walls.

        These are candidate exploration frontiers. A wall/fence boundary is a
        hard obstacle, so cells touching it are suppressed; an unexplored open
        gap remains a soft frontier and is worth approaching.
        """
        reachable = self.reachable_mask(robot_xy, step_tolerance=step_tolerance)
        if not reachable.any():
            return np.empty((0, 3), dtype=np.float32)

        walkable_heights, _ = self.filled_surface()
        hard = self.hard_obstacle_mask()
        known = np.isfinite(walkable_heights) | hard
        unknown = ~known
        frontier = reachable & self._dilate8(unknown)
        if exclude_hard_neighbors:
            frontier &= ~self._dilate8(hard)
        frontier &= np.isfinite(walkable_heights)

        rows, cols = np.nonzero(frontier)
        if rows.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        res = self.config.resolution
        xs = self._origin[0] + (cols.astype(np.float32) + 0.5) * res
        ys = self._origin[1] + (rows.astype(np.float32) + 0.5) * res
        zs = walkable_heights[rows, cols].astype(np.float32)
        return np.column_stack((xs, ys, zs)).astype(np.float32)

    @staticmethod
    def _neighbour_sum(arr: np.ndarray) -> np.ndarray:
        """Sum of the 8 neighbours of every cell (zero-padded edges)."""
        pad = np.pad(arr, 1)
        return (
            pad[0:-2, 0:-2] + pad[0:-2, 1:-1] + pad[0:-2, 2:]
            + pad[1:-1, 0:-2] + pad[1:-1, 2:]
            + pad[2:, 0:-2] + pad[2:, 1:-1] + pad[2:, 2:]
        )

    def wall_mask(self, wall_height: float, floor_band: float) -> np.ndarray:
        """Boolean mask of cells belonging to a *solid, tall* structure.

        A field outer wall is both clearly taller than internal obstacles
        (``height_max >= wall_height``) and solid from the ground up
        (``height_min <= floor_band``). The second test is what separates a
        real wall from an overhead structure such as a height-limit bar, whose
        lowest observed point sits well above the ground (free space below it).
        """
        obs = self.observed_mask()
        tall = self._height_max >= wall_height
        solid = self._height_min <= floor_band
        return obs & tall & solid

    def cell_centres_of(self, mask: np.ndarray) -> np.ndarray:
        """World XY centres of every ``True`` cell in ``mask`` (``(N, 2)``)."""
        rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
        if len(rows) == 0:
            return np.empty((0, 2), dtype=float)
        res = self.config.resolution
        xs = self._origin[0] + (cols + 0.5) * res
        ys = self._origin[1] + (rows + 0.5) * res
        return np.column_stack((xs, ys))


    def consume_dirty(self) -> set[tuple[int, int]]:
        """Return the cells touched since the last call and clear the set.

        Cells are ``(row, col)`` indices into the current grid. Indices stay
        valid as long as the grid is not grown again before they are used; the
        builder keeps them consistent with the live origin (a growth step
        offsets pending dirty indices, see :meth:`_ensure_bounds`).
        """
        dirty = self._dirty
        self._dirty = set()
        return dirty

    def peek_dirty(self) -> set[tuple[int, int]]:
        """Return a copy of the current dirty set without clearing it."""
        return set(self._dirty)

    def dirty_cell_centres(self, dirty: set[tuple[int, int]] | None = None) -> np.ndarray:
        """World XY centres of the given (or current) dirty cells.

        Returns an ``(N, 2)`` array; empty ``(0, 2)`` when there are none.
        """
        if dirty is None:
            dirty = self._dirty
        if not dirty:
            return np.empty((0, 2), dtype=float)
        rows_cols = np.array(sorted(dirty), dtype=float)
        centres = np.empty_like(rows_cols)
        centres[:, 0] = self._origin[0] + (rows_cols[:, 1] + 0.5) * self.config.resolution
        centres[:, 1] = self._origin[1] + (rows_cols[:, 0] + 0.5) * self.config.resolution
        return centres

    def no_go_mask(self) -> np.ndarray:
        """Boolean ``(rows, cols)`` mask of cells that are not traversable.

        Combines optional low-clearance and solid-obstacle rules. Returns an
        all-False mask when both are disabled.
        """
        mask = np.zeros((self._rows, self._cols), dtype=bool)
        obs = self.observed_mask()
        if self.config.clearance_threshold > 0.0:
            # Clearance = top observed height; a cell is no-go if the lowest
            # obstacle above ground is below the clearance threshold while a
            # higher structure also exists.
            clearance = self._height_min
            has_overhead = self._height_max > self.config.clearance_threshold
            too_low = np.isfinite(clearance) & (clearance < self.config.clearance_threshold)
            mask |= obs & has_overhead & too_low

        if self.config.obstacle_height > 0.0:
            obstacle_top = self._hard_obstacle_top()
            span = obstacle_top - self._height_min
            tall = obstacle_top >= self.config.obstacle_height
            grounded = self._height_min <= self.config.obstacle_floor_band
            solid = span >= self.config.obstacle_min_span
            mask |= obs & tall & grounded & solid
        mask |= self._virtual_hard
        return mask

    def fill_holes(self) -> None:
        """Hook for future hole filling (morphology / interpolation).

        Intentionally a no-op in v0. Documented in the plan as a deferred
        optimisation; implementing it here keeps the call site stable.
        """
        return None

    def traversable_boundary(
        self, robot_xy: tuple[float, float] | np.ndarray,
        *,
        step_tolerance: float = 0.30,
    ) -> np.ndarray | None:
        """Contour of the traversable region reachable from ``robot_xy``.

        Flood-fills from the robot through cells that are observed and whose
        walking-surface height does not jump more than ``step_tolerance``
        between 8-neighbours.  Returns an ``(N, 2)`` array of world-XY
        boundary points (closed loop) of the reachable component, or ``None``
        if the robot cell is unobserved.
        """
        flooded = self.reachable_mask(robot_xy, step_tolerance=step_tolerance)
        if not flooded.any():
            return None
        return self._trace_boundary(flooded)

    def clean(self, robot_xy: tuple[float, float] | np.ndarray | None = None) -> int:
        """Periodic morphological cleanup of the observed grid.

        Applies a binary closing to the observed mask (to bridge pinhole gaps
        so a single missing cell does not split the field) and then, if
        ``robot_xy`` is given, keeps only the observed component 8-connected to
        the robot. Observed cells outside that component are reset to
        unobserved, which removes disconnected spurious blobs and far dynamic
        objects that the range gate / persistence let through.

        Returns the number of cells reset. Cells reset here are also dropped
        from the pending dirty set.
        """
        observed = self.observed_mask()
        if not observed.any():
            return 0

        closed = self._binary_close(observed)

        if robot_xy is not None:
            robot_xy = np.asarray(robot_xy, dtype=float)[:2]
            col = int(np.floor((robot_xy[0] - self._origin[0]) / self.config.resolution))
            row = int(np.floor((robot_xy[1] - self._origin[1]) / self.config.resolution))
            keep = self._connected_component(closed, (row, col))
        else:
            # No robot anchor: keep the single largest connected component.
            keep = self._largest_component(closed)

        # Reset observed cells that fall outside the kept component.
        to_reset = observed & ~keep & ~self._virtual_hard
        n_reset = int(to_reset.sum())
        if n_reset:
            self._height_max[to_reset] = -np.inf
            self._height_min[to_reset] = np.inf
            self._count[to_reset] = 0
            self._frames_seen[to_reset] = 0
            if self.config.vertical_layering:
                self._zhist[to_reset] = 0
            reset_cells = set(zip(*np.nonzero(to_reset)))
            if self._dirty:
                self._dirty -= {(int(r), int(c)) for (r, c) in reset_cells}
        if n_reset > 0:
            self._map_dirty = True
        return n_reset

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _compute_n_bins(self) -> int:
        """Number of vertical histogram bins spanning [floor_offset, ceiling_clip]."""
        span = self.config.ceiling_clip - self.config.floor_offset
        return max(1, int(np.ceil(span / max(1e-6, self.config.layer_bin))))

    def _lowest_layer_cap(self) -> np.ndarray:
        """Per-cell upper edge of the lowest contiguous occupied height layer.

        Walks each cell's height histogram upward from its first occupied bin and
        stops at the first vertical gap wider than ``layer_gap``. The returned
        value is the height (above ground) at the top edge of that lowest layer;
        cells with no occupied bin get ``+inf`` (no cap). This is what separates
        the walking surface from a suspended overhead structure: the overhead
        sits beyond the gap and is therefore excluded from the cap.
        """
        occ = self._zhist >= self.config.min_points_per_bin  # (R, C, B)
        n_bins = occ.shape[2]
        gap_bins = max(1, int(round(self.config.layer_gap / self.config.layer_bin)))

        started = np.zeros((self._rows, self._cols), dtype=bool)
        ended = np.zeros((self._rows, self._cols), dtype=bool)
        top_bin = np.zeros((self._rows, self._cols), dtype=np.int64)
        gap = np.zeros((self._rows, self._cols), dtype=np.int64)

        for b in range(n_bins):
            occ_b = occ[:, :, b]
            in_layer = started & ~ended
            # Extend the current layer through an occupied bin.
            set_top = (in_layer | (~started & occ_b)) & occ_b
            top_bin = np.where(set_top, b, top_bin)
            gap = np.where(set_top, 0, gap)
            started |= occ_b
            # Count empty bins inside an open layer; close it past the gap.
            empty = (started & ~ended) & ~occ_b
            gap = np.where(empty, gap + 1, gap)
            ended |= empty & (gap > gap_bins)

        cap = self.config.floor_offset + (top_bin + 1).astype(float) * self.config.layer_bin
        # Cells never occupied have no cap.
        cap = np.where(started, cap, np.inf)
        return cap

    def _walking_surface(self) -> np.ndarray:
        """Per-cell walking-surface height with overhead structures removed.

        Where the global ``height_max`` already lies within the lowest layer
        (the common case: smooth terrain, no overhead) the *continuous*
        ``height_max`` is used, so slopes are not quantised to the bin grid.
        Only where a separate higher layer exists is the surface capped at the
        lowest layer's top edge, which drops a suspended bar/rail back down to
        the passable ground beneath it.
        """
        if not self.config.vertical_layering:
            return self._height_max
        cap = self._lowest_layer_cap()
        return np.minimum(self._height_max, cap)

    def _hard_obstacle_top(self) -> np.ndarray:
        """Per-cell top height used for grounded hard-obstacle classification.

        With vertical layering enabled, suspended upper layers should not turn
        passable ground below them into a hard wall. The top used by the solid
        obstacle rule is therefore capped to the lowest contiguous layer, while
        real solid structures remain tall because they have no separating gap.
        """
        if not self.config.vertical_layering:
            return self._height_max
        cap = self._lowest_layer_cap()
        return np.minimum(self._height_max, cap)

    def _traversability_layer(self) -> np.ndarray:
        """Return the height layer used for planning, applying no-go cells.

        Unobserved cells (and the -inf sentinel) become NaN here; callers fill
        them as needed.
        """
        heights = np.where(self.observed_mask(), self._walking_surface(), np.nan)
        no_go = self.no_go_mask()
        if np.any(no_go):
            # Mark no-go cells with a large height so edge/node validity fails.
            heights[no_go] = self.config.ceiling_clip + 1.0
        return heights

    @staticmethod
    def _dilate8(mask: np.ndarray) -> np.ndarray:
        """8-connected binary dilation implemented with array shifts."""
        out = mask.copy()
        out[:-1, :] |= mask[1:, :]
        out[1:, :] |= mask[:-1, :]
        out[:, :-1] |= mask[:, 1:]
        out[:, 1:] |= mask[:, :-1]
        out[:-1, :-1] |= mask[1:, 1:]
        out[1:, 1:] |= mask[:-1, :-1]
        out[:-1, 1:] |= mask[1:, :-1]
        out[1:, :-1] |= mask[:-1, 1:]
        return out

    def _binary_close(self, mask: np.ndarray) -> np.ndarray:
        """Morphological closing (dilate then erode) with an 8-connected kernel."""
        dilated = self._dilate8(mask)
        # Erosion = complement of the dilation of the complement.
        return ~self._dilate8(~dilated)

    @staticmethod
    def _connected_component(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
        """Boolean mask of cells 8-connected to ``seed`` within ``mask``."""
        rows, cols = mask.shape
        out = np.zeros_like(mask)
        sr, sc = seed
        if not (0 <= sr < rows and 0 <= sc < cols) or not mask[sr, sc]:
            return out
        stack = [(sr, sc)]
        out[sr, sc] = True
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not out[nr, nc]:
                        out[nr, nc] = True
                        stack.append((nr, nc))
        return out

    def _trace_boundary(self, mask: np.ndarray) -> np.ndarray | None:
        """World-XY contour of the outer boundary of a boolean ``mask``.

        Uses Moore-neighbour tracing starting from the first foreground pixel
        found by a left-to-right, top-to-bottom scan.  Returns an ``(N, 2)``
        array of world coordinates (closed loop: first == last), or ``None``
        if the mask is empty.
        """
        rs, cs = np.nonzero(mask)
        if len(rs) == 0:
            return None
        # Start at the leftmost pixel of the topmost row.
        start_r = int(rs[0])
        start_c = int(cs[0])
        # Moore-neighbour directions (8-connected), clockwise from E.
        dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        boundary: list[tuple[int, int]] = []
        r, c = start_r, start_c
        # Find the first boundary edge: scan clockwise from NW until we find a
        # neighbour OUTSIDE the mask (that neighbour is the starting boundary
        # pixel), then the mask pixel (r,c) is the first boundary pixel.
        found = False
        for d in range(8):
            dr, dc = dirs[d]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self._rows and 0 <= nc < self._cols) or not mask[nr, nc]:
                found = True
                break
        if not found:
            # Mask fills entire grid — trace the grid border.
            boundary = [(0, 0), (self._cols - 1, 0),
                        (self._cols - 1, self._rows - 1), (0, self._rows - 1), (0, 0)]
        else:
            # Standard Moore boundary trace.
            cur_r, cur_c = r, c
            start_dir = d
            cur_dir = start_dir
            for _ in range(4 * (self._rows + self._cols)):  # safety bound
                boundary.append((cur_c, cur_r))
                # Search for next boundary pixel: start scanning from the direction
                # that is 90° clockwise from the current entry direction.
                search = (cur_dir + 6) % 8  # two steps clockwise from current
                for k in range(8):
                    sd = (search + k) % 8
                    dr, dc = dirs[sd]
                    nr, nc = cur_r + dr, cur_c + dc
                    if 0 <= nr < self._rows and 0 <= nc < self._cols and mask[nr, nc]:
                        cur_r, cur_c = nr, nc
                        cur_dir = sd
                        break
                if (cur_c, cur_r) == (start_c, start_r):
                    break
            boundary.append((start_c, start_r))  # close the loop

        res = self.config.resolution
        pts = np.array([[self._origin[0] + (c + 0.5) * res,
                         self._origin[1] + (r + 0.5) * res]
                        for c, r in boundary], dtype=float)
        return pts

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Boolean mask of the largest 8-connected component in ``mask``."""
        best = np.zeros_like(mask)
        best_size = 0
        seen = np.zeros_like(mask)
        for seed in zip(*np.nonzero(mask)):
            if seen[seed]:
                continue
            comp = self._connected_component(mask, seed)
            seen |= comp
            size = int(comp.sum())
            if size > best_size:
                best_size = size
                best = comp
        return best

    def _init_origin(self, robot_xy: np.ndarray) -> None:
        """Anchor the grid origin on the first observation.

        The initial window is centred on the first robot pose (size
        ``window_size``); afterwards the origin only moves when the grid grows
        on the low side (see :meth:`_ensure_bounds`).
        """
        if self._origin_initialised:
            return
        half = self.config.window_size / 2.0
        desired_origin = robot_xy - half
        # Snap origin to the resolution grid for stable indexing.
        self._origin = np.floor(desired_origin / self.config.resolution) * self.config.resolution
        self._origin_initialised = True

    def _ensure_bounds(self, xy: np.ndarray) -> None:
        """Grow the grid so all points in ``xy`` fall inside it.

        The grid is padded in whole-cell steps on whichever sides are needed.
        Padding on the low (origin) side shifts the origin down and offsets all
        pending dirty indices so they stay consistent.
        """
        res = self.config.resolution
        col = np.floor((xy[:, 0] - self._origin[0]) / res).astype(np.int64)
        row = np.floor((xy[:, 1] - self._origin[1]) / res).astype(np.int64)

        min_row, max_row = int(row.min()), int(row.max())
        min_col, max_col = int(col.min()), int(col.max())

        pad_low_row = max(0, -min_row)
        pad_low_col = max(0, -min_col)
        pad_high_row = max(0, max_row - (self._rows - 1))
        pad_high_col = max(0, max_col - (self._cols - 1))

        if not (pad_low_row or pad_low_col or pad_high_row or pad_high_col):
            return

        new_rows = self._rows + pad_low_row + pad_high_row
        new_cols = self._cols + pad_low_col + pad_high_col

        def grown(arr: np.ndarray, fill: float) -> np.ndarray:
            out = np.full((new_rows, new_cols), fill, dtype=arr.dtype)
            out[
                pad_low_row : pad_low_row + self._rows,
                pad_low_col : pad_low_col + self._cols,
            ] = arr
            return out

        self._height_max = grown(self._height_max, -np.inf)
        self._height_min = grown(self._height_min, np.inf)
        self._count = grown(self._count, 0)
        self._frames_seen = grown(self._frames_seen, 0)
        self._virtual_hard = grown(self._virtual_hard, False)
        self._virtual_hard_base = grown(self._virtual_hard_base, np.nan)
        self._virtual_hard_top = grown(self._virtual_hard_top, np.nan)

        if self.config.vertical_layering:
            new_hist = np.zeros((new_rows, new_cols, self._n_bins), dtype=self._zhist.dtype)
            new_hist[
                pad_low_row : pad_low_row + self._rows,
                pad_low_col : pad_low_col + self._cols,
            ] = self._zhist
            self._zhist = new_hist

        self._rows = new_rows
        self._cols = new_cols

        # Origin moves down by the low-side padding.
        self._origin = self._origin - np.array(
            [pad_low_col * res, pad_low_row * res], dtype=float
        )

        # Offset pending dirty indices to match the shifted grid.
        if (pad_low_row or pad_low_col) and self._dirty:
            self._dirty = {
                (r + pad_low_row, c + pad_low_col) for (r, c) in self._dirty
            }
