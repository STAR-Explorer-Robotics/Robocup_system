"""Open3D 3D visualization of the elevation map, robot pose and planned path.

Provides geometry builders and two entry points:

- :func:`show_static` for a one-shot view (offline analysis).
- :class:`ElevationVisualizer` for a live-updating window (near-real-time).

Open3D is imported lazily so importing this module never forces the heavy
dependency unless visualization is actually requested.
"""

from __future__ import annotations

import numpy as np

from ..elevation_map import ElevationMap


# Turbo-like colormap control points (low -> high), avoids a matplotlib dependency.
_TURBO = np.array(
    [
        [0.19, 0.07, 0.23],
        [0.27, 0.31, 0.86],
        [0.10, 0.72, 0.85],
        [0.40, 0.91, 0.34],
        [0.93, 0.83, 0.18],
        [0.96, 0.42, 0.12],
        [0.69, 0.01, 0.01],
    ],
    dtype=float,
)


def _colormap(values: np.ndarray) -> np.ndarray:
    """Map normalised values in [0, 1] to RGB using the turbo control points."""
    values = np.clip(values, 0.0, 1.0)
    scaled = values * (len(_TURBO) - 1)
    lo = np.floor(scaled).astype(int)
    hi = np.minimum(lo + 1, len(_TURBO) - 1)
    frac = (scaled - lo)[:, None]
    return _TURBO[lo] * (1.0 - frac) + _TURBO[hi] * frac


def elevation_mesh(
    elevation_map: ElevationMap,
    observed_mask: np.ndarray | None = None,
):
    """Build an Open3D TriangleMesh coloured by height (solid surface, no gaps)."""
    import open3d as o3d  # noqa: PLC0415

    heights = np.asarray(elevation_map.heights, dtype=float)
    rows, cols = heights.shape
    res = elevation_map.resolution
    ox, oy = elevation_map.origin

    valid = np.isfinite(heights)
    if observed_mask is not None:
        valid &= observed_mask
    if not np.any(valid):
        return o3d.geometry.TriangleMesh()

    # vertices at grid corners
    xs = ox + np.arange(cols + 1) * res
    ys = oy + np.arange(rows + 1) * res
    vid = np.full((rows + 1, cols + 1), -1, dtype=int)
    verts: list[tuple[float, float, float]] = []
    # Place a vertex at each corner where at least one adjacent cell is observed.
    for r in range(rows + 1):
        for c in range(cols + 1):
            neigh = []
            if r > 0 and c > 0 and valid[r - 1, c - 1]:
                neigh.append(heights[r - 1, c - 1])
            if r > 0 and c < cols and valid[r - 1, c]:
                neigh.append(heights[r - 1, c])
            if r < rows and c > 0 and valid[r, c - 1]:
                neigh.append(heights[r, c - 1])
            if r < rows and c < cols and valid[r, c]:
                neigh.append(heights[r, c])
            if neigh:
                vid[r, c] = len(verts)
                verts.append((xs[c], ys[r], float(np.mean(neigh))))

    tris: list[tuple[int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            a, b = vid[r, c], vid[r, c + 1]
            d, e = vid[r + 1, c], vid[r + 1, c + 1]
            if -1 in (a, b, d, e):
                continue
            tris.append((a, b, e))
            tris.append((a, e, d))

    if not tris:
        return o3d.geometry.TriangleMesh()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(verts, dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tris, dtype=int))

    # colour by vertex height
    vz = np.array([v[2] for v in verts], dtype=float)
    vmin, vmax = float(vz.min()), float(vz.max())
    norm = (vz - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vz)
    mesh.vertex_colors = o3d.utility.Vector3dVector(_colormap(norm))

    mesh.compute_vertex_normals()
    return mesh


def robot_pose_frame(robot_xy: tuple[float, float], z: float, size: float = 0.8):
    """A coordinate frame placed at the robot position."""
    import open3d as o3d  # noqa: PLC0415

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.translate((float(robot_xy[0]), float(robot_xy[1]), float(z)))
    return frame


def path_line_set(path_xyz: np.ndarray, color=(1.0, 0.0, 0.0)):
    """A LineSet connecting the planned path waypoints."""
    import open3d as o3d  # noqa: PLC0415

    path_xyz = np.asarray(path_xyz, dtype=float)
    if path_xyz.ndim != 2 or path_xyz.shape[0] < 2:
        return o3d.geometry.LineSet()

    lines = [[i, i + 1] for i in range(len(path_xyz) - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(path_xyz)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


def marker_sphere(xyz: tuple[float, float, float], radius: float, color):
    """A small sphere used to mark start/goal positions."""
    import open3d as o3d  # noqa: PLC0415

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate((float(xyz[0]), float(xyz[1]), float(xyz[2])))
    sphere.paint_uniform_color(list(color))
    sphere.compute_vertex_normals()
    return sphere


def _graph_node_positions(graph) -> np.ndarray:
    """(N, 3) array of graph node positions, empty when the graph has none."""
    if not getattr(graph, "nodes", None):
        return np.empty((0, 3), dtype=float)
    return np.array(
        [graph.nodes[nid].position for nid in graph.nodes], dtype=float
    )


def trg_node_cloud(graph, z_offset: float = 0.05):
    """A point cloud of TRG graph nodes coloured by validity state."""
    import open3d as o3d  # noqa: PLC0415

    cloud = o3d.geometry.PointCloud()
    if not getattr(graph, "nodes", None):
        return cloud
    pts = []
    colors = []
    for nid in graph.nodes:
        node = graph.nodes[nid]
        p = np.asarray(node.position, dtype=float).copy()
        p[2] += z_offset
        pts.append(p)
        # green = valid/stable, grey = invalid
        if getattr(node, "state", "valid") == "invalid" or not getattr(node, "stable", True):
            colors.append([0.5, 0.5, 0.5])
        else:
            colors.append([0.1, 0.9, 0.3])
    cloud.points = o3d.utility.Vector3dVector(np.array(pts, dtype=float))
    cloud.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=float))
    return cloud


def trg_edge_line_sets(graph, z_offset: float = 0.05):
    """Two LineSets: (open edges in light blue, blocked edges in red)."""
    import open3d as o3d  # noqa: PLC0415

    open_ls = o3d.geometry.LineSet()
    blocked_ls = o3d.geometry.LineSet()
    if not getattr(graph, "edges", None) or not getattr(graph, "nodes", None):
        return open_ls, blocked_ls

    index_of = {nid: i for i, nid in enumerate(graph.nodes)}
    pts = _graph_node_positions(graph).copy()
    if pts.size == 0:
        return open_ls, blocked_ls
    pts[:, 2] += z_offset

    open_lines, blocked_lines = [], []
    for eid in graph.edges:
        edge = graph.edges[eid]
        if edge.src not in index_of or edge.dst not in index_of:
            continue
        pair = [index_of[edge.src], index_of[edge.dst]]
        if getattr(edge, "blocked", False):
            blocked_lines.append(pair)
        else:
            open_lines.append(pair)

    vec = o3d.utility.Vector3dVector(pts)
    if open_lines:
        open_ls.points = vec
        open_ls.lines = o3d.utility.Vector2iVector(open_lines)
        open_ls.colors = o3d.utility.Vector3dVector(
            [[0.4, 0.6, 0.9]] * len(open_lines)
        )
    if blocked_lines:
        blocked_ls.points = o3d.utility.Vector3dVector(pts)
        blocked_ls.lines = o3d.utility.Vector2iVector(blocked_lines)
        blocked_ls.colors = o3d.utility.Vector3dVector(
            [[0.95, 0.1, 0.1]] * len(blocked_lines)
        )
    return open_ls, blocked_ls


def field_boundary_geometries(boundary, elevation_map, z_offset: float = 0.3):
    """Open3D geometries for the detected field wall + fitted boundary.

    Renders the raw wall cells (dark red points) so detection quality is
    visible, plus the fitted rectilinear boundary (bright yellow loop).
    """
    import open3d as o3d  # noqa: PLC0415

    geoms = []

    wall_xy = np.asarray(getattr(boundary, "wall_xy", np.empty((0, 2))), dtype=float)
    if wall_xy.size:
        z = elevation_map.height_at((0.0, 0.0))
        base_z = z if np.isfinite(z) else 0.0
        zs = np.full((len(wall_xy), 1), base_z + z_offset)
        wall_pts = np.hstack([wall_xy, zs])
        wall_cloud = o3d.geometry.PointCloud()
        wall_cloud.points = o3d.utility.Vector3dVector(wall_pts)
        wall_cloud.colors = o3d.utility.Vector3dVector(
            np.tile([0.6, 0.0, 0.0], (len(wall_pts), 1))
        )
        geoms.append(wall_cloud)

    corners = getattr(boundary, "corners", None)
    if corners is not None and len(corners) >= 3:
        corners = np.asarray(corners, dtype=float)
        loop = []
        for cx, cy in corners:
            cz = elevation_map.height_at((cx, cy))
            loop.append([cx, cy, (cz if np.isfinite(cz) else 0.0) + z_offset + 0.5])
        loop_pts = np.array(loop, dtype=float)
        n = len(loop_pts)
        lines = [[i, (i + 1) % n] for i in range(n)]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(loop_pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[1.0, 0.9, 0.0]] * len(lines))
        geoms.append(ls)

    return geoms


def field_extent_geometries(field_extent, elevation_map, z_offset: float = 0.3):
    """Open3D geometries for the concave-hull field extent.

    Renders the short-structure (fence) points as light-green dots so detection
    quality is visible, plus the extracted extent polygon as a bright-cyan loop.
    """
    import open3d as o3d  # noqa: PLC0415

    geoms = []
    z = elevation_map.height_at((0.0, 0.0))
    base_z = z if np.isfinite(z) else 0.0

    pts_xy = np.asarray(getattr(field_extent, "points_xy", np.empty((0, 2))), dtype=float)
    if pts_xy.size:
        zs = np.full((len(pts_xy), 1), base_z + z_offset)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.hstack([pts_xy, zs]))
        pc.colors = o3d.utility.Vector3dVector(
            np.tile([0.3, 0.9, 0.3], (len(pts_xy), 1))
        )
        geoms.append(pc)

    poly = np.asarray(getattr(field_extent, "polygon", np.empty((0, 2))), dtype=float)
    if len(poly) >= 3:
        loop = []
        for cx, cy in poly:
            cz = elevation_map.height_at((cx, cy))
            loop.append([cx, cy, (cz if np.isfinite(cz) else base_z) + z_offset + 0.4])
        loop_pts = np.array(loop, dtype=float)
        n = len(loop_pts)
        lines = [[i, (i + 1) % n] for i in range(n)]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(loop_pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[0.0, 0.9, 1.0]] * len(lines))
        geoms.append(ls)

    return geoms


def traversable_boundary_geometries(trav_contour, elevation_map, z_offset: float = 0.3):
    """Open3D LineSet for the traversable-region boundary (magenta loop)."""
    import open3d as o3d  # noqa: PLC0415

    contour = np.asarray(trav_contour, dtype=float)
    if len(contour) < 3:
        return []
    z = elevation_map.height_at((contour[0, 0], contour[0, 1]))
    base_z = z if np.isfinite(z) else 0.0
    pts = np.hstack([contour[:, :2], np.full((len(contour), 1), base_z + z_offset + 0.2)])
    n = len(pts)
    lines = [[i, (i + 1) % n] for i in range(n)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 1.0]] * len(lines))
    return [ls]


def manual_boundary_geometries(manual_extent, elevation_map, z_offset: float = 0.3):
    """Open3D LineSet for the manual boundary polygon (orange dashed loop)."""
    import open3d as o3d  # noqa: PLC0415

    poly = np.asarray(getattr(manual_extent, "polygon", np.empty((0, 2))), dtype=float)
    if len(poly) < 3:
        return []
    z = elevation_map.height_at((poly[0, 0], poly[0, 1]))
    base_z = z if np.isfinite(z) else 0.0
    pts = np.hstack([poly, np.full((len(poly), 1), base_z + z_offset + 0.6)])
    n = len(pts)
    lines = [[i, (i + 1) % n] for i in range(n)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[1.0, 0.6, 0.0]] * len(lines))
    return [ls]


def build_scene(
    elevation_map: ElevationMap,
    robot_xy: tuple[float, float] | None = None,
    path_xyz: np.ndarray | None = None,
    goal_xy: tuple[float, float] | None = None,
    graph=None,
    field_extent=None,
    trav_boundary=None,
    manual_boundary=None,
):
    """Assemble the list of Open3D geometries for a scene."""
    geometries = [elevation_mesh(elevation_map)]

    if graph is not None:
        open_ls, blocked_ls = trg_edge_line_sets(graph)
        geometries.append(open_ls)
        geometries.append(blocked_ls)
        geometries.append(trg_node_cloud(graph))

    if field_extent is not None:
        geometries.extend(field_extent_geometries(field_extent, elevation_map))

    if trav_boundary is not None:
        geometries.extend(traversable_boundary_geometries(trav_boundary, elevation_map))

    if manual_boundary is not None:
        geometries.extend(manual_boundary_geometries(manual_boundary, elevation_map))

    if robot_xy is not None:
        z = elevation_map.height_at(robot_xy)
        if not np.isfinite(z):
            z = 0.0
        geometries.append(robot_pose_frame(robot_xy, z))
        geometries.append(marker_sphere((robot_xy[0], robot_xy[1], z + 0.2), 0.2, (0.0, 0.4, 1.0)))

    if path_xyz is not None and len(path_xyz) >= 2:
        geometries.append(path_line_set(path_xyz))

    if goal_xy is not None:
        z = elevation_map.height_at(goal_xy)
        if not np.isfinite(z):
            z = 0.0
        geometries.append(marker_sphere((goal_xy[0], goal_xy[1], z + 0.2), 0.25, (0.0, 1.0, 0.2)))

    return geometries


def show_static(
    elevation_map: ElevationMap,
    robot_xy: tuple[float, float] | None = None,
    path_xyz: np.ndarray | None = None,
    goal_xy: tuple[float, float] | None = None,
    graph=None,
    field_extent=None,
    trav_boundary=None,
    manual_boundary=None,
    window_name: str = "TRG Online — Elevation / Pose / Path",
) -> None:
    """Open a one-shot Open3D window with the assembled scene."""
    import open3d as o3d  # noqa: PLC0415

    geometries = build_scene(
        elevation_map, robot_xy, path_xyz, goal_xy, graph,
        field_extent, trav_boundary, manual_boundary,
    )
    o3d.visualization.draw_geometries(geometries, window_name=window_name)


class ElevationVisualizer:
    """Live-updating Open3D window for near-real-time visualization."""

    def __init__(self, window_name: str = "TRG Online") -> None:
        import open3d as o3d  # noqa: PLC0415

        self._o3d = o3d
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name=window_name)
        self._geoms: list = []
        self._first = True
        self._mesh_cell_count = 0  # cached mesh's observed cell count

    def update(
        self,
        elevation_map: ElevationMap,
        robot_xy: tuple[float, float] | None = None,
        path_xyz: np.ndarray | None = None,
        goal_xy: tuple[float, float] | None = None,
        graph=None,
        field_extent=None,
        trav_boundary=None,
        manual_boundary=None,
    ) -> None:
        for geom in self._geoms:
            self._vis.remove_geometry(geom, reset_bounding_box=False)

        # Only rebuild the heavy mesh when the observed region grew >= 10%
        cur_cells = int(np.isfinite(elevation_map.heights).sum())
        if cur_cells == 0 or abs(cur_cells - self._mesh_cell_count) / max(cur_cells, 1) >= 0.10:
            self._cached_mesh = elevation_mesh(elevation_map)
            self._mesh_cell_count = cur_cells

        self._geoms = [self._cached_mesh]
        self._add_nonmesh_geoms(
            elevation_map, robot_xy, path_xyz, goal_xy, graph,
            field_extent, trav_boundary, manual_boundary,
        )
        reset = self._first
        for geom in self._geoms:
            self._vis.add_geometry(geom, reset_bounding_box=reset)
        self._first = False
        self._vis.poll_events()
        self._vis.update_renderer()

    def _add_nonmesh_geoms(
        self, elevation_map, robot_xy, path_xyz, goal_xy, graph,
        field_extent, trav_boundary, manual_boundary,
    ) -> None:
        """Light weight geometries rebuilt every frame (no mesh)."""
        if graph is not None:
            open_ls, blocked_ls = trg_edge_line_sets(graph)
            self._geoms.append(open_ls)
            self._geoms.append(blocked_ls)
            self._geoms.append(trg_node_cloud(graph))
        if field_extent is not None:
            self._geoms.extend(field_extent_geometries(field_extent, elevation_map))
        if trav_boundary is not None:
            self._geoms.extend(traversable_boundary_geometries(trav_boundary, elevation_map))
        if manual_boundary is not None:
            self._geoms.extend(manual_boundary_geometries(manual_boundary, elevation_map))
        if robot_xy is not None:
            z = elevation_map.height_at(robot_xy)
            if not np.isfinite(z):
                z = 0.0
            self._geoms.append(robot_pose_frame(robot_xy, z))
            self._geoms.append(marker_sphere((robot_xy[0], robot_xy[1], z + 0.2), 0.2, (0.0, 0.4, 1.0)))
        if path_xyz is not None and len(path_xyz) >= 2:
            self._geoms.append(path_line_set(path_xyz))
        if goal_xy is not None:
            z = elevation_map.height_at(goal_xy)
            if not np.isfinite(z):
                z = 0.0
            self._geoms.append(marker_sphere((goal_xy[0], goal_xy[1], z + 0.2), 0.25, (0.0, 1.0, 0.2)))

    def capture(self, path: str) -> None:
        """Save the current framebuffer to ``path`` (PNG)."""
        self._vis.capture_screen_image(path, do_render=True)

    def close(self) -> None:
        self._vis.destroy_window()
