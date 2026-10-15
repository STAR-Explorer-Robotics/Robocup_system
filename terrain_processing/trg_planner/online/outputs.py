"""Persist the final elevation map and TRG graph as reusable artefacts.

Writes, into an output directory:

- ``elevation.npz`` — heights, observed mask, resolution, origin (reloadable).
- ``elevation_mesh.ply`` — a coloured 3D surface mesh of the observed terrain.
- ``trg_graph.json`` — node positions/states and edges with blocked flags.
- ``summary_topdown.png`` — a top-down heightmap with the final path overlaid.

The mesh and PNG writers are pure-numpy / matplotlib so no heavy 3D dependency
is required to produce the artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..elevation_map import ElevationMap


# Same turbo control points as the Open3D visualizer, kept local to avoid a
# circular import.
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
    values = np.clip(values, 0.0, 1.0)
    scaled = values * (len(_TURBO) - 1)
    lo = np.floor(scaled).astype(int)
    hi = np.minimum(lo + 1, len(_TURBO) - 1)
    frac = (scaled - lo)[:, None]
    return _TURBO[lo] * (1.0 - frac) + _TURBO[hi] * frac


def save_elevation_npz(
    path: str | Path, elevation_map: ElevationMap, observed_mask: np.ndarray
) -> None:
    """Save the raw elevation grid so it can be reloaded later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        heights=np.asarray(elevation_map.heights, dtype=float),
        observed=np.asarray(observed_mask, dtype=bool),
        resolution=float(elevation_map.resolution),
        origin=np.asarray(elevation_map.origin, dtype=float),
    )


def save_elevation_mesh(
    path: str | Path,
    elevation_map: ElevationMap,
    observed_mask: np.ndarray,
    *,
    z_scale: float = 1.0,
) -> int:
    """Write a coloured surface mesh (.ply ASCII) of the observed terrain.

    A quad is emitted for each 2x2 block of mutually-observed cells, split into
    two triangles. Returns the number of vertices written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    heights = np.asarray(elevation_map.heights, dtype=float)
    observed = np.asarray(observed_mask, dtype=bool)
    rows, cols = heights.shape
    res = elevation_map.resolution
    ox, oy = elevation_map.origin

    # Vertex per observed cell; map (r, c) -> vertex index.
    vert_index = -np.ones((rows, cols), dtype=int)
    verts: list[tuple[float, float, float]] = []
    hvals: list[float] = []
    for r in range(rows):
        for c in range(cols):
            if not observed[r, c] or not np.isfinite(heights[r, c]):
                continue
            x = ox + (c + 0.5) * res
            y = oy + (r + 0.5) * res
            z = heights[r, c] * z_scale
            vert_index[r, c] = len(verts)
            verts.append((x, y, z))
            hvals.append(heights[r, c])

    if not verts:
        # Still write a valid (empty) PLY so downstream tooling does not break.
        _write_ply_ascii(path, np.empty((0, 3)), np.empty((0, 3)), [])
        return 0

    hvals_arr = np.array(hvals, dtype=float)
    hmin, hmax = float(hvals_arr.min()), float(hvals_arr.max())
    norm = (hvals_arr - hmin) / (hmax - hmin) if hmax > hmin else np.zeros_like(hvals_arr)
    colors = _colormap(norm)

    faces: list[tuple[int, int, int]] = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            a = vert_index[r, c]
            b = vert_index[r, c + 1]
            d = vert_index[r + 1, c]
            e = vert_index[r + 1, c + 1]
            if a >= 0 and b >= 0 and d >= 0:
                faces.append((a, d, b))
            if b >= 0 and d >= 0 and e >= 0:
                faces.append((b, d, e))

    _write_ply_ascii(path, np.array(verts, dtype=float), colors, faces)
    return len(verts)


def _write_ply_ascii(path, verts: np.ndarray, colors: np.ndarray, faces: list) -> None:
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(verts)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write(f"element face {len(faces)}\n")
        fh.write("property list uchar int vertex_indices\n")
        fh.write("end_header\n")
        for i in range(len(verts)):
            x, y, z = verts[i]
            r, g, b = (np.clip(colors[i], 0, 1) * 255).astype(int) if len(colors) else (200, 200, 200)
            fh.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
        for f in faces:
            fh.write(f"3 {f[0]} {f[1]} {f[2]}\n")


def save_graph_json(path: str | Path, graph) -> None:
    """Serialise the TRG graph (nodes + edges with blocked flags) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = []
    for nid in graph.nodes:
        node = graph.nodes[nid]
        nodes.append(
            {
                "id": int(nid),
                "position": [float(v) for v in node.position],
                "stable": bool(getattr(node, "stable", True)),
                "state": getattr(node, "state", "valid"),
            }
        )
    edges = []
    for eid in graph.edges:
        edge = graph.edges[eid]
        edges.append(
            {
                "id": int(eid),
                "src": int(edge.src),
                "dst": int(edge.dst),
                "distance": float(edge.distance),
                "weight": float(edge.weight),
                "blocked": bool(getattr(edge, "blocked", False)),
            }
        )
    with open(path, "w") as fh:
        json.dump({"nodes": nodes, "edges": edges}, fh, indent=2)


def save_topdown_png(
    path: str | Path,
    elevation_map: ElevationMap,
    observed_mask: np.ndarray,
    *,
    robot_xy: tuple[float, float] | None = None,
    goal_xy: tuple[float, float] | None = None,
    path_xyz: np.ndarray | None = None,
    graph=None,
    field_extent=None,
    trav_boundary=None,
    manual_boundary=None,
) -> None:
    """Render a top-down heightmap with graph/extent overlay to ``path``."""
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(path).parent / ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    heights = np.where(observed_mask, elevation_map.heights, np.nan)
    xmin, xmax, ymin, ymax = elevation_map.extent

    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(
        heights, origin="lower", cmap="turbo", aspect="equal",
        extent=[xmin, xmax, ymin, ymax],
    )
    plt.colorbar(im, ax=ax, label="Height above ground (m)", shrink=0.85)

    if graph is not None and getattr(graph, "edges", None):
        for eid in graph.edges:
            edge = graph.edges[eid]
            if edge.src not in graph.nodes or edge.dst not in graph.nodes:
                continue
            p = graph.nodes[edge.src].position
            q = graph.nodes[edge.dst].position
            color = "red" if getattr(edge, "blocked", False) else (0.4, 0.6, 0.9, 0.4)
            ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=color, linewidth=0.6, zorder=7)

    if field_extent is not None:
        pts_xy = np.asarray(getattr(field_extent, "points_xy", np.empty((0, 2))), dtype=float)
        if pts_xy.size:
            ax.plot(pts_xy[:, 0], pts_xy[:, 1], ".", color="limegreen", markersize=0.8,
                    alpha=0.5, label="fence points", zorder=3)
        poly = np.asarray(getattr(field_extent, "polygon", np.empty((0, 2))), dtype=float)
        if len(poly) >= 3:
            loop = np.vstack([poly, poly[:1]])
            ax.plot(loop[:, 0], loop[:, 1], "-", color="cyan", linewidth=2.0,
                    label="field extent", zorder=4)

    if trav_boundary is not None:
        tb = np.asarray(trav_boundary, dtype=float)
        if len(tb) >= 3:
            ax.plot(tb[:, 0], tb[:, 1], "-", color="magenta", linewidth=1.5,
                    label="reachable boundary", zorder=4)

    if manual_boundary is not None:
        mb = np.asarray(getattr(manual_boundary, "polygon", np.empty((0, 2))), dtype=float)
        if len(mb) >= 3:
            loop = np.vstack([mb, mb[:1]])
            ax.plot(loop[:, 0], loop[:, 1], "--", color="orange", linewidth=2.0,
                    label="manual boundary", zorder=7)

    if path_xyz is not None and len(path_xyz) >= 2:
        ax.plot(path_xyz[:, 0], path_xyz[:, 1], "-", color="red", linewidth=2.5,
                label="TRG path", zorder=4)
    if robot_xy is not None:
        ax.plot(robot_xy[0], robot_xy[1], "o", color="dodgerblue", markersize=12,
                label="robot", zorder=5)
    if goal_xy is not None:
        ax.plot(goal_xy[0], goal_xy[1], "*", color="lime", markersize=18,
                label="goal", zorder=5)

    ax.legend(loc="upper right", fontsize=8, markerscale=0.8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("TRG Online Mapping")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_boundary_json(path: str | Path, boundary) -> None:
    """Serialise the detected field boundary (corners + wall cells) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    corners = getattr(boundary, "corners", None)
    payload = {
        "angle": float(getattr(boundary, "angle", 0.0)),
        "corners": (np.asarray(corners, dtype=float).tolist() if corners is not None else None),
        "wall_cell_count": int(len(getattr(boundary, "wall_xy", []))),
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def save_field_extent_json(path: str | Path, field_extent) -> None:
    """Serialise the detected field extent (polygon + area) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    poly = getattr(field_extent, "polygon", None)
    payload = {
        "area": float(getattr(field_extent, "area", 0.0)),
        "polygon": (np.asarray(poly, dtype=float).tolist() if poly is not None else None),
        "vertex_count": int(len(poly)) if poly is not None else 0,
        "is_valid": bool(field_extent.is_valid()) if hasattr(field_extent, "is_valid") else False,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)

