from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Node:
    node_id: int
    position: np.ndarray
    stable: bool = True
    reachable: bool = False
    state: str = "invalid"
    incident_edges: set[int] = field(default_factory=set)
    # Consecutive frames this node failed its stability check. Used by the
    # planner's validation hysteresis so a single noisy frame does not
    # invalidate a node (see TRGPlanner.validate_path).
    invalid_strikes: int = 0

    @property
    def degree(self) -> int:
        return len(self.incident_edges)


@dataclass(slots=True)
class Edge:
    edge_id: int
    src: int
    dst: int
    distance: float
    weight: float
    blocked: bool = False
    # Consecutive frames this edge failed its traversability check. Used by the
    # planner's validation hysteresis so a single noisy frame does not block an
    # edge (see TRGPlanner.validate_path).
    block_strikes: int = 0

    def other(self, node_id: int) -> int:
        if node_id == self.src:
            return self.dst
        if node_id == self.dst:
            return self.src
        raise ValueError(f"node {node_id} is not incident to edge {self.edge_id}")


class TraversalRiskGraph:
    def __init__(self, spatial_bin_size: float = 1.0) -> None:
        self.nodes: dict[int, Node] = {}
        self.edges: dict[int, Edge] = {}
        self._adjacency: dict[int, dict[int, int]] = defaultdict(dict)
        self._edge_lookup: dict[tuple[int, int], int] = {}
        self._next_node_id = 0
        self._next_edge_id = 0

        # Uniform grid spatial hash: cell (ix, iy) -> set of node ids whose XY
        # falls in that cell. Replaces O(n) scans in radius / nearest queries.
        if spatial_bin_size <= 0.0:
            raise ValueError("spatial_bin_size must be positive")
        self._bin_size = float(spatial_bin_size)
        self._spatial: dict[tuple[int, int], set[int]] = defaultdict(set)

    def _cell_of(self, xy: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        return (
            int(np.floor(xy[0] / self._bin_size)),
            int(np.floor(xy[1] / self._bin_size)),
        )

    def add_node(self, position: np.ndarray, stable: bool = True) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        position = np.asarray(position, dtype=float)
        self.nodes[node_id] = Node(
            node_id=node_id,
            position=position,
            stable=stable,
        )
        self._spatial[self._cell_of(position[:2])].add(node_id)
        return node_id

    def connect(self, src: int, dst: int, distance: float, weight: float) -> int | None:
        if src == dst:
            return None

        key = tuple(sorted((src, dst)))
        if key in self._edge_lookup:
            edge_id = self._edge_lookup[key]
            edge = self.edges[edge_id]
            edge.distance = min(edge.distance, distance)
            edge.weight = min(edge.weight, weight)
            return edge_id

        edge_id = self._next_edge_id
        self._next_edge_id += 1
        edge = Edge(
            edge_id=edge_id,
            src=src,
            dst=dst,
            distance=float(distance),
            weight=float(weight),
        )
        self.edges[edge_id] = edge
        self._edge_lookup[key] = edge_id
        self._adjacency[src][dst] = edge_id
        self._adjacency[dst][src] = edge_id
        self.nodes[src].incident_edges.add(edge_id)
        self.nodes[dst].incident_edges.add(edge_id)
        return edge_id

    def remove_node(self, node_id: int) -> bool:
        """Remove a node and all its incident edges.

        Returns ``True`` if the node existed, ``False`` otherwise.
        """
        node = self.nodes.pop(node_id, None)
        if node is None:
            return False
        # Remove from spatial index.
        self._spatial[self._cell_of(node.position[:2])].discard(node_id)
        # Remove all incident edges.
        for edge_id in list(node.incident_edges):
            edge = self.edges.pop(edge_id, None)
            if edge is None:
                continue
            key = tuple(sorted((edge.src, edge.dst)))
            self._edge_lookup.pop(key, None)
            self._adjacency.get(edge.src, {}).pop(edge.dst, None)
            self._adjacency.get(edge.dst, {}).pop(edge.src, None)
            other = edge.dst if edge.src == node_id else edge.src
            other_node = self.nodes.get(other)
            if other_node is not None:
                other_node.incident_edges.discard(edge_id)
        return True

    def neighbors(self, node_id: int) -> list[tuple[int, Edge]]:
        return [
            (neighbor_id, self.edges[edge_id])
            for neighbor_id, edge_id in self._adjacency.get(node_id, {}).items()
        ]

    def node_ids_within_radius(
        self,
        xy: tuple[float, float] | np.ndarray,
        radius: float,
        exclude: set[int] | None = None,
    ) -> list[int]:
        xy = np.asarray(xy, dtype=float)
        exclude = exclude or set()
        ids: list[int] = []
        # Only inspect cells overlapping the query's bounding box.
        cx0 = int(np.floor((xy[0] - radius) / self._bin_size))
        cx1 = int(np.floor((xy[0] + radius) / self._bin_size))
        cy0 = int(np.floor((xy[1] - radius) / self._bin_size))
        cy1 = int(np.floor((xy[1] + radius) / self._bin_size))
        radius_sq = radius * radius
        for ix in range(cx0, cx1 + 1):
            for iy in range(cy0, cy1 + 1):
                bucket = self._spatial.get((ix, iy))
                if not bucket:
                    continue
                for node_id in bucket:
                    if node_id in exclude:
                        continue
                    delta = self.nodes[node_id].position[:2] - xy
                    if float(delta @ delta) <= radius_sq:
                        ids.append(node_id)
        return ids

    def nearest_node(
        self,
        xy: tuple[float, float] | np.ndarray,
        valid_only: bool = False,
    ) -> tuple[int, float] | None:
        xy = np.asarray(xy, dtype=float)
        if not self.nodes:
            return None

        cx, cy = self._cell_of(xy)
        best_id: int | None = None
        best_distance = float("inf")

        # Expand in square rings around the query cell. A node in a cell that is
        # `ring` cells away (Chebyshev) is at least (ring - 1) * bin_size from
        # xy, so we can stop once that lower bound exceeds the best distance.
        max_ring = self._max_ring_extent(cx, cy)
        for ring in range(0, max_ring + 1):
            if best_id is not None and (ring - 1) * self._bin_size > best_distance:
                break
            for ix, iy in self._ring_cells(cx, cy, ring):
                bucket = self._spatial.get((ix, iy))
                if not bucket:
                    continue
                for node_id in bucket:
                    node = self.nodes[node_id]
                    if valid_only and node.state != "valid":
                        continue
                    distance = float(np.linalg.norm(node.position[:2] - xy))
                    if distance < best_distance:
                        best_id = node_id
                        best_distance = distance
        if best_id is None:
            return None
        return best_id, best_distance

    def _max_ring_extent(self, cx: int, cy: int) -> int:
        """Upper bound on rings to search: covers all currently occupied cells."""
        if not self._spatial:
            return 0
        max_d = 0
        for (ix, iy) in self._spatial:
            d = max(abs(ix - cx), abs(iy - cy))
            if d > max_d:
                max_d = d
        return max_d

    @staticmethod
    def _ring_cells(cx: int, cy: int, ring: int):
        """Yield the (ix, iy) cells on the square ring at Chebyshev distance."""
        if ring == 0:
            yield (cx, cy)
            return
        for ix in range(cx - ring, cx + ring + 1):
            yield (ix, cy - ring)
            yield (ix, cy + ring)
        for iy in range(cy - ring + 1, cy + ring):
            yield (cx - ring, iy)
            yield (cx + ring, iy)
