from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq

import numpy as np

from .elevation_map import ElevationMap
from .graph import TraversalRiskGraph


@dataclass(slots=True)
class PlannerConfig:
    robot_radius: float = 0.20
    expansion_radius: float = 0.80
    max_height_step: float = 0.30
    gamma: float = 0.2
    safety_factor: float = 3.0
    samples_per_node: int = 20
    max_nodes: int = 2500
    random_seed: int = 7
    min_points_per_region: int = 4
    # Number of consecutive failing frames before an edge is committed as
    # blocked (or a node invalidated) during validate_path. 1 = react
    # immediately; higher values reject transient sensor noise.
    block_hysteresis: int = 1


@dataclass(slots=True)
class PathResult:
    node_ids: list[int]
    path_xyz: np.ndarray
    total_cost: float
    total_distance: float
    mean_edge_risk: float
    goal_reached_exact: bool
    requested_goal_xy: tuple[float, float]
    target_goal_xy: tuple[float, float]


class TRGPlanner:
    def __init__(self, elevation_map: ElevationMap, config: PlannerConfig | None = None) -> None:
        self.map = elevation_map
        self.config = config or PlannerConfig()
        if not 0.0 <= self.config.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.config.safety_factor < 0.0:
            raise ValueError("safety_factor must be non-negative")

        self.graph = TraversalRiskGraph()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._start_node_id: int | None = None
        # Nodes whose ring has already been expanded (persists across
        # incremental ingest calls; reset on full build_graph).
        self._expanded: set[int] = set()
        self._field_extent = None  # optional polygon for gating expansion

    def set_field_extent(self, field_extent) -> None:
        self._field_extent = field_extent

    def build_graph(self, start_xy: tuple[float, float]) -> TraversalRiskGraph:
        self.graph = TraversalRiskGraph()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._expanded = set()

        start_point = self._stable_point(start_xy)
        if start_point is None:
            raise ValueError("start pose is not on a stable terrain region")

        start_node_id = self.graph.add_node(start_point, stable=True)
        self._start_node_id = start_node_id

        queue: deque[int] = deque([start_node_id])
        self._expand_frontier(queue)

        self._update_reachability()
        return self.graph

    def _expand_frontier(self, queue: deque[int]) -> None:
        """Grow the graph by ring-sampling from queued nodes.

        Shared by the one-shot :meth:`build_graph` and the incremental
        :meth:`ingest`. Expansion stops when the queue drains or the node
        budget (:attr:`PlannerConfig.max_nodes`) is reached. ``self._expanded``
        tracks already-expanded nodes so re-queued nodes are not redone.
        """
        while queue and len(self.graph.nodes) < self.config.max_nodes:
            ref_id = queue.popleft()
            if ref_id in self._expanded:
                continue

            ref_node = self.graph.nodes[ref_id]
            for candidate_xy in self._sample_ring(ref_node.position[:2]):
                if self._field_extent is not None and self._field_extent.is_valid():
                    if not self._field_extent.contains(candidate_xy.reshape(1, 2))[0]:
                        continue
                candidate_point = self._stable_point(tuple(candidate_xy))
                if candidate_point is None:
                    continue

                ref_edge = self._evaluate_edge(ref_node.position, candidate_point)
                if ref_edge is None:
                    continue

                merge_target = self._find_merge_target(candidate_point[:2], exclude={ref_id})
                if merge_target is not None:
                    merged_edge = self._evaluate_edge(
                        ref_node.position,
                        self.graph.nodes[merge_target].position,
                    )
                    if merged_edge is not None:
                        self.graph.connect(ref_id, merge_target, *merged_edge)
                    continue

                node_id = self.graph.add_node(candidate_point, stable=True)
                self.graph.connect(ref_id, node_id, *ref_edge)
                queue.append(node_id)
                self._wire_existing_neighbors(node_id)

            self._expanded.add(ref_id)

    def plan(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> PathResult:
        self.build_graph(start_xy)
        if self._start_node_id is None:
            raise RuntimeError("graph was not initialized with a start node")
        return self._plan_to_goal(self._start_node_id, goal_xy)

    def _plan_to_goal(self, start_node_id: int, goal_xy: tuple[float, float]) -> PathResult:
        """Attach the goal, run A*, and assemble a :class:`PathResult`.

        Assumes the graph and reachability are already up to date. Shared by the
        one-shot :meth:`plan` and the incremental :meth:`replan`.
        """
        goal_node_id, goal_reached_exact = self._attach_or_project_goal(goal_xy)
        if goal_node_id is None:
            raise RuntimeError("failed to identify a reachable goal node")

        target_goal_xy = (
            float(self.graph.nodes[goal_node_id].position[0]),
            float(self.graph.nodes[goal_node_id].position[1]),
        )
        node_ids, total_cost = self._a_star(start_node_id, goal_node_id, target_goal_xy)
        path_xyz = np.vstack([self.graph.nodes[node_id].position for node_id in node_ids])

        total_distance = 0.0
        risks: list[float] = []
        for src, dst in zip(node_ids[:-1], node_ids[1:]):
            edge = self.graph.edges[self.graph._adjacency[src][dst]]
            total_distance += edge.distance
            risks.append(edge.weight)

        mean_edge_risk = float(np.mean(risks)) if risks else 0.0
        return PathResult(
            node_ids=node_ids,
            path_xyz=path_xyz,
            total_cost=total_cost,
            total_distance=total_distance,
            mean_edge_risk=mean_edge_risk,
            goal_reached_exact=goal_reached_exact,
            requested_goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            target_goal_xy=target_goal_xy,
        )

    def _sample_ring(self, center_xy: np.ndarray) -> np.ndarray:
        base_offset = self._rng.uniform(0.0, 2.0 * np.pi)
        angles = base_offset + np.linspace(0.0, 2.0 * np.pi, self.config.samples_per_node, endpoint=False)
        offsets = np.column_stack((np.cos(angles), np.sin(angles))) * self.config.expansion_radius
        return center_xy.reshape(1, 2) + offsets

    def _stable_point(self, xy: tuple[float, float]) -> np.ndarray | None:
        if not self.map.in_bounds_xy(xy):
            return None

        disk_points = self.map.disk_points(xy, self.config.robot_radius)
        if len(disk_points) < self.config.min_points_per_region:
            return None

        heights = disk_points[:, 2]
        median_height = float(np.median(heights))
        if np.any(np.abs(heights - median_height) >= self.config.max_height_step):
            return None

        return np.array([float(xy[0]), float(xy[1]), median_height], dtype=float)

    def _evaluate_edge(self, point_a: np.ndarray, point_b: np.ndarray) -> tuple[float, float] | None:
        horizontal_distance = float(np.linalg.norm(point_b[:2] - point_a[:2]))
        if horizontal_distance <= 1e-9:
            return None

        max_slope = self.config.max_height_step / self.config.robot_radius
        edge_slope = abs(point_a[2] - point_b[2]) / horizontal_distance
        if edge_slope >= max_slope:
            return None

        ellipse_points = self.map.ellipse_points(point_a[:2], point_b[:2], self.config.robot_radius)
        if len(ellipse_points) < self.config.min_points_per_region:
            return None

        heights = ellipse_points[:, 2]
        median_height = float(np.median(heights))
        if np.any(np.abs(heights - median_height) >= self.config.max_height_step):
            return None

        tangent_basis = self._ellipse_tangent_basis(ellipse_points, point_a, point_b)
        if tangent_basis is None:
            return None

        longitudinal, lateral = tangent_basis
        longitudinal_risk = abs(float(longitudinal[2]))
        lateral_risk = abs(float(lateral[2]))
        weight = self.config.gamma * longitudinal_risk + (1.0 - self.config.gamma) * lateral_risk
        distance = float(np.linalg.norm(point_b - point_a))
        return distance, weight

    def _ellipse_tangent_basis(
        self,
        ellipse_points: np.ndarray,
        point_a: np.ndarray,
        point_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        centered = ellipse_points - np.mean(ellipse_points, axis=0)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, int(np.argmin(eigenvalues))]
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            return None
        normal = normal / normal_norm
        if normal[2] < 0.0:
            normal = -normal

        path_direction = point_b - point_a
        longitudinal = path_direction - np.dot(path_direction, normal) * normal
        longitudinal_norm = float(np.linalg.norm(longitudinal))
        if longitudinal_norm <= 1e-12:
            return None
        longitudinal = longitudinal / longitudinal_norm

        lateral = np.cross(normal, longitudinal)
        lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm <= 1e-12:
            return None
        lateral = lateral / lateral_norm
        return longitudinal, lateral

    def _find_merge_target(self, candidate_xy: np.ndarray, exclude: set[int] | None = None) -> int | None:
        node_ids = self.graph.node_ids_within_radius(candidate_xy, self.config.robot_radius, exclude=exclude)
        if not node_ids:
            return None
        distances = [
            (float(np.linalg.norm(self.graph.nodes[node_id].position[:2] - candidate_xy)), node_id)
            for node_id in node_ids
        ]
        distances.sort(key=lambda item: item[0])
        return distances[0][1]

    def _wire_existing_neighbors(self, node_id: int) -> None:
        node = self.graph.nodes[node_id]
        nearby = self.graph.node_ids_within_radius(
            node.position[:2],
            self.config.expansion_radius,
            exclude={node_id},
        )
        for other_id in nearby:
            edge = self._evaluate_edge(node.position, self.graph.nodes[other_id].position)
            if edge is None:
                continue
            self.graph.connect(node_id, other_id, *edge)

    def _update_reachability(self) -> None:
        if self._start_node_id is None:
            return

        reachable: set[int] = set()
        queue: deque[int] = deque([self._start_node_id])
        while queue:
            node_id = queue.popleft()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            for neighbor_id, _ in self.graph.neighbors(node_id):
                if neighbor_id not in reachable:
                    queue.append(neighbor_id)

        for node_id, node in self.graph.nodes.items():
            node.reachable = node_id in reachable
            node.state = "valid" if node.stable and node.reachable else "invalid"

    def _attach_or_project_goal(self, goal_xy: tuple[float, float]) -> tuple[int | None, bool]:
        nearest = self.graph.nearest_node(goal_xy, valid_only=True)
        if nearest is not None and nearest[1] <= self.config.robot_radius:
            return nearest[0], True

        goal_point = self._stable_point(goal_xy)
        if goal_point is not None:
            goal_node_id = self.graph.add_node(goal_point, stable=True)
            candidate_neighbors = self.graph.node_ids_within_radius(
                goal_point[:2],
                1.5 * self.config.expansion_radius,
                exclude={goal_node_id},
            )
            connected = False
            for neighbor_id in candidate_neighbors:
                if self.graph.nodes[neighbor_id].state != "valid":
                    continue
                edge = self._evaluate_edge(goal_point, self.graph.nodes[neighbor_id].position)
                if edge is None:
                    continue
                self.graph.connect(goal_node_id, neighbor_id, *edge)
                connected = True

            if connected:
                self._update_reachability()
                if self.graph.nodes[goal_node_id].state == "valid":
                    return goal_node_id, True

        nearest_valid = self.graph.nearest_node(goal_xy, valid_only=True)
        if nearest_valid is None:
            return None, False
        return nearest_valid[0], False

    def _a_star(
        self,
        start_node_id: int,
        goal_node_id: int,
        target_goal_xy: tuple[float, float],
    ) -> tuple[list[int], float]:
        target_goal_xy = (float(target_goal_xy[0]), float(target_goal_xy[1]))
        open_heap: list[tuple[float, int]] = []
        heapq.heappush(open_heap, (0.0, start_node_id))

        cost_to_come: dict[int, float] = {start_node_id: 0.0}
        came_from: dict[int, int] = {}
        closed: set[int] = set()

        while open_heap:
            _, node_id = heapq.heappop(open_heap)
            if node_id in closed:
                continue
            if node_id == goal_node_id:
                return self._reconstruct_path(came_from, goal_node_id), cost_to_come[goal_node_id]

            closed.add(node_id)
            for neighbor_id, edge in self.graph.neighbors(node_id):
                if edge.blocked:
                    continue
                if self.graph.nodes[neighbor_id].state != "valid" and neighbor_id != goal_node_id:
                    continue

                step_cost = edge.distance * (self.config.safety_factor * edge.weight + 1.0)
                tentative_cost = cost_to_come[node_id] + step_cost
                if tentative_cost >= cost_to_come.get(neighbor_id, float("inf")):
                    continue

                cost_to_come[neighbor_id] = tentative_cost
                came_from[neighbor_id] = node_id
                heuristic = float(np.linalg.norm(self.graph.nodes[neighbor_id].position[:2] - target_goal_xy))
                heapq.heappush(open_heap, (tentative_cost + heuristic, neighbor_id))

        raise RuntimeError("A* failed to find a path on the constructed TRG")

    def _reconstruct_path(self, came_from: dict[int, int], goal_node_id: int) -> list[int]:
        path = [goal_node_id]
        current = goal_node_id
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ------------------------------------------------------------------ #
    # Incremental / online API (task 6)
    #
    # These let the graph persist across frames: the elevation map is swapped
    # in each frame (:meth:`set_map`), newly observed terrain grows the graph
    # locally (:meth:`ingest`), the current path is cheaply re-checked every
    # frame (:meth:`validate_path`), and a blocked path triggers a local
    # :meth:`replan` on the persistent graph instead of a full rebuild.
    # ------------------------------------------------------------------ #
    def set_map(self, elevation_map: ElevationMap) -> None:
        """Swap in the latest elevation map for subsequent operations."""
        self.map = elevation_map

    def seed(self, start_xy: tuple[float, float]) -> int:
        """Initialise the persistent graph with a single start node.

        Unlike :meth:`build_graph`, this does not expand or reset an existing
        graph beyond creating the seed; call :meth:`ingest` to grow it.
        """
        start_point = self._stable_point(start_xy)
        if start_point is None:
            raise ValueError("start pose is not on a stable terrain region")
        start_node_id = self.graph.add_node(start_point, stable=True)
        self._start_node_id = start_node_id
        return start_node_id

    def ingest(self, region_xy: np.ndarray) -> int:
        """Grow the graph into newly observed terrain near ``region_xy``.

        ``region_xy`` is an ``(N, 2)`` array of world XY centres of cells that
        changed this frame (the elevation builder's dirty set). Existing nodes
        near those cells are re-opened for expansion so the frontier advances
        into terrain that has just become observable. Returns the number of new
        nodes added.
        """
        region = np.asarray(region_xy, dtype=float)
        if region.ndim != 2 or region.shape[1] != 2:
            raise ValueError("region_xy must be an (N, 2) array")
        if len(region) == 0:
            return 0

        seeds: set[int] = set()
        for xy in region:
            seeds.update(self.graph.node_ids_within_radius(xy, self.config.expansion_radius))
        if not seeds:
            # No graph nodes near the changed region yet; nothing to grow from.
            return 0

        for sid in seeds:
            self._expanded.discard(sid)

        before = len(self.graph.nodes)
        self._expand_frontier(deque(seeds))
        self._update_reachability()
        return len(self.graph.nodes) - before

    def validate_path(self, node_ids: list[int]) -> dict[str, list]:
        """Re-check a path against the current map (cheap, every frame).

        Refreshes node heights/stability and edge traversability along the
        path. A node/edge that fails is only *committed* (node marked
        ``state="invalid"`` / ``edge.blocked = True``) after
        :attr:`PlannerConfig.block_hysteresis` consecutive failing frames, so a
        single noisy frame does not tear down the graph. A passing frame resets
        the strike counter and clears any prior block. Returns the issues
        committed this call as ``{"invalid_nodes": [...],
        "blocked_edges": [(s, d)]}``.
        """
        hysteresis = max(1, self.config.block_hysteresis)
        invalid_nodes: list[int] = []
        blocked_edges: list[tuple[int, int]] = []

        for nid in node_ids:
            node = self.graph.nodes[nid]
            pt = self._stable_point((float(node.position[0]), float(node.position[1])))
            if pt is None:
                node.invalid_strikes += 1
                if node.invalid_strikes >= hysteresis:
                    node.stable = False
                    node.state = "invalid"
                    invalid_nodes.append(nid)
            else:
                node.invalid_strikes = 0
                node.position[2] = pt[2]
                node.stable = True

        for src, dst in zip(node_ids[:-1], node_ids[1:]):
            edge_id = self.graph._adjacency.get(src, {}).get(dst)
            if edge_id is None:
                blocked_edges.append((src, dst))
                continue
            edge = self.graph.edges[edge_id]
            evaluated = self._evaluate_edge(
                self.graph.nodes[src].position, self.graph.nodes[dst].position
            )
            if evaluated is None:
                edge.block_strikes += 1
                if edge.block_strikes >= hysteresis:
                    edge.blocked = True
                    blocked_edges.append((src, dst))
            else:
                edge.block_strikes = 0
                edge.blocked = False
                edge.distance, edge.weight = evaluated

        return {"invalid_nodes": invalid_nodes, "blocked_edges": blocked_edges}

    def replan(self, start_xy: tuple[float, float], goal_xy: tuple[float, float], heal: bool = True) -> PathResult:
        """Plan on the persistent graph from the robot's current pose.

        Does not rebuild the graph; it picks the nearest stable node to
        ``start_xy`` as the start, refreshes reachability, and runs A* avoiding
        blocked edges and invalid nodes. If no path is found and ``heal`` is
        True, :meth:`_plan_with_healing` optimistically reopens recently blocked
        edges so a falsely blocked unique passage can still be probed.
        """
        if not self.graph.nodes:
            raise RuntimeError("graph is empty; call seed()/ingest() first")
        start_node_id = self._pick_start_node(start_xy)
        if start_node_id is None:
            raise RuntimeError("no stable graph node near the start pose")
        self._start_node_id = start_node_id
        self._update_reachability()
        try:
            return self._plan_to_goal(start_node_id, goal_xy)
        except RuntimeError:
            if not heal:
                raise
            return self._plan_with_healing(start_node_id, goal_xy)

    def _plan_with_healing(self, start_node_id: int, goal_xy: tuple[float, float]) -> PathResult:
        """Recover connectivity by optimistically reopening blocked edges.

        When the normal plan fails, a unique passage may have been (possibly
        falsely) blocked. We temporarily clear every blocked edge, replan an
        optimistic "probe" path, then re-block only the suspect edges that the
        probe path does *not* use. The reopened on-path edges will be
        re-verified by :meth:`validate_path` as the robot approaches, so a truly
        blocked passage simply gets blocked again (after hysteresis) while a
        false positive is permanently healed.
        """
        suspects = [edge for edge in self.graph.edges.values() if edge.blocked]
        if not suspects:
            # Nothing was blocked, so the failure is genuine (no terrain path).
            raise RuntimeError("no path to goal and no blocked edges to reopen")

        for edge in suspects:
            edge.blocked = False
        self._update_reachability()

        try:
            result = self._plan_to_goal(start_node_id, goal_xy)
        except RuntimeError:
            # Reopening did not help; restore the blocks and report failure.
            for edge in suspects:
                edge.blocked = True
            self._update_reachability()
            raise

        path_pairs: set[tuple[int, int]] = set()
        for src, dst in zip(result.node_ids[:-1], result.node_ids[1:]):
            path_pairs.add((src, dst))
            path_pairs.add((dst, src))

        for edge in suspects:
            if (edge.src, edge.dst) not in path_pairs:
                edge.blocked = True
        self._update_reachability()
        return result


    def _pick_start_node(self, start_xy: tuple[float, float]) -> int | None:
        """Nearest stable node to ``start_xy`` (expanding-radius search)."""
        start = np.asarray(start_xy, dtype=float)[:2]
        radius = self.config.expansion_radius
        for _ in range(10):
            ids = self.graph.node_ids_within_radius(start, radius)
            stable = [
                (float(np.linalg.norm(self.graph.nodes[i].position[:2] - start)), i)
                for i in ids
                if self.graph.nodes[i].stable
            ]
            if stable:
                stable.sort(key=lambda item: item[0])
                return stable[0][1]
            radius *= 2.0
        return None

    def prune_outside(self, field_extent) -> int:
        """Remove all nodes that lie outside the field-extent polygon.

        Returns the number of nodes pruned.
        """
        if field_extent is None or not field_extent.is_valid():
            return 0
        to_remove = []
        for node_id, node in self.graph.nodes.items():
            if not field_extent.contains(
                node.position[:2].reshape(1, 2)
            )[0]:
                to_remove.append(node_id)
        for node_id in to_remove:
            self.graph.remove_node(node_id)
        if to_remove:
            self._expanded.difference_update(to_remove)
            self._update_reachability()
        return len(to_remove)
