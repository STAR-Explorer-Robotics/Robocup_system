"""Online TRG runner — incremental elevation + persistent-graph planning.

Wraps the :class:`~trg_planner.planner.TRGPlanner` and the
:class:`~trg_planner.online.elevation_builder.ElevationGridBuilder` so a single
persistent graph and full-field elevation map are maintained across frames.
Each :meth:`OnlineTRGRunner.step` integrates the latest cloud, grows the graph
only into newly observed terrain, cheaply re-validates the current path, and
replans locally (with connectivity self-healing) only when the path is broken.

This class is ROS-free; a ROS 2 node or a bag-replay driver wraps it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..planner import PathResult, PlannerConfig, TRGPlanner
from .elevation_builder import ElevationConfig, ElevationGridBuilder
from .field_extent import FieldExtent


@dataclass
class OnlineTRGResult:
    """Outcome of one planning cycle."""

    path: PathResult | None
    robot_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    error: str | None = None
    new_nodes: int = 0
    invalid_nodes: list[int] = field(default_factory=list)
    blocked_edges: list[tuple[int, int]] = field(default_factory=list)
    replanned: bool = False


class OnlineTRGRunner:
    """Persistent-graph incremental planning loop.

    Typical usage::

        runner = OnlineTRGRunner()
        runner.set_goal((10.5, 10.5))
        for frame in frames:                # cloud + pose stream
            runner.update_cloud(frame.points_world, frame.robot_xy)
            result = runner.step()
    """

    def __init__(
        self,
        elevation_config: ElevationConfig | None = None,
        planner_config: PlannerConfig | None = None,
        *,
        clean_every: int = 0,
    ) -> None:
        self.builder = ElevationGridBuilder(elevation_config)
        self.planner_config = planner_config or PlannerConfig()
        self.planner: TRGPlanner | None = None
        self._robot_xy: np.ndarray | None = None
        self._goal_xy: tuple[float, float] | None = None
        self._last_map = None
        self._current_path: PathResult | None = None
        self._seeded = False
        self._clean_every = max(0, clean_every)
        self._frame_count = 0
        self._field_extent: FieldExtent | None = None

    # ------------------------------------------------------------------ #
    def set_goal(self, goal_xy: tuple[float, float] | np.ndarray | None) -> None:
        if goal_xy is None:
            self._goal_xy = None
            return
        goal_xy = np.asarray(goal_xy, dtype=float)[:2]
        self._goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        # A new goal invalidates the cached path.
        self._current_path = None

    def update_cloud(
        self,
        points_world: np.ndarray,
        robot_xy: tuple[float, float] | np.ndarray,
    ) -> None:
        """Integrate a new map-frame cloud and record the robot position."""
        robot_xy = np.asarray(robot_xy, dtype=float)[:2]
        self._robot_xy = robot_xy
        self.builder.add_cloud(points_world, robot_xy)

    def current_map(self):
        """Return the latest :class:`ElevationMap` (rebuilds on demand)."""
        self._last_map = self.builder.build_map()
        return self._last_map

    def set_field_extent(self, field_extent: FieldExtent | None, *,
                          prune: bool = False) -> None:
        """Update the field-extent polygon used to gate graph growth.

        Pass ``None`` to disable gating (graph grows freely).

        When ``prune=True``, existing nodes outside a valid extent are
        retroactively removed.  Callers should only enable this once the extent
        has stabilised (area large enough) to avoid over-pruning from an
        early, incomplete extent.
        """
        self._field_extent = field_extent
        if self.planner is not None:
            self.planner.set_field_extent(field_extent)
        if (
            prune
            and field_extent is not None
            and field_extent.is_valid()
            and self.planner is not None
        ):
            removed = self.planner.prune_outside(field_extent)
            if removed:
                print(f"[gate] pruned {removed} nodes outside field extent")
                self._current_path = None  # path may reference pruned nodes

    # ------------------------------------------------------------------ #
    def step(self) -> OnlineTRGResult:
        """Run one incremental cycle from the current map and pose."""
        if self._robot_xy is None:
            return OnlineTRGResult(None, (0.0, 0.0), self._goal_xy or (0.0, 0.0), "no pose yet")

        start_xy = (float(self._robot_xy[0]), float(self._robot_xy[1]))
        self._frame_count += 1

        # Periodic flood-fill cleanup of disconnected map debris.
        if self._clean_every and self._frame_count % self._clean_every == 0:
            self.builder.clean(robot_xy=start_xy)

        elevation_map = self.current_map()
        if self.planner is None:
            self.planner = TRGPlanner(elevation_map, self.planner_config)
            self.planner.set_field_extent(self._field_extent)
        else:
            self.planner.set_map(elevation_map)

        # Seed the persistent graph once the robot stands on stable terrain.
        if not self._seeded:
            try:
                self.planner.seed(start_xy)
                self._seeded = True
            except ValueError as exc:
                return OnlineTRGResult(None, start_xy, self._goal_xy or (0.0, 0.0), str(exc))

        # Grow the graph into terrain that changed this frame.
        dirty = self.builder.consume_dirty()
        centres = self.builder.dirty_cell_centres(dirty)
        if len(centres):
            if self._field_extent is not None and self._field_extent.is_valid():
                inside = self._field_extent.contains(centres)
                centres = centres[inside]
        new_nodes = self.planner.ingest(centres) if len(centres) else 0

        result = OnlineTRGResult(
            self._current_path, start_xy, self._goal_xy or (0.0, 0.0), new_nodes=new_nodes
        )

        if self._goal_xy is None:
            return result

        # Cheap per-frame validation; replan only when broken or absent.
        needs_replan = True
        if self._current_path is not None and len(self._current_path.node_ids) >= 1:
            issues = self.planner.validate_path(self._current_path.node_ids)
            result.invalid_nodes = issues["invalid_nodes"]
            result.blocked_edges = issues["blocked_edges"]
            needs_replan = bool(issues["invalid_nodes"] or issues["blocked_edges"])

        if needs_replan:
            result.replanned = True
            try:
                self._current_path = self.planner.replan(start_xy, self._goal_xy, heal=True)
            except (ValueError, RuntimeError) as exc:
                self._current_path = None
                result.path = None
                result.error = str(exc)
                return result

        result.path = self._current_path
        return result
