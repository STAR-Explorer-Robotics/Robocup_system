#!/usr/bin/env python3
"""
goal_selector — choose a forward exploration target and verify TRG reachability.

Subscribes:
  /trg/debug/global_trg      (MarkerArray)  — TRG frontier nodes
  /navigation/soft_frontiers (PointCloud2)  — reachable frontier cells next to unknown space
  /lightning/lio_odom        (Odometry)     — robot pose
  /trg/output/path           (Path)         — path validation after a candidate goal is sent
  /mission/state             (Int32)        — accepts Stage3 only while FSM is INIT

Publishes:
  /goal_pose                 (PoseStamped)  — candidate/selected goal for TRG
  /mission/selected_goal     (PointStamped) — accepted goal visualization
"""

import math
import struct
import threading

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import ParameterDescriptor
from rclpy._rclpy_pybind11 import RCLError
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Int32
from std_srvs.srv import SetBool
from visualization_msgs.msg import MarkerArray


def normalize(v):
    """Return a unit vector."""
    length = math.hypot(v[0], v[1])
    if length < 1e-9:
        return (1.0, 0.0)
    return (v[0] / length, v[1] / length)


def in_sector(point, origin, direction, half_angle_rad):
    """Check whether *point* lies within a sector centred on *direction*."""
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return False
    dx /= dist
    dy /= dist
    dot = dx * direction[0] + dy * direction[1]
    return dot >= math.cos(half_angle_rad)


def pointcloud2_xy(msg: PointCloud2):
    """Extract finite XY tuples from a simple FLOAT32 PointCloud2."""
    if msg.width == 0 or msg.point_step == 0:
        return []
    fields = {field.name: field for field in msg.fields}
    if "x" not in fields or "y" not in fields:
        return []
    if (
        fields["x"].datatype != PointField.FLOAT32
        or fields["y"].datatype != PointField.FLOAT32
    ):
        return []
    point_step = int(msg.point_step)
    total_bytes = int(msg.width) * int(msg.height) * point_step
    if len(msg.data) < total_bytes:
        return []
    endian = ">" if msg.is_bigendian else "<"
    unpack_x = struct.Struct(endian + "f")
    unpack_y = struct.Struct(endian + "f")
    off_x = int(fields["x"].offset)
    off_y = int(fields["y"].offset)
    points = []
    for base in range(0, total_bytes, point_step):
        x = unpack_x.unpack_from(msg.data, base + off_x)[0]
        y = unpack_y.unpack_from(msg.data, base + off_y)[0]
        if math.isfinite(x) and math.isfinite(y):
            points.append((float(x), float(y)))
    return points


class GoalSelector(Node):
    """Selects a forward frontier and accepts it only if TRG returns a path."""

    def __init__(self):
        super().__init__("goal_selector")

        self._load_params()
        self._service_group = MutuallyExclusiveCallbackGroup()
        self._path_group = MutuallyExclusiveCallbackGroup()

        self._robot = (0.0, 0.0)
        self._robot_yaw = 0.0
        self._start = (0.0, 0.0)
        self._start_yaw = 0.0
        self._direction_computed = False
        self._frontiers = []
        self._soft_frontiers = []
        self._direction = (1.0, 0.0)
        self._operator_target = None
        self._mission_state = 0

        self._state_lock = threading.Lock()
        self._path_event = threading.Event()
        self._pending_goal = None
        self._request_stamp_sec = 0.0
        self._request_in_progress = False
        self._path_received = False
        self._path_valid = False
        self._path_length = 0.0
        self._path_endpoint_error = float("inf")
        self._last_valid_source = ""

        self._sub_markers = self.create_subscription(
            MarkerArray, "/trg/debug/global_trg", self._on_markers, 10
        )
        self._sub_odom = self.create_subscription(
            Odometry, "/lightning/lio_odom", self._on_odom, 10
        )
        self._sub_start_direction = self.create_subscription(
            PointStamped,
            self.start_direction_topic,
            self._on_start_direction,
            10,
        )
        self._sub_mission_state = self.create_subscription(
            Int32, "/mission/state", self._on_mission_state, 10
        )
        frontier_qos = QoSProfile(depth=1)
        frontier_qos.reliability = QoSReliabilityPolicy.RELIABLE
        frontier_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._sub_soft_frontiers = self.create_subscription(
            PointCloud2,
            self.soft_frontier_topic,
            self._on_soft_frontiers,
            frontier_qos,
        )

        self._pub_goal = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self._pub_viz = self.create_publisher(PointStamped, "/mission/selected_goal", 10)

        self._sub_path = self.create_subscription(
            Path,
            "/trg/output/path",
            self._on_path,
            10,
            callback_group=self._path_group,
        )
        self._srv = self.create_service(
            SetBool,
            "/mission/select_goal",
            self._on_select_request,
            callback_group=self._service_group,
        )

    def _load_params(self):
        self.declare_parameter(
            "heading_deg", 0.0, ParameterDescriptor(dynamic_typing=True)
        )
        self.declare_parameter("sector_half_angle", math.radians(60))
        self.declare_parameter("max_path_length", 50.0)
        self.declare_parameter("path_validation_timeout", 2.0)
        self.declare_parameter("path_goal_tolerance", 1.0)
        self.declare_parameter("max_candidates_to_try", 8)
        self.declare_parameter("soft_frontier_topic", "/navigation/soft_frontiers")
        self.declare_parameter("start_direction_topic", "/mission/start_direction_point")
        self.declare_parameter("min_start_direction_distance", 0.50)
        self.declare_parameter("operator_target_enabled", True)
        self.declare_parameter("frontier_sector_filter_enabled", True)

        self.heading_deg = float(self.get_parameter("heading_deg").value)
        self.sector_half_angle = (
            self.get_parameter("sector_half_angle").get_parameter_value().double_value
        )
        self.max_path_length = (
            self.get_parameter("max_path_length").get_parameter_value().double_value
        )
        self.path_validation_timeout = float(
            self.get_parameter("path_validation_timeout").value
        )
        self.path_goal_tolerance = float(
            self.get_parameter("path_goal_tolerance").value
        )
        self.max_candidates_to_try = max(
            1, int(self.get_parameter("max_candidates_to_try").value)
        )
        self.soft_frontier_topic = str(
            self.get_parameter("soft_frontier_topic").value
        )
        self.start_direction_topic = str(
            self.get_parameter("start_direction_topic").value
        )
        self.min_start_direction_distance = float(
            self.get_parameter("min_start_direction_distance").value
        )
        self.operator_target_enabled = bool(
            self.get_parameter("operator_target_enabled").value
        )
        self.frontier_sector_filter_enabled = bool(
            self.get_parameter("frontier_sector_filter_enabled").value
        )

    # -- callbacks --------------------------------------------------------

    def _on_markers(self, msg: MarkerArray):
        frontiers = []
        for marker in msg.markers:
            if marker.id == 0:
                for i, pt in enumerate(marker.points):
                    if i < len(marker.colors) and marker.colors[i].r > 0.9:
                        frontiers.append((float(pt.x), float(pt.y)))
        with self._state_lock:
            self._frontiers = frontiers

    def _on_soft_frontiers(self, msg: PointCloud2):
        points = pointcloud2_xy(msg)
        with self._state_lock:
            self._soft_frontiers = points

    def _on_odom(self, msg: Odometry):
        robot = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        with self._state_lock:
            self._robot = robot
            self._robot_yaw = yaw
            if self._direction_computed:
                return
            self._start = robot
            self._start_yaw = yaw
            rad = math.radians(self.heading_deg) + self._start_yaw
            self._direction = normalize((math.cos(rad), math.sin(rad)))
            self._direction_computed = True
        self.get_logger().info(
            f"heading={self.heading_deg}° + start_yaw={math.degrees(self._start_yaw):.0f}° "
            f"→ map_dir=({self._direction[0]:.2f},{self._direction[1]:.2f}) "
            f"sector=±{math.degrees(self.sector_half_angle):.0f}° "
            f"frontier_filter={'sector' if self.frontier_sector_filter_enabled else 'all'}"
        )

    def _on_mission_state(self, msg: Int32):
        with self._state_lock:
            self._mission_state = int(msg.data)

    def _on_path(self, msg: Path):
        with self._state_lock:
            if not self._request_in_progress or self._pending_goal is None:
                return
            pending = self._pending_goal
            request_stamp = self._request_stamp_sec

        msg_stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if msg_stamp > 0.0 and msg_stamp + 0.05 < request_stamp:
            return

        if len(msg.poses) < 2:
            total = 0.0
            endpoint_error = float("inf")
            valid = False
        else:
            total = 0.0
            for i in range(1, len(msg.poses)):
                dx = msg.poses[i].pose.position.x - msg.poses[i - 1].pose.position.x
                dy = msg.poses[i].pose.position.y - msg.poses[i - 1].pose.position.y
                total += math.hypot(dx, dy)
            end = msg.poses[-1].pose.position
            endpoint_error = math.hypot(end.x - pending[0], end.y - pending[1])
            valid = (
                total <= self.max_path_length
                and endpoint_error <= self.path_goal_tolerance
            )

        with self._state_lock:
            if not self._request_in_progress:
                return
            self._path_received = True
            self._path_valid = bool(valid)
            self._path_length = float(total)
            self._path_endpoint_error = float(endpoint_error)
        self._path_event.set()

    def _on_start_direction(self, msg: PointStamped):
        with self._state_lock:
            mission_state = self._mission_state
            if not self._direction_computed:
                ready = False
                start = self._robot
                robot_yaw = self._robot_yaw
            else:
                ready = True
                start = self._robot
                robot_yaw = self._robot_yaw
        if not ready:
            self.get_logger().warn(
                "ignoring mission direction point before odom/start pose is ready",
                throttle_duration_sec=2.0,
            )
            return
        if mission_state != 0:
            self.get_logger().warn(
                "ignoring Stage3 leg target because mission_fsm is not waiting "
                f"in INIT (state={mission_state})",
                throttle_duration_sec=2.0,
            )
            return
        dx = float(msg.point.x) - start[0]
        dy = float(msg.point.y) - start[1]
        dist = math.hypot(dx, dy)
        if dist < self.min_start_direction_distance:
            self.get_logger().warn(
                "ignoring mission direction point: horizontal distance "
                f"{dist:.2f}m < {self.min_start_direction_distance:.2f}m"
            )
            return
        direction = normalize((dx, dy))
        with self._state_lock:
            self._start = start
            self._direction = direction
            self._operator_target = (float(msg.point.x), float(msg.point.y))
        rel_heading = math.degrees(math.atan2(dy, dx) - robot_yaw)
        self.get_logger().warn(
            "mission leg target updated from RViz point: "
            f"leg_start=({start[0]:.2f},{start[1]:.2f}), "
            f"point=({msg.point.x:.2f},{msg.point.y:.2f}), "
            f"map_dir=({direction[0]:.2f},{direction[1]:.2f}), "
            f"body_relative_heading={rel_heading:.1f}deg"
        )

    # -- service ----------------------------------------------------------

    def _on_select_request(self, req: SetBool.Request, resp: SetBool.Response):
        if not req.data:
            resp.success = False
            resp.message = "request.data must be True"
            return resp

        direct_result = self._try_operator_target()
        if direct_result is not None:
            source, goal = direct_result
            self._publish_selected_goal(goal[0], goal[1])
            resp.success = True
            resp.message = (
                f"goal selected from {source} at ({goal[0]:.2f}, {goal[1]:.2f}); "
                f"path={self._path_length:.2f}m endpoint_error={self._path_endpoint_error:.2f}m"
            )
            return resp

        candidates = self._ranked_candidates()
        if not candidates:
            resp.success = False
            if self.frontier_sector_filter_enabled:
                resp.message = (
                    "operator target invalid and no TRG/soft frontier candidate in leg sector"
                )
            else:
                resp.message = "operator target invalid and no TRG/soft frontier candidate"
            return resp

        for source, goal in candidates[: self.max_candidates_to_try]:
            if self._validate_candidate(source, goal):
                self._publish_selected_goal(goal[0], goal[1])
                resp.success = True
                resp.message = (
                    f"goal selected from {source} at ({goal[0]:.2f}, {goal[1]:.2f}); "
                    f"path={self._path_length:.2f}m endpoint_error={self._path_endpoint_error:.2f}m"
                )
                return resp

        resp.success = False
        resp.message = (
            f"no candidate produced a valid TRG path "
            f"(tried {min(len(candidates), self.max_candidates_to_try)})"
        )
        return resp

    # -- core logic -------------------------------------------------------

    def _try_operator_target(self):
        if not self.operator_target_enabled:
            return None
        with self._state_lock:
            target = self._operator_target
            start = self._start
        if target is None:
            return None
        dist = math.hypot(target[0] - start[0], target[1] - start[1])
        if dist < self.min_start_direction_distance:
            return None
        if self._validate_candidate("operator_target", target):
            return ("operator_target", target)
        self.get_logger().warn(
            "operator target did not produce a valid path; falling back to "
            "TRG/soft-frontier exploration in the same direction",
            throttle_duration_sec=1.0,
        )
        return None

    def _ranked_candidates(self):
        with self._state_lock:
            start = self._start
            direction = self._direction
            filter_enabled = self.frontier_sector_filter_enabled
            sources = [
                ("trg_frontier", list(self._frontiers)),
                ("soft_frontier", list(self._soft_frontiers)),
            ]

        per_source = []
        seen = set()
        for source, points in sources:
            source_candidates = []
            for point in points:
                if filter_enabled and not in_sector(point, start, direction, self.sector_half_angle):
                    continue
                key = (round(point[0], 2), round(point[1], 2))
                if key in seen:
                    continue
                seen.add(key)
                dist = math.hypot(point[0] - start[0], point[1] - start[1])
                source_candidates.append((dist, source, point))
            source_candidates.sort(key=lambda item: item[0], reverse=True)
            per_source.append(source_candidates)

        if not any(per_source):
            filter_name = "forward sector" if filter_enabled else "frontier set"
            self.get_logger().warn(
                f"no TRG frontier or soft frontier in {filter_name}",
                throttle_duration_sec=2.0,
            )
            return []

        candidates = []
        max_len = max(len(items) for items in per_source)
        for i in range(max_len):
            for items in per_source:
                if i < len(items):
                    _, source, point = items[i]
                    candidates.append((source, point))
        best_source, best = candidates[0]
        filter_name = "sector" if filter_enabled else "all"
        self.get_logger().info(
            f"goal candidates ({filter_name}): total={len(candidates)} "
            f"trg={len(sources[0][1])} soft={len(sources[1][1])}; "
            f"first={best_source} ({best[0]:.2f},{best[1]:.2f})"
        )
        return candidates

    def _validate_candidate(self, source: str, goal):
        self._path_event.clear()
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        with self._state_lock:
            self._pending_goal = goal
            self._request_stamp_sec = now_sec
            self._request_in_progress = True
            self._path_received = False
            self._path_valid = False
            self._path_length = 0.0
            self._path_endpoint_error = float("inf")

        self._publish_goal_request(goal[0], goal[1])
        got_path = self._path_event.wait(timeout=self.path_validation_timeout)
        with self._state_lock:
            valid = bool(got_path and self._path_valid)
            path_length = self._path_length
            endpoint_error = self._path_endpoint_error
            self._request_in_progress = False
            self._pending_goal = None
        if valid:
            self._last_valid_source = source
            self.get_logger().info(
                f"accepted {source} candidate ({goal[0]:.2f},{goal[1]:.2f}) "
                f"path={path_length:.2f}m endpoint_error={endpoint_error:.2f}m"
            )
            return True

        reason = "timeout" if not got_path else "invalid_path"
        self.get_logger().warn(
            f"rejected {source} candidate ({goal[0]:.2f},{goal[1]:.2f}): "
            f"{reason}, path={path_length:.2f}m endpoint_error={endpoint_error:.2f}m",
            throttle_duration_sec=0.5,
        )
        return False

    def _publish_goal_request(self, x: float, y: float):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self._pub_goal.publish(msg)

    def _publish_selected_goal(self, x: float, y: float):
        viz = PointStamped()
        viz.header.stamp = self.get_clock().now().to_msg()
        viz.header.frame_id = "map"
        viz.point.x = float(x)
        viz.point.y = float(y)
        viz.point.z = 0.3
        self._pub_viz.publish(viz)

    def set_direction(self, dx: float, dy: float):
        """Update forward direction (called by mission_fsm)."""
        with self._state_lock:
            self._direction = normalize((dx, dy))
            direction = self._direction
        self.get_logger().info(
            f"direction updated to ({direction[0]:.2f},{direction[1]:.2f})"
        )

    def set_start(self, x: float, y: float):
        """Update start pose (called by mission_fsm)."""
        with self._state_lock:
            self._start = (float(x), float(y))


def main(args=None):
    rclpy.init(args=args)
    node = GoalSelector()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        _shutdown_rclpy()


def _shutdown_rclpy():
    if not rclpy.ok():
        return
    try:
        rclpy.shutdown()
    except RCLError as exc:
        if "rcl_shutdown already called" not in str(exc):
            raise


if __name__ == "__main__":
    main()
