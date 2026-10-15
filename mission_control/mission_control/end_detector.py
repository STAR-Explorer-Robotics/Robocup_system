#!/usr/bin/env python3
"""
end_detector — four-condition joint end-of-corridor detection.

Subscribes:
  /trg/debug/global_trg   (MarkerArray)  — TRG frontier nodes
  /lightning/lio_odom     (Odometry)     — robot pose
  /mission/state          (Int32)         — accepts Stage3 only while FSM is INIT

Publishes:
  /mission/end_status     (String)       — detailed verdict
  /mission/is_at_end      (Bool)         — True when end detected
"""

import math
import struct
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String, Bool, Int32


def normalize(v):
    length = math.hypot(v[0], v[1])
    if length < 1e-9:
        return (1.0, 0.0)
    return (v[0] / length, v[1] / length)


def count_frontiers_in_sector(frontiers, origin, direction, half_angle_rad):
    """Count frontier points inside the sector."""
    cnt = 0
    for f in frontiers:
        dx = f[0] - origin[0]
        dy = f[1] - origin[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        dot = (dx / dist) * direction[0] + (dy / dist) * direction[1]
        if dot >= math.cos(half_angle_rad):
            cnt += 1
    return cnt


def pointcloud2_xy(msg: PointCloud2):
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


class EndDetector(Node):
    """Detect whether the robot has reached the end of the corridor."""

    def __init__(self):
        super().__init__("end_detector")

        self._load_params()

        # Subscribers
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

        # Publishers
        self._pub_status = self.create_publisher(String, "/mission/end_status", 10)
        self._pub_is_end = self.create_publisher(Bool, "/mission/is_at_end", 10)

        # Internal state
        self._robot = (0.0, 0.0)
        self._robot_yaw = 0.0
        self._start = (0.0, 0.0)
        self._start_yaw = 0.0
        self._direction_computed = False
        self._frontiers = []
        self._soft_frontiers = []
        self._direction = (1.0, 0.0)
        self._reverse_dir = (-1.0, 0.0)
        self._mission_state = 0

        # Condition-D: hold timer
        self._conditions_met_since = None
        self._timer = self.create_timer(1.0 / self.check_rate, self._on_timer)

        self.get_logger().info(
            f"end_detector ready: min_travel={self.min_travel}m hold={self.hold_duration}s"
        )

    def _load_params(self):
        self.declare_parameter(
            "heading_deg", 0.0, ParameterDescriptor(dynamic_typing=True)
        )
        self.declare_parameter("sector_half_angle", math.radians(60))
        self.declare_parameter("reverse_half_angle", math.radians(60))
        self.declare_parameter("min_travel_distance", 3.0)
        self.declare_parameter("hold_duration", 5.0)
        self.declare_parameter("check_rate", 2.0)
        self.declare_parameter("soft_frontier_topic", "/navigation/soft_frontiers")
        self.declare_parameter("start_direction_topic", "/mission/start_direction_point")
        self.declare_parameter("min_start_direction_distance", 0.50)

        self.heading_deg = float(self.get_parameter("heading_deg").value)
        self.sector_half_angle = (
            self.get_parameter("sector_half_angle").get_parameter_value().double_value
        )
        self.reverse_half_angle = (
            self.get_parameter("reverse_half_angle").get_parameter_value().double_value
        )
        self.min_travel = (
            self.get_parameter("min_travel_distance").get_parameter_value().double_value
        )
        self.hold_duration = (
            self.get_parameter("hold_duration").get_parameter_value().double_value
        )
        self.check_rate = (
            self.get_parameter("check_rate").get_parameter_value().double_value
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

    # -- callbacks --------------------------------------------------------

    def _on_markers(self, msg: MarkerArray):
        frontiers = []
        for marker in msg.markers:
            if marker.id == 0:
                for i, pt in enumerate(marker.points):
                    if i < len(marker.colors) and marker.colors[i].r > 0.9:
                        frontiers.append((pt.x, pt.y))
        self._frontiers = frontiers

    def _on_soft_frontiers(self, msg: PointCloud2):
        self._soft_frontiers = pointcloud2_xy(msg)

    def _on_odom(self, msg: Odometry):
        self._robot = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny, cosy)
        if not self._direction_computed:
            self._start = self._robot
            self._start_yaw = self._robot_yaw
            rad = math.radians(self.heading_deg) + self._start_yaw
            self._direction = normalize((math.cos(rad), math.sin(rad)))
            self._reverse_dir = (-self._direction[0], -self._direction[1])
            self._direction_computed = True
            self.get_logger().info(
                f"heading={self.heading_deg}° + start_yaw={math.degrees(self._start_yaw):.0f}° "
                f"→ map_dir=({self._direction[0]:.2f},{self._direction[1]:.2f})"
            )

    def _on_mission_state(self, msg: Int32):
        self._mission_state = int(msg.data)

    def _on_start_direction(self, msg: PointStamped):
        if not self._direction_computed:
            self.get_logger().warn(
                "ignoring mission direction point before odom/start pose is ready",
                throttle_duration_sec=2.0,
            )
            return
        if self._mission_state != 0:
            self.get_logger().warn(
                "ignoring Stage3 leg target because mission_fsm is not waiting "
                f"in INIT (state={self._mission_state})",
                throttle_duration_sec=2.0,
            )
            return
        leg_start = self._robot
        dx = float(msg.point.x) - leg_start[0]
        dy = float(msg.point.y) - leg_start[1]
        dist = math.hypot(dx, dy)
        if dist < self.min_start_direction_distance:
            self.get_logger().warn(
                "ignoring mission leg target point: horizontal distance "
                f"{dist:.2f}m < {self.min_start_direction_distance:.2f}m"
            )
            return
        self._start = leg_start
        self._start_yaw = self._robot_yaw
        self._direction = normalize((dx, dy))
        self._reverse_dir = (-self._direction[0], -self._direction[1])
        self._conditions_met_since = None
        rel_heading = math.degrees(math.atan2(dy, dx) - self._robot_yaw)
        self.get_logger().warn(
            "mission end direction updated from RViz leg target: "
            f"leg_start=({leg_start[0]:.2f},{leg_start[1]:.2f}), "
            f"map_dir=({self._direction[0]:.2f},{self._direction[1]:.2f}), "
            f"body_relative_heading={rel_heading:.1f}deg"
        )

    # -- timer ------------------------------------------------------------

    def _on_timer(self):
        verdict = self._evaluate()
        self._publish(verdict["is_end"], verdict["reason"])

    # -- core logic -------------------------------------------------------

    def _evaluate(self):
        """Return dict with is_end and reason."""

        # B: must be far enough from start
        dist_from_start = math.hypot(
            self._robot[0] - self._start[0],
            self._robot[1] - self._start[1],
        )
        if dist_from_start < self.min_travel:
            return {"is_end": False, "reason": f"too_close_to_start ({dist_from_start:.1f}m < {self.min_travel}m)"}

        frontiers = list(self._frontiers) + list(self._soft_frontiers)
        if not frontiers:
            return {"is_end": False, "reason": "no_frontiers_yet"}

        # A: no frontier in forward sector
        fwd_cnt = count_frontiers_in_sector(
            frontiers, self._robot, self._direction, self.sector_half_angle
        )
        if fwd_cnt > 0:
            self._conditions_met_since = None
            return {"is_end": False, "reason": f"forward_frontiers={fwd_cnt}"}

        # C: three-side check — sides (not reverse) should have no frontiers
        # We check left and right (perpendicular to forward direction)
        left_dir = (-self._direction[1], self._direction[0])
        right_dir = (self._direction[1], -self._direction[0])

        left_cnt = count_frontiers_in_sector(
            frontiers, self._robot, left_dir, self.reverse_half_angle
        )
        right_cnt = count_frontiers_in_sector(
            frontiers, self._robot, right_dir, self.reverse_half_angle
        )

        if left_cnt > 0 or right_cnt > 0:
            self._conditions_met_since = None
            return {
                "is_end": False,
                "reason": f"side_frontiers left={left_cnt} right={right_cnt} (not fully blocked)",
            }

        # A + B + C all met → start hold timer (D)
        now = self.get_clock().now()
        if self._conditions_met_since is None:
            self._conditions_met_since = now
            return {"is_end": False, "reason": "conditions_met_counting"}

        elapsed = (now - self._conditions_met_since).nanoseconds / 1e9
        if elapsed < self.hold_duration:
            return {
                "is_end": False,
                "reason": f"holding {elapsed:.1f}s / {self.hold_duration}s",
            }

        return {"is_end": True, "reason": "all_conditions_met"}

    def _publish(self, is_end: bool, reason: str):
        msg = String()
        msg.data = reason
        self._pub_status.publish(msg)

        b = Bool()
        b.data = is_end
        self._pub_is_end.publish(b)

    def set_direction(self, dx: float, dy: float):
        self._direction = normalize((dx, dy))
        self._reverse_dir = (-self._direction[0], -self._direction[1])

    def set_start(self, x: float, y: float):
        self._start = (x, y)


def main(args=None):
    rclpy.init(args=args)
    node = EndDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
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
