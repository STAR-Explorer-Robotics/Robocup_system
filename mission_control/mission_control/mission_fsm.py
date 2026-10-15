#!/usr/bin/env python3
"""
mission_fsm — RoboCup rescue mission state machine.
             One-click launch, fully autonomous, end-detection driven.

States: INIT -> FORWARD -> NAVIGATING -> ARRIVED -> CHECK_END -> INIT ...
         \                         \          \
          AUTO_RECOVER             RETRY      (retry/next)

Operator interaction (competition rules):
  Stage3: /mission/start_direction_point starts the next scoring leg
  Stage5: /mission/start_recorded_path starts a leg from the recorded path
  Stage4: /mission/complete_leg ends the current leg and waits for Stage3 again
  End:    /mission/confirm_done (only when DONE, no penalty)
  Emergency: /rpp/set_enabled false (anytime, no penalty)
             /mission/mark_done (anytime, penalty)
"""

import math
import enum
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PolygonStamped, PointStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import MarkerArray
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Int32, Bool, String
from std_srvs.srv import SetBool, Trigger


class FsmState(enum.IntEnum):
    INIT = 0
    FORWARD = 1
    NAVIGATING = 2
    ARRIVED = 3
    CHECK_END = 4
    DONE = 5
    AUTO_RECOVER = 6
    RETRY = 7


RPP_IDLE = 0
RPP_TRACKING = 1
RPP_GOAL_REACHED = 2
RPP_INPUT_STALE = 3
RPP_PATH_DEVIATION = 4


class MissionFSM(Node):
    """Top-level mission orchestrator — fully autonomous after launch."""

    def __init__(self):
        super().__init__("mission_fsm")
        self._load_params()

        self._sub_odom = self.create_subscription(Odometry, "/lightning/lio_odom", self._on_odom, 10)
        self._sub_path = self.create_subscription(Path, "/trg/output/path", self._on_path, 10)
        self._sub_markers = self.create_subscription(MarkerArray, "/trg/debug/global_trg", self._on_markers, 10)
        allowed_qos = QoSProfile(depth=1)
        allowed_qos.reliability = QoSReliabilityPolicy.RELIABLE
        allowed_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._sub_allowed_area = self.create_subscription(
            PolygonStamped,
            self.allowed_area_topic,
            self._on_allowed_area,
            allowed_qos,
        )
        self._sub_rpp = self.create_subscription(Int32, "/rpp/state", self._on_rpp_state, 10)
        self._sub_end = self.create_subscription(Bool, "/mission/is_at_end", self._on_end_status, 10)
        self._sub_start_direction = self.create_subscription(
            PointStamped,
            self.start_direction_topic,
            self._on_start_direction,
            10,
        )

        self._pub_goal = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self._pub_state = self.create_publisher(Int32, "/mission/state", 10)
        self._pub_leg_count = self.create_publisher(Int32, "/mission/leg_count", 10)
        self._pub_leg_status = self.create_publisher(String, "/mission/leg_status", 10)

        # Services: leg completion, end-confirmation, and emergency skip.
        self._srv_complete_leg = self.create_service(
            Trigger, "/mission/complete_leg", self._on_complete_leg
        )
        self._srv_start_recorded = self.create_service(
            Trigger, "/mission/start_recorded_path", self._on_start_recorded_path
        )
        self._srv_confirm = self.create_service(Trigger, "/mission/confirm_done", self._on_confirm_done)
        self._srv_mark = self.create_service(Trigger, "/mission/mark_done", self._on_mark_done)

        self._cli_select = self.create_client(SetBool, "/mission/select_goal")
        self._cli_rpp = self.create_client(SetBool, "/rpp/set_enabled")
        self._cli_path_planned = self.create_client(
            Trigger, self.path_source_use_planned_service
        )
        self._cli_path_recorded = self.create_client(
            Trigger, self.path_source_use_recorded_service
        )

        self._state = FsmState.INIT
        self._recover_target = FsmState.FORWARD
        self._robot = (0.0, 0.0)
        self._robot_yaw = 0.0
        self._start = (0.0, 0.0)
        self._start_recorded = False
        self._frontiers = []
        self._allowed_area_ready = False
        self._start_yaw = 0.0
        self._direction = (1.0, 0.0)
        self._direction_computed = False
        self._start_signal_received = not self.require_start_signal
        self._waiting_for_leg_start = self.require_start_signal
        self._start_wait_logged = False
        self._rpp_state = RPP_IDLE
        self._is_at_end = False
        self._leg_count = 0
        self._current_goal_source = ""
        self._last_leg_result = "waiting"
        self._retry_count = 0
        self._select_future = None
        self._select_start_time = self.get_clock().now()
        self._state_enter_time = self.get_clock().now()
        self._nav_start_time = self.get_clock().now()

        period = 1.0 / max(1.0, self.control_rate)
        self._timer = self.create_timer(period, self._on_timer)
        self._publish_state()
        self._publish_leg_status()
        self.get_logger().info(f"mission_fsm ready: state=INIT rate={self.control_rate}Hz")

    def _load_params(self):
        self.declare_parameter(
            "heading_deg", 0.0, ParameterDescriptor(dynamic_typing=True)
        )
        self.declare_parameter("sector_half_angle", math.radians(60))
        self.declare_parameter("navigate_timeout", 120.0)
        self.declare_parameter("goal_select_timeout", 20.0)
        self.declare_parameter("arrive_settle_duration", 2.0)
        self.declare_parameter("end_hold_duration", 5.0)
        self.declare_parameter("max_retry_count", 3)
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("allowed_area_topic", "/navigation/detected_allowed_area")
        self.declare_parameter(
            "path_source_use_planned_service", "/mission/path_source/use_planned"
        )
        self.declare_parameter(
            "path_source_use_recorded_service", "/mission/path_source/use_recorded"
        )
        self.declare_parameter("auto_recover_timeout", 5.0)
        self.declare_parameter("rpp_input_stale_grace", 2.0)
        self.declare_parameter("require_start_signal", False)
        self.declare_parameter("start_direction_topic", "/mission/start_direction_point")
        self.declare_parameter("min_start_direction_distance", 0.50)
        p = lambda n, d: self.get_parameter(n).get_parameter_value()
        self.heading_deg = float(self.get_parameter("heading_deg").value)
        self.sector_half_angle = p("sector_half_angle", None).double_value
        self.navigate_timeout = p("navigate_timeout", None).double_value
        self.goal_select_timeout = p("goal_select_timeout", None).double_value
        self.arrive_settle_duration = p("arrive_settle_duration", None).double_value
        self.end_hold_duration = p("end_hold_duration", None).double_value
        self.max_retry_count = p("max_retry_count", None).integer_value
        self.control_rate = p("control_rate", None).double_value
        self.allowed_area_topic = str(self.get_parameter("allowed_area_topic").value)
        self.path_source_use_planned_service = str(
            self.get_parameter("path_source_use_planned_service").value
        )
        self.path_source_use_recorded_service = str(
            self.get_parameter("path_source_use_recorded_service").value
        )
        self.auto_recover_timeout = float(
            self.get_parameter("auto_recover_timeout").value
        )
        self.rpp_input_stale_grace = float(
            self.get_parameter("rpp_input_stale_grace").value
        )
        self.require_start_signal = bool(
            self.get_parameter("require_start_signal").value
        )
        self.start_direction_topic = str(
            self.get_parameter("start_direction_topic").value
        )
        self.min_start_direction_distance = float(
            self.get_parameter("min_start_direction_distance").value
        )

    # -- callbacks --------------------------------------------------------
    def _on_odom(self, msg: Odometry):
        import tf2_geometry_msgs  # noqa — ensure quaternion helpers
        from tf2_ros import TransformBroadcaster  # not needed, use tf2_geometry_msgs
        self._robot = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny, cosy)
        if not self._start_recorded:
            self._start = self._robot
            # Extract yaw from odom quaternion
            self._start_yaw = self._robot_yaw
            self._start_recorded = True
            self._compute_map_direction()
            self.get_logger().info(
                f"start: ({self._start[0]:.2f},{self._start[1]:.2f}) yaw={math.degrees(self._start_yaw):.0f}°"
            )
            if self.require_start_signal:
                self.get_logger().warn(
                    "mission waiting for RViz Stage3 leg target point on "
                    f"{self.start_direction_topic} or Stage5 recorded path"
                )

    def _compute_map_direction(self):
        """Convert heading_deg (relative to start yaw) to map-frame unit vector."""
        rad = math.radians(self.heading_deg) + self._start_yaw
        self._direction = (math.cos(rad), math.sin(rad))
        if not self._direction_computed:
            self.get_logger().info(
                f"heading={self.heading_deg}° + start_yaw={math.degrees(self._start_yaw):.0f}° "
                f"→ map_dir=({self._direction[0]:.2f},{self._direction[1]:.2f})"
            )
            self._direction_computed = True

    def _on_path(self, msg: Path): pass

    def _on_markers(self, msg: MarkerArray):
        frontiers = []
        for marker in msg.markers:
            if marker.id == 0:
                for i, pt in enumerate(marker.points):
                    if i < len(marker.colors) and marker.colors[i].r > 0.9:
                        frontiers.append((pt.x, pt.y))
        self._frontiers = frontiers

    def _on_allowed_area(self, msg: PolygonStamped):
        count = 0
        for point in msg.polygon.points:
            if math.isfinite(point.x) and math.isfinite(point.y):
                count += 1
        was_ready = self._allowed_area_ready
        self._allowed_area_ready = count >= 3
        if self._allowed_area_ready and not was_ready:
            self.get_logger().info("allowed-area boundary ready for TRG growth constraint")

    def _has_forward_frontier(self):
        if not self._frontiers or not self._start_recorded:
            return False
        cos_limit = math.cos(self.sector_half_angle)
        for point in self._frontiers:
            dx = point[0] - self._start[0]
            dy = point[1] - self._start[1]
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            dot = (dx / dist) * self._direction[0] + (dy / dist) * self._direction[1]
            if dot >= cos_limit:
                return True
        return False

    def _on_rpp_state(self, msg: Int32): self._rpp_state = msg.data
    def _on_end_status(self, msg: Bool): self._is_at_end = msg.data

    def _on_start_direction(self, msg: PointStamped):
        if not self._start_recorded:
            self.get_logger().warn(
                "ignoring mission start point before odom/start pose is ready",
                throttle_duration_sec=2.0,
            )
            return
        if self._state != FsmState.INIT:
            self.get_logger().warn(
                "ignoring Stage3 leg target because mission_fsm is not waiting "
                f"in INIT (state={self._state.name})",
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
        self._direction = self._normalize((dx, dy))
        self.heading_deg = math.degrees(math.atan2(dy, dx) - self._robot_yaw)
        self._start_signal_received = True
        self._start_wait_logged = False
        self._current_goal_source = ""
        self._last_leg_result = "leg_target_received"
        self._call_path_source(self._cli_path_planned, "planned")
        self._publish_leg_status()
        self.get_logger().warn(
            "mission leg target accepted from RViz point: "
            f"leg={self._leg_count + 1}, "
            f"leg_start=({leg_start[0]:.2f},{leg_start[1]:.2f}), "
            f"point=({msg.point.x:.2f},{msg.point.y:.2f}), "
            f"map_dir=({self._direction[0]:.2f},{self._direction[1]:.2f}), "
            f"body_relative_heading={self.heading_deg:.1f}deg"
        )

    def _on_complete_leg(self, req: Trigger.Request, resp: Trigger.Response):
        del req
        if self._state == FsmState.DONE:
            resp.success = False
            resp.message = "mission already in DONE state"
            return resp
        if self._state == FsmState.INIT:
            resp.success = False
            resp.message = "no active leg; waiting for Stage3 target"
            return resp
        self._complete_leg("operator_complete_leg")
        resp.success = True
        resp.message = f"leg {self._leg_count} complete; waiting for Stage3/S5"
        return resp

    def _on_start_recorded_path(self, req: Trigger.Request, resp: Trigger.Response):
        del req
        if not self._start_recorded:
            resp.success = False
            resp.message = "cannot start recorded path before odom/start pose is ready"
            return resp
        if self._state == FsmState.DONE:
            resp.success = False
            resp.message = "mission already in DONE state"
            return resp
        if self._state != FsmState.INIT:
            resp.success = False
            resp.message = (
                "cannot start recorded path while a leg is active "
                f"(state={self._state.name})"
            )
            return resp

        self._start = self._robot
        self._start_yaw = self._robot_yaw
        self._start_signal_received = True
        self._waiting_for_leg_start = False
        self._start_wait_logged = False
        self._current_goal_source = "recorded_path"
        self._last_leg_result = "recorded_path_started"
        self._retry_count = 0
        self._select_future = None
        self._call_path_source(self._cli_path_recorded, "recorded")
        self._call_rpp(True)
        self._nav_start_time = self.get_clock().now()
        self._transition(FsmState.NAVIGATING)
        self._publish_leg_status()
        resp.success = True
        resp.message = f"recorded path leg {self._leg_count + 1} started"
        self.get_logger().warn(resp.message)
        return resp

    def _on_confirm_done(self, req: Trigger.Request, resp: Trigger.Response):
        if self._state == FsmState.DONE:
            resp.success = True
            resp.message = "mission confirmed complete"
            self.get_logger().warn("OPERATOR CONFIRMED MISSION COMPLETE")
        else:
            resp.success = False
            resp.message = f"not in DONE state (current: {self._state.name})"
        return resp

    def _on_mark_done(self, req: Trigger.Request, resp: Trigger.Response):
        self._call_rpp(False)
        self._transition(FsmState.DONE)
        resp.success = True
        resp.message = "EMERGENCY: marked done (penalty applies)"
        self.get_logger().warn("EMERGENCY mark_done triggered (penalty)")
        return resp

    # -- FSM timer --------------------------------------------------------
    def _on_timer(self):
        self._publish_state()
        self._publish_leg_count()
        handlers = {
            FsmState.INIT: self._handle_init,
            FsmState.FORWARD: self._handle_forward,
            FsmState.NAVIGATING: self._handle_navigating,
            FsmState.ARRIVED: self._handle_arrived,
            FsmState.CHECK_END: self._handle_check_end,
            FsmState.AUTO_RECOVER: self._handle_auto_recover,
            FsmState.RETRY: self._handle_retry,
        }
        handler = handlers.get(self._state)
        if handler:
            handler()

    def _handle_init(self):
        if not self._start_recorded:
            return
        if self._waiting_for_leg_start and not self._start_signal_received:
            if not self._start_wait_logged:
                self.get_logger().warn(
                    "INIT: waiting for Stage3 leg target or Stage5 recorded path before enabling RPP"
                )
                self._start_wait_logged = True
            return
        if not self._has_forward_frontier() and not self._allowed_area_ready:
            return
        self._transition(FsmState.FORWARD)

    def _handle_forward(self):
        now = self.get_clock().now()
        if self._select_future is None:
            self._select_future = self._call_select_goal()
            self._select_start_time = now
            if self._select_future is None:
                self._transition(FsmState.RETRY)
            return

        elapsed = (now - self._select_start_time).nanoseconds / 1e9
        if elapsed > self.goal_select_timeout:
            self.get_logger().warn(
                f"goal selection timeout ({elapsed:.1f}s); retrying"
            )
            self._select_future = None
            self._transition(FsmState.RETRY)
            return

        if not self._select_future.done():
            return

        future = self._select_future
        self._select_future = None
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().warn(f"goal selection failed: {exc}")
            self._transition(FsmState.RETRY)
            return

        if result is None or not result.success:
            message = "" if result is None else result.message
            self.get_logger().warn(f"goal selection rejected: {message}")
            self._transition(FsmState.RETRY)
            return

        self.get_logger().info(f"goal selection accepted: {result.message}")
        self._current_goal_source = self._goal_source_from_message(result.message)
        self._retry_count = 0
        self._call_rpp(True)
        self._nav_start_time = self.get_clock().now()
        self._transition(FsmState.NAVIGATING)

    def _handle_navigating(self):
        now = self.get_clock().now()
        elapsed = (now - self._nav_start_time).nanoseconds / 1e9
        if elapsed > self.navigate_timeout:
            self.get_logger().warn(f"nav timeout ({elapsed:.0f}s)")
            self._transition(FsmState.RETRY)
            return
        if self._rpp_state == RPP_GOAL_REACHED:
            self.get_logger().info("follower: goal reached")
            if self._current_goal_source == "operator_target":
                self._complete_leg("operator_target_reached")
                return
            if self._current_goal_source == "recorded_path":
                self._call_rpp(False)
                self._last_leg_result = "recorded_path_goal_reached_waiting_for_stage4"
                self._transition(FsmState.ARRIVED)
                return
            self._transition(FsmState.ARRIVED)
            return
        if self._rpp_state == RPP_INPUT_STALE and elapsed < self.rpp_input_stale_grace:
            return
        if self._rpp_state in (RPP_INPUT_STALE, RPP_PATH_DEVIATION):
            self.get_logger().warn(f"follower issue state={self._rpp_state}, auto-recovering")
            self._recover_target = FsmState.RETRY
            self._transition(FsmState.AUTO_RECOVER)

    def _handle_arrived(self):
        if self._current_goal_source == "recorded_path":
            return
        elapsed = (self.get_clock().now() - self._state_enter_time).nanoseconds / 1e9
        if elapsed < self.arrive_settle_duration:
            return
        self._transition(FsmState.CHECK_END)

    def _handle_check_end(self):
        elapsed = (self.get_clock().now() - self._state_enter_time).nanoseconds / 1e9
        if self._is_at_end:
            if elapsed >= self.end_hold_duration:
                self._complete_leg("auto_detected_end")
            return
        self._retry_count = 0
        self._transition(FsmState.FORWARD)

    def _handle_auto_recover(self):
        if self._rpp_state in (RPP_INPUT_STALE, RPP_PATH_DEVIATION):
            elapsed = (self.get_clock().now() - self._state_enter_time).nanoseconds / 1e9
            if elapsed > self.auto_recover_timeout:
                self.get_logger().warn("auto-recover timeout; retrying goal selection")
                self._transition(FsmState.RETRY)
            return  # still waiting
        if not self._start_recorded:
            return
        self._transition(self._recover_target)

    def _handle_retry(self):
        self._retry_count += 1
        if self._retry_count > self.max_retry_count:
            self.get_logger().warn(f"max retries ({self.max_retry_count}), checking end")
            self._transition(FsmState.CHECK_END)
            return
        self.get_logger().info(f"retry {self._retry_count}/{self.max_retry_count}")
        self._transition(FsmState.FORWARD)

    def _transition(self, new_state: FsmState):
        old = self._state
        self._state = new_state
        self._state_enter_time = self.get_clock().now()
        if old == FsmState.FORWARD and new_state != FsmState.FORWARD:
            self._select_future = None
        if old != new_state:
            if new_state in (FsmState.FORWARD, FsmState.RETRY, FsmState.CHECK_END):
                self._call_rpp(False)
            self._publish_state()
            self.get_logger().info(f"{old.name} -> {new_state.name}")
            self._publish_leg_status()

    def _publish_state(self):
        self._pub_state.publish(Int32(data=int(self._state)))

    def _publish_leg_count(self):
        self._pub_leg_count.publish(Int32(data=int(self._leg_count)))

    def _publish_leg_status(self):
        self._publish_leg_count()
        msg = String()
        msg.data = (
            f"leg_count={self._leg_count}; "
            f"state={self._state.name}; "
            f"goal_source={self._current_goal_source or 'none'}; "
            f"last_result={self._last_leg_result}"
        )
        self._pub_leg_status.publish(msg)

    def _complete_leg(self, reason: str):
        self._call_rpp(False)
        self._leg_count += 1
        self._last_leg_result = reason
        self._current_goal_source = ""
        self._retry_count = 0
        self._select_future = None
        self._is_at_end = False
        self._start_signal_received = False
        self._waiting_for_leg_start = True
        self._start_wait_logged = False
        self.get_logger().warn(
            f"LEG {self._leg_count} COMPLETE: {reason}; waiting for next Stage3/S5"
        )
        self._transition(FsmState.INIT)
        self._publish_leg_status()

    @staticmethod
    def _goal_source_from_message(message: str) -> str:
        prefix = "goal selected from "
        if not message.startswith(prefix):
            return ""
        rest = message[len(prefix):]
        return rest.split(" ", 1)[0].strip()

    def _call_select_goal(self):
        if not self._cli_select.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("goal_selector service unavailable")
            return None
        return self._cli_select.call_async(SetBool.Request(data=True))

    def _call_rpp(self, enable: bool):
        if not self._cli_rpp.wait_for_service(timeout_sec=0.5):
            return
        self._cli_rpp.call_async(SetBool.Request(data=enable))

    def _call_path_source(self, client, source: str):
        if not client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(
                f"path source mux unavailable while selecting {source}",
                throttle_duration_sec=2.0,
            )
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda fut, src=source: self._on_path_source_response(fut, src)
        )

    def _on_path_source_response(self, future, source: str):
        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - ROS callback safety
            self.get_logger().error(f"path source {source} service failed: {exc}")
            return
        message = (
            f"path source {source} response: "
            f"success={response.success} message={response.message}"
        )
        if response.success:
            self.get_logger().warn(message)
        else:
            self.get_logger().error(message)

    @staticmethod
    def _normalize(v):
        length = math.hypot(v[0], v[1])
        return (v[0] / length, v[1] / length) if length > 1e-9 else (1.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = MissionFSM()
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
