#!/usr/bin/env python3
"""Select whether the follower tracks TRG planning or the recorded path."""

from __future__ import annotations

from typing import Optional

import rclpy
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class PathSourceMux(Node):
    def __init__(self) -> None:
        super().__init__("path_source_mux")
        self._load_params()

        self._source = "planned"
        self._latest_planned: Optional[Path] = None
        self._latest_recorded: Optional[Path] = None

        self._sub_planned = self.create_subscription(
            Path, self.planned_path_topic, self._on_planned_path, 10
        )
        self._sub_recorded = self.create_subscription(
            Path, self.recorded_path_topic, self._on_recorded_path, 10
        )
        self._pub_output = self.create_publisher(Path, self.output_path_topic, 10)
        self._pub_status = self.create_publisher(String, self.status_topic, 10)

        self._srv_planned = self.create_service(
            Trigger, self.use_planned_service, self._on_use_planned
        )
        self._srv_recorded = self.create_service(
            Trigger, self.use_recorded_service, self._on_use_recorded
        )
        self._timer = self.create_timer(1.0, self._publish_status)

        self.get_logger().info(
            "path_source_mux ready: "
            f"planned={self.planned_path_topic}, recorded={self.recorded_path_topic}, "
            f"output={self.output_path_topic}, source={self._source}"
        )

    def _load_params(self) -> None:
        self.declare_parameter("planned_path_topic", "/trg/output/path")
        self.declare_parameter("recorded_path_topic", "/mission/recorded_path/active")
        self.declare_parameter("output_path_topic", "/mission/follower_path")
        self.declare_parameter("status_topic", "/mission/path_source/status")
        self.declare_parameter("use_planned_service", "/mission/path_source/use_planned")
        self.declare_parameter("use_recorded_service", "/mission/path_source/use_recorded")

        self.planned_path_topic = str(self.get_parameter("planned_path_topic").value)
        self.recorded_path_topic = str(self.get_parameter("recorded_path_topic").value)
        self.output_path_topic = str(self.get_parameter("output_path_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.use_planned_service = str(self.get_parameter("use_planned_service").value)
        self.use_recorded_service = str(self.get_parameter("use_recorded_service").value)

    def _on_planned_path(self, msg: Path) -> None:
        self._latest_planned = msg
        if self._source == "planned":
            self._publish_path(msg)

    def _on_recorded_path(self, msg: Path) -> None:
        self._latest_recorded = msg
        if self._source == "recorded":
            self._publish_path(msg)

    def _on_use_planned(
        self, req: Trigger.Request, resp: Trigger.Response
    ) -> Trigger.Response:
        del req
        self._source = "planned"
        if self._latest_planned is not None:
            self._publish_path(self._latest_planned)
            resp.message = (
                f"using planned path with {len(self._latest_planned.poses)} points"
            )
        else:
            resp.message = "using planned path; waiting for next TRG path"
        resp.success = True
        self.get_logger().warn(resp.message)
        self._publish_status()
        return resp

    def _on_use_recorded(
        self, req: Trigger.Request, resp: Trigger.Response
    ) -> Trigger.Response:
        del req
        if self._latest_recorded is None or len(self._latest_recorded.poses) < 2:
            resp.success = False
            resp.message = "recorded path source unavailable"
            self.get_logger().error(resp.message)
            self._publish_status()
            return resp
        self._source = "recorded"
        self._publish_path(self._latest_recorded)
        resp.success = True
        resp.message = (
            f"using recorded path with {len(self._latest_recorded.poses)} points"
        )
        self.get_logger().warn(resp.message)
        self._publish_status()
        return resp

    def _publish_path(self, msg: Path) -> None:
        out = Path()
        out.header = msg.header
        out.header.stamp = self.get_clock().now().to_msg()
        out.poses = msg.poses
        self._pub_output.publish(out)

    def _publish_status(self) -> None:
        msg = String()
        planned_count = 0 if self._latest_planned is None else len(self._latest_planned.poses)
        recorded_count = (
            0 if self._latest_recorded is None else len(self._latest_recorded.poses)
        )
        msg.data = (
            f"source={self._source}; "
            f"planned_points={planned_count}; "
            f"recorded_points={recorded_count}; "
            f"output={self.output_path_topic}"
        )
        self._pub_status.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathSourceMux()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
