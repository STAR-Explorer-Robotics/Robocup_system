"""ROS2 node for the latest confirmed result operator panel."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from qrc_ros2_perception.image_codec import bgr_to_image_msg, image_msg_to_bgr
from qrc_ros2_perception.local_overlay_window import opencv_window_flags
from qrc_ros2_perception.result_panel import render_latest_confirmed_panel
from qrc_ros2_perception.result_panel_config import (
    ResultPanelConfig,
    apply_result_panel_overrides,
    load_result_panel_config,
)
from qrc_ros2_perception.result_panel_payload import parse_json_object


class QrcResultPanelNode(Node):
    """Display sticky latest confirmed sensing results at low refresh rate."""

    def __init__(self) -> None:
        super().__init__("qrc_result_panel")
        self.declare_parameter("config", "")
        config_path = str(self.get_parameter("config").value).strip()
        base_config = load_result_panel_config(Path(config_path) if config_path else None)
        self.config = self._declare_and_load_parameters(base_config)

        self._latest_payload: dict[str, Any] | None = None
        self._latest_evidence_image_bgr = None
        self._status_payload: dict[str, Any] = {}
        self._window_enabled = bool(self.config.show_window)
        self._window_created = False
        self._window_warning_logged = False

        self._latest_subscription = self.create_subscription(
            String,
            self.config.latest_confirmed_topic,
            self._handle_latest_confirmed,
            latest_confirmed_qos(),
        )
        self._latest_overlay_subscription = self.create_subscription(
            Image,
            self.config.latest_confirmed_overlay_topic,
            self._handle_latest_confirmed_overlay,
            latest_confirmed_qos(),
        )
        self._status_subscription = self.create_subscription(
            String,
            self.config.status_topic,
            self._handle_status,
            10,
        )
        self._panel_publisher = (
            self.create_publisher(Image, self.config.panel_image_topic, 10)
            if self.config.publish_panel_image
            else None
        )
        refresh_period = 1.0 / max(float(self.config.refresh_fps), 0.1)
        self._refresh_timer = self.create_timer(refresh_period, self._refresh_panel)

        self.get_logger().info(
            "QRC latest confirmed panel ready: "
            f"latest_confirmed_topic={self.config.latest_confirmed_topic} "
            f"latest_confirmed_overlay_topic={self.config.latest_confirmed_overlay_topic} "
            f"status_topic={self.config.status_topic} "
            f"panel_image_topic={self.config.panel_image_topic} "
            f"publish_panel_image={self.config.publish_panel_image} "
            f"show_window={self.config.show_window} "
            f"refresh_fps={self.config.refresh_fps}"
        )

    def _declare_and_load_parameters(self, config: ResultPanelConfig) -> ResultPanelConfig:
        self.declare_parameter("latest_confirmed_topic", config.latest_confirmed_topic)
        self.declare_parameter("latest_confirmed_overlay_topic", config.latest_confirmed_overlay_topic)
        self.declare_parameter("status_topic", config.status_topic)
        self.declare_parameter("panel_image_topic", config.panel_image_topic)
        self.declare_parameter("publish_panel_image", config.publish_panel_image)
        self.declare_parameter("show_window", config.show_window)
        self.declare_parameter("window_name", config.window_name)
        self.declare_parameter("refresh_fps", config.refresh_fps)
        self.declare_parameter("stale_threshold_sec", config.stale_threshold_sec)
        self.declare_parameter("panel_width", config.panel_width)
        self.declare_parameter("panel_height", config.panel_height)

        return apply_result_panel_overrides(
            config,
            latest_confirmed_topic=_parameter_text(self, "latest_confirmed_topic", config.latest_confirmed_topic),
            latest_confirmed_overlay_topic=_parameter_text(
                self,
                "latest_confirmed_overlay_topic",
                config.latest_confirmed_overlay_topic,
            ),
            status_topic=_parameter_text(self, "status_topic", config.status_topic),
            panel_image_topic=_parameter_text(self, "panel_image_topic", config.panel_image_topic),
            publish_panel_image=bool(self.get_parameter("publish_panel_image").value),
            show_window=bool(self.get_parameter("show_window").value),
            window_name=_parameter_text(self, "window_name", config.window_name),
            refresh_fps=float(self.get_parameter("refresh_fps").value),
            stale_threshold_sec=float(self.get_parameter("stale_threshold_sec").value),
            panel_width=int(self.get_parameter("panel_width").value),
            panel_height=int(self.get_parameter("panel_height").value),
        )

    def _handle_latest_confirmed(self, message: String) -> None:
        payload = parse_json_object(message.data)
        if payload is None:
            self._status_payload["last_panel_error"] = "Malformed latest_confirmed_json payload"
            return
        payload = dict(payload)
        payload["_panel_received_at_sec"] = time.time()
        self._latest_payload = payload
        self._status_payload.pop("last_panel_error", None)

    def _handle_latest_confirmed_overlay(self, message: Image) -> None:
        try:
            self._latest_evidence_image_bgr = image_msg_to_bgr(message)
        except Exception as exc:
            self._status_payload["last_panel_error"] = f"Malformed latest confirmed overlay image: {exc}"

    def _handle_status(self, message: String) -> None:
        payload = parse_json_object(message.data)
        if payload is None:
            self._status_payload["last_status_error"] = "Malformed status payload"
            return
        self._status_payload = payload

    def _refresh_panel(self) -> None:
        panel = render_latest_confirmed_panel(
            self._latest_payload,
            evidence_image_bgr=self._latest_evidence_image_bgr,
            status=self._status_payload,
            now=time.time(),
            width=self.config.panel_width,
            height=self.config.panel_height,
            stale_threshold_sec=self.config.stale_threshold_sec,
        )
        if self._panel_publisher is not None:
            self._panel_publisher.publish(bgr_to_image_msg(panel))
        if self._window_enabled:
            self._show_window(panel)

    def _show_window(self, panel_bgr) -> None:
        try:
            if not self._window_created:
                cv2.namedWindow(self.config.window_name, opencv_window_flags())
                self._window_created = True
            cv2.imshow(self.config.window_name, panel_bgr)
            cv2.waitKey(1)
        except cv2.error as exc:
            self._window_enabled = False
            if not self._window_warning_logged:
                self.get_logger().warning(f"Disabling local result panel window: {exc}")
                self._window_warning_logged = True

    def destroy_node(self) -> bool:
        if self._window_created:
            try:
                cv2.destroyWindow(self.config.window_name)
            except cv2.error:
                pass
        return super().destroy_node()


def latest_confirmed_qos() -> QoSProfile:
    """QoS profile compatible with sensing node latest confirmed publisher."""

    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


def _parameter_text(node: Node, name: str, default: str) -> str:
    text = str(node.get_parameter(name).value).strip()
    return text or default


def main(args: list[str] | None = None) -> None:
    """ROS2 entry point."""

    node: QrcResultPanelNode | None = None
    rclpy.init(args=args)
    try:
        node = QrcResultPanelNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":  # pragma: no cover
    main()
