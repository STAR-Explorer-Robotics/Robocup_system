"""ROS2 publishers for qrc_sensing_node outputs."""

from __future__ import annotations

import json
import time
from typing import Any

from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String

from qrc_ros2_perception.config import PerceptionBridgeConfig
from qrc_ros2_perception.image_codec import bgr_to_image_msg


class QrcSensingPublishers:
    """Own the sensing node's ROS output publishers and publisher-side throttles."""

    def __init__(self, *, node: Any, config: PerceptionBridgeConfig) -> None:
        self._config = config
        self.overlay = node.create_publisher(Image, config.overlay_topic, 10) if config.publish_overlay else None
        self.result = node.create_publisher(String, config.result_topic, 10) if config.publish_result_json else None
        self.latest_confirmed = (
            node.create_publisher(String, config.latest_confirmed_topic, latest_confirmed_qos())
            if config.publish_latest_confirmed_json
            else None
        )
        self.latest_confirmed_overlay = (
            node.create_publisher(Image, config.latest_confirmed_overlay_topic, latest_confirmed_qos())
            if config.publish_latest_confirmed_json
            else None
        )
        self.status = node.create_publisher(String, config.status_topic, 10)
        self.last_overlay_publish_monotonic = 0.0

    def should_publish_overlay_now(self) -> bool:
        if self.overlay is None:
            return False
        fps = max(float(self._config.overlay_publish_fps), 0.0)
        now = time.monotonic()
        if fps <= 0.0:
            self.last_overlay_publish_monotonic = now
            return True
        min_interval = 1.0 / fps
        if now - self.last_overlay_publish_monotonic >= min_interval:
            self.last_overlay_publish_monotonic = now
            return True
        return False

    def publish_overlay(self, image_bgr, *, header) -> None:
        if self.overlay is not None and image_bgr is not None:
            self.overlay.publish(bgr_to_image_msg(image_bgr, header=header))

    def publish_result_json(self, result_json: str) -> None:
        if self.result is not None:
            self.result.publish(String(data=result_json))

    def publish_latest_confirmed_event(self, payload: dict[str, object]) -> None:
        if self.latest_confirmed is None:
            return
        self.latest_confirmed.publish(String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))

    def publish_latest_confirmed_overlay(self, image_bgr, *, header) -> None:
        if self.latest_confirmed_overlay is None or image_bgr is None:
            return
        self.latest_confirmed_overlay.publish(bgr_to_image_msg(image_bgr, header=header))

    def publish_status(self, status_json: str) -> None:
        self.status.publish(String(data=status_json))


def latest_confirmed_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )
