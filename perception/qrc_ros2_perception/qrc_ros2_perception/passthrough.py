"""Raw camera passthrough publishing for direct RealSense mode."""

from __future__ import annotations

import time
from typing import Any

from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from qrc_ros2_perception.config import PerceptionBridgeConfig
from qrc_ros2_perception.image_codec import bgr_to_image_msg


class CameraPassthrough:
    """Throttle and publish raw camera frames without overlay decoration."""

    def __init__(self, *, node: Any, config: PerceptionBridgeConfig) -> None:
        self._config = config
        self.publisher = (
            node.create_publisher(Image, config.camera_passthrough_topic, qos_profile_sensor_data)
            if config.publish_camera_passthrough
            else None
        )
        self.last_publish_monotonic = 0.0
        self.frames_published = 0
        self.frames_skipped_no_subscriber = 0

    def should_publish_now(self) -> bool:
        if self.publisher is None:
            return False
        fps = max(float(self._config.camera_passthrough_fps), 0.0)
        now = time.monotonic()
        if fps <= 0.0:
            self.last_publish_monotonic = now
            return True
        min_interval = 1.0 / fps
        if now - self.last_publish_monotonic >= min_interval:
            self.last_publish_monotonic = now
            return True
        return False

    def publish_raw_frame(self, image_bgr, *, header) -> None:
        if self.publisher is None:
            return
        if self.publisher.get_subscription_count() <= 0:
            self.frames_skipped_no_subscriber += 1
            return
        self.publisher.publish(bgr_to_image_msg(image_bgr, header=header))
        self.frames_published += 1

    def subscription_count(self) -> int:
        if self.publisher is None:
            return 0
        return int(self.publisher.get_subscription_count())
