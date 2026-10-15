"""Input source management for qrc_sensing_node."""

from __future__ import annotations

from typing import Any

import rclpy
from rclpy.qos import (
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from qrc_ros2_perception.config import PerceptionBridgeConfig
from qrc_ros2_perception.image_codec import image_msg_to_bgr
from qrc_ros2_perception.pyrealsense_source import PyRealSense2ColorSource
from qrc_ros2_perception.status import SensingNodeStatus


def resolve_source_topic(config: PerceptionBridgeConfig) -> str:
    if config.input_mode == "pyrealsense2":
        return "pyrealsense2://color"
    return config.image_topic


class QrcInputManager:
    """Start, poll, and stop the configured sensing input source."""

    def __init__(
        self,
        *,
        node: Any,
        config: PerceptionBridgeConfig,
        status: SensingNodeStatus,
        processor: Any,
        passthrough: Any,
        source_topic: str,
    ) -> None:
        self._node = node
        self._config = config
        self._status = status
        self._processor = processor
        self._passthrough = passthrough
        self._source_topic = source_topic
        self.image_subscription = None
        self.pyrealsense_source: PyRealSense2ColorSource | None = None
        self.pyrealsense_timer = None

    def start(self) -> None:
        if self._config.input_mode == "ros_image_topic":
            qos_profile = build_input_qos_profile(self._config.input_qos)
            self.image_subscription = self._node.create_subscription(
                Image,
                self._config.image_topic,
                self.handle_image,
                qos_profile,
            )
            return

        if self._config.input_mode != "pyrealsense2":
            raise ValueError(f"Unsupported input_mode: {self._config.input_mode}")

        source = PyRealSense2ColorSource(
            width=self._config.pyrealsense_width,
            height=self._config.pyrealsense_height,
            fps=self._config.pyrealsense_fps,
            serial=self._config.pyrealsense_serial,
            color_format=self._config.pyrealsense_format,
        )
        self._status.pyrealsense_requested_profile = source.requested_profile
        self._node.get_logger().info(
            "requested pyrealsense2 profile: "
            f"{self._config.pyrealsense_width}x{self._config.pyrealsense_height}"
            f"@{self._config.pyrealsense_fps} {self._config.pyrealsense_format}"
            f" serial={self._config.pyrealsense_serial or ''}"
        )
        source.open()
        self.pyrealsense_source = source
        self._status.pyrealsense_actual_profile = source.actual_profile
        self._node.get_logger().info(f"actual pyrealsense2 profile: {self._status.pyrealsense_actual_profile}")
        self.pyrealsense_timer = self._node.create_timer(1.0 / self._config.pyrealsense_fps, self.poll_pyrealsense_frame)

    def handle_image(self, message: Image) -> None:
        self._status.frames_received += 1
        if self._status.busy and self._config.drop_frames_when_busy:
            self._status.frames_dropped += 1
            return
        if self._processor.should_throttle():
            self._status.frames_dropped += 1
            return

        self._status.busy = True
        try:
            image_bgr = image_msg_to_bgr(message)
            self._processor.process_bgr_frame(image_bgr, header=message.header, source_topic=self._config.image_topic)
        except Exception as exc:
            self._status.last_error = str(exc)
            if rclpy.ok():
                self._node.get_logger().error(f"Failed to process sensing frame: {exc}")
        finally:
            self._status.busy = False

    def poll_pyrealsense_frame(self) -> None:
        if self.pyrealsense_source is None:
            return
        if self._status.busy:
            self._status.frames_dropped += 1
            return

        self._status.busy = True
        try:
            frame = self.pyrealsense_source.read_bgr_frame()
            if frame is None:
                return
            self._status.frames_received += 1
            if self._processor.should_throttle():
                self._status.frames_dropped += 1
                return

            header = Header()
            header.stamp = self._node.get_clock().now().to_msg()
            header.frame_id = "pyrealsense2_color"
            passthrough_header = Header()
            passthrough_header.stamp = header.stamp
            passthrough_header.frame_id = self._config.camera_passthrough_frame_id
            if self._passthrough.should_publish_now():
                self._passthrough.publish_raw_frame(frame.image_bgr, header=passthrough_header)
            self._processor.process_bgr_frame(frame.image_bgr, header=header, source_topic=self._source_topic)
        except Exception as exc:
            self._status.last_error = str(exc)
            if rclpy.ok():
                self._node.get_logger().error(f"Failed to process pyrealsense2 sensing frame: {exc}")
        finally:
            self._status.busy = False

    def stop(self) -> None:
        if self.pyrealsense_timer is not None:
            try:
                self._node.destroy_timer(self.pyrealsense_timer)
            except Exception:
                pass
            self.pyrealsense_timer = None
        if self.pyrealsense_source is not None:
            try:
                self.pyrealsense_source.close()
            except Exception as exc:
                self._node.get_logger().warning(f"Failed to stop pyrealsense2 source: {exc}")
            self.pyrealsense_source = None


def build_input_qos_profile(input_qos: str) -> QoSProfile:
    normalized = input_qos.strip().lower()
    if normalized in {"sensor_data", "best_effort"}:
        return qos_profile_sensor_data
    if normalized == "reliable":
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
    raise ValueError("input_qos must be one of: sensor_data, best_effort, reliable")
