"""Status counters and payload construction for qrc_sensing_node."""

from __future__ import annotations

import time

from qrc_ros2_perception.config import PerceptionBridgeConfig


class SensingNodeStatus:
    """Mutable counters used by the ROS2 sensing node."""

    def __init__(self, *, config: PerceptionBridgeConfig, source_topic: str) -> None:
        self.source_topic = source_topic
        self.busy = False
        self.last_processed_monotonic = 0.0
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.last_error = ""
        self.last_processing_time_ms = 0.0
        self.started_monotonic = time.monotonic()
        self.shutdown_requested = False
        self.pyrealsense_requested_profile: dict[str, object] | None = None
        self.pyrealsense_actual_profile: dict[str, object] | None = None
        self._config = config

    def build_payload(
        self,
        *,
        camera_passthrough_frames_published: int,
        camera_passthrough_frames_skipped_no_subscriber: int,
        camera_passthrough_subscription_count: int,
    ) -> dict[str, object]:
        config = self._config
        elapsed = max(time.monotonic() - self.started_monotonic, 1e-6)
        return {
            "node": "qrc_sensing_node",
            "input_mode": config.input_mode,
            "image_topic": config.image_topic,
            "source_topic": self.source_topic,
            "inspection_config": str(config.inspection_config),
            "pyrealsense_requested_profile": self.pyrealsense_requested_profile,
            "pyrealsense_actual_profile": self.pyrealsense_actual_profile,
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "max_frames": config.max_frames,
            "received_fps": self.frames_received / elapsed,
            "processing_fps": self.frames_processed / elapsed,
            "last_processing_time_ms": self.last_processing_time_ms,
            "last_error": self.last_error,
            "publish_overlay": config.publish_overlay,
            "overlay_publish_fps": config.overlay_publish_fps,
            "enable_local_overlay_window": config.enable_local_overlay_window,
            "local_overlay_window_fps": config.local_overlay_window_fps,
            "publish_camera_passthrough": config.publish_camera_passthrough,
            "camera_passthrough_topic": config.camera_passthrough_topic,
            "camera_passthrough_fps": config.camera_passthrough_fps,
            "camera_passthrough_frame_id": config.camera_passthrough_frame_id,
            "camera_passthrough_encoding": config.camera_passthrough_encoding,
            "camera_passthrough_frames_published": camera_passthrough_frames_published,
            "camera_passthrough_frames_skipped_no_subscriber": (
                camera_passthrough_frames_skipped_no_subscriber
            ),
            "camera_passthrough_subscription_count": camera_passthrough_subscription_count,
            "publish_result_json": config.publish_result_json,
            "publish_latest_confirmed_json": config.publish_latest_confirmed_json,
        }
