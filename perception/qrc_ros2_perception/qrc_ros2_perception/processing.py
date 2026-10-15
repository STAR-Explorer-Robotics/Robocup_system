"""Single-frame processing orchestration for qrc_sensing_node."""

from __future__ import annotations

import time
from typing import Any, Callable

from qrc_ros2_perception.config import PerceptionBridgeConfig
from qrc_ros2_perception.result_schema import result_to_json
from qrc_ros2_perception.status import SensingNodeStatus


class FrameProcessor:
    """Run the bridge on BGR frames and fan out results to publishers/artifacts."""

    def __init__(
        self,
        *,
        config: PerceptionBridgeConfig,
        bridge: Any,
        publishers: Any,
        status: SensingNodeStatus,
        outputs: Any | None,
        local_overlay_window: Any | None,
        request_shutdown_if_done: Callable[[], None],
    ) -> None:
        self._config = config
        self._bridge = bridge
        self._publishers = publishers
        self._status = status
        self._outputs = outputs
        self._local_overlay_window = local_overlay_window
        self._request_shutdown_if_done = request_shutdown_if_done
        self.last_local_overlay_show_monotonic = 0.0

    def process_bgr_frame(self, image_bgr, *, header, source_topic: str) -> None:
        should_publish_ros_overlay = self._publishers.should_publish_overlay_now()
        should_show_local_overlay = self.should_show_local_overlay_now()
        should_write_overlay_video = self._outputs is not None and self._config.save_overlay_video
        should_render_overlay = should_publish_ros_overlay or should_show_local_overlay or should_write_overlay_video
        result = self._bridge.process_bgr_frame(
            image_bgr,
            stamp=header.stamp,
            frame_id=header.frame_id,
            source_topic=source_topic,
            render_overlay=should_render_overlay,
        )
        self._status.frames_processed += 1
        self._status.last_processed_monotonic = time.monotonic()
        self._status.last_processing_time_ms = result.processing_time_ms
        self._status.last_error = ""
        result_json = result_to_json(result)

        if self._local_overlay_window is not None and should_show_local_overlay and result.overlay_bgr is not None:
            self._local_overlay_window.show(result.overlay_bgr)
        if should_publish_ros_overlay and result.overlay_bgr is not None:
            self._publishers.publish_overlay(result.overlay_bgr, header=header)
        self._publishers.publish_result_json(result_json)
        if self._outputs is not None:
            record_result = self._outputs.record_frame_result(
                result,
                frame_bgr=image_bgr,
                overlay_bgr=result.overlay_bgr,
            )
            if record_result.latest_refresh is not None:
                self._publishers.publish_latest_confirmed_event(record_result.latest_refresh.payload)
                self._publishers.publish_latest_confirmed_overlay(
                    record_result.latest_refresh.evidence_overlay_bgr,
                    header=header,
                )
            if result.overlay_bgr is not None:
                self._outputs.write_overlay(result.overlay_bgr)
        self._request_shutdown_if_done()

    def should_throttle(self) -> bool:
        if not self._config.max_processing_fps:
            return False
        min_interval = 1.0 / self._config.max_processing_fps
        return (time.monotonic() - self._status.last_processed_monotonic) < min_interval

    def should_show_local_overlay_now(self) -> bool:
        if self._local_overlay_window is None:
            return False
        if not getattr(self._local_overlay_window, "enabled", True):
            return False
        fps = max(float(self._config.local_overlay_window_fps), 0.0)
        now = time.monotonic()
        if fps <= 0.0:
            self.last_local_overlay_show_monotonic = now
            return True
        min_interval = 1.0 / fps
        if now - self.last_local_overlay_show_monotonic >= min_interval:
            self.last_local_overlay_show_monotonic = now
            return True
        return False
