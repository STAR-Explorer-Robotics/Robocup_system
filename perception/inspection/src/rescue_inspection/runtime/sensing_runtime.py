"""Stable facade for external single-frame sensing callers."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rescue_inspection.config import AppConfig, load_config
from rescue_inspection.models import FramePacket
from rescue_inspection.runtime.frame_metadata import SensingFrameMetadata
from rescue_inspection.runtime.paths import resolve_runtime_config_path
from rescue_inspection.runtime.sensing_schema import SensingFrameResult


class SensingRuntime:
    """Thin facade around the current combined QR + Hazmat frame processor."""

    def __init__(self, config: AppConfig, *, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.config = _sensing_config(config)
        self._frame_index = 0

        from rescue_inspection.tasks.sensing import SensingFrameProcessor

        self._processor = SensingFrameProcessor(self.config, self.logger)

    @classmethod
    def from_config_path(cls, config_path: str | Path, *, logger: logging.Logger | None = None) -> "SensingRuntime":
        """Build a sensing runtime from an inspection config path."""

        return cls(load_config(resolve_runtime_config_path(config_path)), logger=logger)

    @classmethod
    def from_config(cls, config: AppConfig, *, logger: logging.Logger | None = None) -> "SensingRuntime":
        """Build a sensing runtime from an already loaded config."""

        return cls(config, logger=logger)

    def process_bgr_frame(
        self,
        image_bgr: np.ndarray,
        *,
        frame_metadata: SensingFrameMetadata | None = None,
        render_overlay: bool = True,
    ) -> SensingFrameResult:
        """Run sensing on one processed frame.

        Input Frame Contract:
          - image_bgr: OpenCV standard HWC BGR image. MUST be a 3D numpy array
            of shape (height, width, 3).
          - Data type MUST be np.uint8.

        Output Result Contract:
          - Returns a SensingFrameResult.
          - Callers should convert this result to a public, JSON-safe structure
            using `result_to_payload` or `result_to_json`. The serialized representation
            is guaranteed to contain only standard JSON-safe types (no numpy arrays,
            OpenCV objects, or internal detector references).
        """

        _validate_image(image_bgr)
        metadata = frame_metadata or SensingFrameMetadata()
        stamp_sec, stamp_nanosec = _resolve_stamp(metadata)
        packet = FramePacket(
            frame_id=self._frame_index,
            timestamp=_timestamp_iso(stamp_sec, stamp_nanosec),
            image=image_bgr,
            source_type=metadata.source_type,
        )
        self._frame_index += 1

        started = time.perf_counter()
        frame_output = self._processor.process_packet(packet, render_overlay=render_overlay, debug_artifact_dir=None)
        processing_time_ms = (time.perf_counter() - started) * 1000.0
        height, width = image_bgr.shape[:2]
        return SensingFrameResult(
            stamp_sec=stamp_sec,
            stamp_nanosec=stamp_nanosec,
            frame_id=metadata.frame_id,
            source_topic=metadata.source_topic,
            processing_time_ms=processing_time_ms,
            image_width=width,
            image_height=height,
            qr_result=frame_output.qr_result,
            qr_vote=frame_output.qr_vote,
            hazmat_result=frame_output.hazmat_result,
            hazmat_vote=frame_output.hazmat_vote,
            overlay_bgr=frame_output.overlay,
            apriltag_result=getattr(frame_output, "apriltag_result", None),
            apriltag_vote=getattr(frame_output, "apriltag_vote", None),
        )


def _sensing_config(config: AppConfig) -> AppConfig:
    return config.model_copy(update={"task": config.task.model_copy(update={"type": "sensing"})})


def _validate_image(image_bgr: np.ndarray) -> None:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected OpenCV BGR/HWC image with 3 channels, got shape={image_bgr.shape}")
    if image_bgr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 BGR image, got dtype={image_bgr.dtype}")


def _resolve_stamp(metadata: SensingFrameMetadata) -> tuple[int, int]:
    if metadata.stamp_sec is not None and metadata.stamp_nanosec is not None:
        return int(metadata.stamp_sec), int(metadata.stamp_nanosec)
    now = time.time()
    sec = int(now)
    return sec, int((now - sec) * 1_000_000_000)


def _timestamp_iso(sec: int, nanosec: int) -> str:
    dt = datetime.fromtimestamp(sec + nanosec / 1_000_000_000.0, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")
