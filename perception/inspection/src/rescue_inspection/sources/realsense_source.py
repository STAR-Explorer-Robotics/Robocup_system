"""Optional Intel RealSense source adapter."""

from __future__ import annotations

import numpy as np

from rescue_inspection.models import FramePacket
from rescue_inspection.sources.base import FrameSource
from rescue_inspection.utils.time_utils import iso_timestamp


class RealSenseSource(FrameSource):
    """Read color frames from a RealSense device when pyrealsense2 is available."""

    def __init__(self, width: int, height: int, fps: int, frame_stride: int = 1) -> None:
        super().__init__(source_type="realsense")
        self.width = width
        self.height = height
        self._fps = fps
        self.frame_stride = max(1, frame_stride)
        self._frame_id = 0
        self._rs = None
        self._pipeline = None

    @property
    def fps(self) -> float | None:
        return float(self._fps)

    def open(self) -> None:
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError(
                "RealSense support is unavailable because pyrealsense2 is not installed."
            ) from exc

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self._fps)
        self._pipeline.start(config)
        self._frame_id = 0

    def read(self) -> FramePacket | None:
        if self._pipeline is None or self._rs is None:
            return None

        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        image = np.asanyarray(color_frame.get_data())
        packet = FramePacket(
            frame_id=self._frame_id,
            timestamp=iso_timestamp(),
            source_time_sec=float(color_frame.get_timestamp()) / 1000.0,
            image=image,
            source_type=self.source_type,
        )
        self._frame_id += 1

        for _ in range(self.frame_stride - 1):
            self._pipeline.wait_for_frames()
            self._frame_id += 1

        return packet

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._rs = None
