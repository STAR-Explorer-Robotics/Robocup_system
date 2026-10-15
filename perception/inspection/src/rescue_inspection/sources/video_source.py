"""Video file source adapter."""

from __future__ import annotations

from pathlib import Path

import cv2

from rescue_inspection.models import FramePacket
from rescue_inspection.sources.base import FrameSource
from rescue_inspection.utils.time_utils import iso_timestamp


class VideoSource(FrameSource):
    """Read frames from a video file with configurable sampling."""

    def __init__(self, path: Path, frame_stride: int = 1) -> None:
        super().__init__(source_type="video")
        self.path = Path(path)
        self.frame_stride = max(1, frame_stride)
        self._capture: cv2.VideoCapture | None = None
        self._frame_id = 0
        self._fps = 0.0

    @property
    def fps(self) -> float | None:
        return self._fps or None

    def open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise ValueError(f"Failed to open video file: {self.path}")

        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._frame_id = 0

    def read(self) -> FramePacket | None:
        if self._capture is None:
            return None

        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None

        packet = FramePacket(
            frame_id=self._frame_id,
            timestamp=iso_timestamp(),
            source_time_sec=float(self._capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0,
            image=frame,
            source_type=self.source_type,
        )
        self._frame_id += 1

        for _ in range(self.frame_stride - 1):
            if not self._capture.grab():
                break
            self._frame_id += 1

        return packet

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
