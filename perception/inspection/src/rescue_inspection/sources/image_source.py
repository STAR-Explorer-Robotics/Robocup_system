"""Single-image source adapter."""

from __future__ import annotations

from pathlib import Path

import cv2

from rescue_inspection.models import FramePacket
from rescue_inspection.sources.base import FrameSource
from rescue_inspection.utils.time_utils import iso_timestamp


class ImageSource(FrameSource):
    """Emit a single image as one frame packet."""

    def __init__(self, path: Path) -> None:
        super().__init__(source_type="image")
        self.path = Path(path)
        self._image = None
        self._emitted = False

    def open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Image file does not exist: {self.path}")

        image = cv2.imread(str(self.path))
        if image is None:
            raise ValueError(f"Failed to read image file: {self.path}")

        self._image = image
        self._emitted = False

    def read(self) -> FramePacket | None:
        if self._image is None or self._emitted:
            return None

        self._emitted = True
        return FramePacket(
            frame_id=0,
            timestamp=iso_timestamp(),
            source_time_sec=None,
            image=self._image.copy(),
            source_type=self.source_type,
        )

    def close(self) -> None:
        self._image = None
