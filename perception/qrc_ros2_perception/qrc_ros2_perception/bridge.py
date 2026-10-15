"""Non-ROS-heavy bridge around the inspection single-frame sensing processor."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from rescue_inspection.runtime import SensingFrameMetadata, SensingFrameResult, SensingRuntime


class QrcSensingBridge:
    """Adapt BGR frames into the current inspection sensing pipeline."""

    def __init__(self, config_path: str | Path, *, logger: logging.Logger | None = None) -> None:
        self.config_path = Path(config_path)
        self.logger = logger or logging.getLogger(__name__)
        self.runtime = SensingRuntime.from_config_path(self.config_path, logger=self.logger)
        self.config = self.runtime.config

    def process_bgr_frame(
        self,
        image_bgr: np.ndarray,
        *,
        stamp: Any = None,
        frame_id: str | None = None,
        source_topic: str = "",
        render_overlay: bool = True,
    ) -> SensingFrameResult:
        """Run inspection sensing on one OpenCV BGR/HWC/uint8 frame."""

        stamp_sec, stamp_nanosec = _stamp_parts(stamp)
        metadata = SensingFrameMetadata(
            stamp_sec=stamp_sec,
            stamp_nanosec=stamp_nanosec,
            frame_id=frame_id or "",
            source_topic=source_topic,
            source_type="ros2",
        )
        return self.runtime.process_bgr_frame(image_bgr, frame_metadata=metadata, render_overlay=render_overlay)


def _stamp_parts(stamp: Any) -> tuple[int, int]:
    if stamp is None:
        now = time.time()
        sec = int(now)
        return sec, int((now - sec) * 1_000_000_000)
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    return sec, nanosec
