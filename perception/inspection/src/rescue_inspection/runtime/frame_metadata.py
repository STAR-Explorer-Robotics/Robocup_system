"""Plain metadata for external single-frame sensing callers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensingFrameMetadata:
    """Metadata attached to one externally supplied sensing frame."""

    stamp_sec: int | None = None
    stamp_nanosec: int | None = None
    frame_id: str = ""
    source_topic: str = ""
    source_type: str = "ros2"
