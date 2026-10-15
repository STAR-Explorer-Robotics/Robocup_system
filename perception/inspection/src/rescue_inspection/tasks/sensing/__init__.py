"""Combined sensing pipeline."""

from __future__ import annotations

from rescue_inspection.tasks.sensing.overlay_recording import SensingOverlayRecorder
from rescue_inspection.tasks.sensing.pipeline import SensingPipeline
from rescue_inspection.tasks.sensing.processor import SensingFrameOutput, SensingFrameProcessor

__all__ = [
    "SensingFrameOutput",
    "SensingFrameProcessor",
    "SensingOverlayRecorder",
    "SensingPipeline",
]
