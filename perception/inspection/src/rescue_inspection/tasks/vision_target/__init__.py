"""Generic visual target ROI detector utilities."""

from rescue_inspection.tasks.vision_target.detector import TargetROIDetector
from rescue_inspection.tasks.vision_target.types import TargetROIDetection, TargetROIDetectorDisabled

__all__ = ["TargetROIDetection", "TargetROIDetector", "TargetROIDetectorDisabled"]
