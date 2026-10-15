"""Shared types for generic visual target ROI detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rescue_inspection.models import Point2D


TARGET_CLASS_NAMES = ["hazmat_roi", "qr_roi"]


class TargetROIDetectorDisabled(RuntimeError):
    """Raised when the configured ROI detector cannot run and policy disables it."""


@dataclass(frozen=True)
class LetterboxInfo:
    """Geometry needed to map between original and model input coordinates."""

    original_shape: tuple[int, int]
    input_size: int
    scale: float
    pad_x: float
    pad_y: float
    resized_shape: tuple[int, int]


@dataclass
class TargetROIDetection:
    """A single generic visual ROI detection in original image coordinates."""

    class_name: str
    class_id: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    polygon: list[Point2D] = field(default_factory=list)
    crop: np.ndarray | None = None
    square_crop: np.ndarray | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
