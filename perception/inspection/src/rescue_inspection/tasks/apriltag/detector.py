"""AprilTag Standard41h12 detector adapter built on pupil-apriltags."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from rescue_inspection.config import DetectorConfig
from rescue_inspection.models import (
    AprilTagDetectionItem,
    AprilTagDetectionResult,
    FramePacket,
    Point2D,
)


APRILTAG_DEPENDENCY_ERROR = (
    "pupil-apriltags is required for AprilTag Standard41h12 detection. "
    "Rebuild the Docker image or install pupil-apriltags in the runtime environment."
)


class AprilTagDetector:
    """Detect Standard41h12 AprilTags in one BGR frame."""

    def __init__(self, config: DetectorConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        try:
            from pupil_apriltags import Detector
        except ImportError as exc:
            raise RuntimeError(APRILTAG_DEPENDENCY_ERROR) from exc

        self._detector = Detector(
            families=config.apriltag_family,
            nthreads=config.apriltag_nthreads,
            quad_decimate=config.apriltag_quad_decimate,
            quad_sigma=config.apriltag_quad_sigma,
            refine_edges=config.apriltag_refine_edges,
            decode_sharpening=config.apriltag_decode_sharpening,
            debug=config.apriltag_debug,
        )

    def detect(self, packet: FramePacket) -> AprilTagDetectionResult:
        """Detect AprilTags from an OpenCV BGR frame."""

        gray = _to_gray(packet.image)
        raw_detections = self._detector.detect(gray)
        items = [_item_from_detection(detection, fallback_family=self.config.apriltag_family) for detection in raw_detections]
        if not items:
            return AprilTagDetectionResult(
                timestamp=packet.timestamp,
                frame_id=packet.frame_id,
                detected=False,
                family=self.config.apriltag_family,
                source_type=packet.source_type,
                failure_reason="no_apriltag_detected",
            )

        primary = max(items, key=_rank_item)
        return AprilTagDetectionResult(
            timestamp=packet.timestamp,
            frame_id=packet.frame_id,
            detected=True,
            tag_id=primary.tag_id,
            family=primary.family,
            center=primary.center,
            corners=primary.corners,
            bbox_xyxy=primary.bbox_xyxy,
            decision_margin=primary.decision_margin,
            hamming=primary.hamming,
            items=items,
            method_used="apriltag",
            source_type=packet.source_type,
        )


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _item_from_detection(detection: Any, *, fallback_family: str) -> AprilTagDetectionItem:
    corners = [_point(x, y) for x, y in np.asarray(getattr(detection, "corners", []), dtype=np.float32).reshape(-1, 2)]
    center_raw = np.asarray(getattr(detection, "center", []), dtype=np.float32).reshape(-1)
    center = _point(center_raw[0], center_raw[1]) if center_raw.size >= 2 else _center_from_corners(corners)
    family = getattr(detection, "tag_family", fallback_family) or fallback_family
    if isinstance(family, bytes):
        family = family.decode("utf-8", errors="replace")
    return AprilTagDetectionItem(
        tag_id=str(getattr(detection, "tag_id", "")),
        family=str(family),
        center=center,
        corners=corners,
        bbox_xyxy=_bbox_from_corners(corners),
        decision_margin=float(getattr(detection, "decision_margin", 0.0) or 0.0),
        hamming=int(getattr(detection, "hamming", 0) or 0),
    )


def _point(x: float, y: float) -> Point2D:
    return Point2D(x=float(x), y=float(y))


def _center_from_corners(corners: list[Point2D]) -> Point2D | None:
    if not corners:
        return None
    return Point2D(
        x=sum(point.x for point in corners) / len(corners),
        y=sum(point.y for point in corners) / len(corners),
    )


def _bbox_from_corners(corners: list[Point2D]) -> list[float] | None:
    if not corners:
        return None
    xs = [point.x for point in corners]
    ys = [point.y for point in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def _rank_item(item: AprilTagDetectionItem) -> tuple[float, float]:
    bbox = item.bbox_xyxy or [0.0, 0.0, 0.0, 0.0]
    area = max(float(bbox[2]) - float(bbox[0]), 0.0) * max(float(bbox[3]) - float(bbox[1]), 0.0)
    return float(item.decision_margin), area
