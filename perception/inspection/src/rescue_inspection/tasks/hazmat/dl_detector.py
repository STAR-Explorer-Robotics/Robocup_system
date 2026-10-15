"""Hazmat adapter over the generic visual target ROI detector."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from rescue_inspection.config import HazmatDetectorConfig, TargetDetectorConfig
from rescue_inspection.models import FramePacket, HazmatDetectorDiagnostics, Point2D
from rescue_inspection.tasks.hazmat.detector import HazmatCandidateExtraction
from rescue_inspection.tasks.vision_target import TargetROIDetector, TargetROIDetectorDisabled


class DLHazmatDetector:
    """Use the generic target ROI detector to produce Hazmat classifier crops."""

    def __init__(
        self,
        config: HazmatDetectorConfig,
        target_config: TargetDetectorConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.target_config = target_config
        self.logger = logger or logging.getLogger(__name__)
        self._disabled_reason: str | None = None
        self._target_detector: TargetROIDetector | None = None
        try:
            self._target_detector = TargetROIDetector(target_config, logger=self.logger)
        except TargetROIDetectorDisabled as exc:
            self._disabled_reason = str(exc)

    def detect_candidates_with_diagnostics(
        self,
        packet: FramePacket,
        ignore_polygons: list[list[Point2D]] | None = None,
    ) -> HazmatCandidateExtraction:
        """Return Hazmat ROI crops from the deployed generic target detector."""

        if self._target_detector is None:
            raise RuntimeError(self._disabled_reason or "target ROI detector is disabled")

        ignore_mask = _create_ignore_mask(
            ignore_polygons,
            packet.image.shape,
            pad_px=self.config.qr_ignore_pad_px,
            pad_ratio=self.config.qr_ignore_pad_ratio,
        )
        detections = self._target_detector.detect(packet, class_names={"hazmat_roi"})
        candidates = []
        rejected_candidates = []
        qr_overlap_rejected = 0
        for detection in detections:
            patch = detection.square_crop if detection.square_crop is not None else detection.crop
            if patch is None:
                continue
            overlap_ratio = _polygon_ignore_overlap_ratio(detection.polygon, ignore_mask, packet.image.shape)
            if overlap_ratio > self.config.qr_ignore_overlap_threshold:
                qr_overlap_rejected += 1
                rejected_candidates.append((patch, detection.polygon, "overlaps_qr_ignore_mask", overlap_ratio))
                continue
            candidates.append((patch, detection.polygon))

        diagnostics = HazmatDetectorDiagnostics(
            backend="dl",
            status="ok",
            candidates=len(candidates),
            qr_overlap_rejected=qr_overlap_rejected,
            rejection_summary={"qr_overlap": qr_overlap_rejected},
            class_counts={"hazmat_roi": len(candidates)},
            model_path=str(self._target_detector.model_path),
        )
        return HazmatCandidateExtraction(
            candidates=candidates,
            diagnostics=diagnostics,
            rejected_candidates=rejected_candidates,
        )

    def detect_candidates(self, packet: FramePacket) -> list[tuple[object, list[Point2D]]]:
        """Compatibility wrapper mirroring the classical detector API."""

        return self.detect_candidates_with_diagnostics(packet).candidates


def _create_ignore_mask(
    ignore_polygons: list[list[Point2D]] | None,
    img_shape: tuple[int, ...],
    *,
    pad_px: int,
    pad_ratio: float,
) -> np.ndarray | None:
    """Create a padded mask for decoded QR regions."""

    if not ignore_polygons:
        return None

    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for poly in ignore_polygons:
        if not poly:
            continue
        pts = np.array([[point.x, point.y] for point in poly], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        pad = max(pad_px, int(pad_ratio * max(w, h)))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_shape[1], x + w + pad)
        y2 = min(img_shape[0], y + h + pad)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask


def _polygon_ignore_overlap_ratio(
    polygon: list[Point2D],
    ignore_mask: np.ndarray | None,
    img_shape: tuple[int, ...],
) -> float:
    """Return the fraction of a candidate polygon covered by the QR ignore mask."""

    if ignore_mask is None or not polygon:
        return 0.0
    points = np.array([[point.x, point.y] for point in polygon], dtype=np.float32)
    area = cv2.contourArea(points)
    if area <= 0.0:
        return 0.0
    candidate_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(candidate_mask, [points.astype(np.int32)], 255)
    intersection = cv2.bitwise_and(candidate_mask, ignore_mask)
    return float(cv2.countNonZero(intersection) / area)
