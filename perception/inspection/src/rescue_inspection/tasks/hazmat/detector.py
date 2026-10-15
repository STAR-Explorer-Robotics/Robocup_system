"""Hazmat candidate detector using classical CV contour extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from rescue_inspection.config import HazmatDetectorConfig
from rescue_inspection.models import FramePacket, HazmatDetectorDiagnostics, Point2D


@dataclass
class HazmatCandidateExtraction:
    """Candidate patches plus diagnostic counters for one frame."""

    candidates: list[tuple[np.ndarray, list[Point2D]]]
    diagnostics: HazmatDetectorDiagnostics
    rejected_candidates: list[tuple[np.ndarray, list[Point2D], str, float]] = None


class HazmatCandidateDetector:
    """Extracts candidate patches from images by looking for diamond/square contours."""

    def __init__(self, config: HazmatDetectorConfig) -> None:
        self.config = config

    def detect_candidates(self, packet: FramePacket) -> list[tuple[np.ndarray, list[Point2D]]]:
        """
        Returns a list of (warped_patch, original_polygon) tuples for each candidate
        found in the frame.
        """
        return self.detect_candidates_with_diagnostics(packet).candidates

    def detect_candidates_with_diagnostics(
        self, packet: FramePacket, ignore_polygons: list[list[Point2D]] | None = None
    ) -> HazmatCandidateExtraction:
        """Return candidates and the rejection counters used to produce them."""
        img = packet.image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ignore_mask = self._create_ignore_mask(ignore_polygons, img.shape)
        
        # 1. Edge Detection & Morphology
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.config.canny_low_threshold, self.config.canny_high_threshold)
        kernel_size = self.config.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 2. Contour Extraction
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        rejected_candidates = []
        area_rejected = 0
        non_quad_rejected = 0
        non_convex_rejected = 0
        qr_overlap_rejected = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_contour_area:
                area_rejected += 1
                continue

            # 3. Polygonal Approximation
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.config.approx_epsilon_factor * peri, True)

            # 4. Geometric Constraints (Look for quadrilaterals/diamond shapes)
            if len(approx) != 4:
                non_quad_rejected += 1
                continue

            points = approx.reshape(4, 2).astype(np.float32)

            # Check convexity
            if not cv2.isContourConvex(points):
                non_convex_rejected += 1
                continue

            cand_area = cv2.contourArea(points)
            if cand_area > 0 and ignore_mask is not None:
                cand_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(cand_mask, [points.astype(np.int32)], 255)
                intersection = cv2.bitwise_and(cand_mask, ignore_mask)
                overlap_area = cv2.countNonZero(intersection)
                ratio = overlap_area / cand_area
                if ratio > self.config.qr_ignore_overlap_threshold:
                    qr_overlap_rejected += 1
                    polygon = [Point2D(x=float(p[0]), y=float(p[1])) for p in points]
                    warped = self._warp_patch(img, points)
                    rejected_candidates.append((warped, polygon, "overlaps_qr_ignore_mask", float(ratio)))
                    continue

            # Rectify the patch
            warped = self._warp_patch(img, points)
            polygon = [Point2D(x=float(p[0]), y=float(p[1])) for p in points]
            candidates.append((warped, polygon))

        diagnostics = HazmatDetectorDiagnostics(
            contours_found=len(contours),
            area_rejected=area_rejected,
            non_quad_rejected=non_quad_rejected,
            non_convex_rejected=non_convex_rejected,
            qr_overlap_rejected=qr_overlap_rejected,
            candidates=len(candidates),
            min_contour_area=self.config.min_contour_area,
            canny_low_threshold=self.config.canny_low_threshold,
            canny_high_threshold=self.config.canny_high_threshold,
            morph_kernel_size=kernel_size,
            approx_epsilon_factor=self.config.approx_epsilon_factor,
            rejection_summary={
                "area": area_rejected,
                "non_quad": non_quad_rejected,
                "non_convex": non_convex_rejected,
                "qr_overlap": qr_overlap_rejected,
            },
        )

        return HazmatCandidateExtraction(
            candidates=candidates,
            diagnostics=diagnostics,
            rejected_candidates=rejected_candidates,
        )

    def _create_ignore_mask(self, ignore_polygons: list[list[Point2D]] | None, img_shape: tuple[int, ...]) -> np.ndarray | None:
        """Create a boolean mask of ignored regions padded by configuration."""
        if not ignore_polygons:
            return None

        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        pad_px = self.config.qr_ignore_pad_px
        pad_ratio = self.config.qr_ignore_pad_ratio

        for poly in ignore_polygons:
            if not poly:
                continue
            pts = np.array([[p.x, p.y] for p in poly], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            pad = max(pad_px, int(pad_ratio * max(w, h)))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_shape[1], x + w + pad)
            y2 = min(img_shape[0], y + h + pad)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        return mask

    def _warp_patch(self, img: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Rectify the quadrilateral into a square patch for classification."""
        ordered = self._order_points(points)
        ts = float(self.config.warped_patch_size)
        dst = np.array([
            [0.0, 0.0],
            [ts - 1, 0.0],
            [ts - 1, ts - 1],
            [0.0, ts - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(img, matrix, (self.config.warped_patch_size, self.config.warped_patch_size))

    def _order_points(self, points: np.ndarray) -> np.ndarray:
        """Order into top-left, top-right, bottom-right, bottom-left."""
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)

        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = points[np.argmin(sums)]   # Top-left
        ordered[2] = points[np.argmax(sums)]   # Bottom-right
        ordered[1] = points[np.argmin(diffs)]  # Top-right
        ordered[3] = points[np.argmax(diffs)]  # Bottom-left
        return ordered
