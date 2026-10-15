"""Hazmat classifier using template matching and unknown gating."""

from __future__ import annotations

import cv2
import numpy as np

from rescue_inspection.config import HazmatClassifierConfig
from rescue_inspection.models import HazmatClassificationDebug
from rescue_inspection.tasks.hazmat.templates import HazmatTemplateManager


class HazmatClassifier:
    """Matches a rectified patch against a known set of templates."""

    def __init__(self, config: HazmatClassifierConfig, template_manager: HazmatTemplateManager) -> None:
        self.config = config
        self.manager = template_manager

    def classify(self, patch: np.ndarray) -> tuple[str, float]:
        """Classify a rectified patch, returning the label and confidence score.
        Returns ("UNKNOWN", max_score) if the unknown gating criteria are met.
        """
        result = self.classify_with_debug(patch)
        return result.top1_label if result.gated_reason is None else "UNKNOWN", result.top1_score

    def classify_with_debug(self, patch: np.ndarray) -> HazmatClassificationDebug:
        """Classify a patch and return the top ranking details used for gating."""
        if not self.manager.templates:
            return HazmatClassificationDebug(gated_reason="no_templates")

        target_size = self.manager.target_size
        if patch.shape[:2] != (target_size, target_size):
            patch = cv2.resize(patch, (target_size, target_size))

        scores = []
        for label, template in self.manager.templates.items():
            score = self._compute_similarity(patch, template)
            scores.append((label, score))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)
        top1_label, top1_score = scores[0]
        top2_label = scores[1][0] if len(scores) > 1 else ""
        top2_score = scores[1][1] if len(scores) > 1 else 0.0
        margin = top1_score - top2_score if len(scores) > 1 else top1_score
        gated_reason = None

        if top1_score < self.config.top1_min_score:
            gated_reason = "top1_below_min_score"

        if gated_reason is None and len(scores) > 1 and margin < self.config.top1_top2_min_margin:
            gated_reason = "top1_top2_margin_too_small"

        return HazmatClassificationDebug(
            top1_label=top1_label,
            top1_score=top1_score,
            top2_label=top2_label,
            top2_score=top2_score,
            margin=margin,
            raw_label=top1_label,
            backend="template",
            gated_reason=gated_reason,
        )

    def _compute_similarity(self, patch: np.ndarray, template: np.ndarray) -> float:
        """Compute visual similarity using color histograms."""
        
        # Hazmat signs are heavily color-coded. HSV histograms provide a robust baseline 
        # without being highly sensitive to small rotations or exact texts.
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

        # Compare using Hue and Saturation
        channels = [0, 1]
        hist_size = [50, 60]
        ranges = [0, 180, 0, 256]

        hist_patch = cv2.calcHist([hsv_patch], channels, None, hist_size, ranges)
        cv2.normalize(hist_patch, hist_patch, 0, 1, cv2.NORM_MINMAX)

        hist_template = cv2.calcHist([hsv_template], channels, None, hist_size, ranges)
        cv2.normalize(hist_template, hist_template, 0, 1, cv2.NORM_MINMAX)

        # Distance metric: Bhattacharyya (0 = perfect match, 1 = maximum mismatch)
        dist = cv2.compareHist(hist_patch, hist_template, cv2.HISTCMP_BHATTACHARYYA)
        
        # Convert metric to similarity score (1 = perfect, 0 = mismatch)
        return float(1.0 - dist)
