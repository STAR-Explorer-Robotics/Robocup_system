"""Hazmat recognizer abstraction shared by Hazmat and Sensing pipelines."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from rescue_inspection.config import AppConfig
from rescue_inspection.models import FramePacket, HazmatDetectionItem, HazmatDetectionResult, Point2D
from rescue_inspection.project_paths import resolve_project_path
from rescue_inspection.tasks.hazmat.classifier import HazmatClassifier
from rescue_inspection.tasks.hazmat.color_evidence import (
    HazmatColorMatch,
    HazmatColorProfileStore,
    extract_color_evidence,
    serialize_color_match_for_debug,
    write_standard_color_profile,
)
from rescue_inspection.tasks.hazmat.cnn_classifier import CNNHazmatClassifier
from rescue_inspection.tasks.hazmat.detector import HazmatCandidateDetector, HazmatCandidateExtraction
from rescue_inspection.tasks.hazmat.dl_detector import DLHazmatDetector
from rescue_inspection.tasks.hazmat.fusion import fuse_classifier_and_color
from rescue_inspection.tasks.hazmat.label_registry import LabelRegistry, UNKNOWN_LABEL
from rescue_inspection.tasks.hazmat.ocr_evidence import (
    CachedOcrEvidence,
    HazmatOcrEvidence,
    OcrCacheContext,
    can_reuse_ocr_cache,
    extract_ocr_text,
    match_hazmat_ocr_keywords,
    normalize_ocr_text,
    reused_ocr_evidence,
    serialize_ocr_evidence_for_debug,
)
from rescue_inspection.tasks.hazmat.prototype_classifier import PrototypeHazmatClassifier
from rescue_inspection.tasks.hazmat.templates import HazmatTemplateManager


class SupportsHazmatClassifier(Protocol):
    """Protocol for classifiers that expose classify_with_debug()."""

    def classify_with_debug(self, patch: np.ndarray): ...


class BaseHazmatRecognizer:
    """Per-frame Hazmat detection facade."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.registry = LabelRegistry.from_file(config.hazmat.labels_file)
        self.active_label_set = config.hazmat.active_label_set
        self.last_extraction: HazmatCandidateExtraction | None = None
        self.last_candidate_patches: list[np.ndarray] = []
        self.detector = None
        self.classifier: SupportsHazmatClassifier | None = None
        self.color_profiles = self._load_color_profiles()
        self._ocr_cache: dict[int, CachedOcrEvidence] = {}
        self.method_prefix = "hazmat"

    def detect(
        self,
        packet: FramePacket,
        ignore_polygons: list[list[Point2D]] | None = None,
    ) -> HazmatDetectionResult:
        raise NotImplementedError

    def _detect_candidates(
        self,
        packet: FramePacket,
        ignore_polygons: list[list[Point2D]] | None = None,
    ) -> HazmatCandidateExtraction:
        """Call the detector while preserving backwards compatibility in tests."""

        try:
            return self.detector.detect_candidates_with_diagnostics(packet, ignore_polygons=ignore_polygons)
        except TypeError:
            if ignore_polygons:
                raise
            return self.detector.detect_candidates_with_diagnostics(packet)

    def _classify_candidates(
        self,
        packet: FramePacket,
        extraction: HazmatCandidateExtraction,
    ) -> HazmatDetectionResult:
        items: list[HazmatDetectionItem] = []
        classification_debug = []
        self.last_candidate_patches = [patch for patch, _ in extraction.candidates]

        for candidate_index, (patch, polygon) in enumerate(extraction.candidates):
            debug = self.classifier.classify_with_debug(patch)
            raw_label = debug.top1_label or UNKNOWN_LABEL
            color_match = self._match_color(patch)
            ocr_evidence = self._match_ocr(
                patch=patch,
                polygon=polygon,
                packet=packet,
                candidate_index=candidate_index,
                classifier_debug=debug,
            )
            fusion_result = None
            if self.config.hazmat.fusion.enabled:
                fusion_result = fuse_classifier_and_color(
                    classifier_debug=debug,
                    color_match=color_match,
                    config=self.config.hazmat.fusion,
                    ocr_evidence=ocr_evidence,
                )
                debug_update = {
                    "fused_top1_label": fusion_result.fused_top1_label,
                    "fused_top1_score": fusion_result.fused_top1_score,
                    "fused_top2_label": fusion_result.fused_top2_label,
                    "fused_top2_score": fusion_result.fused_top2_score,
                    "fused_margin": fusion_result.fused_margin,
                    "fusion_method": fusion_result.fusion_method,
                    "fusion_reason": fusion_result.fusion_reason,
                    "color_weight_used": fusion_result.color_weight_used,
                    "quality_penalty": fusion_result.quality_penalty,
                }
                if color_match is not None:
                    debug_update.update(
                        serialize_color_match_for_debug(
                            color_match,
                            debug_config=self.config.hazmat.debug,
                        )
                    )
                if ocr_evidence is not None:
                    debug_update.update(
                        serialize_ocr_evidence_for_debug(
                            ocr_evidence,
                            ocr_config=self.config.hazmat.ocr,
                            debug_config=self.config.hazmat.debug,
                        )
                    )
                debug = debug.model_copy(update=debug_update)
                resolved_label = fusion_result.label
                confidence = fusion_result.confidence
            else:
                resolved_label = raw_label if debug.gated_reason is None else UNKNOWN_LABEL
                confidence = debug.top1_score

            if resolved_label != UNKNOWN_LABEL:
                mapped_label = self.registry.map_prediction(resolved_label, self.active_label_set)
                if mapped_label == UNKNOWN_LABEL:
                    debug = debug.model_copy(update={"gated_reason": debug.gated_reason or "inactive_label"})
                resolved_label = mapped_label
            item = HazmatDetectionItem(
                label=resolved_label,
                raw_label=raw_label,
                confidence=confidence,
                polygon=polygon,
                method_used=self._method_used(resolved_label, debug),
                debug=debug,
            )
            items.append(item)
            classification_debug.append(debug)

        accepted_items = [item for item in items if item.label != UNKNOWN_LABEL]
        best_debug = max(classification_debug, key=lambda item: item.top1_score, default=None)
        candidates_count = len(extraction.candidates)
        rejected_by_qr_mask = extraction.diagnostics.qr_overlap_rejected
        if accepted_items and best_debug is not None:
            final_decision = "accept"
            final_decision_reason = "accepted_top1"
        elif not extraction.candidates:
            final_decision = "reject"
            final_decision_reason = "no_candidates"
        else:
            final_decision = "reject"
            final_decision_reason = best_debug.gated_reason if best_debug is not None else "gated_all"
        return HazmatDetectionResult(
            timestamp=packet.timestamp,
            frame_id=packet.frame_id,
            detected=bool(accepted_items),
            items=items,
            source_type=packet.source_type,
            failure_reason="no_candidates" if not extraction.candidates else ("gated_all" if not accepted_items else None),
            candidate_count=candidates_count,
            candidates_count=candidates_count,
            rejected_by_qr_mask=rejected_by_qr_mask,
            detector_debug=extraction.diagnostics,
            classification_debug=classification_debug,
            classifier_top1_label=best_debug.top1_label if best_debug else "",
            classifier_top1_score=best_debug.top1_score if best_debug else 0.0,
            classifier_top2_label=best_debug.top2_label if best_debug else "",
            classifier_top2_score=best_debug.top2_score if best_debug else 0.0,
            best_candidate_label=best_debug.top1_label if best_debug else "",
            best_candidate_score=best_debug.top1_score if best_debug else 0.0,
            best_candidate_margin=best_debug.margin if best_debug else 0.0,
            final_decision=final_decision,
            final_decision_reason=final_decision_reason,
        )

    def _load_color_profiles(self) -> HazmatColorProfileStore | None:
        """Load configured color profiles, or disable color fusion gracefully."""

        if not self.config.hazmat.color.enabled or not self.config.hazmat.fusion.color_enabled:
            return None
        profile_path = resolve_project_path(self.config.hazmat.color.profile_path)
        try:
            if profile_path.exists():
                return HazmatColorProfileStore.load(profile_path)
            if self.config.hazmat.color.auto_generate_standard_profile:
                self.logger.warning(
                    "Hazmat color profile %s does not exist; generating because "
                    "auto_generate_standard_profile=true.",
                    profile_path,
                )
                write_standard_color_profile(
                    output_path=profile_path,
                    config=self.config.hazmat.color,
                    registry=self.registry,
                    manifest_path=self.config.hazmat.labels_file,
                )
                return HazmatColorProfileStore.load(profile_path)
            self.logger.warning(
                "Hazmat color profile %s does not exist; color fusion disabled for this run.",
                profile_path,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self.logger.warning("Failed to load Hazmat color profile %s: %s; color fusion disabled.", profile_path, exc)
        return None

    def _match_color(self, patch: np.ndarray) -> HazmatColorMatch | None:
        """Return color profile match details when profiles are available."""

        if self.color_profiles is None:
            return None
        evidence = extract_color_evidence(patch, self.config.hazmat.color)
        return self.color_profiles.match(evidence)

    def _match_ocr(
        self,
        *,
        patch: np.ndarray,
        polygon: list[Point2D],
        packet: FramePacket,
        candidate_index: int,
        classifier_debug,
    ) -> HazmatOcrEvidence | None:
        """Return optional OCR evidence when the OCR branch is enabled."""

        config = self.config.hazmat.ocr
        if not config.enabled:
            return None
        context = OcrCacheContext(
            frame_id=packet.frame_id,
            candidate_index=candidate_index,
            bbox_xyxy=_polygon_bbox_xyxy(polygon),
            classifier_top1_label=classifier_debug.top1_label,
            classifier_top2_label=classifier_debug.top2_label,
        )
        should_run = packet.frame_id % config.run_every_n_frames == 0
        if not should_run:
            cached = self._ocr_cache.get(candidate_index)
            if cached is not None and can_reuse_ocr_cache(cached=cached, context=context, config=config):
                return reused_ocr_evidence(cached.evidence)
            return None

        raw_evidence = extract_ocr_text(patch, config, logger=self.logger)
        normalized = normalize_ocr_text(raw_evidence.raw_text)
        evidence = match_hazmat_ocr_keywords(
            normalized,
            self.registry,
            config,
            raw_text=raw_evidence.raw_text,
            backend=raw_evidence.ocr_backend,
            base_evidence=raw_evidence,
        )
        self._ocr_cache[candidate_index] = CachedOcrEvidence(context=context, evidence=evidence)
        return evidence

    def _method_used(self, resolved_label: str, debug) -> str:
        """Return the method name for a classified candidate."""

        if resolved_label == UNKNOWN_LABEL:
            return "none"
        base = f"{self.method_prefix}_{self.config.hazmat.classifier.backend}"
        if debug.fusion_method == "ocr_fusion":
            return f"{base}_ocr_fusion"
        if debug.fusion_method == "color_fusion":
            return f"{base}_color_fusion"
        return base


def _polygon_bbox_xyxy(polygon: list[Point2D]) -> tuple[float, float, float, float]:
    if not polygon:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [point.x for point in polygon]
    ys = [point.y for point in polygon]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


class ClassicalHazmatRecognizer(BaseHazmatRecognizer):
    """Existing contour detector + HSV template classifier."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        super().__init__(config, logger)
        self.detector = HazmatCandidateDetector(config.hazmat.detector)
        template_manager = HazmatTemplateManager(
            target_size=config.hazmat.detector.warped_patch_size,
            label_registry=self.registry,
        )
        self.classifier = HazmatClassifier(config.hazmat.classifier, template_manager)
        self.method_prefix = "classical"

    def detect(
        self,
        packet: FramePacket,
        ignore_polygons: list[list[Point2D]] | None = None,
    ) -> HazmatDetectionResult:
        extraction = self._detect_candidates(packet, ignore_polygons=ignore_polygons)
        self.last_extraction = extraction
        return self._classify_candidates(packet, extraction)


class ClassifierOnlyHazmatRecognizer(BaseHazmatRecognizer):
    """Classical detector with neural classifier backends."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        super().__init__(config, logger)
        self.detector = HazmatCandidateDetector(config.hazmat.detector)
        if config.hazmat.classifier.backend == "template":
            template_manager = HazmatTemplateManager(
                target_size=config.hazmat.detector.warped_patch_size,
                label_registry=self.registry,
            )
            self.classifier = HazmatClassifier(config.hazmat.classifier, template_manager)
        elif config.hazmat.classifier.backend == "prototype":
            self.classifier = PrototypeHazmatClassifier(
                config.hazmat.classifier,
                self.registry,
                active_label_set=self.active_label_set,
                logger=logger,
            )
        else:
            self.classifier = CNNHazmatClassifier(
                config.hazmat.classifier,
                self.registry,
                active_label_set=self.active_label_set,
                logger=logger,
            )
        self.method_prefix = "classifier_only"

    def detect(
        self,
        packet: FramePacket,
        ignore_polygons: list[list[Point2D]] | None = None,
    ) -> HazmatDetectionResult:
        extraction = self._detect_candidates(packet, ignore_polygons=ignore_polygons)
        self.last_extraction = extraction
        return self._classify_candidates(packet, extraction)


class DLHazmatRecognizer(ClassifierOnlyHazmatRecognizer):
    """Reserved DL detector + neural classifier path."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        super().__init__(config, logger)
        self.detector = DLHazmatDetector(config.hazmat.detector, config.target_detector, logger=logger)
        self.method_prefix = "dl"


class HybridHazmatRecognizer(ClassifierOnlyHazmatRecognizer):
    """DL detector first, with optional classical fallback."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        super().__init__(config, logger)
        self.dl_detector = DLHazmatDetector(config.hazmat.detector, config.target_detector, logger=logger)
        self.classical_detector = HazmatCandidateDetector(config.hazmat.detector)
        self.detector = self.dl_detector
        self.method_prefix = "hybrid"

    def detect(
        self,
        packet: FramePacket,
        ignore_polygons: list[list[Point2D]] | None = None,
    ) -> HazmatDetectionResult:
        try:
            extraction = self.dl_detector.detect_candidates_with_diagnostics(packet, ignore_polygons=ignore_polygons)
            self.last_extraction = extraction
            return self._classify_candidates(packet, extraction)
        except RuntimeError:
            if not self.config.hazmat.detector.allow_classical_fallback:
                raise
            self.logger.info("Falling back to classical Hazmat detector inside hybrid mode.")
            extraction = self.classical_detector.detect_candidates_with_diagnostics(packet, ignore_polygons=ignore_polygons)
            self.last_extraction = extraction
            return self._classify_candidates(packet, extraction)


def build_hazmat_recognizer(config: AppConfig, logger: logging.Logger) -> BaseHazmatRecognizer:
    """Create the configured Hazmat recognizer implementation."""

    mode = config.hazmat.mode

    if mode == "classical":
        return ClassicalHazmatRecognizer(config, logger)
    if mode == "classifier_only":
        return ClassifierOnlyHazmatRecognizer(config, logger)
    if mode == "dl":
        return DLHazmatRecognizer(config, logger)
    if mode == "hybrid":
        return HybridHazmatRecognizer(config, logger)
    raise ValueError(f"Unsupported hazmat mode: {mode}")
