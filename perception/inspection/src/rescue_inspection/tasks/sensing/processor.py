"""Single-frame processing for combined QR + Hazmat sensing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rescue_inspection.config import AppConfig
from rescue_inspection.models import (
    AprilTagDetectionResult,
    FramePacket,
    HazmatDetectionResult,
    Point2D,
    QRDetectionItem,
    QRDetectionResult,
)
from rescue_inspection.tasks.apriltag import (
    AprilTagDetectionStabilizer,
    AprilTagDetector,
    StableAprilTagVoteResult,
)
from rescue_inspection.tasks.hazmat.debug_artifacts import (
    save_classifier_debug_bundle,
    save_color_evidence_debug,
    save_ocr_evidence_debug,
)
from rescue_inspection.tasks.hazmat.recognizer import build_hazmat_recognizer
from rescue_inspection.tasks.hazmat.temporal import HazmatDetectionStabilizer, StableHazmatVoteResult
from rescue_inspection.tasks.qr.detector import QRCodeBaselineDetector
from rescue_inspection.tasks.qr.postprocess import QRDetectionStabilizer, StableVoteResult
from rescue_inspection.utils.io import ensure_dir
from rescue_inspection.visualization.overlay import (
    draw_apriltag_overlay,
    draw_hazmat_overlay,
    draw_qr_overlay,
    draw_sensing_status_overlay,
)


@dataclass
class SensingFrameOutput:
    """Single-frame combined sensing output."""

    qr_result: QRDetectionResult
    qr_vote: StableVoteResult
    hazmat_result: HazmatDetectionResult
    hazmat_vote: StableHazmatVoteResult
    apriltag_result: AprilTagDetectionResult | None = None
    apriltag_vote: StableAprilTagVoteResult | None = None
    overlay: np.ndarray | None = None


class SensingFrameProcessor:
    """Reusable single-frame processor for combined QR + Hazmat sensing."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

        self.visual_code_backend = config.detector.backend
        self.qr_detector = None
        self.qr_stabilizer = None
        self.apriltag_detector = None
        self.apriltag_stabilizer = None
        if self.visual_code_backend == "apriltag":
            self.apriltag_detector = AprilTagDetector(config.detector, logger=logger)
            self.apriltag_stabilizer = AprilTagDetectionStabilizer(config.stabilizer)
        else:
            try:
                self.qr_detector = QRCodeBaselineDetector(config.detector, config.target_detector, logger=logger)
            except TypeError:
                self.qr_detector = QRCodeBaselineDetector(config.detector)
            self.qr_stabilizer = QRDetectionStabilizer(config.stabilizer)

        self.hazmat_recognizer = build_hazmat_recognizer(config, logger)
        self.hazmat_detector = getattr(self.hazmat_recognizer, "detector", None)
        self.hazmat_classifier = getattr(self.hazmat_recognizer, "classifier", None)
        self.hazmat_stabilizer = HazmatDetectionStabilizer(config.hazmat.stabilizer)

    def process_packet(
        self,
        packet: FramePacket,
        *,
        render_overlay: bool = True,
        debug_artifact_dir: Path | None = None,
    ) -> SensingFrameOutput:
        """Run QR, Hazmat, temporal voting, and optional overlay for one frame packet."""

        apriltag_result: AprilTagDetectionResult | None = None
        apriltag_vote: StableAprilTagVoteResult | None = None
        if self.visual_code_backend == "apriltag":
            assert self.apriltag_detector is not None
            assert self.apriltag_stabilizer is not None
            apriltag_result = self.apriltag_detector.detect(packet)
            apriltag_vote = self.apriltag_stabilizer.update(apriltag_result)
            if packet.source_type == "image" and apriltag_result.detected:
                apriltag_vote = StableAprilTagVoteResult(
                    is_stable=True,
                    stable_tag_id=apriltag_result.tag_id,
                    stability_score=1.0,
                    representative_result=apriltag_result,
                )
            apriltag_result = apriltag_result.model_copy(update={"stability_score": apriltag_vote.stability_score})
            qr_result = _qr_alias_from_apriltag(apriltag_result)
            qr_vote = StableVoteResult(
                is_stable=apriltag_vote.is_stable,
                stable_text=apriltag_vote.stable_tag_id,
                stability_score=apriltag_vote.stability_score,
                representative_result=qr_result if apriltag_vote.representative_result is not None else None,
            )
        else:
            assert self.qr_detector is not None
            assert self.qr_stabilizer is not None
            qr_result = self.qr_detector.detect(packet)
            qr_vote = self.qr_stabilizer.update(qr_result)

            if packet.source_type == "image" and qr_result.detected:
                qr_vote = StableVoteResult(
                    is_stable=True,
                    stable_text=qr_result.decoded_text,
                    stability_score=1.0,
                    representative_result=qr_result,
                )
            qr_result = qr_result.model_copy(update={"stability_score": qr_vote.stability_score})

        ignore_polygons = _qr_ignore_polygons(qr_result)

        hazmat_result = self.hazmat_recognizer.detect(packet, ignore_polygons=ignore_polygons)
        extraction = self.hazmat_recognizer.last_extraction
        accepted_items = [item for item in hazmat_result.items if item.label != "UNKNOWN"]

        if (
            self.config.output.debug_visualization
            and debug_artifact_dir is not None
            and extraction
            and extraction.rejected_candidates
        ):
            for i, (patch, _, reason, ratio) in enumerate(extraction.rejected_candidates):
                crop_name = f"frame_{packet.frame_id:04d}_rejected_detector_{i}_{reason}_ratio_{ratio:.2f}.png"
                cv2.imwrite(str(ensure_dir(debug_artifact_dir) / crop_name), patch)

        if self.config.output.debug_visualization and debug_artifact_dir is not None:
            debug_root = ensure_dir(debug_artifact_dir)
            for i, item in enumerate(hazmat_result.items):
                if item.debug is not None and hasattr(self.hazmat_classifier, "diagnose_patch"):
                    patch = self.hazmat_recognizer.last_candidate_patches[i]
                    diagnose_payload = self.hazmat_classifier.diagnose_patch(patch, topk=5)
                    save_classifier_debug_bundle(
                        debug_root=debug_root,
                        frame_id=packet.frame_id,
                        candidate_index=i,
                        frame_bgr=packet.image,
                        polygon=item.polygon,
                        rectified_crop_bgr=patch,
                        classifier_input_bgr=diagnose_payload["classifier_input_bgr"],
                        preprocess_meta=diagnose_payload["preprocess_meta"],
                    )
                if item.label == "UNKNOWN" and item.debug is not None:
                    patch = self.hazmat_recognizer.last_candidate_patches[i]
                    crop_name = (
                        f"frame_{packet.frame_id:04d}_candidate_{i}_"
                        f"rejected_classifier_{item.debug.gated_reason}.png"
                    )
                    cv2.imwrite(str(debug_root / crop_name), patch)
                if self.config.hazmat.debug.write_color_evidence_files and item.debug is not None:
                    save_color_evidence_debug(
                        debug_root=debug_root,
                        frame_id=packet.frame_id,
                        candidate_index=i,
                        payload=_color_debug_payload(item.debug),
                    )
                if self.config.hazmat.debug.write_ocr_evidence_files and item.debug is not None:
                    save_ocr_evidence_debug(
                        debug_root=debug_root,
                        frame_id=packet.frame_id,
                        candidate_index=i,
                        payload=_ocr_debug_payload(item.debug),
                    )

        hazmat_vote = self.hazmat_stabilizer.update(hazmat_result)

        if packet.source_type == "image" and hazmat_result.detected:
            best_item = max(accepted_items, key=lambda k: k.confidence)
            hazmat_vote = StableHazmatVoteResult(
                is_stable=True,
                stable_label=best_item.label,
                stability_score=1.0,
                representative_items=[best_item],
                representative_result=hazmat_result,
            )

        hazmat_result = hazmat_result.model_copy(
            update={
                "best_label": hazmat_vote.stable_label,
                "stability_score": hazmat_vote.stability_score,
            }
        )

        overlay = None
        if render_overlay:
            overlay = packet.image.copy()
            if apriltag_result is not None and apriltag_vote is not None:
                overlay = draw_apriltag_overlay(
                    overlay,
                    apriltag_result,
                    apriltag_vote.stable_tag_id,
                    apriltag_vote.stability_score,
                    self.config.output.debug_visualization,
                    apriltag_vote.is_stable,
                    draw_status=False,
                )
            else:
                overlay = draw_qr_overlay(
                    overlay,
                    qr_result,
                    qr_vote.stable_text,
                    qr_vote.stability_score,
                    self.config.output.debug_visualization,
                    qr_vote.is_stable,
                    draw_status=False,
                )
            overlay = draw_hazmat_overlay(
                overlay,
                hazmat_result,
                hazmat_vote.stable_label,
                hazmat_vote.stability_score,
                self.config.output.debug_visualization,
                hazmat_vote.is_stable,
                draw_status=False,
            )
            overlay = draw_sensing_status_overlay(
                overlay,
                qr_detection=qr_result,
                qr_stable_text=qr_vote.stable_text,
                qr_stability_score=qr_vote.stability_score,
                qr_is_stable=qr_vote.is_stable,
                apriltag_detection=apriltag_result,
                apriltag_stable_tag_id=apriltag_vote.stable_tag_id if apriltag_vote is not None else "",
                apriltag_stability_score=apriltag_vote.stability_score if apriltag_vote is not None else 0.0,
                apriltag_is_stable=apriltag_vote.is_stable if apriltag_vote is not None else False,
                hazmat_detection=hazmat_result,
                hazmat_stable_label=hazmat_vote.stable_label,
                hazmat_stability_score=hazmat_vote.stability_score,
                hazmat_is_stable=hazmat_vote.is_stable,
                debug=self.config.output.debug_visualization,
            )

        return SensingFrameOutput(
            qr_result=qr_result,
            qr_vote=qr_vote,
            hazmat_result=hazmat_result,
            hazmat_vote=hazmat_vote,
            apriltag_result=apriltag_result,
            apriltag_vote=apriltag_vote,
            overlay=overlay,
        )


def _color_debug_payload(debug) -> dict[str, object]:
    """Return color/fusion fields for explicit debug file output."""

    return {
        "color_top1_label": debug.color_top1_label,
        "color_top1_score": debug.color_top1_score,
        "color_top2_label": debug.color_top2_label,
        "color_top2_score": debug.color_top2_score,
        "color_margin": debug.color_margin,
        "quality_flags": debug.quality_flags,
        "quality_penalty": debug.quality_penalty,
        "fused_top1_label": debug.fused_top1_label,
        "fused_top1_score": debug.fused_top1_score,
        "fused_top2_label": debug.fused_top2_label,
        "fused_top2_score": debug.fused_top2_score,
        "fused_margin": debug.fused_margin,
        "fusion_method": debug.fusion_method,
        "fusion_reason": debug.fusion_reason,
        "color_weight_used": debug.color_weight_used,
        "color_scores_by_label": debug.color_scores_by_label,
        "color_evidence": debug.color_evidence,
    }


def _qr_ignore_polygons(qr_result: QRDetectionResult) -> list[list[Point2D]]:
    """Return decoded and suspect QR polygons that must mask Hazmat candidates."""

    ignore_polygons: list[list[Point2D]] = []
    if qr_result.polygon:
        ignore_polygons.append(qr_result.polygon)
    for item in qr_result.items:
        if item.polygon and (item.decoded_text or item.is_suspect):
            ignore_polygons.append(item.polygon)
    return ignore_polygons


def _qr_alias_from_apriltag(result: AprilTagDetectionResult) -> QRDetectionResult:
    """Expose AprilTag detections through the legacy QR fields for compatibility."""

    tag_id = result.tag_id if result.detected else ""
    items = [
        QRDetectionItem(decoded_text=item.tag_id, polygon=item.corners)
        for item in result.items
        if item.tag_id
    ]
    return QRDetectionResult(
        timestamp=result.timestamp,
        frame_id=result.frame_id,
        detected=result.detected,
        decoded_text=tag_id,
        polygon=result.corners,
        items=items,
        method_used="apriltag_legacy_qr_alias",
        stability_score=result.stability_score,
        source_type=result.source_type,
        failure_reason=result.failure_reason,
    )


def _ocr_debug_payload(debug) -> dict[str, object]:
    """Return full OCR/fusion fields for explicit debug file output."""

    return {
        "ocr_raw_text": debug.ocr_raw_text,
        "ocr_normalized_text": debug.ocr_normalized_text,
        "ocr_top1_label": debug.ocr_top1_label,
        "ocr_top1_score": debug.ocr_top1_score,
        "ocr_top2_label": debug.ocr_top2_label,
        "ocr_top2_score": debug.ocr_top2_score,
        "ocr_margin": debug.ocr_margin,
        "ocr_matched_keywords": debug.ocr_matched_keywords,
        "ocr_matched_numbers": debug.ocr_matched_numbers,
        "ocr_quality_flags": debug.ocr_quality_flags,
        "ocr_available": debug.ocr_available,
        "ocr_disabled_reason": debug.ocr_disabled_reason,
        "ocr_backend": debug.ocr_backend,
        "ocr_backend_error": debug.ocr_backend_error,
        "ocr_reused": debug.ocr_reused,
        "ocr_scores_by_label": debug.ocr_scores_by_label,
        "fused_top1_label": debug.fused_top1_label,
        "fused_top1_score": debug.fused_top1_score,
        "fused_top2_label": debug.fused_top2_label,
        "fused_top2_score": debug.fused_top2_score,
        "fused_margin": debug.fused_margin,
        "fusion_method": debug.fusion_method,
        "fusion_reason": debug.fusion_reason,
    }
