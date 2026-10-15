"""Shared result and packet models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Point2D(BaseModel):
    """2D point in image coordinates."""

    x: float
    y: float


class FramePacket(BaseModel):
    """Source-emitted frame payload."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_id: int
    timestamp: str
    source_time_sec: float | None = None
    image: np.ndarray
    source_type: str


class QRDetectionItem(BaseModel):
    """A single QR candidate within a frame."""

    decoded_text: str = ""
    polygon: list[Point2D] = Field(default_factory=list)
    is_suspect: bool = False
    failure_reason: str | None = None


class QRDetectionResult(BaseModel):
    """Per-frame QR detection output."""

    timestamp: str
    frame_id: int
    detected: bool
    decoded_text: str = ""
    polygon: list[Point2D] = Field(default_factory=list)
    items: list[QRDetectionItem] = Field(default_factory=list)
    method_used: str = "none"
    stability_score: float = 0.0
    source_type: str
    failure_reason: str | None = None
    roi_candidate_count: int = 0
    roi_diagnostics: dict[str, object] = Field(default_factory=dict)


class AprilTagDetectionItem(BaseModel):
    """A single AprilTag candidate within a frame."""

    tag_id: str = ""
    family: str = "tagStandard41h12"
    center: Point2D | None = None
    corners: list[Point2D] = Field(default_factory=list)
    bbox_xyxy: list[float] | None = None
    decision_margin: float = 0.0
    hamming: int = 0


class AprilTagDetectionResult(BaseModel):
    """Per-frame AprilTag detection output."""

    timestamp: str
    frame_id: int
    detected: bool
    tag_id: str = ""
    family: str = "tagStandard41h12"
    center: Point2D | None = None
    corners: list[Point2D] = Field(default_factory=list)
    bbox_xyxy: list[float] | None = None
    decision_margin: float = 0.0
    hamming: int = 0
    items: list[AprilTagDetectionItem] = Field(default_factory=list)
    method_used: str = "apriltag"
    stability_score: float = 0.0
    source_type: str
    failure_reason: str | None = None


class HazmatDetectorDiagnostics(BaseModel):
    """Candidate extraction counters for a single Hazmat frame."""

    backend: str = "classical"
    status: str = "ok"
    message: str | None = None
    contours_found: int = 0
    area_rejected: int = 0
    non_quad_rejected: int = 0
    non_convex_rejected: int = 0
    qr_overlap_rejected: int = 0
    candidates: int = 0
    min_contour_area: int = 0
    canny_low_threshold: int = 0
    canny_high_threshold: int = 0
    morph_kernel_size: int = 0
    approx_epsilon_factor: float = 0.0
    rejection_summary: dict[str, int] = Field(default_factory=dict)
    class_counts: dict[str, int] = Field(default_factory=dict)
    model_path: str | None = None


class HazmatClassificationDebug(BaseModel):
    """Template ranking details for one Hazmat candidate."""

    top1_label: str = ""
    top1_score: float = 0.0
    top2_label: str = ""
    top2_score: float = 0.0
    margin: float = 0.0
    raw_label: str = ""
    backend: str = "template"
    prototype_similarity: float = 0.0
    gated_reason: str | None = None
    color_top1_label: str = ""
    color_top1_score: float = 0.0
    color_top2_label: str = ""
    color_top2_score: float = 0.0
    color_margin: float = 0.0
    fused_top1_label: str = ""
    fused_top1_score: float = 0.0
    fused_top2_label: str = ""
    fused_top2_score: float = 0.0
    fused_margin: float = 0.0
    fusion_method: str = ""
    fusion_reason: str = ""
    color_weight_used: float = 0.0
    quality_penalty: float = 0.0
    quality_flags: list[str] = Field(default_factory=list)
    color_evidence: dict[str, object] = Field(default_factory=dict)
    color_scores_by_label: dict[str, float] = Field(default_factory=dict)
    ocr_raw_text: str = ""
    ocr_normalized_text: str = ""
    ocr_top1_label: str = ""
    ocr_top1_score: float = 0.0
    ocr_top2_label: str = ""
    ocr_top2_score: float = 0.0
    ocr_margin: float = 0.0
    ocr_matched_keywords: list[str] = Field(default_factory=list)
    ocr_matched_numbers: list[str] = Field(default_factory=list)
    ocr_quality_flags: list[str] = Field(default_factory=list)
    ocr_available: bool = False
    ocr_disabled_reason: str | None = None
    ocr_backend: str = ""
    ocr_backend_error: str | None = None
    ocr_reused: bool = False
    ocr_scores_by_label: dict[str, float] = Field(default_factory=dict)


class HazmatDetectionItem(BaseModel):
    """A single hazmat candidate within a frame."""

    label: str
    confidence: float
    polygon: list[Point2D]
    method_used: str
    raw_label: str = ""
    debug: HazmatClassificationDebug | None = None


class HazmatDetectionResult(BaseModel):
    """Per-frame Hazmat detection output."""

    timestamp: str
    frame_id: int
    detected: bool
    items: list[HazmatDetectionItem] = Field(default_factory=list)
    best_label: str = ""
    stability_score: float = 0.0
    source_type: str
    failure_reason: str | None = None
    candidate_count: int = 0
    candidates_count: int = 0
    rejected_by_qr_mask: int = 0
    detector_debug: HazmatDetectorDiagnostics | None = None
    classification_debug: list[HazmatClassificationDebug] = Field(default_factory=list)
    classifier_top1_label: str = ""
    classifier_top1_score: float = 0.0
    classifier_top2_label: str = ""
    classifier_top2_score: float = 0.0
    best_candidate_label: str = ""
    best_candidate_score: float = 0.0
    best_candidate_margin: float = 0.0
    final_decision: str = "reject"
    final_decision_reason: str | None = None


class RunSummary(BaseModel):
    """Run-level summary artifact."""

    run_id: str
    source_type: str
    input_path: str | None = None
    total_frames: int = 0
    frames_with_detection: int = 0
    final_detected: bool = False
    final_decoded_text: str = ""
    final_stability_score: float = 0.0
    best_frame_id: int | None = None
    artifact_dir: Path
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class SensingRunSummary(BaseModel):
    """Run-level summary artifact for sensing mode."""

    run_id: str
    source_type: str
    input_path: str | None = None
    total_frames: int = 0
    frames_with_detection: int = 0
    frames_with_any_detection: int = 0
    qr_frames_with_detection: int = 0
    hazmat_frames_with_detection: int = 0
    frames_with_both_qr_and_hazmat: int = 0
    best_frame_id: int | None = None
    submission_frame_id: int | None = None
    qr_detected: bool = False
    qr_message: str = ""
    qr_stability_score: float = 0.0
    apriltag_detected: bool = False
    apriltag_tag_id: str = ""
    apriltag_stability_score: float = 0.0
    hazmat_detected: bool = False
    hazmat_label: str = ""
    hazmat_stability_score: float = 0.0
    artifact_dir: Path
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class SensingDetectionResult(BaseModel):
    """Per-frame combined sensing output."""

    timestamp: str
    frame_id: int
    source_type: str
    qr_result: QRDetectionResult
    hazmat_result: HazmatDetectionResult
    apriltag_result: AprilTagDetectionResult | None = None
