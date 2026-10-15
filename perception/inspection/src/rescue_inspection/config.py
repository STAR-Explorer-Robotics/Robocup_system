"""Configuration models and YAML loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


SourceType = Literal["image", "video", "realsense"]
TaskType = Literal["qr", "hazmat", "sensing", "combined"]
HazmatMode = Literal["classical", "classifier_only", "dl", "hybrid"]
HazmatDetectorBackend = Literal["classical", "dl"]
HazmatClassifierBackend = Literal["template", "cnn", "prototype"]
HazmatOcrBackend = Literal["none", "tesseract", "external"]
HazmatBackbone = Literal["mobilenet_v3_small", "resnet18"]
PreviewMode = Literal["opencv", "none"]
TargetDetectorFailurePolicy = Literal["fail_fast", "warn_disable"]
TargetDetectorModelFamily = Literal["yolo"]
TargetDetectorOutputFormat = Literal["xywh_conf_class"]
DetectorBackend = Literal["qr", "apriltag"]


class TaskConfig(BaseModel):
    """Task configuration."""

    type: TaskType = "qr"


class RealSenseConfig(BaseModel):
    """Runtime settings for the optional RealSense source."""

    enabled: bool = False
    width: int = 1280
    height: int = 720
    fps: int = 30


class SourceConfig(BaseModel):
    """Input source settings."""

    type: SourceType = "image"
    input_path: Path | None = None
    frame_stride: int = Field(default=1, ge=1)
    loop: bool = False
    realsense: RealSenseConfig = Field(default_factory=RealSenseConfig)


class OutputConfig(BaseModel):
    """Artifact saving settings."""

    root_dir: Path = Path("outputs")
    save_overlay: bool = True
    save_jsonl: bool = True
    save_summary: bool = True
    debug_visualization: bool = True


class PreviewConfig(BaseModel):
    """Optional runtime preview settings."""

    enabled: bool = False
    mode: PreviewMode = "opencv"
    window_name: str = "QRC Inspection Preview"
    wait_key_delay_ms: int = Field(default=1, ge=1, le=1000)

    @model_validator(mode="after")
    def enforce_disabled_mode(self) -> "PreviewConfig":
        if not self.enabled:
            self.mode = "none"
        return self


class DetectorConfig(BaseModel):
    """Visual code detector configuration for QR or AprilTag backends."""

    backend: DetectorBackend = "qr"
    enable_preprocess_fallback: bool = True
    try_curved: bool = True
    enable_candidate_decode_fallback: bool = True
    warped_patch_size: int = Field(default=512, ge=64)
    qr_roi_decode_pad_px: int = Field(default=24, ge=0)
    qr_roi_decode_pad_ratio: float = Field(default=0.18, ge=0.0, le=1.0)
    qr_roi_decode_white_border_px: int = Field(default=24, ge=0)
    qr_roi_decode_scale: float = Field(default=2.0, ge=1.0, le=8.0)
    apriltag_family: str = "tagStandard41h12"
    apriltag_nthreads: int = Field(default=1, ge=1)
    apriltag_quad_decimate: float = Field(default=1.0, gt=0.0)
    apriltag_quad_sigma: float = Field(default=0.0, ge=0.0)
    apriltag_refine_edges: int = Field(default=1, ge=0, le=1)
    apriltag_decode_sharpening: float = Field(default=0.25, ge=0.0)
    apriltag_debug: int = Field(default=0, ge=0, le=1)


class StabilizerConfig(BaseModel):
    """Sliding window vote configuration (QR)."""

    window_size: int = Field(default=5, ge=1)
    min_votes: int = Field(default=3, ge=1)
    stability_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class TargetDetectorConfig(BaseModel):
    """Generic visual target ROI detector settings."""

    enabled: bool = False
    model_path: Path = Path("models/targets/qrc_target_detector.onnx")
    metadata_path: Path = Path("models/targets/qrc_target_detector_metadata.json")
    classes: list[str] = Field(default_factory=lambda: ["hazmat_roi", "qr_roi"])
    input_size: int = Field(default=640, ge=64)
    score_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=20, ge=1)
    model_family: TargetDetectorModelFamily = "yolo"
    output_format: TargetDetectorOutputFormat = "xywh_conf_class"
    execution_providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])
    missing_model_policy: TargetDetectorFailurePolicy = "warn_disable"
    runtime_unavailable_policy: TargetDetectorFailurePolicy = "warn_disable"
    metadata_mismatch_policy: TargetDetectorFailurePolicy = "fail_fast"

    @model_validator(mode="after")
    def validate_classes(self) -> "TargetDetectorConfig":
        if self.classes != ["hazmat_roi", "qr_roi"]:
            raise ValueError("target_detector.classes must be exactly ['hazmat_roi', 'qr_roi'] in class-id order")
        return self


class HazmatDetectorConfig(BaseModel):
    """Hazmat candidate detection settings."""

    backend: HazmatDetectorBackend = "classical"
    model_path: Path | None = None
    input_size: int = Field(default=640, ge=64)
    score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=20, ge=1)
    allow_classical_fallback: bool = True
    warped_patch_size: int = Field(default=256, ge=64)
    min_contour_area: int = Field(default=1000, ge=100)
    canny_low_threshold: int = Field(default=50, ge=0)
    canny_high_threshold: int = Field(default=150, ge=1)
    morph_kernel_size: int = Field(default=5, ge=1)
    approx_epsilon_factor: float = Field(default=0.04, ge=0.001, le=0.2)
    qr_ignore_pad_px: int = Field(default=20, ge=0)
    qr_ignore_pad_ratio: float = Field(default=0.08, ge=0.0, le=1.0)
    qr_ignore_overlap_threshold: float = Field(default=0.2, ge=0.0, le=1.0)


class HazmatClassifierConfig(BaseModel):
    """Hazmat classification and unknown gating settings."""

    backend: HazmatClassifierBackend = "template"
    backbone: HazmatBackbone = "mobilenet_v3_small"
    model_path: Path | None = None
    prototypes_path: Path | None = None
    input_size: int = Field(default=224, ge=32)
    embedding_dim: int = Field(default=128, ge=8)
    batch_size: int = Field(default=64, ge=1)
    top1_min_score: float = Field(default=0.45, ge=0.0, le=1.0)
    top1_top2_min_margin: float = Field(default=0.05, ge=0.0, le=1.0)
    prototype_min_similarity: float = Field(default=0.40, ge=-1.0, le=1.0)
    temperature: float = Field(default=1.0, gt=0.0)
    use_amp: bool = True
    execution_providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])


class HazmatStabilizerConfig(BaseModel):
    """Sliding window vote configuration (Hazmat)."""

    window_size: int = Field(default=10, ge=1)
    min_votes: int = Field(default=5, ge=1)
    stability_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class HazmatColorConfig(BaseModel):
    """Color evidence and profile loading settings."""

    enabled: bool = True
    profile_path: Path = Path("data/processed/hazmat_color_profiles_standard.json")
    auto_generate_standard_profile: bool = False
    hue_bins: int = Field(default=12, ge=4, le=36)
    alpha_threshold: int = Field(default=10, ge=0, le=255)
    min_value: int = Field(default=20, ge=0, le=255)
    black_value_threshold: int = Field(default=55, ge=0, le=255)
    white_value_threshold: int = Field(default=190, ge=0, le=255)
    low_valid_pixel_ratio_threshold: float = Field(default=0.12, ge=0.0, le=1.0)
    high_white_ratio_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    low_saturation_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    shadow_value_threshold: int = Field(default=55, ge=0, le=255)


class HazmatOcrConfig(BaseModel):
    """Optional OCR evidence settings."""

    enabled: bool = False
    backend: HazmatOcrBackend = "none"
    min_crop_size: int = Field(default=48, ge=1)
    run_every_n_frames: int = Field(default=1, ge=1)
    max_ocr_bonus: float = Field(default=0.20, ge=0.0, le=1.0)
    strong_keyword_bonus: float = Field(default=0.18, ge=0.0, le=1.0)
    weak_keyword_bonus: float = Field(default=0.06, ge=0.0, le=1.0)
    strong_number_bonus: float = Field(default=0.20, ge=0.0, le=1.0)
    weak_number_bonus: float = Field(default=0.05, ge=0.0, le=1.0)
    medium_number_bonus: float = Field(default=0.12, ge=0.0, le=1.0)
    fuzzy_match_enabled: bool = True
    fuzzy_match_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    allow_ocr_only_decision: bool = False
    allow_ocr_top2_flip: bool = True
    high_confidence_ocr_override: bool = False
    debug_text_max_chars: int = Field(default=96, ge=8, le=1024)
    cache_min_iou: float = Field(default=0.85, ge=0.0, le=1.0)
    cache_min_size_ratio: float = Field(default=0.90, ge=0.0, le=1.0)


class HazmatFusionConfig(BaseModel):
    """Classifier/color fusion settings."""

    enabled: bool = True
    color_enabled: bool = True
    ocr_enabled: bool = True
    classifier_weight: float = Field(default=0.85, ge=0.0, le=1.0)
    color_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    ocr_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    max_color_delta: float = Field(default=0.12, ge=0.0, le=1.0)
    max_ocr_bonus: float = Field(default=0.20, ge=0.0, le=1.0)
    allow_color_top2_flip: bool = True
    allow_color_accept_gated_classifier: bool = True
    allow_color_only_decision: bool = False
    allow_ocr_top2_flip: bool = True
    allow_ocr_only_decision: bool = False
    high_confidence_ocr_override: bool = False
    min_fused_score: float = Field(default=0.45, ge=0.0, le=1.0)
    min_fused_margin: float = Field(default=0.01, ge=0.0, le=1.0)
    high_confidence_classifier_score: float = Field(default=0.75, ge=0.0, le=1.0)
    high_confidence_classifier_margin: float = Field(default=0.20, ge=0.0, le=1.0)
    quality_penalty_weight: float = Field(default=0.10, ge=0.0, le=1.0)


class HazmatDebugConfig(BaseModel):
    """Hazmat debug output controls."""

    write_color_evidence_files: bool = False
    include_full_color_scores: bool = False
    include_color_histograms: bool = False
    write_ocr_evidence_files: bool = False
    include_full_ocr_scores: bool = False


class HazmatConfig(BaseModel):
    """Settings specific to the Hazmat task."""

    mode: HazmatMode = "classical"
    labels_file: Path = Path("data/manifests/hazmat_labels.yaml")
    active_label_set: str | None = None
    detector: HazmatDetectorConfig = Field(default_factory=HazmatDetectorConfig)
    classifier: HazmatClassifierConfig = Field(default_factory=HazmatClassifierConfig)
    color: HazmatColorConfig = Field(default_factory=HazmatColorConfig)
    ocr: HazmatOcrConfig = Field(default_factory=HazmatOcrConfig)
    fusion: HazmatFusionConfig = Field(default_factory=HazmatFusionConfig)
    debug: HazmatDebugConfig = Field(default_factory=HazmatDebugConfig)
    stabilizer: HazmatStabilizerConfig = Field(default_factory=HazmatStabilizerConfig)


class RuntimeConfig(BaseModel):
    """General runtime options."""

    max_frames: int | None = Field(default=None, ge=1)
    log_level: str = "INFO"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    task: TaskConfig = Field(default_factory=TaskConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    preview: PreviewConfig = Field(default_factory=PreviewConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    stabilizer: StabilizerConfig = Field(default_factory=StabilizerConfig)
    target_detector: TargetDetectorConfig = Field(default_factory=TargetDetectorConfig)
    hazmat: HazmatConfig = Field(default_factory=HazmatConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


def load_config(path: Path) -> AppConfig:
    """Load YAML config from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle) or {}

    return AppConfig.model_validate(raw_data)


def apply_overrides(
    config: AppConfig,
    *,
    task_type: TaskType | None = None,
    source_type: SourceType | None = None,
    input_path: Path | None = None,
    output_root: Path | None = None,
    max_frames: int | None = None,
    show_preview: bool | None = None,
    preview_window_name: str | None = None,
) -> AppConfig:
    """Return a new config with CLI overrides applied."""

    update: dict[str, object] = {}

    if task_type is not None:
        update["task"] = config.task.model_copy(update={"type": task_type})

    if source_type is not None:
        source_update: dict[str, object] = {"type": source_type}
        if source_type == "realsense":
            source_update["realsense"] = config.source.realsense.model_copy(update={"enabled": True})
        update["source"] = config.source.model_copy(update=source_update)

    if input_path is not None:
        source_cfg = update.get("source", config.source)
        if isinstance(source_cfg, SourceConfig):
            update["source"] = source_cfg.model_copy(update={"input_path": input_path})

    if output_root is not None:
        update["output"] = config.output.model_copy(update={"root_dir": output_root})

    if max_frames is not None:
        update["runtime"] = config.runtime.model_copy(update={"max_frames": max_frames})

    if show_preview is not None or preview_window_name is not None:
        preview_update: dict[str, object] = {}
        if show_preview is not None:
            preview_update["enabled"] = show_preview
            preview_update["mode"] = "opencv" if show_preview else "none"
        if preview_window_name is not None:
            preview_update["window_name"] = preview_window_name
        update["preview"] = config.preview.model_copy(update=preview_update)

    return config.model_copy(update=update)
