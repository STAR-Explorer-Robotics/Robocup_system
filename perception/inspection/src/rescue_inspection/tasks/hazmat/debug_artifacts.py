"""Hazmat runtime debug artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from rescue_inspection.models import Point2D
from rescue_inspection.tasks.hazmat.preprocessing import (
    channel_statistics,
    polygon_bbox_crop,
    save_bgr_debug_image,
)
from rescue_inspection.utils.io import ensure_dir


def candidate_debug_dir(debug_root: Path, frame_id: int, candidate_index: int) -> Path:
    """Return the per-candidate debug directory."""

    return debug_root / f"frame_{frame_id:04d}_candidate_{candidate_index}"


def save_classifier_debug_bundle(
    *,
    debug_root: Path,
    frame_id: int,
    candidate_index: int,
    frame_bgr: np.ndarray,
    polygon: list[Point2D],
    rectified_crop_bgr: np.ndarray,
    classifier_input_bgr: np.ndarray,
    preprocess_meta: dict[str, object],
) -> Path:
    """Persist raw frame, bbox crop, rectified crop, and classifier input artifacts."""

    bundle_dir = ensure_dir(candidate_debug_dir(debug_root, frame_id, candidate_index))
    polygon_points = [(point.x, point.y) for point in polygon]
    candidate_crop = polygon_bbox_crop(frame_bgr, polygon_points)
    raw_frame_path = bundle_dir / "raw_frame_rgb.png"
    candidate_crop_path = bundle_dir / "candidate_crop.png"
    rectified_crop_path = bundle_dir / "rectified_crop.png"
    classifier_input_path = bundle_dir / "classifier_input.png"
    meta_path = bundle_dir / "classifier_input_meta.json"
    candidate_crop_was_empty = _is_empty_image(candidate_crop)
    candidate_crop_for_debug = rectified_crop_bgr if candidate_crop_was_empty else candidate_crop

    save_bgr_debug_image(raw_frame_path, frame_bgr)
    save_bgr_debug_image(candidate_crop_path, candidate_crop_for_debug)
    save_bgr_debug_image(rectified_crop_path, rectified_crop_bgr)
    save_bgr_debug_image(classifier_input_path, classifier_input_bgr)

    meta_payload = {
        **preprocess_meta,
        "polygon_points": [{"x": point.x, "y": point.y} for point in polygon],
        "candidate_crop_empty_fallback_used": candidate_crop_was_empty,
        "raw_frame_stats": channel_statistics(frame_bgr),
        "candidate_crop_stats": channel_statistics(candidate_crop_for_debug),
        "rectified_crop_stats": channel_statistics(rectified_crop_bgr),
        "classifier_input_stats": channel_statistics(classifier_input_bgr),
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    return bundle_dir


def save_color_evidence_debug(
    *,
    debug_root: Path,
    frame_id: int,
    candidate_index: int,
    payload: dict[str, object],
) -> Path:
    """Persist explicit color evidence debug details for one candidate."""

    bundle_dir = ensure_dir(candidate_debug_dir(debug_root, frame_id, candidate_index))
    output_path = bundle_dir / "color_evidence.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def save_ocr_evidence_debug(
    *,
    debug_root: Path,
    frame_id: int,
    candidate_index: int,
    payload: dict[str, object],
) -> Path:
    """Persist explicit OCR evidence debug details for one candidate."""

    bundle_dir = ensure_dir(candidate_debug_dir(debug_root, frame_id, candidate_index))
    output_path = bundle_dir / "ocr_evidence.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _is_empty_image(image: np.ndarray) -> bool:
    """Return true when OpenCV cannot safely save or summarize an image."""

    return image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0


def infer_run_root_from_crop(input_path: Path) -> Path | None:
    """Infer outputs/<run_id> from a debug crop path when possible."""

    for parent in input_path.parents:
        if parent.name == "debug_candidates":
            return parent.parent
    return None


def write_realsense_failure_report(
    *,
    report_path: Path,
    crop_path: Path,
    diagnose_payload: dict[str, object],
    classifier_input_meta: dict[str, object] | None,
) -> Path:
    """Write a small markdown report for a RealSense Hazmat failure case."""

    classifier_debug = diagnose_payload["classifier_debug"]
    topk_softmax = diagnose_payload["topk_softmax"]
    topk_prototype = diagnose_payload["topk_prototype"]
    active_label_set = diagnose_payload["active_label_set"] or "<default>"
    reject_reason = diagnose_payload["reject_reason"] or "accepted"

    raw_stats = (classifier_input_meta or {}).get("raw_frame_stats")
    rectified_stats = (classifier_input_meta or {}).get("rectified_crop_stats")
    classifier_stats = (classifier_input_meta or {}).get("classifier_input_stats")
    raw_cast = raw_stats.get("dominant_bgr_channel") if isinstance(raw_stats, dict) else "unknown"
    rectified_cast = rectified_stats.get("dominant_bgr_channel") if isinstance(rectified_stats, dict) else "unknown"
    classifier_cast = classifier_stats.get("dominant_bgr_channel") if isinstance(classifier_stats, dict) else "unknown"

    report_lines = [
        "# Hazmat RealSense Failure Report",
        "",
        f"- Crop: `{crop_path}`",
        f"- Active label set: `{active_label_set}`",
        f"- Final decision: `{diagnose_payload['final_decision']}`",
        f"- Reject reason: `{reject_reason}`",
        "",
        "## Answers",
        "",
        "1. Direct classifier top5",
        *(f"   - `{item['label']}`: {item['score']:.6f}" for item in topk_softmax),
        "",
        "2. Prototype similarity",
        *(f"   - `{item['label']}`: {item['similarity']:.6f}" for item in topk_prototype[:5]),
        "",
        "3. RGB/BGR or IR/depth misuse",
        "   - RealSense source is configured for `rs.stream.color` with `rs.format.bgr8`.",
        "   - No IR/depth/colorized frame path is used in the current adapter.",
        "   - The classifier runtime now applies the same `BGR -> RGB -> ImageNet normalize` preprocessing used in training.",
        "",
        "4. Raw frame cyan check",
        f"   - Raw frame dominant BGR channel: `{raw_cast}`",
        f"   - Rectified crop dominant BGR channel: `{rectified_cast}`",
        f"   - Classifier input dominant BGR channel: `{classifier_cast}`",
        "",
        "5. Fix direction",
        f"   - Current top1 is `{classifier_debug['top1_label']}` with score {classifier_debug['top1_score']:.6f}.",
        f"   - Current prototype similarity for top1 is {classifier_debug.get('prototype_similarity', 0.0):.6f}.",
        "   - If raw frame and rectified crop are both color-shifted, this points to RealSense domain gap more than a debug-image save bug.",
        "   - If raw frame looks normal but rectified/classifier input shifts, focus on crop/rectify/preprocess.",
        "",
        "6. Recommended next step",
        "   - Keep the preprocessing fix.",
        "   - Collect a small set of manually verified RealSense crops and fine-tune from the existing checkpoint instead of retraining from scratch.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report_path
