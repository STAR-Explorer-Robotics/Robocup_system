"""Common submission exporter module."""

from __future__ import annotations

import json
from pathlib import Path

from rescue_inspection.models import (
    AprilTagDetectionItem,
    AprilTagDetectionResult,
    HazmatDetectionItem,
    HazmatDetectionResult,
    Point2D,
    QRDetectionItem,
    QRDetectionResult,
)
from rescue_inspection.utils.io import ensure_dir


def _polygon_to_xy(polygon: list[Point2D]) -> list[list[float]]:
    return [[point.x, point.y] for point in polygon]


def _bbox_from_polygon(polygon: list[Point2D]) -> list[float] | None:
    if not polygon:
        return None
    xs = [point.x for point in polygon]
    ys = [point.y for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _write_submission_text(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_submission_json(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _submission_paths(
    submission_dir: Path,
    *,
    overlay_name: str | None,
) -> dict[str, str]:
    paths = {
        "submission_decoded_message": str(submission_dir / "decoded_message.txt"),
        "submission_result_json": str(submission_dir / "result.json"),
    }
    if overlay_name:
        paths["submission_overlay_image"] = str(submission_dir / overlay_name)
    return paths


def _qr_detection_payload(
    item: QRDetectionItem,
    *,
    frame_id: int | None,
    method_used: str,
    stability_score: float | None = None,
) -> dict[str, object]:
    message = item.decoded_text.strip() or "UNKNOWN"
    payload: dict[str, object] = {
        "kind": "qr",
        "type": "qr_code",
        "detected": bool(item.decoded_text.strip()),
        "message": message,
        "label": message,
        "confidence": 1.0 if item.decoded_text.strip() else 0.0,
        "polygon": _polygon_to_xy(item.polygon),
        "method_used": method_used,
    }
    if stability_score is not None:
        payload["stability_score"] = stability_score
    bbox = _bbox_from_polygon(item.polygon)
    if bbox is not None:
        payload["bbox_xyxy"] = bbox
    if frame_id is not None:
        payload["frame_id"] = frame_id
    return payload


def _qr_detections(result: QRDetectionResult | None, *, stability_score: float | None = None) -> list[dict[str, object]]:
    if result is None:
        return []
    if result.items:
        return [
            _qr_detection_payload(
                item,
                frame_id=result.frame_id,
                method_used=result.method_used,
                stability_score=stability_score,
            )
            for item in result.items
            if item.decoded_text.strip() and not item.is_suspect
        ]
    if result.polygon and result.decoded_text.strip():
        return [
            _qr_detection_payload(
                QRDetectionItem(decoded_text=result.decoded_text, polygon=result.polygon),
                frame_id=result.frame_id,
                method_used=result.method_used,
                stability_score=stability_score,
            )
        ]
    return []


def _apriltag_detection_payload(
    item: AprilTagDetectionItem,
    *,
    frame_id: int | None,
    method_used: str,
    stability_score: float | None = None,
) -> dict[str, object]:
    tag_id = item.tag_id.strip() or "UNKNOWN"
    payload: dict[str, object] = {
        "kind": "apriltag",
        "type": "ar_code",
        "detected": bool(item.tag_id.strip()),
        "message": tag_id,
        "label": tag_id,
        "name": tag_id,
        "tag_id": tag_id,
        "family": item.family,
        "confidence": item.decision_margin,
        "decision_margin": item.decision_margin,
        "hamming": item.hamming,
        "polygon": _polygon_to_xy(item.corners),
        "method_used": method_used,
    }
    if item.center is not None:
        payload["center"] = [item.center.x, item.center.y]
    if stability_score is not None:
        payload["stability_score"] = stability_score
    if item.bbox_xyxy is not None:
        payload["bbox_xyxy"] = item.bbox_xyxy
    if frame_id is not None:
        payload["frame_id"] = frame_id
    return payload


def _apriltag_detections(
    result: AprilTagDetectionResult | None,
    *,
    stability_score: float | None = None,
) -> list[dict[str, object]]:
    if result is None:
        return []
    if result.items:
        return [
            _apriltag_detection_payload(
                item,
                frame_id=result.frame_id,
                method_used=result.method_used,
                stability_score=stability_score,
            )
            for item in result.items
            if item.tag_id.strip()
        ]
    if result.tag_id.strip():
        return [
            _apriltag_detection_payload(
                AprilTagDetectionItem(
                    tag_id=result.tag_id,
                    family=result.family,
                    center=result.center,
                    corners=result.corners,
                    bbox_xyxy=result.bbox_xyxy,
                    decision_margin=result.decision_margin,
                    hamming=result.hamming,
                ),
                frame_id=result.frame_id,
                method_used=result.method_used,
                stability_score=stability_score,
            )
        ]
    return []


def _hazmat_detection_payload(
    item: HazmatDetectionItem,
    *,
    frame_id: int | None,
    stability_score: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "hazmat",
        "type": "hazmat_sign",
        "detected": item.label != "UNKNOWN",
        "message": item.label,
        "label": item.label,
        "confidence": item.confidence,
        "classifier_confidence": item.confidence,
        "polygon": _polygon_to_xy(item.polygon),
        "method_used": item.method_used,
    }
    if stability_score is not None:
        payload["stability_score"] = stability_score
    bbox = _bbox_from_polygon(item.polygon)
    if bbox is not None:
        payload["bbox_xyxy"] = bbox
    if frame_id is not None:
        payload["frame_id"] = frame_id
    return payload


def _hazmat_detections(result: HazmatDetectionResult | None, *, stability_score: float | None = None) -> list[dict[str, object]]:
    if result is None:
        return []
    return [
        _hazmat_detection_payload(item, frame_id=result.frame_id, stability_score=stability_score)
        for item in result.items
        if item.label != "UNKNOWN"
    ]


def export_qr_submission(
    artifact_dir: Path,
    *,
    detected: bool,
    message: str,
    best_frame_id: int | None,
    stability_score: float,
    source_type: str,
    representative_result: QRDetectionResult | None,
    overlay_name: str | None,
) -> dict[str, str]:
    """Export QR submission files and return their paths."""

    submission_dir = ensure_dir(artifact_dir / "submission")
    message_str = message if detected and message else "UNKNOWN"
    detections = _qr_detections(representative_result)

    lines = [
        "task=QR_CODE_ACUITY",
        "type=qr_code",
        f"detected={str(detected).lower()}",
        f"message={message_str}",
    ]
    if best_frame_id is not None:
        lines.append(f"best_frame_id={best_frame_id}")
    if overlay_name:
        lines.append(f"overlay={overlay_name}")
    _write_submission_text(submission_dir / "decoded_message.txt", lines)

    result: dict[str, object] = {
        "task": "QR_CODE_ACUITY",
        "type": "qr_code",
        "detected": detected,
        "message": message_str,
        "label": message_str,
        "confidence": stability_score,
        "source": source_type,
        "detections": detections,
    }
    if best_frame_id is not None:
        result["best_frame_id"] = best_frame_id
    if overlay_name:
        result["overlay_path"] = overlay_name
    _write_submission_json(submission_dir / "result.json", result)

    return _submission_paths(submission_dir, overlay_name=overlay_name)


def export_hazmat_submission(
    artifact_dir: Path,
    *,
    detected: bool,
    label: str,
    best_frame_id: int | None,
    stability_score: float,
    source_type: str,
    representative_result: HazmatDetectionResult | None,
    overlay_name: str | None,
) -> dict[str, str]:
    """Export Hazmat submission files and return their paths."""

    submission_dir = ensure_dir(artifact_dir / "submission")
    message_str = label if detected and label else "UNKNOWN"
    detections = _hazmat_detections(representative_result)

    lines = [
        "task=PARTIAL_IMAGE_RECOGNITION_HAZMAT",
        "type=hazmat_sign",
        f"detected={str(detected).lower()}",
        f"message={message_str}",
    ]
    if best_frame_id is not None:
        lines.append(f"best_frame_id={best_frame_id}")
    if overlay_name:
        lines.append(f"overlay={overlay_name}")
    _write_submission_text(submission_dir / "decoded_message.txt", lines)

    result: dict[str, object] = {
        "task": "PARTIAL_IMAGE_RECOGNITION_HAZMAT",
        "type": "hazmat_sign",
        "detected": detected,
        "message": message_str,
        "label": message_str,
        "confidence": stability_score,
        "source": source_type,
        "detections": detections,
    }
    if best_frame_id is not None:
        result["best_frame_id"] = best_frame_id
    if overlay_name:
        result["overlay_path"] = overlay_name
    _write_submission_json(submission_dir / "result.json", result)

    return _submission_paths(submission_dir, overlay_name=overlay_name)


def export_sensing_submission(
    artifact_dir: Path,
    *,
    qr_detected: bool,
    qr_message: str,
    qr_stability_score: float,
    qr_result: QRDetectionResult | None,
    apriltag_detected: bool | None = None,
    apriltag_tag_id: str = "",
    apriltag_stability_score: float = 0.0,
    apriltag_result: AprilTagDetectionResult | None = None,
    hazmat_detected: bool,
    hazmat_label: str,
    hazmat_stability_score: float,
    hazmat_result: HazmatDetectionResult | None,
    best_frame_id: int | None,
    source_type: str,
    overlay_name: str | None,
) -> dict[str, str]:
    """Export Combined Sensing submission files and return their paths."""

    submission_dir = ensure_dir(artifact_dir / "submission")
    qr_msg_str = qr_message if qr_detected and qr_message else "UNKNOWN"
    ar_detected = bool(apriltag_detected)
    ar_tag_id = apriltag_tag_id if ar_detected and apriltag_tag_id else "UNKNOWN"
    hazmat_msg_str = hazmat_label if hazmat_detected and hazmat_label else "UNKNOWN"
    apriltag_detections = _apriltag_detections(apriltag_result, stability_score=apriltag_stability_score)
    qr_detections = (
        apriltag_detections
        if apriltag_result is not None
        else _qr_detections(qr_result, stability_score=qr_stability_score)
    )
    hazmat_detections = _hazmat_detections(hazmat_result, stability_score=hazmat_stability_score)
    detections = qr_detections + hazmat_detections

    lines = [
        "task=COMBINED_SENSING",
        f"apriltag_detected={str(ar_detected).lower()}",
        f"apriltag_tag_id={ar_tag_id}",
        f"qr_detected={str(qr_detected).lower()}",
        f"qr_messages={qr_msg_str}",
        f"hazmat_detected={str(hazmat_detected).lower()}",
        f"hazmat_labels={hazmat_msg_str}",
    ]
    if best_frame_id is not None:
        lines.append(f"best_frame_id={best_frame_id}")
    if overlay_name:
        lines.append(f"overlay={overlay_name}")
    _write_submission_text(submission_dir / "decoded_message.txt", lines)

    result: dict[str, object] = {
        "task": "COMBINED_SENSING",
        "source": source_type,
        "apriltag": {
            "type": "ar_code",
            "detected": ar_detected,
            "message": ar_tag_id,
            "label": ar_tag_id,
            "name": ar_tag_id,
            "tag_id": ar_tag_id,
            "stability_score": apriltag_stability_score,
            "confidence": apriltag_stability_score,
            "best_frame_id": apriltag_result.frame_id if apriltag_result is not None else best_frame_id,
            "submission_frame_id": best_frame_id,
            "detections": apriltag_detections,
        },
        "qr": {
            "type": "ar_code" if apriltag_result is not None else "qr_code",
            "detected": qr_detected,
            "message": qr_msg_str,
            "label": qr_msg_str,
            "stability_score": qr_stability_score,
            "confidence": qr_stability_score,
            "best_frame_id": qr_result.frame_id if qr_result is not None else best_frame_id,
            "submission_frame_id": best_frame_id,
            "detections": qr_detections,
        },
        "hazmat": {
            "type": "hazmat_sign",
            "detected": hazmat_detected,
            "message": hazmat_msg_str,
            "label": hazmat_msg_str,
            "stability_score": hazmat_stability_score,
            "confidence": hazmat_stability_score,
            "best_frame_id": hazmat_result.frame_id if hazmat_result is not None else best_frame_id,
            "submission_frame_id": best_frame_id,
            "detections": hazmat_detections,
        },
        "detections": detections,
    }
    if best_frame_id is not None:
        result["best_frame_id"] = best_frame_id
        result["submission_frame_id"] = best_frame_id
    if overlay_name:
        result["overlay_path"] = overlay_name
    _write_submission_json(submission_dir / "result.json", result)

    return _submission_paths(submission_dir, overlay_name=overlay_name)
