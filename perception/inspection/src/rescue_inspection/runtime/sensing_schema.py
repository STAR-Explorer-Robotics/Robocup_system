"""Stable JSON schema adapter for single-frame sensing results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SensingFrameResult:
    """External result for one processed sensing frame.

    Attributes:
        stamp_sec: Timestamp seconds.
        stamp_nanosec: Timestamp nanoseconds.
        frame_id: Frame identifier string.
        source_topic: Source ROS2/input topic/path string.
        processing_time_ms: Total frame sensing processing duration in milliseconds.
        image_width: Input frame image width.
        image_height: Input frame image height.
        qr_result: Internal QR detector result object (not guaranteed to be JSON-safe).
        qr_vote: Internal QR temporal stabilizer state.
        apriltag_result: Internal AprilTag detector result object, when enabled.
        apriltag_vote: Internal AprilTag temporal stabilizer state, when enabled.
        hazmat_result: Internal Hazmat detector result object (not guaranteed to be JSON-safe).
        hazmat_vote: Internal Hazmat temporal stabilizer state.
        overlay_bgr: OpenCV standard HWC BGR overlay image (np.ndarray) or None.
    """

    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    source_topic: str
    processing_time_ms: float
    image_width: int
    image_height: int
    qr_result: Any
    qr_vote: Any
    hazmat_result: Any
    hazmat_vote: Any
    overlay_bgr: np.ndarray | None
    apriltag_result: Any | None = None
    apriltag_vote: Any | None = None


QrcSensingResult = SensingFrameResult


def result_to_payload(result: SensingFrameResult) -> dict[str, Any]:
    """Convert one sensing frame result to the public JSON payload.

    This is the official stable serializer. It converts the SensingFrameResult
    into a dictionary containing only standard JSON-safe types (no numpy arrays,
    OpenCV objects, or internal detector references).
    """

    apriltag_result = getattr(result, "apriltag_result", None)
    apriltag_vote = getattr(result, "apriltag_vote", None)
    qr_bbox = _qr_bbox(result.qr_result, apriltag_result)
    hazmat_items = [item for item in getattr(result.hazmat_result, "items", []) if item.label != "UNKNOWN"]
    best_hazmat_item = max(hazmat_items, key=lambda item: float(item.confidence), default=None)
    hazmat_bbox = _bbox_from_polygon(getattr(best_hazmat_item, "polygon", [])) if best_hazmat_item else None

    payload = {
        "stamp": {
            "sec": result.stamp_sec,
            "nanosec": result.stamp_nanosec,
        },
        "frame_id": result.frame_id,
        "source_topic": result.source_topic,
        "processing_time_ms": result.processing_time_ms,
        "image_width": result.image_width,
        "image_height": result.image_height,
        "qr": {
            # Legacy schema alias: when AprilTag backend is active, these fields
            # carry the detected tag id without invoking any QR decoder.
            "detected": bool(getattr(apriltag_result or result.qr_result, "detected", False)),
            "decoded_text": _legacy_qr_text(result.qr_result, apriltag_result),
            "bbox": qr_bbox,
            "confidence": None,
            "stable_text": _legacy_qr_stable_text(result.qr_vote, apriltag_vote),
            "is_stable": bool(getattr(apriltag_vote or result.qr_vote, "is_stable", False)),
            "stability_score": float(getattr(apriltag_vote or result.qr_vote, "stability_score", 0.0)),
        },
        "apriltag": _apriltag_payload(apriltag_result, apriltag_vote),
        "hazmat": {
            "detected": bool(getattr(result.hazmat_result, "detected", False)),
            "label": getattr(best_hazmat_item, "label", "") if best_hazmat_item else "",
            "confidence": float(best_hazmat_item.confidence) if best_hazmat_item else None,
            "bbox": hazmat_bbox,
            "stable_label": getattr(result.hazmat_vote, "stable_label", "") or "",
            "is_stable": bool(getattr(result.hazmat_vote, "is_stable", False)),
            "stability_score": float(getattr(result.hazmat_vote, "stability_score", 0.0)),
        },
        "detections": [],
    }

    if apriltag_result is not None:
        payload["detections"].extend(_apriltag_detections(apriltag_result))
    else:
        payload["detections"].extend(_qr_detections(result.qr_result))
    payload["detections"].extend(_hazmat_detections(result.hazmat_result))
    return payload


def result_to_json(result: SensingFrameResult) -> str:
    """Serialize one sensing result to compact JSON."""

    return json.dumps(result_to_payload(result), ensure_ascii=False, separators=(",", ":"))


def _qr_detections(qr_result: Any) -> list[dict[str, Any]]:
    detections = []
    items = getattr(qr_result, "items", []) or []
    if not items and getattr(qr_result, "polygon", []):
        items = [qr_result]
    for item in items:
        if getattr(item, "is_suspect", False) or not (getattr(item, "decoded_text", "") or "").strip():
            continue
        bbox = _bbox_from_polygon(getattr(item, "polygon", []))
        if bbox is None:
            continue
        detections.append(
            {
                "target_type": "qr",
                "label": "qr_code",
                "confidence": None,
                "bbox": bbox,
                "decoded_text": getattr(item, "decoded_text", "") or "",
            }
        )
    return detections


def _apriltag_payload(apriltag_result: Any | None, apriltag_vote: Any | None) -> dict[str, Any]:
    if apriltag_result is None:
        return {
            "detected": False,
            "tag_id": "",
            "family": "",
            "bbox": None,
            "center": None,
            "corners": [],
            "decision_margin": None,
            "hamming": None,
            "stable_tag_id": "",
            "is_stable": False,
            "stability_score": 0.0,
        }
    return {
        "detected": bool(getattr(apriltag_result, "detected", False)),
        "tag_id": getattr(apriltag_result, "tag_id", "") or "",
        "family": getattr(apriltag_result, "family", "") or "",
        "bbox": _bbox_from_apriltag(apriltag_result),
        "center": _point_payload(getattr(apriltag_result, "center", None)),
        "corners": [_point_payload(point) for point in getattr(apriltag_result, "corners", [])],
        "decision_margin": _float_or_none(getattr(apriltag_result, "decision_margin", None)),
        "hamming": _int_or_none(getattr(apriltag_result, "hamming", None)),
        "stable_tag_id": getattr(apriltag_vote, "stable_tag_id", "") or "",
        "is_stable": bool(getattr(apriltag_vote, "is_stable", False)),
        "stability_score": float(getattr(apriltag_vote, "stability_score", 0.0)),
    }


def _apriltag_detections(apriltag_result: Any) -> list[dict[str, Any]]:
    detections = []
    items = getattr(apriltag_result, "items", []) or []
    if not items and getattr(apriltag_result, "tag_id", ""):
        items = [apriltag_result]
    for item in items:
        tag_id = str(getattr(item, "tag_id", "") or "")
        if not tag_id:
            continue
        bbox = _bbox_from_apriltag(item)
        if bbox is None:
            continue
        detections.append(
            {
                "target_type": "ar_code",
                "type": "ar_code",
                "label": "ar_code",
                "name": tag_id,
                "tag_id": tag_id,
                "family": getattr(item, "family", "") or "",
                "confidence": _float_or_none(getattr(item, "decision_margin", None)),
                "bbox": bbox,
                "center": _point_payload(getattr(item, "center", None)),
                "corners": [_point_payload(point) for point in getattr(item, "corners", [])],
                "decision_margin": _float_or_none(getattr(item, "decision_margin", None)),
                "hamming": _int_or_none(getattr(item, "hamming", None)),
                "decoded_text": tag_id,
            }
        )
    return detections


def _hazmat_detections(hazmat_result: Any) -> list[dict[str, Any]]:
    detections = []
    for item in getattr(hazmat_result, "items", []) or []:
        if item.label == "UNKNOWN":
            continue
        bbox = _bbox_from_polygon(getattr(item, "polygon", []))
        if bbox is None:
            continue
        detections.append(
            {
                "target_type": "hazmat",
                "label": item.label,
                "confidence": float(item.confidence),
                "bbox": bbox,
                "decoded_text": "",
            }
        )
    return detections


def _qr_bbox(qr_result: Any, apriltag_result: Any | None) -> list[int] | None:
    if apriltag_result is not None:
        return _bbox_from_apriltag(apriltag_result)
    return _bbox_from_polygon(getattr(qr_result, "polygon", []))


def _legacy_qr_text(qr_result: Any, apriltag_result: Any | None) -> str:
    if apriltag_result is not None:
        return getattr(apriltag_result, "tag_id", "") or ""
    return getattr(qr_result, "decoded_text", "") or ""


def _legacy_qr_stable_text(qr_vote: Any, apriltag_vote: Any | None) -> str:
    if apriltag_vote is not None:
        return getattr(apriltag_vote, "stable_tag_id", "") or ""
    return getattr(qr_vote, "stable_text", "") or ""


def _bbox_from_apriltag(item: Any) -> list[int] | None:
    bbox = getattr(item, "bbox_xyxy", None)
    if bbox:
        return [int(np.floor(float(bbox[0]))), int(np.floor(float(bbox[1]))), int(np.ceil(float(bbox[2]))), int(np.ceil(float(bbox[3])))]
    return _bbox_from_polygon(getattr(item, "corners", []))


def _point_payload(point: Any | None) -> list[float] | None:
    if point is None:
        return None
    return [float(point.x), float(point.y)]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _bbox_from_polygon(polygon: list[Any] | None) -> list[int] | None:
    if not polygon:
        return None
    xs = [float(point.x) for point in polygon]
    ys = [float(point.y) for point in polygon]
    return [
        int(np.floor(min(xs))),
        int(np.floor(min(ys))),
        int(np.ceil(max(xs))),
        int(np.ceil(max(ys))),
    ]
