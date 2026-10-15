"""Pure OpenCV renderer for the latest confirmed evidence image viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np


PanelStatus = Literal["WAITING", "CONFIRMED", "STALE"]

PANEL_TITLE = "Latest Confirmed Evidence"
EVIDENCE_HINT = "submission/overlay.png is final evidence; latest_overlay.png is live debug snapshot."


@dataclass(frozen=True)
class PanelState:
    """Derived display state for the latest confirmed payload."""

    status: PanelStatus
    task_type: str
    age_sec: float | None
    updated_at_sec: float | None
    has_payload: bool


def compute_panel_state(
    payload: dict[str, Any] | None,
    *,
    now: float | None = None,
    stale_threshold_sec: float = 10.0,
) -> PanelState:
    """Compute WAITING/CONFIRMED/STALE without mutating confirmed semantics."""

    if not isinstance(payload, dict) or not _has_confirmed_payload(payload):
        return PanelState(
            status="WAITING",
            task_type="none",
            age_sec=None,
            updated_at_sec=None,
            has_payload=False,
        )

    updated_at = _latest_timestamp_sec(payload)
    age_sec = max(float(now) - updated_at, 0.0) if now is not None and updated_at is not None else None
    status: PanelStatus = "CONFIRMED"
    if age_sec is not None and age_sec > max(float(stale_threshold_sec), 0.0):
        status = "STALE"
    return PanelState(
        status=status,
        task_type=_task_type(payload),
        age_sec=age_sec,
        updated_at_sec=updated_at,
        has_payload=True,
    )


def render_waiting_placeholder(width: int = 1280, height: int = 720) -> np.ndarray:
    """Render a waiting placeholder when no confirmed evidence image exists."""

    width = max(int(width), 480)
    height = max(int(height), 300)
    image = np.full((height, width, 3), (28, 35, 45), dtype=np.uint8)
    _put_text(image, PANEL_TITLE, (34, 58), 0.9, (245, 248, 252), 2)
    _put_text(image, "WAITING FOR CONFIRMED EVIDENCE IMAGE", (34, height // 2), 0.75, (180, 195, 215), 2)
    _put_text(image, "Listening for /qrc/sensing/latest_confirmed_overlay/image_raw", (34, height // 2 + 42), 0.5, (140, 154, 174), 1)
    _put_text(image, EVIDENCE_HINT, (34, height - 34), 0.45, (150, 165, 185), 1)
    return image


def render_evidence_view(
    image_bgr: np.ndarray | None,
    *,
    payload: dict[str, Any] | None = None,
    status: dict[str, Any] | None = None,
    now: float | None = None,
    width: int = 1280,
    height: int = 720,
    stale_threshold_sec: float = 10.0,
) -> np.ndarray:
    """Render the latest confirmed evidence image with small UI chrome."""

    width = max(int(width), 480)
    height = max(int(height), 300)
    state = compute_panel_state(payload, now=now, stale_threshold_sec=stale_threshold_sec)
    if image_bgr is None:
        return render_waiting_placeholder(width=width, height=height)

    canvas = np.full((height, width, 3), (16, 20, 28), dtype=np.uint8)
    evidence = _as_bgr_uint8(image_bgr)
    scaled = _fit_image(evidence, max_width=width, max_height=height)
    y0 = max((height - scaled.shape[0]) // 2, 0)
    x0 = max((width - scaled.shape[1]) // 2, 0)
    canvas[y0 : y0 + scaled.shape[0], x0 : x0 + scaled.shape[1]] = scaled

    return canvas


def render_latest_confirmed_panel(
    payload: dict[str, Any] | None,
    *,
    evidence_image_bgr: np.ndarray | None = None,
    status: dict[str, Any] | None = None,
    now: float | None = None,
    width: int = 1280,
    height: int = 720,
    stale_threshold_sec: float = 10.0,
) -> np.ndarray:
    """Compatibility wrapper for the result panel node."""

    return render_evidence_view(
        evidence_image_bgr,
        payload=payload,
        status=status,
        now=now,
        width=width,
        height=height,
        stale_threshold_sec=stale_threshold_sec,
    )


def _fit_image(image: np.ndarray, *, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    new_size = (max(int(width * scale), 1), max(int(height * scale), 1))
    if new_size == (width, height):
        return image.copy()
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _as_bgr_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return np.ascontiguousarray(image)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Expected image shape HxW, HxWx3, or HxWx4, got {image.shape}")


def _draw_badge(image: np.ndarray, state: PanelStatus) -> None:
    color = {"WAITING": (75, 165, 255), "CONFIRMED": (70, 190, 115), "STALE": (55, 145, 235)}[state]
    x2 = image.shape[1] - 24
    x1 = max(x2 - 170, 24)
    cv2.rectangle(image, (x1, 20), (x2, 56), color, thickness=-1)
    _put_text(image, state, (x1 + 14, 45), 0.55, (255, 255, 255), 2)


def _draw_footer(image: np.ndarray, payload: dict[str, Any], status: dict[str, Any], state: PanelState) -> None:
    footer_top = image.shape[0] - 42
    cv2.rectangle(image, (0, footer_top), (image.shape[1], image.shape[0]), (20, 26, 35), thickness=-1)
    left = f"{state.task_type.upper()} | age={_age_text(state.age_sec)} | frame={_frame_id(payload)} | value={_event_value(payload)}"
    right = f"frames={_value_text(status.get('frames_processed'))} | {EVIDENCE_HINT}"
    _put_text(image, _clip_text(left, max(image.shape[1] // 13, 40)), (24, footer_top + 27), 0.45, (230, 236, 245), 1)
    _put_text(
        image,
        _clip_text(right, max(image.shape[1] // 13, 40)),
        (max(image.shape[1] // 2, 24), footer_top + 27),
        0.43,
        (155, 170, 190),
        1,
    )


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(image, str(text), origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _has_confirmed_payload(payload: dict[str, Any]) -> bool:
    return bool(
        _mapping_or_empty(payload.get("apriltag"))
        or _mapping_or_empty(payload.get("qr"))
        or _mapping_or_empty(payload.get("hazmat"))
    )


def _task_type(payload: dict[str, Any]) -> str:
    has_apriltag = bool(_mapping_or_empty(payload.get("apriltag")))
    has_qr = bool(_mapping_or_empty(payload.get("qr")))
    has_hazmat = bool(_mapping_or_empty(payload.get("hazmat")))
    if (has_apriltag or has_qr) and has_hazmat:
        return "combined"
    if has_apriltag:
        return "apriltag"
    if has_qr:
        return "qr"
    if has_hazmat:
        return "hazmat"
    return "none"


def _latest_timestamp_sec(payload: dict[str, Any]) -> float | None:
    candidates = [
        _float_or_none(payload.get("_panel_received_at_sec")),
        _float_or_none(payload.get("panel_received_at_sec")),
        _stamp_to_sec(_mapping_or_empty(_mapping_or_empty(payload.get("event")).get("timestamp"))),
        _stamp_to_sec(
            _mapping_or_empty(_mapping_or_empty(_mapping_or_empty(payload.get("apriltag")).get("last_seen")).get("stamp"))
        ),
        _stamp_to_sec(_mapping_or_empty(_mapping_or_empty(_mapping_or_empty(payload.get("qr")).get("last_seen")).get("stamp"))),
        _stamp_to_sec(
            _mapping_or_empty(_mapping_or_empty(_mapping_or_empty(payload.get("hazmat")).get("last_seen")).get("stamp"))
        ),
    ]
    valid = [value for value in candidates if value is not None]
    return max(valid) if valid else None


def _stamp_to_sec(stamp: dict[str, Any]) -> float | None:
    if not stamp:
        return None
    try:
        return float(stamp.get("sec", 0)) + float(stamp.get("nanosec", 0)) / 1_000_000_000.0
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _frame_id(payload: dict[str, Any]) -> str:
    event = _mapping_or_empty(payload.get("event"))
    if event.get("frame_id") not in (None, ""):
        return str(event.get("frame_id"))
    for section in ("hazmat", "apriltag", "qr"):
        last_seen = _mapping_or_empty(_mapping_or_empty(payload.get(section)).get("last_seen"))
        if last_seen.get("frame_id") not in (None, ""):
            return str(last_seen.get("frame_id"))
    return "N/A"


def _event_value(payload: dict[str, Any]) -> str:
    event = _mapping_or_empty(payload.get("event"))
    if event.get("value") not in (None, ""):
        return str(event.get("value"))
    apriltag = _mapping_or_empty(payload.get("apriltag"))
    qr = _mapping_or_empty(payload.get("qr"))
    hazmat = _mapping_or_empty(payload.get("hazmat"))
    return str(
        apriltag.get("tag_id")
        or apriltag.get("stable_tag_id")
        or qr.get("decoded_text")
        or qr.get("stable_text")
        or hazmat.get("label")
        or hazmat.get("stable_label")
        or "N/A"
    )


def _age_text(age_sec: float | None) -> str:
    return "N/A" if age_sec is None else f"{age_sec:.1f}s"


def _value_text(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    return str(value)


def _clip_text(text: str, max_chars: int) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."
