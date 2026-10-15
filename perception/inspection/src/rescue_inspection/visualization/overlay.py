"""Overlay rendering utilities."""

from __future__ import annotations

import cv2
import numpy as np

from rescue_inspection.models import AprilTagDetectionResult, QRDetectionResult


STATUS_LEFT = 20
STATUS_TOP = 20
STATUS_LINE_HEIGHT = 26
STATUS_BOX_PADDING_X = 10
STATUS_BOX_PADDING_Y = 8
STATUS_SECTION_GAP = 8


def _draw_status_box(
    overlay: np.ndarray,
    *,
    title: str,
    lines: list[str],
    origin: tuple[int, int],
    text_color: tuple[int, int, int] = (0, 255, 255),
    title_color: tuple[int, int, int] = (255, 255, 255),
) -> int:
    """Draw a semi-transparent status box and return the next y anchor."""

    if not lines:
        return origin[1]

    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.6
    text_scale = 0.6
    thickness = 2
    content_lines = [f"{title}"] + lines
    widths = [
        cv2.getTextSize(line, font, title_scale if index == 0 else text_scale, thickness)[0][0]
        for index, line in enumerate(content_lines)
    ]
    box_width = max(widths) + STATUS_BOX_PADDING_X * 2
    box_height = STATUS_BOX_PADDING_Y * 2 + len(content_lines) * STATUS_LINE_HEIGHT
    x1 = x - STATUS_BOX_PADDING_X
    y1 = y - STATUS_BOX_PADDING_Y
    x2 = x1 + box_width
    y2 = y1 + box_height

    shaded = overlay.copy()
    cv2.rectangle(shaded, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(shaded, 0.45, overlay, 0.55, 0.0, dst=overlay)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 60, 60), 1)

    for index, line in enumerate(content_lines):
        line_y = y + index * STATUS_LINE_HEIGHT
        cv2.putText(
            overlay,
            line,
            (x, line_y),
            font,
            title_scale if index == 0 else text_scale,
            title_color if index == 0 else text_color,
            thickness,
            cv2.LINE_AA,
        )

    return y2 + STATUS_SECTION_GAP + STATUS_BOX_PADDING_Y


def build_sensing_status_layout(
    qr_lines: list[str],
    hazmat_lines: list[str],
    *,
    origin: tuple[int, int] = (STATUS_LEFT, STATUS_TOP + STATUS_LINE_HEIGHT),
    visual_code_title: str = "QR:",
) -> list[tuple[str, list[str], int]]:
    """Build non-overlapping stacked status box anchors for sensing overlays."""

    x, start_y = origin
    layout: list[tuple[str, list[str], int]] = []
    next_y = start_y
    if qr_lines:
        layout.append((visual_code_title, qr_lines, next_y))
        next_y += STATUS_BOX_PADDING_Y * 2 + (len(qr_lines) + 1) * STATUS_LINE_HEIGHT + STATUS_SECTION_GAP
    if hazmat_lines:
        layout.append(("Hazmat:", hazmat_lines, next_y))
    return layout


def _qr_status_lines(
    detection: QRDetectionResult,
    stable_text: str,
    stability_score: float,
    debug: bool,
    is_stable: bool,
) -> list[str]:
    lines = []
    if debug:
        lines.extend([
            f"Frame: {detection.frame_id} | Source: {detection.source_type}",
            f"Decoded: {detection.decoded_text or 'NOT DETECTED'}",
            f"Method: {detection.method_used}",
        ])
        if detection.failure_reason:
            lines.append(f"Failure: {detection.failure_reason}")
        if detection.roi_candidate_count:
            lines.append(f"ROI candidates: {detection.roi_candidate_count}")

    if stable_text:
        lines.append(f"Stable: {stable_text} ({stability_score:.2f})")
    elif is_stable:
        lines.append(f"Stable: {stable_text} ({stability_score:.2f})")
    elif debug:
        lines.append(f"Stable: pending ({stability_score:.2f})")
    return lines


def _apriltag_status_lines(
    detection: AprilTagDetectionResult,
    stable_tag_id: str,
    stability_score: float,
    debug: bool,
    is_stable: bool,
) -> list[str]:
    lines = []
    if debug:
        lines.extend([
            f"Frame: {detection.frame_id} | Source: {detection.source_type}",
            f"Tag: {_apriltag_label(detection.tag_id) if detection.tag_id else 'NOT DETECTED'}",
            f"Family: {detection.family}",
        ])
        if detection.failure_reason:
            lines.append(f"Failure: {detection.failure_reason}")
        if detection.items:
            lines.append(f"Candidates: {len(detection.items)}")

    if stable_tag_id:
        lines.append(f"Stable: {_apriltag_label(stable_tag_id)} ({stability_score:.2f})")
    elif is_stable:
        lines.append(f"Stable: {_apriltag_label(stable_tag_id)} ({stability_score:.2f})")
    elif debug:
        lines.append(f"Stable: pending ({stability_score:.2f})")
    return lines


def _hazmat_status_lines(
    detection: "HazmatDetectionResult",  # type: ignore
    stable_label: str,
    stability_score: float,
    debug: bool,
    is_stable: bool,
) -> list[str]:
    lines = []
    if debug:
        lines.extend([
            f"Frame: {detection.frame_id} | Source: {detection.source_type}",
            f"Detected: {'YES' if detection.detected else 'NO'}",
            f"Candidates: {detection.candidates_count or detection.candidate_count}",
        ])
        lines.append(f"Rejected by QR mask: {detection.rejected_by_qr_mask}")
        if detection.failure_reason:
            lines.append(f"Failure: {detection.failure_reason}")
        if detection.classifier_top1_label:
            lines.append(f"Top1: {detection.classifier_top1_label} ({detection.classifier_top1_score:.2f})")
        if detection.classifier_top2_label:
            lines.append(f"Top2: {detection.classifier_top2_label} ({detection.classifier_top2_score:.2f})")
        if detection.final_decision_reason:
            lines.append(f"Decision: {detection.final_decision} ({detection.final_decision_reason})")

    if stable_label:
        lines.append(f"Stable: {stable_label} ({stability_score:.2f})")
    elif is_stable:
        lines.append(f"Stable: {stable_label} ({stability_score:.2f})")
    elif debug:
        lines.append(f"Stable: pending ({stability_score:.2f})")
    return lines


def draw_sensing_status_overlay(
    frame: np.ndarray,
    *,
    qr_detection: QRDetectionResult,
    qr_stable_text: str,
    qr_stability_score: float,
    qr_is_stable: bool,
    apriltag_detection: AprilTagDetectionResult | None = None,
    apriltag_stable_tag_id: str = "",
    apriltag_stability_score: float = 0.0,
    apriltag_is_stable: bool = False,
    hazmat_detection: "HazmatDetectionResult",  # type: ignore
    hazmat_stable_label: str,
    hazmat_stability_score: float,
    hazmat_is_stable: bool,
    debug: bool,
) -> np.ndarray:
    """Draw stacked visual-code and Hazmat status blocks without overlap."""

    overlay = frame.copy()
    visual_code_title = "QR:"
    if apriltag_detection is not None:
        qr_lines = _apriltag_status_lines(
            apriltag_detection,
            apriltag_stable_tag_id,
            apriltag_stability_score,
            debug,
            apriltag_is_stable,
        )
        visual_code_title = "AprilTag:"
    else:
        qr_lines = _qr_status_lines(qr_detection, qr_stable_text, qr_stability_score, debug, qr_is_stable)
    hazmat_lines = _hazmat_status_lines(
        hazmat_detection,
        hazmat_stable_label,
        hazmat_stability_score,
        debug,
        hazmat_is_stable,
    )
    for title, lines, y in build_sensing_status_layout(
        qr_lines,
        hazmat_lines,
        visual_code_title=visual_code_title,
    ):
        _draw_status_box(overlay, title=title, lines=lines, origin=(STATUS_LEFT, y))
    return overlay


def draw_apriltag_overlay(
    frame: np.ndarray,
    detection: AprilTagDetectionResult,
    stable_tag_id: str,
    stability_score: float,
    debug: bool,
    is_stable: bool,
    *,
    draw_status: bool = True,
) -> np.ndarray:
    """Draw AprilTag polygon and textual status on a frame."""

    overlay = frame.copy()

    if detection.corners:
        points = np.array([[point.x, point.y] for point in detection.corners], dtype=np.int32)
        cv2.polylines(overlay, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        cx, cy = points[0][0], points[0][1]
        display_tag = stable_tag_id if (stable_tag_id and is_stable) else detection.tag_id
        if not display_tag and debug:
            display_tag = "pending"
        if display_tag:
            cv2.putText(
                overlay,
                _apriltag_label(display_tag),
                (cx, max(cy - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    if draw_status:
        lines = _apriltag_status_lines(detection, stable_tag_id, stability_score, debug, is_stable)
        _draw_status_box(
            overlay,
            title="AprilTag:",
            lines=lines,
            origin=(STATUS_LEFT, STATUS_TOP + STATUS_LINE_HEIGHT),
        )

    return overlay


def draw_qr_overlay(
    frame: np.ndarray,
    detection: QRDetectionResult,
    stable_text: str,
    stability_score: float,
    debug: bool,
    is_stable: bool,
    *,
    draw_status: bool = True,
) -> np.ndarray:
    """Draw polygon and textual QR status on a frame."""

    overlay = frame.copy()

    if detection.polygon:
        if debug or detection.detected:
            points = np.array([[point.x, point.y] for point in detection.polygon], dtype=np.int32)
            cv2.polylines(overlay, [points], isClosed=True, color=(0, 255, 0), thickness=2)

            cx, cy = points[0][0], points[0][1]
            display_text = stable_text if (stable_text and is_stable) else detection.decoded_text
            if not display_text and debug:
                display_text = "pending"
            if display_text:
                cv2.putText(
                    overlay,
                    str(display_text),
                    (cx, max(cy - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

    if draw_status:
        lines = _qr_status_lines(detection, stable_text, stability_score, debug, is_stable)
        _draw_status_box(overlay, title="QR:", lines=lines, origin=(STATUS_LEFT, STATUS_TOP + STATUS_LINE_HEIGHT))

    return overlay


def _apriltag_label(tag_id: str) -> str:
    return f"AprilTag #{tag_id}" if tag_id else "AprilTag #UNKNOWN"


def draw_hazmat_overlay(
    frame: np.ndarray,
    detection: "HazmatDetectionResult",  # type: ignore # Lazy import hint for typing
    stable_label: str,
    stability_score: float,
    debug: bool,
    is_stable: bool,
    *,
    draw_status: bool = True,
) -> np.ndarray:
    """Draw polygon and textual Hazmat status on a frame."""

    overlay = frame.copy()

    for item in detection.items:
        if not debug and item.label == "UNKNOWN":
            continue

        if item.polygon:
            points = np.array([[point.x, point.y] for point in item.polygon], dtype=np.int32)
            color = (0, 255, 0) if item.label != "UNKNOWN" else (0, 0, 255)
            cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=2)

            cx, cy = points[0][0], points[0][1]
            display_label = stable_label if (item.label != "UNKNOWN" and stable_label and is_stable) else item.label
            cv2.putText(
                overlay,
                f"{display_label} ({item.confidence:.2f})",
                (cx, max(cy - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    if draw_status:
        lines = _hazmat_status_lines(detection, stable_label, stability_score, debug, is_stable)
        _draw_status_box(
            overlay,
            title="Hazmat:",
            lines=lines,
            origin=(STATUS_LEFT, STATUS_TOP + STATUS_LINE_HEIGHT),
        )

    return overlay
