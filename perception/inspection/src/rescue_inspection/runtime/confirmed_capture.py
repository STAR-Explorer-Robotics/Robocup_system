"""Shared confirmed-result capture for continuous sensing runs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from rescue_inspection.models import (
    AprilTagDetectionItem,
    AprilTagDetectionResult,
    HazmatDetectionItem,
    HazmatDetectionResult,
    QRDetectionResult,
)
from rescue_inspection.runtime.sensing_schema import SensingFrameResult, result_to_payload
from rescue_inspection.submission import export_sensing_submission
from rescue_inspection.utils.io import ensure_dir
from rescue_inspection.visualization.overlay import (
    draw_hazmat_overlay,
    draw_apriltag_overlay,
    draw_qr_overlay,
)


ConfirmedKind = Literal["apriltag", "qr", "hazmat"]


@dataclass(frozen=True)
class ConfirmedEvent:
    """One stable-result transition captured for evidence/submission output."""

    event_id: str
    kind: ConfirmedKind
    value: str
    frame_id: str
    submission_frame_id: int | None
    timestamp: dict[str, int]
    payload: dict[str, Any]
    event_dir: Path
    evidence_overlay_bgr: np.ndarray | None = None


@dataclass(frozen=True)
class LatestConfirmedRefresh:
    """One stable episode refresh for latest-confirmed UI/topic consumers."""

    kind: ConfirmedKind
    value: str
    frame_id: str
    timestamp: dict[str, int]
    payload: dict[str, Any]
    evidence_overlay_bgr: np.ndarray | None = None


@dataclass(frozen=True)
class SensingConfirmedRecordResult:
    """Confirmed artifact events plus the optional latest-confirmed refresh."""

    events: tuple[ConfirmedEvent, ...] = ()
    latest_refresh: LatestConfirmedRefresh | None = None


class SensingConfirmedCapture:
    """Capture sticky confirmed sensing results and event-driven submission artifacts."""

    def __init__(self, run_dir: Path, *, save_overlays: bool = True) -> None:
        self.run_dir = run_dir
        self.submission_dir = ensure_dir(run_dir / "submission")
        self.latest_confirmed_dir = ensure_dir(run_dir / "latest_confirmed")
        self.confirmed_events_dir = ensure_dir(run_dir / "confirmed_events")
        self.save_overlays = save_overlays
        self.latest_confirmed_apriltag: dict[str, Any] | None = None
        self.latest_confirmed_qr: dict[str, Any] | None = None
        self.latest_confirmed_hazmat: dict[str, Any] | None = None
        self.confirmed_events: list[ConfirmedEvent] = []
        self._last_apriltag_id = ""
        self._last_qr_text = ""
        self._last_hazmat_label = ""
        self._last_refresh_episode_keys: tuple[tuple[ConfirmedKind, str], ...] = ()

    def record_frame(
        self,
        result: SensingFrameResult,
        *,
        frame_bgr: np.ndarray | None = None,
        overlay_bgr: np.ndarray | None = None,
    ) -> SensingConfirmedRecordResult:
        """Record one frame and refresh latest-confirmed once per stable episode."""

        events: list[ConfirmedEvent] = []
        latest_refresh: LatestConfirmedRefresh | None = None
        apriltag_vote = getattr(result, "apriltag_vote", None)
        apriltag_tag_id = str(getattr(apriltag_vote, "stable_tag_id", "") or "")
        apriltag_is_stable = bool(getattr(apriltag_vote, "is_stable", False)) and bool(apriltag_tag_id)
        if apriltag_is_stable:
            self.latest_confirmed_apriltag = self._confirmed_apriltag_payload(result, apriltag_tag_id)
        if apriltag_is_stable and apriltag_tag_id != self._last_apriltag_id:
            self._last_apriltag_id = apriltag_tag_id
            events.append(
                self._write_event(
                    result,
                    kind="apriltag",
                    value=apriltag_tag_id,
                    frame_bgr=frame_bgr,
                    overlay_bgr=overlay_bgr,
                )
            )

        qr_text = "" if apriltag_is_stable else str(getattr(result.qr_vote, "stable_text", "") or "")
        qr_is_stable = bool(getattr(result.qr_vote, "is_stable", False)) and bool(qr_text) and not apriltag_is_stable
        if qr_is_stable:
            self.latest_confirmed_qr = self._confirmed_qr_payload(result, qr_text)
        if qr_is_stable and qr_text != self._last_qr_text:
            self._last_qr_text = qr_text
            events.append(
                self._write_event(
                    result,
                    kind="qr",
                    value=qr_text,
                    frame_bgr=frame_bgr,
                    overlay_bgr=overlay_bgr,
                )
            )

        hazmat_label = str(getattr(result.hazmat_vote, "stable_label", "") or "")
        hazmat_is_stable = (
            bool(getattr(result.hazmat_vote, "is_stable", False))
            and bool(hazmat_label)
            and hazmat_label != "UNKNOWN"
        )
        if hazmat_is_stable:
            self.latest_confirmed_hazmat = self._confirmed_hazmat_payload(result, hazmat_label)
        if (
            hazmat_is_stable
            and hazmat_label != self._last_hazmat_label
        ):
            self._last_hazmat_label = hazmat_label
            events.append(
                self._write_event(
                    result,
                    kind="hazmat",
                    value=hazmat_label,
                    frame_bgr=frame_bgr,
                    overlay_bgr=overlay_bgr,
                )
            )

        active_keys = _active_stable_keys(
            apriltag_tag_id=apriltag_tag_id if apriltag_is_stable else "",
            qr_text=qr_text if qr_is_stable else "",
            hazmat_label=hazmat_label if hazmat_is_stable else "",
        )
        if active_keys:
            refresh_key = _select_refresh_key(active_keys, self._last_refresh_episode_keys)
            if active_keys != self._last_refresh_episode_keys and refresh_key is not None:
                event_refresh = next(
                    (
                        event
                        for event in reversed(events)
                        if event.kind == refresh_key[0] and event.value == refresh_key[1]
                    ),
                    None,
                )
                if event_refresh is not None:
                    latest_refresh = _refresh_from_event(event_refresh)
                else:
                    latest_refresh = self._write_latest_refresh(
                        result,
                        kind=refresh_key[0],
                        value=refresh_key[1],
                        frame_bgr=frame_bgr,
                    )
            self._last_refresh_episode_keys = active_keys
        else:
            self._last_refresh_episode_keys = ()

        return SensingConfirmedRecordResult(events=tuple(events), latest_refresh=latest_refresh)

    def latest_payload(self) -> dict[str, Any]:
        """Return the sticky latest confirmed state."""

        return {
            "apriltag": self.latest_confirmed_apriltag,
            "qr": self.latest_confirmed_qr,
            "hazmat": self.latest_confirmed_hazmat,
            "confirmed_events_count": len(self.confirmed_events),
        }

    @property
    def latest_submission_frame_id(self) -> int | None:
        """Return the frame id that last updated submission evidence."""

        if not self.confirmed_events:
            return None
        event = self.confirmed_events[-1]
        if event.submission_frame_id is not None:
            return event.submission_frame_id
        try:
            return int(event.frame_id)
        except (TypeError, ValueError):
            return None

    def finalize_empty_submission(self) -> None:
        """Ensure submission files exist when no confirmed event was ever seen."""

        result_path = self.submission_dir / "result.json"
        text_path = self.submission_dir / "decoded_message.txt"
        if not result_path.exists():
            result_path.write_text("{}\n", encoding="utf-8")
        if not text_path.exists():
            text_path.write_text("", encoding="utf-8")

    def _write_event(
        self,
        result: SensingFrameResult,
        *,
        kind: ConfirmedKind,
        value: str,
        frame_bgr: np.ndarray | None,
        overlay_bgr: np.ndarray | None,
    ) -> ConfirmedEvent:
        event_id = f"{len(self.confirmed_events) + 1:04d}_{kind}_{_slug(value)}"
        event_dir = ensure_dir(self.confirmed_events_dir / event_id)
        payload = self.latest_payload()
        payload["event"] = {
            "event_id": event_id,
            "kind": kind,
            "value": value,
            "frame_id": result.frame_id,
            "submission_frame_id": _int_frame_id(result),
            "timestamp": {"sec": result.stamp_sec, "nanosec": result.stamp_nanosec},
        }

        overlay_name, evidence_overlay = self._write_overlay(
            result,
            event_dir,
            kind=kind,
            frame_bgr=frame_bgr,
            overlay_bgr=overlay_bgr,
        )
        if overlay_name:
            payload["paths"] = {
                "confirmed_event_result_json": str(event_dir / "result.json"),
                "confirmed_event_overlay": str(event_dir / overlay_name),
                "latest_confirmed_result_json": str(self.latest_confirmed_dir / "result.json"),
                "latest_confirmed_overlay": str(self.latest_confirmed_dir / overlay_name),
                "submission_result_json": str(self.submission_dir / "result.json"),
                "submission_overlay": str(self.submission_dir / overlay_name),
            }

        self._write_json(event_dir / "result.json", payload)
        self._write_text(event_dir / "result.txt", payload)
        self._write_json(self.latest_confirmed_dir / "result.json", payload)
        self._write_text(self.latest_confirmed_dir / "result.txt", payload)

        if overlay_name:
            shutil.copyfile(event_dir / overlay_name, self.latest_confirmed_dir / overlay_name)

        apriltag_result = _submission_apriltag_result(result)
        qr_result = _submission_qr_result(result)
        hazmat_result = _submission_hazmat_result(result)
        export_sensing_submission(
            artifact_dir=self.run_dir,
            qr_detected=self.latest_confirmed_apriltag is not None or self.latest_confirmed_qr is not None,
            qr_message=str(
                (self.latest_confirmed_apriltag or {}).get("tag_id")
                or (self.latest_confirmed_qr or {}).get("decoded_text")
                or ""
            ),
            qr_stability_score=float(
                (self.latest_confirmed_apriltag or {}).get("stability_score")
                or (self.latest_confirmed_qr or {}).get("stability_score")
                or 0.0
            ),
            qr_result=qr_result,
            apriltag_detected=self.latest_confirmed_apriltag is not None,
            apriltag_tag_id=str((self.latest_confirmed_apriltag or {}).get("tag_id") or ""),
            apriltag_stability_score=float((self.latest_confirmed_apriltag or {}).get("stability_score") or 0.0),
            apriltag_result=apriltag_result,
            hazmat_detected=self.latest_confirmed_hazmat is not None,
            hazmat_label=str((self.latest_confirmed_hazmat or {}).get("label") or ""),
            hazmat_stability_score=float((self.latest_confirmed_hazmat or {}).get("stability_score") or 0.0),
            hazmat_result=hazmat_result,
            best_frame_id=_int_frame_id(result),
            source_type=_source_type(result),
            overlay_name=overlay_name,
        )
        if overlay_name:
            shutil.copyfile(event_dir / overlay_name, self.submission_dir / overlay_name)

        event = ConfirmedEvent(
            event_id=event_id,
            kind=kind,
            value=value,
            frame_id=result.frame_id,
            submission_frame_id=_int_frame_id(result),
            timestamp={"sec": result.stamp_sec, "nanosec": result.stamp_nanosec},
            payload=payload,
            event_dir=event_dir,
            evidence_overlay_bgr=evidence_overlay,
        )
        self.confirmed_events.append(event)
        return event

    def _write_latest_refresh(
        self,
        result: SensingFrameResult,
        *,
        kind: ConfirmedKind,
        value: str,
        frame_bgr: np.ndarray | None,
    ) -> LatestConfirmedRefresh:
        payload = self.latest_payload()
        payload["event"] = {
            "event_id": None,
            "kind": kind,
            "value": value,
            "frame_id": result.frame_id,
            "submission_frame_id": self.latest_submission_frame_id,
            "timestamp": {"sec": result.stamp_sec, "nanosec": result.stamp_nanosec},
            "latest_refresh_only": True,
        }
        evidence_overlay = _draw_evidence_overlay(result, frame_bgr, kind=kind) if frame_bgr is not None else None
        if evidence_overlay is not None and self.save_overlays:
            cv2.imwrite(str(self.latest_confirmed_dir / "overlay.png"), evidence_overlay)
            payload["paths"] = {
                "latest_confirmed_result_json": str(self.latest_confirmed_dir / "result.json"),
                "latest_confirmed_overlay": str(self.latest_confirmed_dir / "overlay.png"),
                "submission_result_json": str(self.submission_dir / "result.json"),
                "submission_overlay": str(self.submission_dir / "overlay.png"),
            }
        self._write_json(self.latest_confirmed_dir / "result.json", payload)
        self._write_text(self.latest_confirmed_dir / "result.txt", payload)
        return LatestConfirmedRefresh(
            kind=kind,
            value=value,
            frame_id=result.frame_id,
            timestamp={"sec": result.stamp_sec, "nanosec": result.stamp_nanosec},
            payload=payload,
            evidence_overlay_bgr=evidence_overlay,
        )

    def _write_overlay(
        self,
        result: SensingFrameResult,
        event_dir: Path,
        *,
        kind: ConfirmedKind,
        frame_bgr: np.ndarray | None,
        overlay_bgr: np.ndarray | None,
    ) -> tuple[str | None, np.ndarray | None]:
        if not self.save_overlays:
            return None, None
        if overlay_bgr is not None:
            cv2.imwrite(str(event_dir / "debug_overlay.png"), overlay_bgr)
        image = _draw_evidence_overlay(result, frame_bgr, kind=kind) if frame_bgr is not None else None
        if image is None and overlay_bgr is not None:
            image = overlay_bgr.copy()
        if image is None:
            return None, None
        cv2.imwrite(str(event_dir / "overlay.png"), image)
        return "overlay.png", image

    def _confirmed_qr_payload(self, result: SensingFrameResult, text: str) -> dict[str, Any]:
        payload = result_to_payload(result)["qr"]
        payload["decoded_text"] = text
        payload["stable_text"] = text
        payload["is_stable"] = True
        payload["last_seen"] = {"frame_id": result.frame_id, "stamp": {"sec": result.stamp_sec, "nanosec": result.stamp_nanosec}}
        payload["source_topic"] = result.source_topic
        return payload

    def _confirmed_apriltag_payload(self, result: SensingFrameResult, tag_id: str) -> dict[str, Any]:
        payload = result_to_payload(result)["apriltag"]
        payload["tag_id"] = tag_id
        payload["stable_tag_id"] = tag_id
        payload["is_stable"] = True
        payload["last_seen"] = {"frame_id": result.frame_id, "stamp": {"sec": result.stamp_sec, "nanosec": result.stamp_nanosec}}
        payload["source_topic"] = result.source_topic
        return payload

    def _confirmed_hazmat_payload(self, result: SensingFrameResult, label: str) -> dict[str, Any]:
        payload = result_to_payload(result)["hazmat"]
        payload["label"] = label
        payload["stable_label"] = label
        payload["is_stable"] = True
        payload["last_seen"] = {"frame_id": result.frame_id, "stamp": {"sec": result.stamp_sec, "nanosec": result.stamp_nanosec}}
        payload["source_topic"] = result.source_topic
        return payload

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _write_text(path: Path, payload: dict[str, Any]) -> None:
        lines = ["task=COMBINED_SENSING"]
        apriltag = payload.get("apriltag") or {}
        qr = payload.get("qr") or {}
        hazmat = payload.get("hazmat") or {}
        if apriltag:
            lines.append(f"apriltag_confirmed={apriltag.get('tag_id') or apriltag.get('stable_tag_id') or 'UNKNOWN'}")
        if qr:
            lines.append(f"qr_confirmed={qr.get('decoded_text') or qr.get('stable_text') or 'UNKNOWN'}")
        if hazmat:
            lines.append(f"hazmat_confirmed={hazmat.get('label') or hazmat.get('stable_label') or 'UNKNOWN'}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _submission_qr_result(result: SensingFrameResult) -> QRDetectionResult | None:
    if getattr(result, "apriltag_vote", None) is not None:
        text = str(getattr(result.apriltag_vote, "stable_tag_id", "") or "")
        if not text:
            return None
        qr_result = result.qr_result
        items = [
            item.model_copy(update={"decoded_text": text})
            for item in getattr(qr_result, "items", []) or []
            if not getattr(item, "is_suspect", False)
        ]
        return qr_result.model_copy(
            update={
                "decoded_text": text,
                "items": items,
                "stability_score": float(getattr(result.apriltag_vote, "stability_score", 0.0)),
            }
        )
    text = str(getattr(result.qr_vote, "stable_text", "") or "")
    if not text:
        return None
    qr_result = getattr(result.qr_vote, "representative_result", None) or result.qr_result
    items = [
        item.model_copy(update={"decoded_text": text})
        for item in getattr(qr_result, "items", []) or []
        if not getattr(item, "is_suspect", False)
    ]
    return qr_result.model_copy(
        update={
            "decoded_text": text,
            "items": items,
            "stability_score": float(getattr(result.qr_vote, "stability_score", 0.0)),
        }
    )


def _submission_apriltag_result(result: SensingFrameResult) -> AprilTagDetectionResult | None:
    vote = getattr(result, "apriltag_vote", None)
    tag_id = str(getattr(vote, "stable_tag_id", "") or "")
    if not tag_id:
        return None
    apriltag_result = getattr(vote, "representative_result", None) or getattr(result, "apriltag_result", None)
    if apriltag_result is None:
        return None
    items: list[AprilTagDetectionItem] = []
    for item in getattr(apriltag_result, "items", []) or []:
        if item.tag_id == getattr(apriltag_result, "tag_id", ""):
            items.append(item.model_copy(update={"tag_id": tag_id}))
        else:
            items.append(item)
    return apriltag_result.model_copy(
        update={
            "tag_id": tag_id,
            "items": items,
            "stability_score": float(getattr(vote, "stability_score", 0.0)),
        }
    )


def _submission_hazmat_result(result: SensingFrameResult) -> HazmatDetectionResult | None:
    label = str(getattr(result.hazmat_vote, "stable_label", "") or "")
    if not label or label == "UNKNOWN":
        return None
    hazmat_result = getattr(result.hazmat_vote, "representative_result", None) or result.hazmat_result
    items: list[HazmatDetectionItem] = []
    for item in getattr(hazmat_result, "items", []) or []:
        if item.label == "UNKNOWN":
            items.append(item)
        else:
            items.append(item.model_copy(update={"label": label, "raw_label": label}))
    return hazmat_result.model_copy(
        update={
            "best_label": label,
            "items": items,
            "stability_score": float(getattr(result.hazmat_vote, "stability_score", 0.0)),
        }
    )


def _draw_evidence_overlay(result: SensingFrameResult, frame_bgr: np.ndarray, *, kind: ConfirmedKind) -> np.ndarray:
    """Draw clean confirmed evidence overlay without debug status blocks."""

    overlay = frame_bgr.copy()
    if kind == "apriltag":
        apriltag_result = getattr(result, "apriltag_result", None)
        apriltag_vote = getattr(result, "apriltag_vote", None)
        if apriltag_result is not None and apriltag_vote is not None:
            tag_id = str(getattr(apriltag_vote, "stable_tag_id", "") or "")
            score = float(getattr(apriltag_vote, "stability_score", 0.0))
            is_stable = bool(getattr(apriltag_vote, "is_stable", False))
            overlay = draw_apriltag_overlay(overlay, apriltag_result, tag_id, score, False, is_stable, draw_status=False)
    if kind == "qr":
        qr_text = str(getattr(result.qr_vote, "stable_text", "") or "")
        qr_score = float(getattr(result.qr_vote, "stability_score", 0.0))
        qr_is_stable = bool(getattr(result.qr_vote, "is_stable", False))
        overlay = draw_qr_overlay(overlay, result.qr_result, qr_text, qr_score, False, qr_is_stable, draw_status=False)
    if kind == "hazmat":
        hazmat_label = str(getattr(result.hazmat_vote, "stable_label", "") or "")
        hazmat_score = float(getattr(result.hazmat_vote, "stability_score", 0.0))
        hazmat_is_stable = bool(getattr(result.hazmat_vote, "is_stable", False))
        overlay = draw_hazmat_overlay(
            overlay,
            result.hazmat_result,
            hazmat_label,
            hazmat_score,
            False,
            hazmat_is_stable,
            draw_status=False,
        )
    return overlay


def _active_stable_keys(
    *,
    apriltag_tag_id: str = "",
    qr_text: str,
    hazmat_label: str,
) -> tuple[tuple[ConfirmedKind, str], ...]:
    keys: list[tuple[ConfirmedKind, str]] = []
    if apriltag_tag_id:
        keys.append(("apriltag", apriltag_tag_id))
    if qr_text:
        keys.append(("qr", qr_text))
    if hazmat_label and hazmat_label != "UNKNOWN":
        keys.append(("hazmat", hazmat_label))
    return tuple(keys)


def _select_refresh_key(
    active_keys: tuple[tuple[ConfirmedKind, str], ...],
    previous_keys: tuple[tuple[ConfirmedKind, str], ...],
) -> tuple[ConfirmedKind, str] | None:
    if active_keys == previous_keys:
        return None
    previous_set = set(previous_keys)
    for key in reversed(active_keys):
        if key not in previous_set:
            return key
    return active_keys[-1]


def _refresh_from_event(event: ConfirmedEvent) -> LatestConfirmedRefresh:
    return LatestConfirmedRefresh(
        kind=event.kind,
        value=event.value,
        frame_id=event.frame_id,
        timestamp=event.timestamp,
        payload=event.payload,
        evidence_overlay_bgr=event.evidence_overlay_bgr,
    )


def _source_type(result: SensingFrameResult) -> str:
    return str(getattr(result.qr_result, "source_type", "") or getattr(result.hazmat_result, "source_type", "") or "ros2")


def _int_frame_id(result: SensingFrameResult) -> int | None:
    try:
        return int(result.frame_id)
    except (TypeError, ValueError):
        pass
    for detection_result in (result.qr_result, result.hazmat_result):
        frame_id = getattr(detection_result, "frame_id", None)
        if frame_id is not None:
            return int(frame_id)
    return None


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    return slug[:48] or "unknown"
