"""Run artifact writer for external sensing runtime integrations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rescue_inspection.runtime.confirmed_capture import SensingConfirmedCapture, SensingConfirmedRecordResult
from rescue_inspection.runtime.sensing_schema import SensingFrameResult, result_to_json
from rescue_inspection.utils.io import create_video_writer


class SensingRunArtifactWriter:
    """Write inspection-like artifacts for a continuous sensing run."""

    def __init__(
        self,
        *,
        output_root: Path,
        image_topic: str,
        overlay_topic: str,
        result_topic: str,
        status_topic: str,
        inspection_config: Path,
        save_overlay_video: bool,
        output_video_fps: float,
        run_id_prefix: str = "ros2_sensing",
        latest_result_write_period_sec: float = 0.0,
        save_confirmed_overlays: bool = True,
    ) -> None:
        run_id = time.strftime(f"{run_id_prefix}_%Y%m%d_%H%M%S")
        self.run_dir = output_root / run_id
        self.submission_dir = self.run_dir / "submission"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.submission_dir.mkdir(parents=True, exist_ok=True)

        self.image_topic = image_topic
        self.overlay_topic = overlay_topic
        self.result_topic = result_topic
        self.status_topic = status_topic
        self.inspection_config = inspection_config
        self.save_overlay_video = save_overlay_video
        self.output_video_fps = max(float(output_video_fps), 0.1)
        self.latest_result_write_period_sec = max(float(latest_result_write_period_sec), 0.0)
        self._last_latest_result_write_monotonic = 0.0
        self._video_writer: Any | None = None
        self._latest_result_payload: dict[str, Any] | None = None
        self._latest_overlay_payload: dict[str, Any] | None = None
        self._latest_inspection_frame_id: int | None = None
        self.confirmed_capture = SensingConfirmedCapture(self.run_dir, save_overlays=save_confirmed_overlays)

    def write_result(self, payload: str) -> None:
        """Append per-frame result JSON and refresh latest result."""

        with (self.run_dir / "detections.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        if self._should_write_latest_result():
            (self.run_dir / "latest_result.json").write_text(payload + "\n", encoding="utf-8")
        try:
            self._latest_result_payload = json.loads(payload)
        except json.JSONDecodeError:
            self._latest_result_payload = None

    def record_frame_result(
        self,
        result: SensingFrameResult,
        *,
        frame_bgr: np.ndarray | None = None,
        overlay_bgr: np.ndarray | None = None,
    ) -> SensingConfirmedRecordResult:
        """Write per-frame JSON and capture any new confirmed stable result."""

        self.write_result(result_to_json(result))
        self._latest_inspection_frame_id = _result_inspection_frame_id(result)
        return self.confirmed_capture.record_frame(result, frame_bgr=frame_bgr, overlay_bgr=overlay_bgr)

    def _should_write_latest_result(self) -> bool:
        now = time.monotonic()
        if self.latest_result_write_period_sec <= 0.0:
            self._last_latest_result_write_monotonic = now
            return True
        if now - self._last_latest_result_write_monotonic >= self.latest_result_write_period_sec:
            self._last_latest_result_write_monotonic = now
            return True
        return False

    def write_status(self, payload: str) -> None:
        """Append node status JSON and refresh latest status."""

        with (self.run_dir / "status.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        (self.run_dir / "latest_status.json").write_text(payload + "\n", encoding="utf-8")
        try:
            status_payload = json.loads(payload)
        except json.JSONDecodeError:
            status_payload = {}
        self.write_summary(node_summary=status_payload)

    def write_overlay(self, image_bgr: np.ndarray) -> None:
        """Write latest overlay PNG and append overlay video frames."""

        overlay_path = self.run_dir / "latest_overlay.png"
        cv2.imwrite(str(overlay_path), image_bgr)
        self._write_latest_overlay_metadata(overlay_path=overlay_path)
        if not self.save_overlay_video:
            return
        if self._video_writer is None:
            height, width = image_bgr.shape[:2]
            self._video_writer = create_video_writer(
                self.run_dir / "overlay.mp4",
                fps=self.output_video_fps,
                size=(width, height),
            )
        self._video_writer.write(image_bgr)

    def finalize(self, *, node_summary: dict[str, Any]) -> None:
        """Release video resources and write final summary.json."""

        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        self.write_summary(node_summary=node_summary)

    def write_summary(self, *, node_summary: dict[str, Any]) -> None:
        """Refresh summary.json with the latest known run state."""

        summary = {
            "run_id": self.run_dir.name,
            "image_topic": self.image_topic,
            "overlay_topic": self.overlay_topic,
            "result_topic": self.result_topic,
            "status_topic": self.status_topic,
            "inspection_config": str(self.inspection_config),
            "output_dir": str(self.run_dir),
            "latest_result": self._latest_result_payload,
            "latest_overlay": self._latest_overlay_payload,
            "latest_confirmed": self.confirmed_capture.latest_payload(),
            "best_frame_id": self.confirmed_capture.latest_submission_frame_id,
            "submission_frame_id": self.confirmed_capture.latest_submission_frame_id,
            "best_apriltag_tag_id": str(
                (self.confirmed_capture.latest_confirmed_apriltag or {}).get("tag_id") or ""
            ),
            "best_qr_text": str(
                (self.confirmed_capture.latest_confirmed_apriltag or {}).get("tag_id")
                or (self.confirmed_capture.latest_confirmed_qr or {}).get("decoded_text")
                or ""
            ),
            "best_qr_stability_score": float(
                (self.confirmed_capture.latest_confirmed_apriltag or {}).get("stability_score")
                or (self.confirmed_capture.latest_confirmed_qr or {}).get("stability_score")
                or 0.0
            ),
            **node_summary,
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not (self.submission_dir / "result.json").exists():
            (self.submission_dir / "result.json").write_text("{}\n", encoding="utf-8")
        if not (self.submission_dir / "decoded_message.txt").exists():
            (self.submission_dir / "decoded_message.txt").write_text("", encoding="utf-8")

    def _write_latest_overlay_metadata(self, *, overlay_path: Path) -> None:
        frame_id = _latest_result_frame_id(self._latest_result_payload)
        stamp = None
        if isinstance(self._latest_result_payload, dict):
            stamp = self._latest_result_payload.get("stamp")
        payload = {
            "overlay_path": overlay_path.name,
            "frame_id": frame_id,
            "inspection_frame_id": self._latest_inspection_frame_id,
            "stamp": stamp,
            "reason": "latest rendered live overlay snapshot",
            "final_evidence_overlay": "submission/overlay.png",
        }
        self._latest_overlay_payload = payload
        (self.run_dir / "latest_overlay.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _latest_result_frame_id(payload: dict[str, Any] | None) -> Any:
    if not isinstance(payload, dict):
        return None
    frame_id = payload.get("frame_id")
    if frame_id not in (None, ""):
        return frame_id
    for section in ("qr", "hazmat"):
        section_payload = payload.get(section)
        if isinstance(section_payload, dict) and section_payload.get("frame_id") not in (None, ""):
            return section_payload.get("frame_id")
    return None


def _result_inspection_frame_id(result: SensingFrameResult) -> int | None:
    for detection_result in (result.qr_result, result.hazmat_result):
        frame_id = getattr(detection_result, "frame_id", None)
        if frame_id is not None:
            return int(frame_id)
    return None
