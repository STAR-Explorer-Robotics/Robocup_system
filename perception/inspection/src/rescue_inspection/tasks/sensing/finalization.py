"""Final artifact export for native combined sensing runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2

from rescue_inspection.config import AppConfig
from rescue_inspection.models import SensingRunSummary
from rescue_inspection.runtime.confirmed_capture import SensingConfirmedCapture
from rescue_inspection.submission import export_sensing_submission
from rescue_inspection.tasks.sensing.aggregation import SensingRunAggregationResult
from rescue_inspection.utils.io import ensure_dir, write_json
from rescue_inspection.visualization.overlay import (
    draw_apriltag_overlay,
    draw_hazmat_overlay,
    draw_qr_overlay,
)


class SensingRunFinalizer:
    """Write native sensing summary and submission artifacts after the source loop."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def finalize_run(
        self,
        *,
        run_id: str,
        source: Any,
        artifact_dir: Path,
        artifact_paths: dict[str, str],
        aggregation: SensingRunAggregationResult,
        confirmed_capture: SensingConfirmedCapture | None = None,
    ) -> SensingRunSummary:
        """Finalize a native sensing run without owning the source loop or stream writer."""

        counters = aggregation.counters
        final_qr_vote = aggregation.final_qr_vote
        final_apriltag_vote = aggregation.final_apriltag_vote
        final_hazmat_vote = aggregation.final_hazmat_vote
        selected_frame_id = aggregation.selected_frame_id
        if confirmed_capture is not None and confirmed_capture.latest_submission_frame_id is not None:
            selected_frame_id = confirmed_capture.latest_submission_frame_id
        submission_qr_result = aggregation.submission_qr_result
        submission_apriltag_result = aggregation.submission_apriltag_result
        submission_hazmat_result = aggregation.submission_hazmat_result
        submission_overlay_name = None
        if confirmed_capture is None or not confirmed_capture.confirmed_events:
            submission_overlay_name = self._write_submission_overlay(
                artifact_dir=artifact_dir,
                source_type=source.source_type,
                aggregation=aggregation,
            )

        if self.config.output.save_summary:
            artifact_paths["summary_json"] = str(artifact_dir / "summary.json")

        summary = SensingRunSummary(
            run_id=run_id,
            source_type=source.source_type,
            input_path=str(getattr(source, "path", "")) or None,
            total_frames=counters.total_frames,
            frames_with_detection=counters.frames_with_detection,
            frames_with_any_detection=counters.frames_with_any_detection,
            qr_frames_with_detection=counters.qr_frames_with_detection,
            hazmat_frames_with_detection=counters.hazmat_frames_with_detection,
            frames_with_both_qr_and_hazmat=counters.frames_with_both_qr_and_hazmat,
            best_frame_id=selected_frame_id,
            submission_frame_id=selected_frame_id,
            qr_detected=_confirmed_qr_detected(confirmed_capture, final_qr_vote.is_stable),
            qr_message=_confirmed_qr_message(confirmed_capture, final_qr_vote.stable_text),
            qr_stability_score=_confirmed_qr_score(confirmed_capture, final_qr_vote.stability_score),
            apriltag_detected=_confirmed_apriltag_detected(
                confirmed_capture,
                bool(final_apriltag_vote and final_apriltag_vote.is_stable),
            ),
            apriltag_tag_id=_confirmed_apriltag_tag_id(
                confirmed_capture,
                final_apriltag_vote.stable_tag_id if final_apriltag_vote is not None else "",
            ),
            apriltag_stability_score=_confirmed_apriltag_score(
                confirmed_capture,
                final_apriltag_vote.stability_score if final_apriltag_vote is not None else 0.0,
            ),
            hazmat_detected=_confirmed_hazmat_detected(confirmed_capture, final_hazmat_vote.is_stable),
            hazmat_label=_confirmed_hazmat_label(confirmed_capture, final_hazmat_vote.stable_label),
            hazmat_stability_score=_confirmed_hazmat_score(confirmed_capture, final_hazmat_vote.stability_score),
            artifact_dir=artifact_dir,
            artifact_paths=artifact_paths,
        )

        if confirmed_capture is not None and confirmed_capture.confirmed_events:
            artifact_paths.update(_existing_confirmed_submission_paths(artifact_dir))
            artifact_paths["latest_confirmed_result_json"] = str(artifact_dir / "latest_confirmed" / "result.json")
        else:
            artifact_paths.update(
                export_sensing_submission(
                    artifact_dir=artifact_dir,
                    qr_detected=summary.qr_detected,
                    qr_message=summary.qr_message,
                    qr_stability_score=summary.qr_stability_score,
                    qr_result=submission_qr_result,
                    apriltag_detected=summary.apriltag_detected,
                    apriltag_tag_id=summary.apriltag_tag_id,
                    apriltag_stability_score=summary.apriltag_stability_score,
                    apriltag_result=submission_apriltag_result,
                    hazmat_detected=summary.hazmat_detected,
                    hazmat_label=summary.hazmat_label,
                    hazmat_stability_score=summary.hazmat_stability_score,
                    hazmat_result=submission_hazmat_result,
                    best_frame_id=summary.submission_frame_id,
                    source_type=summary.source_type,
                    overlay_name=submission_overlay_name,
                )
            )
        if self.config.output.save_summary:
            summary = summary.model_copy(update={"artifact_paths": artifact_paths})
            write_json(artifact_dir / "summary.json", summary)

        self.logger.info(
            "Finished Sensing run: apriltag_detected=%s apriltag_tag_id='%s' qr_alias_detected=%s qr_alias_message='%s' "
            "hazmat_detected=%s hazmat_label='%s' artifacts=%s",
            summary.apriltag_detected,
            summary.apriltag_tag_id,
            summary.qr_detected,
            summary.qr_message,
            summary.hazmat_detected,
            summary.hazmat_label,
            artifact_dir,
        )
        return summary

    def _write_submission_overlay(
        self,
        *,
        artifact_dir: Path,
        source_type: str,
        aggregation: SensingRunAggregationResult,
    ) -> str | None:
        if aggregation.submission_candidate.frame_image is None or not self.config.output.save_overlay:
            return None

        final_qr_vote = aggregation.final_qr_vote
        final_apriltag_vote = aggregation.final_apriltag_vote
        final_hazmat_vote = aggregation.final_hazmat_vote
        submission_qr_result = aggregation.submission_qr_result
        submission_apriltag_result = aggregation.submission_apriltag_result
        submission_hazmat_result = aggregation.submission_hazmat_result

        submission_dir = ensure_dir(artifact_dir / "submission")
        submission_overlay_path = submission_dir / "overlay.png"
        overlay = aggregation.submission_candidate.frame_image.copy()
        if submission_apriltag_result is not None and final_apriltag_vote is not None:
            overlay = draw_apriltag_overlay(
                overlay,
                submission_apriltag_result,
                final_apriltag_vote.stable_tag_id or "UNKNOWN",
                final_apriltag_vote.stability_score,
                False,
                final_apriltag_vote.is_stable,
                draw_status=False,
            )
        elif submission_qr_result is not None:
            overlay = draw_qr_overlay(
                overlay,
                submission_qr_result,
                final_qr_vote.stable_text or "UNKNOWN",
                final_qr_vote.stability_score,
                False,
                final_qr_vote.is_stable,
                draw_status=False,
            )
        if submission_hazmat_result is not None:
            overlay = draw_hazmat_overlay(
                overlay,
                submission_hazmat_result,
                final_hazmat_vote.stable_label or "UNKNOWN",
                final_hazmat_vote.stability_score,
                False,
                final_hazmat_vote.is_stable,
                draw_status=False,
            )
        cv2.imwrite(str(submission_overlay_path), overlay)
        return submission_overlay_path.name


def _existing_confirmed_submission_paths(artifact_dir: Path) -> dict[str, str]:
    submission_dir = artifact_dir / "submission"
    paths = {
        "submission_decoded_message": str(submission_dir / "decoded_message.txt"),
        "submission_result_json": str(submission_dir / "result.json"),
    }
    overlay_path = submission_dir / "overlay.png"
    if overlay_path.exists():
        paths["submission_overlay_image"] = str(overlay_path)
    return paths


def _confirmed_qr_detected(capture: SensingConfirmedCapture | None, fallback: bool) -> bool:
    if capture is not None and getattr(capture, "latest_confirmed_apriltag", None) is not None:
        return True
    if capture is not None and capture.confirmed_events:
        return capture.latest_confirmed_qr is not None
    return fallback


def _confirmed_qr_message(capture: SensingConfirmedCapture | None, fallback: str) -> str:
    if capture is not None and getattr(capture, "latest_confirmed_apriltag", None) is not None:
        apriltag = getattr(capture, "latest_confirmed_apriltag")
        return str(apriltag.get("tag_id") or apriltag.get("stable_tag_id") or "")
    if capture is not None and capture.latest_confirmed_qr is not None:
        return str(capture.latest_confirmed_qr.get("decoded_text") or capture.latest_confirmed_qr.get("stable_text") or "")
    return fallback


def _confirmed_qr_score(capture: SensingConfirmedCapture | None, fallback: float) -> float:
    if capture is not None and getattr(capture, "latest_confirmed_apriltag", None) is not None:
        return float(getattr(capture, "latest_confirmed_apriltag").get("stability_score") or 0.0)
    if capture is not None and capture.latest_confirmed_qr is not None:
        return float(capture.latest_confirmed_qr.get("stability_score") or 0.0)
    return fallback


def _confirmed_apriltag_detected(capture: SensingConfirmedCapture | None, fallback: bool) -> bool:
    if capture is not None and capture.confirmed_events:
        return getattr(capture, "latest_confirmed_apriltag", None) is not None
    return fallback


def _confirmed_apriltag_tag_id(capture: SensingConfirmedCapture | None, fallback: str) -> str:
    if capture is not None and getattr(capture, "latest_confirmed_apriltag", None) is not None:
        apriltag = getattr(capture, "latest_confirmed_apriltag")
        return str(apriltag.get("tag_id") or apriltag.get("stable_tag_id") or "")
    return fallback


def _confirmed_apriltag_score(capture: SensingConfirmedCapture | None, fallback: float) -> float:
    if capture is not None and getattr(capture, "latest_confirmed_apriltag", None) is not None:
        return float(getattr(capture, "latest_confirmed_apriltag").get("stability_score") or 0.0)
    return fallback


def _confirmed_hazmat_detected(capture: SensingConfirmedCapture | None, fallback: bool) -> bool:
    if capture is not None and capture.confirmed_events:
        return capture.latest_confirmed_hazmat is not None
    return fallback


def _confirmed_hazmat_label(capture: SensingConfirmedCapture | None, fallback: str) -> str:
    if capture is not None and capture.latest_confirmed_hazmat is not None:
        return str(capture.latest_confirmed_hazmat.get("label") or capture.latest_confirmed_hazmat.get("stable_label") or "")
    return fallback


def _confirmed_hazmat_score(capture: SensingConfirmedCapture | None, fallback: float) -> float:
    if capture is not None and capture.latest_confirmed_hazmat is not None:
        return float(capture.latest_confirmed_hazmat.get("stability_score") or 0.0)
    return fallback
