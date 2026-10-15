"""QR inspection pipeline."""

from __future__ import annotations

import logging

import cv2

from rescue_inspection.config import AppConfig
from rescue_inspection.models import QRDetectionResult, RunSummary
from rescue_inspection.sources.base import FrameSource
from rescue_inspection.submission import export_qr_submission
from rescue_inspection.tasks.qr.detector import QRCodeBaselineDetector
from rescue_inspection.tasks.qr.postprocess import QRDetectionStabilizer, StableVoteResult
from rescue_inspection.utils.io import (
    append_jsonl,
    create_video_writer,
    ensure_dir,
    estimate_output_fps,
    write_json,
)
from rescue_inspection.utils.time_utils import make_run_id
from rescue_inspection.visualization.overlay import draw_qr_overlay


class QRInspectionPipeline:
    """End-to-end QR inspection pipeline."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.detector = QRCodeBaselineDetector(config.detector, config.target_detector, logger=logger)
        self.stabilizer = QRDetectionStabilizer(config.stabilizer)

    def run(self, source: FrameSource) -> RunSummary:
        """Run the detector on a source and persist artifacts."""

        run_id = make_run_id(prefix="qr")
        artifact_dir = ensure_dir(self.config.output.root_dir / run_id)
        jsonl_path = artifact_dir / "detections.jsonl"
        overlay_image_path = artifact_dir / "overlay.png"
        overlay_video_path = artifact_dir / "overlay.mp4"
        artifact_paths: dict[str, str] = {}

        total_frames = 0
        frames_with_detection = 0
        best_candidate_vote = StableVoteResult(False, "", 0.0, None)
        best_stable_vote: StableVoteResult | None = None
        best_candidate_frame_image = None
        best_stable_frame_image = None
        last_frame_image = None
        frame_cache: dict[int, object] = {}
        cached_frame_ids: list[int] = []
        writer = None
        pending_overlay = None
        pending_overlay_time_sec: float | None = None

        self.logger.info("Starting QR inspection run in %s mode", source.source_type)

        try:
            with source:
                while True:
                    if self.config.runtime.max_frames is not None and total_frames >= self.config.runtime.max_frames:
                        self.logger.info("Reached max_frames=%s", self.config.runtime.max_frames)
                        break

                    packet = source.read()
                    if packet is None:
                        break

                    total_frames += 1
                    frame_copy = packet.image.copy()
                    last_frame_image = frame_copy
                    frame_cache[packet.frame_id] = frame_copy
                    cached_frame_ids.append(packet.frame_id)
                    while len(cached_frame_ids) > self.config.stabilizer.window_size:
                        stale_frame_id = cached_frame_ids.pop(0)
                        frame_cache.pop(stale_frame_id, None)
                    result = self.detector.detect(packet)
                    vote = self.stabilizer.update(result)

                    if source.source_type == "image" and result.detected:
                        vote = StableVoteResult(
                            is_stable=True,
                            stable_text=result.decoded_text,
                            stability_score=1.0,
                            representative_result=result,
                        )

                    result = result.model_copy(update={"stability_score": vote.stability_score})

                    if result.detected:
                        frames_with_detection += 1
                        self.logger.info(
                            "Frame %s decoded '%s' via %s",
                            result.frame_id,
                            result.decoded_text,
                            result.method_used,
                        )
                    else:
                        self.logger.debug(
                            "Frame %s had no QR decode (%s)",
                            result.frame_id,
                            result.failure_reason,
                        )

                    if vote.stability_score >= best_candidate_vote.stability_score:
                        best_candidate_vote = vote
                        if vote.representative_result is not None:
                            best_candidate_frame_image = frame_cache.get(
                                vote.representative_result.frame_id,
                                frame_copy,
                            )

                    if vote.is_stable and (
                        best_stable_vote is None or vote.stability_score >= best_stable_vote.stability_score
                    ):
                        best_stable_vote = vote
                        if vote.representative_result is not None:
                            best_stable_frame_image = frame_cache.get(
                                vote.representative_result.frame_id,
                                frame_copy,
                            )

                    overlay = draw_qr_overlay(
                        packet.image,
                        result,
                        vote.stable_text,
                        vote.stability_score,
                        self.config.output.debug_visualization,
                        vote.is_stable,
                    )

                    if self.config.output.save_jsonl:
                        append_jsonl(jsonl_path, result)
                        artifact_paths["detections_jsonl"] = str(jsonl_path)

                    if self.config.output.save_overlay:
                        if source.source_type == "image":
                            cv2.imwrite(str(overlay_image_path), overlay)
                            artifact_paths["overlay_image"] = str(overlay_image_path)
                        else:
                            fallback_fps = source.fps or 10.0
                            if writer is None and pending_overlay is None:
                                pending_overlay = overlay.copy()
                                pending_overlay_time_sec = packet.source_time_sec
                            elif writer is None:
                                writer = create_video_writer(
                                    overlay_video_path,
                                    fps=estimate_output_fps(
                                        pending_overlay_time_sec,
                                        packet.source_time_sec,
                                        fallback_fps,
                                    ),
                                    size=(overlay.shape[1], overlay.shape[0]),
                                )
                                artifact_paths["overlay_video"] = str(overlay_video_path)
                                writer.write(pending_overlay)
                                writer.write(overlay)
                                pending_overlay = None
                                pending_overlay_time_sec = None
                            else:
                                writer.write(overlay)
        finally:
            if self.config.output.save_overlay and writer is None and pending_overlay is not None:
                writer = create_video_writer(
                    overlay_video_path,
                    fps=source.fps or 10.0,
                    size=(pending_overlay.shape[1], pending_overlay.shape[0]),
                )
                artifact_paths["overlay_video"] = str(overlay_video_path)
                writer.write(pending_overlay)
            if writer is not None:
                writer.close()

        final_vote = best_stable_vote or StableVoteResult(
            False,
            "",
            best_candidate_vote.stability_score,
            best_candidate_vote.representative_result,
        )
        representative_result = final_vote.representative_result or best_candidate_vote.representative_result
        best_frame_id = representative_result.frame_id if representative_result is not None else None
        submission_frame = best_stable_frame_image
        if submission_frame is None:
            submission_frame = best_candidate_frame_image
        if submission_frame is None:
            submission_frame = last_frame_image
        submission_overlay_name: str | None = None
        submission_result = representative_result
        if submission_result is None and submission_frame is not None:
            submission_result = QRDetectionResult(
                timestamp="",
                frame_id=best_frame_id or 0,
                detected=False,
                source_type=source.source_type,
                failure_reason="no_qr_detected",
            )
        if submission_frame is not None and submission_result is not None and self.config.output.save_overlay:
            submission_dir = ensure_dir(artifact_dir / "submission")
            submission_overlay_path = submission_dir / "overlay.png"
            submission_overlay = draw_qr_overlay(
                submission_frame,
                submission_result,
                final_vote.stable_text or "UNKNOWN",
                final_vote.stability_score,
                False,
                final_vote.is_stable,
            )
            cv2.imwrite(str(submission_overlay_path), submission_overlay)
            submission_overlay_name = submission_overlay_path.name

        summary = RunSummary(
            run_id=run_id,
            source_type=source.source_type,
            input_path=str(getattr(source, "path", "")) or None,
            total_frames=total_frames,
            frames_with_detection=frames_with_detection,
            final_detected=final_vote.is_stable,
            final_decoded_text=final_vote.stable_text,
            final_stability_score=final_vote.stability_score,
            best_frame_id=best_frame_id,
            artifact_dir=artifact_dir,
            artifact_paths=artifact_paths,
        )

        artifact_paths.update(
            export_qr_submission(
                artifact_dir=artifact_dir,
                detected=summary.final_detected,
                message=summary.final_decoded_text,
                best_frame_id=summary.best_frame_id,
                stability_score=summary.final_stability_score,
                source_type=summary.source_type,
                representative_result=representative_result,
                overlay_name=submission_overlay_name,
            )
        )
        if self.config.output.save_summary:
            artifact_paths["summary_json"] = str(artifact_dir / "summary.json")
            summary = summary.model_copy(update={"artifact_paths": artifact_paths})
            write_json(artifact_dir / "summary.json", summary)

        self.logger.info(
            "Finished run: final_detected=%s final_decoded_text='%s' stability=%.2f artifacts=%s",
            summary.final_detected,
            summary.final_decoded_text,
            summary.final_stability_score,
            artifact_dir,
        )
        return summary
