"""Hazmat inspection pipeline."""

from __future__ import annotations

import logging

import cv2

from rescue_inspection.config import AppConfig
from rescue_inspection.models import HazmatDetectionResult, RunSummary
from rescue_inspection.preview.base import NullPreviewBackend, PreviewBackend
from rescue_inspection.sources.base import FrameSource
from rescue_inspection.submission import export_hazmat_submission
from rescue_inspection.tasks.hazmat.debug_artifacts import (
    save_classifier_debug_bundle,
    save_color_evidence_debug,
    save_ocr_evidence_debug,
)
from rescue_inspection.tasks.hazmat.recognizer import build_hazmat_recognizer
from rescue_inspection.tasks.hazmat.temporal import HazmatDetectionStabilizer, StableHazmatVoteResult
from rescue_inspection.utils.io import (
    append_jsonl,
    create_video_writer,
    ensure_dir,
    estimate_output_fps,
    write_json,
)
from rescue_inspection.utils.time_utils import make_run_id
from rescue_inspection.visualization.overlay import draw_hazmat_overlay


class HazmatInspectionPipeline:
    """End-to-end Hazmat inspection pipeline."""

    def __init__(
        self,
        config: AppConfig,
        logger: logging.Logger,
        preview: PreviewBackend | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.preview = preview or NullPreviewBackend()
        self.recognizer = build_hazmat_recognizer(config, logger)
        self.detector = getattr(self.recognizer, "detector", None)
        self.classifier = getattr(self.recognizer, "classifier", None)
        self.stabilizer = HazmatDetectionStabilizer(config.hazmat.stabilizer)

    def run(self, source: FrameSource) -> RunSummary:
        """Run the detector on a source and persist artifacts."""
        run_id = make_run_id(prefix="hazmat")
        artifact_dir = ensure_dir(self.config.output.root_dir / run_id)
        jsonl_path = artifact_dir / "detections.jsonl"
        overlay_image_path = artifact_dir / "overlay.png"
        overlay_video_path = artifact_dir / "overlay.mp4"
        artifact_paths: dict[str, str] = {}

        total_frames = 0
        frames_with_detection = 0
        best_candidate_vote = StableHazmatVoteResult(False, "UNKNOWN", 0.0, [], None)
        best_stable_vote: StableHazmatVoteResult | None = None
        best_candidate_frame_image = None
        best_stable_frame_image = None
        last_frame_image = None
        frame_cache: dict[int, object] = {}
        cached_frame_ids: list[int] = []
        writer = None
        pending_overlay = None
        pending_overlay_time_sec: float | None = None
        interrupted_by_user = False

        self.logger.info("Starting Hazmat inspection run in %s mode", source.source_type)

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
                    while len(cached_frame_ids) > self.config.hazmat.stabilizer.window_size:
                        stale_frame_id = cached_frame_ids.pop(0)
                        frame_cache.pop(stale_frame_id, None)

                    result = self.recognizer.detect(packet)
                    extraction = self.recognizer.last_extraction
                    accepted_items = [item for item in result.items if item.label != "UNKNOWN"]

                    if self.config.output.debug_visualization and extraction and extraction.rejected_candidates:
                        for i, (patch, _, reason, ratio) in enumerate(extraction.rejected_candidates):
                            crop_name = f"frame_{packet.frame_id:04d}_rejected_detector_{i}_{reason}_ratio_{ratio:.2f}.png"
                            cv2.imwrite(str(ensure_dir(artifact_dir / "debug_candidates") / crop_name), patch)

                    if self.config.output.debug_visualization:
                        debug_root = ensure_dir(artifact_dir / "debug_candidates")
                        for i, item in enumerate(result.items):
                            if item.debug is not None and hasattr(self.classifier, "diagnose_patch"):
                                patch = self.recognizer.last_candidate_patches[i]
                                diagnose_payload = self.classifier.diagnose_patch(patch, topk=5)
                                save_classifier_debug_bundle(
                                    debug_root=debug_root,
                                    frame_id=packet.frame_id,
                                    candidate_index=i,
                                    frame_bgr=packet.image,
                                    polygon=item.polygon,
                                    rectified_crop_bgr=patch,
                                    classifier_input_bgr=diagnose_payload["classifier_input_bgr"],
                                    preprocess_meta=diagnose_payload["preprocess_meta"],
                                )
                            if item.label == "UNKNOWN" and item.debug is not None:
                                patch = self.recognizer.last_candidate_patches[i]
                                crop_name = (
                                    f"frame_{packet.frame_id:04d}_candidate_{i}_"
                                    f"rejected_classifier_{item.debug.gated_reason}.png"
                                )
                                cv2.imwrite(str(debug_root / crop_name), patch)
                            if self.config.hazmat.debug.write_color_evidence_files and item.debug is not None:
                                save_color_evidence_debug(
                                    debug_root=debug_root,
                                    frame_id=packet.frame_id,
                                    candidate_index=i,
                                    payload=_color_debug_payload(item.debug),
                                )
                            if self.config.hazmat.debug.write_ocr_evidence_files and item.debug is not None:
                                save_ocr_evidence_debug(
                                    debug_root=debug_root,
                                    frame_id=packet.frame_id,
                                    candidate_index=i,
                                    payload=_ocr_debug_payload(item.debug),
                                )

                    # 3. Temporal smoothing
                    vote = self.stabilizer.update(result)

                    if source.source_type == "image" and result.detected:
                        # Instant stability for image mode
                        best_item = max(accepted_items, key=lambda k: k.confidence)
                        vote = StableHazmatVoteResult(
                            is_stable=True,
                            stable_label=best_item.label,
                            stability_score=1.0,
                            representative_items=[best_item],
                            representative_result=result,
                        )

                    result = result.model_copy(update={"best_label": vote.stable_label, "stability_score": vote.stability_score})

                    if result.detected:
                        frames_with_detection += 1
                        labels = [item.label for item in result.items if item.label != "UNKNOWN"]
                        self.logger.info(
                            "Frame %s detected %s candidates=%s top=%s score=%.3f margin=%.3f",
                            result.frame_id,
                            labels,
                            result.candidate_count,
                            result.best_candidate_label,
                            result.best_candidate_score,
                            result.best_candidate_margin,
                        )
                    else:
                        self.logger.debug(
                            "Frame %s had no hazmat detection (%s); candidates=%s contours=%s "
                            "area_rejected=%s non_quad_rejected=%s top=%s score=%.3f margin=%.3f",
                            result.frame_id,
                            result.failure_reason,
                            result.candidate_count,
                            result.detector_debug.contours_found if result.detector_debug else 0,
                            result.detector_debug.area_rejected if result.detector_debug else 0,
                            result.detector_debug.non_quad_rejected if result.detector_debug else 0,
                            result.best_candidate_label or "none",
                            result.best_candidate_score,
                            result.best_candidate_margin,
                        )

                    if total_frames % 30 == 0:
                        self.logger.info(
                            "Hazmat progress: frames=%s frames_with_detection=%s last_failure=%s "
                            "last_candidates=%s",
                            total_frames,
                            frames_with_detection,
                            result.failure_reason or "none",
                            result.candidate_count,
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

                    # 4. Rendering
                    overlay = draw_hazmat_overlay(
                        packet.image,
                        result,
                        vote.stable_label,
                        vote.stability_score,
                        self.config.output.debug_visualization,
                        vote.is_stable,
                    )

                    if self.config.output.save_jsonl:
                        append_jsonl(jsonl_path, result)
                        artifact_paths["detections_jsonl"] = str(jsonl_path)

                    should_show_preview = source.source_type in {"video", "realsense"} and self.preview.enabled
                    should_render_overlay = self.config.output.save_overlay or should_show_preview
                    if should_render_overlay:
                        if source.source_type == "image":
                            if self.config.output.save_overlay:
                                cv2.imwrite(str(overlay_image_path), overlay)
                                artifact_paths["overlay_image"] = str(overlay_image_path)
                        else:
                            if self.config.output.save_overlay:
                                fallback_fps = getattr(source, "fps", 10.0) or 10.0
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
                            if should_show_preview:
                                self.preview.show(overlay)
                                action = self.preview.poll_action()
                                if action == "quit":
                                    interrupted_by_user = True
                                    self.logger.info(
                                        "Preview requested stop; finalizing Hazmat artifacts before exit."
                                    )
                                    break
        except KeyboardInterrupt:
            interrupted_by_user = True
            self.logger.info("Keyboard interrupt received; finalizing Hazmat artifacts before exit.")
        finally:
            if self.config.output.save_overlay and writer is None and pending_overlay is not None:
                writer = create_video_writer(
                    overlay_video_path,
                    fps=getattr(source, "fps", 10.0) or 10.0,
                    size=(pending_overlay.shape[1], pending_overlay.shape[0]),
                )
                artifact_paths["overlay_video"] = str(overlay_video_path)
                writer.write(pending_overlay)
            if writer is not None:
                writer.close(interrupted=interrupted_by_user, logger=self.logger)
            self.preview.close()

        final_vote = best_stable_vote or StableHazmatVoteResult(
            False,
            best_candidate_vote.stable_label,
            best_candidate_vote.stability_score,
            best_candidate_vote.representative_items,
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
            submission_result = HazmatDetectionResult(
                timestamp="",
                frame_id=best_frame_id or 0,
                detected=False,
                source_type=source.source_type,
                failure_reason="no_hazmat_detected",
            )
        if submission_frame is not None and submission_result is not None and self.config.output.save_overlay:
            submission_dir = ensure_dir(artifact_dir / "submission")
            submission_overlay_path = submission_dir / "overlay.png"
            submission_overlay = draw_hazmat_overlay(
                submission_frame,
                submission_result,
                final_vote.stable_label or "UNKNOWN",
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
            final_decoded_text=final_vote.stable_label,
            final_stability_score=final_vote.stability_score,
            best_frame_id=best_frame_id,
            artifact_dir=artifact_dir,
            artifact_paths=artifact_paths,
        )

        artifact_paths.update(
            export_hazmat_submission(
                artifact_dir=artifact_dir,
                detected=summary.final_detected,
                label=summary.final_decoded_text,
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
            "Finished Hazmat run: final_detected=%s final_label='%s' stability=%.2f "
            "frames=%s frames_with_detection=%s artifacts=%s",
            summary.final_detected,
            summary.final_decoded_text,
            summary.final_stability_score,
            summary.total_frames,
            summary.frames_with_detection,
            artifact_dir,
        )
        return summary


def _color_debug_payload(debug) -> dict[str, object]:
    """Return color/fusion fields for explicit debug file output."""

    return {
        "color_top1_label": debug.color_top1_label,
        "color_top1_score": debug.color_top1_score,
        "color_top2_label": debug.color_top2_label,
        "color_top2_score": debug.color_top2_score,
        "color_margin": debug.color_margin,
        "quality_flags": debug.quality_flags,
        "quality_penalty": debug.quality_penalty,
        "fused_top1_label": debug.fused_top1_label,
        "fused_top1_score": debug.fused_top1_score,
        "fused_top2_label": debug.fused_top2_label,
        "fused_top2_score": debug.fused_top2_score,
        "fused_margin": debug.fused_margin,
        "fusion_method": debug.fusion_method,
        "fusion_reason": debug.fusion_reason,
        "color_weight_used": debug.color_weight_used,
        "color_scores_by_label": debug.color_scores_by_label,
        "color_evidence": debug.color_evidence,
    }


def _ocr_debug_payload(debug) -> dict[str, object]:
    """Return full OCR/fusion fields for explicit debug file output."""

    return {
        "ocr_raw_text": debug.ocr_raw_text,
        "ocr_normalized_text": debug.ocr_normalized_text,
        "ocr_top1_label": debug.ocr_top1_label,
        "ocr_top1_score": debug.ocr_top1_score,
        "ocr_top2_label": debug.ocr_top2_label,
        "ocr_top2_score": debug.ocr_top2_score,
        "ocr_margin": debug.ocr_margin,
        "ocr_matched_keywords": debug.ocr_matched_keywords,
        "ocr_matched_numbers": debug.ocr_matched_numbers,
        "ocr_quality_flags": debug.ocr_quality_flags,
        "ocr_available": debug.ocr_available,
        "ocr_disabled_reason": debug.ocr_disabled_reason,
        "ocr_backend": debug.ocr_backend,
        "ocr_backend_error": debug.ocr_backend_error,
        "ocr_reused": debug.ocr_reused,
        "ocr_scores_by_label": debug.ocr_scores_by_label,
        "fused_top1_label": debug.fused_top1_label,
        "fused_top1_score": debug.fused_top1_score,
        "fused_top2_label": debug.fused_top2_label,
        "fused_top2_score": debug.fused_top2_score,
        "fused_margin": debug.fused_margin,
        "fusion_method": debug.fusion_method,
        "fusion_reason": debug.fusion_reason,
    }
