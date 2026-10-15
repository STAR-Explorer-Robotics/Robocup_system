"""Combined Sensing inspection pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from rescue_inspection.config import AppConfig
from rescue_inspection.models import (
    FramePacket,
    SensingDetectionResult,
    SensingRunSummary,
)
from rescue_inspection.preview.base import NullPreviewBackend, PreviewBackend
from rescue_inspection.runtime import SensingConfirmedCapture, SensingFrameResult
from rescue_inspection.sources.base import FrameSource
from rescue_inspection.tasks.sensing.aggregation import SensingRunAggregator
from rescue_inspection.tasks.sensing.finalization import SensingRunFinalizer
from rescue_inspection.tasks.sensing.overlay_recording import SensingOverlayRecorder
from rescue_inspection.tasks.sensing.processor import SensingFrameOutput, SensingFrameProcessor
from rescue_inspection.utils.io import (
    append_jsonl,
    ensure_dir,
)
from rescue_inspection.utils.time_utils import make_run_id


class SensingPipeline:
    """End-to-end combined Sensing inspection pipeline."""

    def __init__(
        self,
        config: AppConfig,
        logger: logging.Logger,
        preview: PreviewBackend | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.preview = preview or NullPreviewBackend()
        self.frame_processor = SensingFrameProcessor(config, logger)
        self.qr_detector = self.frame_processor.qr_detector
        self.qr_stabilizer = self.frame_processor.qr_stabilizer
        self.hazmat_recognizer = self.frame_processor.hazmat_recognizer
        self.hazmat_detector = self.frame_processor.hazmat_detector
        self.hazmat_classifier = self.frame_processor.hazmat_classifier
        self.hazmat_stabilizer = self.frame_processor.hazmat_stabilizer

    def run(self, source: FrameSource) -> SensingRunSummary:
        """Run the combined sensing pipeline on a source."""
        run_id = make_run_id(prefix="sensing")
        artifact_dir = ensure_dir(self.config.output.root_dir / run_id)
        jsonl_path = artifact_dir / "detections.jsonl"
        artifact_paths: dict[str, str] = {}

        aggregator = SensingRunAggregator()
        confirmed_capture = SensingConfirmedCapture(artifact_dir, save_overlays=self.config.output.save_overlay)
        recorder = SensingOverlayRecorder(
            config=self.config,
            artifact_dir=artifact_dir,
            artifact_paths=artifact_paths,
            source=source,
            logger=self.logger,
        )
        interrupted_by_user = False

        self.logger.info("Starting combined Sensing inspection run in %s mode", source.source_type)

        try:
            with source:
                while True:
                    if (
                        self.config.runtime.max_frames is not None
                        and aggregator.counters.total_frames >= self.config.runtime.max_frames
                    ):
                        self.logger.info("Reached max_frames=%s", self.config.runtime.max_frames)
                        break

                    packet = source.read()
                    if packet is None:
                        break

                    aggregator.record_frame_seen()

                    should_show_preview = source.source_type in {"video", "realsense"} and self.preview.enabled
                    should_render_overlay = self.config.output.save_overlay or should_show_preview
                    frame_output = self.frame_processor.process_packet(
                        packet,
                        render_overlay=should_render_overlay,
                        debug_artifact_dir=artifact_dir / "debug_candidates",
                    )
                    qr_result = frame_output.qr_result
                    qr_vote = frame_output.qr_vote
                    apriltag_result = frame_output.apriltag_result
                    apriltag_vote = frame_output.apriltag_vote
                    hazmat_result = frame_output.hazmat_result
                    hazmat_vote = frame_output.hazmat_vote

                    aggregator.record_frame_result(
                        packet=packet,
                        qr_result=qr_result,
                        qr_vote=qr_vote,
                        apriltag_result=apriltag_result,
                        apriltag_vote=apriltag_vote,
                        hazmat_result=hazmat_result,
                        hazmat_vote=hazmat_vote,
                    )

                    sensing_result = SensingDetectionResult(
                        timestamp=packet.timestamp,
                        frame_id=packet.frame_id,
                        source_type=packet.source_type,
                        qr_result=qr_result,
                        hazmat_result=hazmat_result,
                        apriltag_result=apriltag_result,
                    )

                    if self.config.output.save_jsonl:
                        append_jsonl(jsonl_path, sensing_result)
                        artifact_paths["detections_jsonl"] = str(jsonl_path)

                    height, width = packet.image.shape[:2]
                    confirmed_capture.record_frame(
                        SensingFrameResult(
                            stamp_sec=0,
                            stamp_nanosec=0,
                            frame_id=str(packet.frame_id),
                            source_topic=str(getattr(source, "path", "") or source.source_type),
                            processing_time_ms=0.0,
                            image_width=width,
                            image_height=height,
                            qr_result=qr_result,
                            qr_vote=qr_vote,
                            hazmat_result=hazmat_result,
                            hazmat_vote=hazmat_vote,
                            overlay_bgr=frame_output.overlay,
                            apriltag_result=apriltag_result,
                            apriltag_vote=apriltag_vote,
                        ),
                        frame_bgr=packet.image,
                        overlay_bgr=frame_output.overlay,
                    )

                    if should_render_overlay and frame_output.overlay is not None:
                        overlay = frame_output.overlay
                        recorder.record(packet, overlay)
                        if should_show_preview:
                            self.preview.show(overlay)
                            action = self.preview.poll_action()
                            if action == "quit":
                                interrupted_by_user = True
                                self.logger.info(
                                    "Preview requested stop; finalizing Sensing artifacts before exit."
                                )
                                break
                            
                    if aggregator.counters.total_frames % 30 == 0:
                        self.logger.info(
                            "Sensing progress: frames=%s detection=%s "
                            "code='%s' hz_best='%s'",
                            aggregator.counters.total_frames,
                            aggregator.counters.frames_with_detection,
                            (apriltag_result.tag_id if apriltag_result is not None else qr_result.decoded_text),
                            hazmat_result.best_label,
                        )
        except KeyboardInterrupt:
            interrupted_by_user = True
            self.logger.info("Keyboard interrupt received; finalizing Sensing artifacts before exit.")
        finally:
            recorder.close(interrupted=interrupted_by_user)
            self.preview.close()

        return SensingRunFinalizer(self.config, self.logger).finalize_run(
            run_id=run_id,
            source=source,
            artifact_dir=artifact_dir,
            artifact_paths=artifact_paths,
            aggregation=aggregator.finalize(),
            confirmed_capture=confirmed_capture,
        )
