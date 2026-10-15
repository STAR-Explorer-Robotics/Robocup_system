"""Overlay recording for native combined sensing runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rescue_inspection.config import AppConfig
from rescue_inspection.models import FramePacket
from rescue_inspection.utils.io import create_video_writer, estimate_output_fps


class SensingOverlayRecorder:
    """Recorder for saving overlay images and overlay videos in native sensing runs."""

    def __init__(
        self,
        config: AppConfig,
        artifact_dir: Path,
        artifact_paths: dict[str, str],
        source: Any,
        logger: logging.Logger,
        output_video_fps: float | None = None,
    ) -> None:
        self.config = config
        self.artifact_dir = artifact_dir
        self.artifact_paths = artifact_paths
        self.source = source
        self.logger = logger

        # Explicit parameter or config field override
        self.output_video_fps = output_video_fps or getattr(config.output, "output_video_fps", None)

        self.writer = None
        self.pending_overlay = None
        self.pending_overlay_time_sec = None

    def record(self, packet: FramePacket, overlay_bgr: np.ndarray) -> None:
        """Record an overlay frame (image or stream video)."""
        if not self.config.output.save_overlay:
            return

        latest_overlay_path = self.artifact_dir / "latest_overlay.png"
        cv2.imwrite(str(latest_overlay_path), overlay_bgr)
        self._write_latest_overlay_metadata(
            packet=packet,
            reason="latest rendered live overlay snapshot",
            overlay_path=latest_overlay_path,
        )

        if self.source.source_type == "image":
            self.artifact_paths["overlay_image"] = str(latest_overlay_path)
        else:
            overlay_video_path = self.artifact_dir / "overlay.mp4"
            fallback_fps = getattr(self.source, "fps", 10.0) or 10.0

            if self.writer is None and self.pending_overlay is None:
                self.pending_overlay = overlay_bgr.copy()
                self.pending_overlay_time_sec = packet.source_time_sec
            elif self.writer is None:
                fps = (
                    self.output_video_fps
                    if self.output_video_fps is not None
                    else estimate_output_fps(
                        self.pending_overlay_time_sec,
                        packet.source_time_sec,
                        fallback_fps,
                    )
                )
                self.writer = create_video_writer(
                    overlay_video_path,
                    fps=fps,
                    size=(overlay_bgr.shape[1], overlay_bgr.shape[0]),
                )
                self.artifact_paths["overlay_video"] = str(overlay_video_path)
                self.writer.write(self.pending_overlay)
                self.writer.write(overlay_bgr)
                self.pending_overlay = None
                self.pending_overlay_time_sec = None
            else:
                self.writer.write(overlay_bgr)

    def close(self, interrupted: bool = False) -> None:
        """Finalize and close the overlay recorder, writing any pending frames."""
        if self.config.output.save_overlay:
            if self.writer is None and self.pending_overlay is not None:
                overlay_video_path = self.artifact_dir / "overlay.mp4"
                
                if self.output_video_fps is not None:
                    fps = self.output_video_fps
                else:
                    fps = getattr(self.source, "fps", 10.0) or 10.0
                
                self.writer = create_video_writer(
                    overlay_video_path,
                    fps=fps,
                    size=(self.pending_overlay.shape[1], self.pending_overlay.shape[0]),
                )
                self.artifact_paths["overlay_video"] = str(overlay_video_path)
                self.writer.write(self.pending_overlay)
                self.pending_overlay = None
                self.pending_overlay_time_sec = None

        if self.writer is not None:
            self.writer.close(interrupted=interrupted, logger=self.logger)
            self.writer = None

    def _write_latest_overlay_metadata(
        self,
        *,
        packet: FramePacket,
        reason: str,
        overlay_path: Path,
    ) -> None:
        payload = {
            "overlay_path": overlay_path.name,
            "frame_id": packet.frame_id,
            "timestamp": packet.timestamp,
            "source_type": packet.source_type,
            "source_time_sec": packet.source_time_sec,
            "reason": reason,
            "final_evidence_overlay": "submission/overlay.png",
        }
        (self.artifact_dir / "latest_overlay.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.artifact_paths["latest_overlay_image"] = str(overlay_path)
        self.artifact_paths["latest_overlay_metadata"] = str(self.artifact_dir / "latest_overlay.json")
