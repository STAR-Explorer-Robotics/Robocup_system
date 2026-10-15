"""Application assembly helpers."""

from __future__ import annotations

import os
from pathlib import Path

from rescue_inspection.config import AppConfig, SourceType, apply_overrides, load_config
from rescue_inspection.logging_utils import configure_logging
from rescue_inspection.models import RunSummary
from rescue_inspection.preview.base import NullPreviewBackend, PreviewBackend
from rescue_inspection.preview.opencv_window import OpenCVPreviewBackend
from rescue_inspection.sources.base import FrameSource
from rescue_inspection.sources.image_source import ImageSource
from rescue_inspection.sources.realsense_source import RealSenseSource
from rescue_inspection.sources.video_source import VideoSource


def build_preview(config: AppConfig, logger) -> PreviewBackend:
    """Construct the configured preview backend."""

    if not config.preview.enabled or config.preview.mode == "none":
        return NullPreviewBackend()

    if config.source.type not in {"video", "realsense"}:
        logger.info("Preview is only enabled for video/realsense sources; skipping %s source.", config.source.type)
        return NullPreviewBackend()

    if not os.environ.get("DISPLAY"):
        logger.warning("Preview requested but DISPLAY is unavailable; continuing without live preview.")
        return NullPreviewBackend()

    if config.preview.mode == "opencv":
        return OpenCVPreviewBackend(
            window_name=config.preview.window_name,
            logger=logger,
            wait_key_delay_ms=config.preview.wait_key_delay_ms,
        )

    raise ValueError(f"Unsupported preview mode: {config.preview.mode}")


def build_pipeline(config: AppConfig, logger, preview: PreviewBackend | None = None):
    """Construct the configured task pipeline."""

    task_type = config.task.type
    if task_type == "qr":
        from rescue_inspection.tasks.qr.pipeline import QRInspectionPipeline
        return QRInspectionPipeline(config=config, logger=logger)
    
    if task_type == "hazmat":
        from rescue_inspection.tasks.hazmat.pipeline import HazmatInspectionPipeline
        return HazmatInspectionPipeline(config=config, logger=logger, preview=preview)

    if task_type in ("sensing", "combined"):
        from rescue_inspection.tasks.sensing.pipeline import SensingPipeline
        return SensingPipeline(config=config, logger=logger, preview=preview)

    raise ValueError(f"Unsupported task type: {task_type}")


def build_source(config: AppConfig) -> FrameSource:
    """Construct the configured input source."""

    source_type = config.source.type
    input_path = config.source.input_path

    if source_type == "image":
        if input_path is None:
            raise ValueError("Image source requires an input path.")
        return ImageSource(path=input_path)

    if source_type == "video":
        if input_path is None:
            raise ValueError("Video source requires an input path.")
        return VideoSource(path=input_path, frame_stride=config.source.frame_stride)

    if source_type == "realsense":
        if not config.source.realsense.enabled:
            raise RuntimeError("RealSense source selected but source.realsense.enabled is false.")
        return RealSenseSource(
            width=config.source.realsense.width,
            height=config.source.realsense.height,
            fps=config.source.realsense.fps,
            frame_stride=config.source.frame_stride,
        )

    raise ValueError(f"Unsupported source type: {source_type}")


def run_app(
    config_path: Path,
    *,
    task_type: str | None = None,
    source_type: SourceType | None = None,
    input_path: Path | None = None,
    output_root: Path | None = None,
    max_frames: int | None = None,
    show_preview: bool | None = None,
    preview_window_name: str | None = None,
):
    """Load config, create dependencies, and execute the task pipeline."""

    config = load_config(config_path)
    config = apply_overrides(
        config,
        task_type=task_type,
        source_type=source_type,
        input_path=input_path,
        output_root=output_root,
        max_frames=max_frames,
        show_preview=show_preview,
        preview_window_name=preview_window_name,
    )
    logger = configure_logging(config.runtime.log_level)
    preview = build_preview(config, logger)
    source = build_source(config)
    pipeline = build_pipeline(config, logger, preview=preview)
    return pipeline.run(source)
