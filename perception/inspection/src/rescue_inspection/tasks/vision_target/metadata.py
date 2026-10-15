"""Metadata loading and validation for deployed ROI detector models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rescue_inspection.config import TargetDetectorConfig
from rescue_inspection.project_paths import resolve_project_path


@dataclass(frozen=True)
class TargetDetectorMetadata:
    """Deploy-time contract for a target ROI detector."""

    class_names: list[str]
    input_size: int
    model_family: str
    output_format: str


def load_target_detector_metadata(path: Path) -> TargetDetectorMetadata:
    """Load detector metadata from JSON."""

    metadata_path = resolve_project_path(path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return TargetDetectorMetadata(
        class_names=list(payload.get("class_names", [])),
        input_size=int(payload.get("input_size", 0)),
        model_family=str(payload.get("model_family", "")),
        output_format=str(payload.get("output_format", "")),
    )


def validate_target_detector_metadata(
    metadata: TargetDetectorMetadata,
    config: TargetDetectorConfig,
) -> None:
    """Raise ValueError when model metadata does not match runtime config."""

    errors: list[str] = []
    if metadata.class_names != config.classes:
        errors.append(f"class_names mismatch: metadata={metadata.class_names} config={config.classes}")
    if metadata.input_size != config.input_size:
        errors.append(f"input_size mismatch: metadata={metadata.input_size} config={config.input_size}")
    if metadata.model_family != config.model_family:
        errors.append(f"model_family mismatch: metadata={metadata.model_family} config={config.model_family}")
    if metadata.output_format != config.output_format:
        errors.append(f"output_format mismatch: metadata={metadata.output_format} config={config.output_format}")
    if errors:
        raise ValueError("; ".join(errors))
