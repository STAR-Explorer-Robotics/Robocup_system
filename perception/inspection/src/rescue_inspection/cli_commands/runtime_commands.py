"""Runtime command handlers for inspection-run."""

from __future__ import annotations

import argparse
from pathlib import Path


def _validate_args(source_type: str | None, input_path: Path | None) -> None:
    """Validate CLI-only requirements before running the app."""

    if source_type in {"image", "video"} and input_path is None:
        raise ValueError(f"{source_type} source requires --input.")


def run_runtime_command(args: argparse.Namespace) -> dict[str, object]:
    """Run the existing runtime inspection flow."""

    from rescue_inspection.app import run_app

    task_type = getattr(args, "task", None)
    source_type = args.source
    _validate_args(source_type, args.input)
    summary = run_app(
        args.config,
        task_type=task_type,
        source_type=source_type,
        input_path=args.input,
        output_root=args.output_root,
        max_frames=args.max_frames,
        show_preview=args.preview,
        preview_window_name=args.preview_window_name,
    )
    payload: dict[str, object] = {
        "best_frame_id": summary.best_frame_id,
        "total_frames": summary.total_frames,
        "frames_with_detection": summary.frames_with_detection,
        "artifacts_dir": str(summary.artifact_dir),
        "artifact_paths": summary.artifact_paths,
    }

    if hasattr(summary, "qr_detected"):
        payload["qr_detected"] = summary.qr_detected
        payload["qr_message"] = summary.qr_message
        payload["qr_stability_score"] = summary.qr_stability_score
        payload["hazmat_detected"] = summary.hazmat_detected
        payload["hazmat_label"] = summary.hazmat_label
        payload["hazmat_stability_score"] = summary.hazmat_stability_score
    else:
        payload["final_detected"] = summary.final_detected
        payload["final_decoded_text"] = summary.final_decoded_text
        payload["stability_score"] = summary.final_stability_score

    if (task_type or "").lower() == "hazmat" or summary.run_id.startswith("hazmat_"):
        payload["final_label"] = getattr(summary, "final_decoded_text", "")
    return payload


def run_check_runtime_assets_command(args: argparse.Namespace) -> dict[str, object]:
    """Check required sensing runtime assets and print their status."""

    from dataclasses import asdict
    from rescue_inspection.runtime.assets import validate_runtime_assets

    result = validate_runtime_assets(args.config)
    return asdict(result)

