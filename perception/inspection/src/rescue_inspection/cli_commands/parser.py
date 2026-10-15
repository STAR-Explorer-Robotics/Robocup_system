"""Argument parser construction for inspection-run."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""

    parser = argparse.ArgumentParser(description="Run inspection tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the inspection baseline.")
    run_parser.add_argument("--config", required=True, type=Path, help="Path to YAML config.")
    run_parser.add_argument(
        "--task",
        choices=["qr", "hazmat", "sensing", "combined"],
        help="Override task type from config.",
    )
    run_parser.add_argument(
        "--source",
        choices=["image", "video", "realsense"],
        help="Override source type from config.",
    )
    run_parser.add_argument("--input", type=Path, help="Override image/video input path.")
    run_parser.add_argument("--output-root", type=Path, help="Override artifact root directory.")
    run_parser.add_argument("--max-frames", type=int, help="Maximum frames to process.")
    run_parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the live overlay preview for video/realsense sources.",
    )
    run_parser.add_argument(
        "--preview-window-name",
        type=str,
        help="Override the preview window title when preview is enabled.",
    )

    subparsers.add_parser(
        "check-onnx-providers",
        help="Print ONNX Runtime execution providers available in the current container.",
    )

    label_parser = subparsers.add_parser(
        "build-hazmat-labels",
        help="Write the Hazmat labels manifest.",
    )
    label_parser.add_argument("--output", type=Path, default=Path("data/manifests/hazmat_labels.yaml"))
    label_parser.add_argument("--template-root", default="samples/Hazmat2020")


    classify_crop_parser = subparsers.add_parser(
        "classify-hazmat-crop",
        help="Run the configured Hazmat classifier directly on a rectified crop.",
    )
    classify_crop_parser.add_argument("--config", required=True, type=Path)
    classify_crop_parser.add_argument("--input", required=True, type=Path)
    classify_crop_parser.add_argument("--topk", type=int, default=5)
    classify_crop_parser.add_argument("--report", type=Path)

    color_profiles_parser = subparsers.add_parser(
        "debug-hazmat-color-profiles",
        help="Generate the standard Hazmat color profile JSON from configured templates.",
    )
    color_profiles_parser.add_argument("--config", required=True, type=Path)
    color_profiles_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON path, for example data/processed/hazmat_color_profiles_standard.json.",
    )

    color_evidence_parser = subparsers.add_parser(
        "debug-hazmat-color-evidence",
        help="Write a CSV report with Hazmat color evidence for image files.",
    )
    color_evidence_parser.add_argument("--config", required=True, type=Path)
    color_evidence_parser.add_argument("--input", required=True, type=Path)
    color_evidence_parser.add_argument("--output", required=True, type=Path)

    ocr_evidence_parser = subparsers.add_parser(
        "debug-hazmat-ocr-evidence",
        help="Write a CSV report with optional Hazmat OCR keyword evidence for image files.",
    )
    ocr_evidence_parser.add_argument("--config", required=True, type=Path)
    ocr_evidence_parser.add_argument("--input", required=True, type=Path)
    ocr_evidence_parser.add_argument("--output", required=True, type=Path)

    check_assets_parser = subparsers.add_parser(
        "check-runtime-assets",
        help="Check that all required runtime model assets and configuration paths exist.",
    )
    check_assets_parser.add_argument("--config", required=True, type=Path, help="Path to YAML config.")
    return parser

