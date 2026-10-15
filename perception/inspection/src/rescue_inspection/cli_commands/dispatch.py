"""Command dispatch for inspection-run."""

from __future__ import annotations

import argparse


def dispatch_command(args: argparse.Namespace) -> dict[str, object]:
    """Dispatch parsed CLI args to the selected command handler."""

    if args.command == "run":
        from rescue_inspection.cli_commands.runtime_commands import run_runtime_command

        return run_runtime_command(args)
    if args.command == "check-onnx-providers":
        from rescue_inspection.cli_commands.provider_commands import run_check_onnx_providers_command

        return run_check_onnx_providers_command()
    if args.command == "build-hazmat-labels":
        from rescue_inspection.cli_commands.label_commands import run_build_hazmat_labels_command

        return run_build_hazmat_labels_command(args)
    if args.command == "classify-hazmat-crop":
        from rescue_inspection.cli_commands.debug_commands import run_classify_hazmat_crop_command

        return run_classify_hazmat_crop_command(args)
    if args.command == "debug-hazmat-color-profiles":
        from rescue_inspection.cli_commands.debug_commands import run_debug_hazmat_color_profiles_command

        return run_debug_hazmat_color_profiles_command(args)
    if args.command == "debug-hazmat-color-evidence":
        from rescue_inspection.cli_commands.debug_commands import run_debug_hazmat_color_evidence_command

        return run_debug_hazmat_color_evidence_command(args)
    if args.command == "debug-hazmat-ocr-evidence":
        from rescue_inspection.cli_commands.debug_commands import run_debug_hazmat_ocr_evidence_command

        return run_debug_hazmat_ocr_evidence_command(args)
    if args.command == "check-runtime-assets":
        from rescue_inspection.cli_commands.runtime_commands import run_check_runtime_assets_command

        return run_check_runtime_assets_command(args)
    raise ValueError(f"Unsupported command: {args.command}")
