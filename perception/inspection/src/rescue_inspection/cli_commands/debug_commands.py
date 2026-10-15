"""Runtime debug command handlers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def run_classify_hazmat_crop_command(args: argparse.Namespace) -> dict[str, object]:
    """Run the configured Hazmat classifier directly on a rectified crop."""

    from rescue_inspection.config import load_config
    from rescue_inspection.logging_utils import configure_logging
    from rescue_inspection.tasks.hazmat.debug_artifacts import (
        infer_run_root_from_crop,
        write_realsense_failure_report,
    )
    from rescue_inspection.tasks.hazmat.recognizer import build_hazmat_recognizer

    config = load_config(args.config)
    logger = configure_logging(config.runtime.log_level)
    recognizer = build_hazmat_recognizer(config, logger)
    classifier = getattr(recognizer, "classifier", None)
    if classifier is None or not hasattr(classifier, "diagnose_patch"):
        raise RuntimeError("Configured Hazmat backend does not support direct crop diagnosis.")

    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(args.input)
    diagnose_payload = classifier.diagnose_patch(image, topk=args.topk)
    diagnose_payload.pop("classifier_input_bgr", None)
    report_path = args.report
    run_root = infer_run_root_from_crop(args.input.resolve())
    if report_path is None and run_root is not None:
        report_path = run_root / "hazmat_realsense_failure_report.md"

    if report_path is not None:
        bundle_dir = args.input.parent / args.input.stem.split("_rejected_classifier_", 1)[0]
        meta_path = bundle_dir / "classifier_input_meta.json"
        classifier_input_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        report_path = write_realsense_failure_report(
            report_path=report_path,
            crop_path=args.input,
            diagnose_payload=diagnose_payload,
            classifier_input_meta=classifier_input_meta,
        )
        diagnose_payload["report_path"] = str(report_path)
    return diagnose_payload


def run_debug_hazmat_color_profiles_command(args: argparse.Namespace) -> dict[str, object]:
    """Generate the configured standard Hazmat color profile JSON."""

    from rescue_inspection.config import load_config
    from rescue_inspection.project_paths import resolve_project_path
    from rescue_inspection.tasks.hazmat.color_evidence import write_standard_color_profile
    from rescue_inspection.tasks.hazmat.label_registry import LabelRegistry

    config = load_config(args.config)
    registry = LabelRegistry.from_file(config.hazmat.labels_file)
    payload = write_standard_color_profile(
        output_path=args.output,
        config=config.hazmat.color,
        registry=registry,
        manifest_path=config.hazmat.labels_file,
    )
    output_path = resolve_project_path(args.output)
    return {
        "output": str(output_path),
        "schema_version": payload["schema_version"],
        "source": payload["source"],
        "profile_count": len(payload["profiles"]),
        "source_file_count": len(payload["source_files"]),
        "color_extraction_config_hash": payload["color_extraction_config_hash"],
    }


def run_debug_hazmat_color_evidence_command(args: argparse.Namespace) -> dict[str, object]:
    """Write a CSV report for Hazmat color evidence over a file or directory."""

    from rescue_inspection.config import load_config
    from rescue_inspection.project_paths import resolve_project_path
    from rescue_inspection.tasks.hazmat.color_evidence import (
        COLOR_RATIO_KEYS,
        HazmatColorProfileStore,
        extract_color_evidence,
    )
    from rescue_inspection.utils.io import ensure_dir

    config = load_config(args.config)
    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)
    image_paths = _iter_image_paths(input_path)
    profile_store = None
    profile_path = resolve_project_path(config.hazmat.color.profile_path)
    if profile_path.exists():
        profile_store = HazmatColorProfileStore.load(profile_path)

    fieldnames = [
        "image",
        "true_label",
        "classifier_top1",
        "classifier_score",
        "color_top1",
        "color_score",
        "fused_top1",
        "fused_score",
        *COLOR_RATIO_KEYS,
        "quality_flags",
        "fusion_reason",
    ]
    rows = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        evidence = extract_color_evidence(image, config.hazmat.color)
        match = profile_store.match(evidence) if profile_store is not None else None
        rows.append(
            {
                "image": str(image_path),
                "true_label": _infer_label_from_path(image_path),
                "classifier_top1": "",
                "classifier_score": "",
                "color_top1": match.top1_label if match else "",
                "color_score": f"{match.top1_score:.6f}" if match else "",
                "fused_top1": "",
                "fused_score": "",
                **{key: f"{evidence.ratios[key]:.6f}" for key in COLOR_RATIO_KEYS},
                "quality_flags": "|".join(evidence.quality_flags),
                "fusion_reason": "report_only",
            }
        )

    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"output": str(output_path), "image_count": len(rows), "profile_loaded": profile_store is not None}


def run_debug_hazmat_ocr_evidence_command(args: argparse.Namespace) -> dict[str, object]:
    """Write a CSV report for optional Hazmat OCR evidence over a file or directory."""

    from rescue_inspection.config import load_config
    from rescue_inspection.logging_utils import configure_logging
    from rescue_inspection.project_paths import resolve_project_path
    from rescue_inspection.tasks.hazmat.label_registry import LabelRegistry
    from rescue_inspection.tasks.hazmat.ocr_evidence import (
        extract_ocr_text,
        full_ocr_evidence_payload,
        match_hazmat_ocr_keywords,
        normalize_ocr_text,
    )
    from rescue_inspection.utils.io import ensure_dir

    config = load_config(args.config)
    logger = configure_logging(config.runtime.log_level)
    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)
    registry = LabelRegistry.from_file(config.hazmat.labels_file)
    image_paths = _iter_image_paths(input_path)
    fieldnames = [
        "image",
        "true_label",
        "classifier_top1",
        "classifier_score",
        "color_top1",
        "color_score",
        "ocr_raw_text",
        "ocr_top1",
        "ocr_score",
        "matched_keywords",
        "matched_numbers",
        "fused_top1",
        "fused_score",
        "fusion_reason",
        "ocr_available",
        "ocr_disabled_reason",
        "ocr_backend",
        "ocr_backend_error",
        "ocr_reused",
    ]
    rows = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        raw_evidence = extract_ocr_text(image, config.hazmat.ocr, logger=logger)
        normalized = normalize_ocr_text(raw_evidence.raw_text)
        evidence = match_hazmat_ocr_keywords(
            normalized,
            registry,
            config.hazmat.ocr,
            raw_text=raw_evidence.raw_text,
            backend=raw_evidence.ocr_backend,
            base_evidence=raw_evidence,
        )
        payload = full_ocr_evidence_payload(evidence)
        rows.append(
            {
                "image": str(image_path),
                "true_label": _infer_label_from_path(image_path),
                "classifier_top1": "",
                "classifier_score": "",
                "color_top1": "",
                "color_score": "",
                "ocr_raw_text": payload["ocr_raw_text"],
                "ocr_top1": payload["ocr_top1_label"],
                "ocr_score": f"{payload['ocr_top1_score']:.6f}",
                "matched_keywords": "|".join(payload["ocr_matched_keywords"]),
                "matched_numbers": "|".join(payload["ocr_matched_numbers"]),
                "fused_top1": "",
                "fused_score": "",
                "fusion_reason": "report_only",
                "ocr_available": payload["ocr_available"],
                "ocr_disabled_reason": payload["ocr_disabled_reason"] or "",
                "ocr_backend": payload["ocr_backend"],
                "ocr_backend_error": payload["ocr_backend_error"] or "",
                "ocr_reused": payload["ocr_reused"],
            }
        )

    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"output": str(output_path), "image_count": len(rows), "ocr_enabled": config.hazmat.ocr.enabled}


def _iter_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in suffixes and path.is_file())


def _infer_label_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("placard-"):
            return Path(part).stem.replace("placard-", "")
    return ""
