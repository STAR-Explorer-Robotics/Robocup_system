"""Runtime asset discovery and validation logic."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import yaml

from rescue_inspection.config import AppConfig, load_config
from rescue_inspection.runtime.paths import (
    resolve_inspection_root,
    resolve_runtime_model_path,
    resolve_inspection_relative_path,
)


@dataclass
class RuntimeAsset:
    """Represents a single runtime resource asset."""

    name: str
    configured_path: str
    resolved_path: str
    exists: bool
    required: bool
    source: str
    note: str = ""


@dataclass
class RuntimeAssetValidationResult:
    """Summary of runtime assets validation."""

    ok: bool
    config_path: str
    path_policy: dict[str, dict[str, str]]
    assets: list[RuntimeAsset]
    missing_required: list[str]


def collect_runtime_assets(config: AppConfig) -> list[RuntimeAsset]:
    """Inspect config and collect resolved paths and metadata for all potential runtime assets."""

    assets: list[RuntimeAsset] = []
    task_type = config.task.type
    is_hazmat_active = task_type in {"hazmat", "sensing", "combined"}

    # 1. Target Detector ONNX Model
    td_enabled = config.target_detector.enabled
    td_model_configured = str(config.target_detector.model_path)
    td_model_resolved = resolve_runtime_model_path(config.target_detector.model_path)
    assets.append(
        RuntimeAsset(
            name="target_detector_model",
            configured_path=td_model_configured,
            resolved_path=str(td_model_resolved),
            exists=td_model_resolved.exists() and td_model_resolved.is_file(),
            required=bool(td_enabled),
            source="target_detector.model_path",
            note="Required if target_detector.enabled is true",
        )
    )

    # 2. Target Detector Metadata
    td_meta_configured = str(config.target_detector.metadata_path)
    td_meta_resolved = resolve_runtime_model_path(config.target_detector.metadata_path)
    assets.append(
        RuntimeAsset(
            name="target_detector_metadata",
            configured_path=td_meta_configured,
            resolved_path=str(td_meta_resolved),
            exists=td_meta_resolved.exists() and td_meta_resolved.is_file(),
            required=bool(td_enabled),
            source="target_detector.metadata_path",
            note="Required if target_detector.enabled is true (no fallback)",
        )
    )

    # 3. Hazmat Labels File
    labels_configured = str(config.hazmat.labels_file)
    labels_resolved = resolve_inspection_relative_path(config.hazmat.labels_file)
    assets.append(
        RuntimeAsset(
            name="hazmat_labels_file",
            configured_path=labels_configured,
            resolved_path=str(labels_resolved),
            exists=labels_resolved.exists() and labels_resolved.is_file(),
            required=is_hazmat_active,
            source="hazmat.labels_file",
            note="Required for mapping predictions if hazmat/sensing/combined task is active",
        )
    )

    # Read manifest contents if labels file exists, to extract template_root
    template_root = "samples/Hazmat2020"
    if labels_resolved.exists() and labels_resolved.is_file():
        try:
            with open(labels_resolved, "r", encoding="utf-8") as f:
                manifest_data = yaml.safe_load(f) or {}
                if isinstance(manifest_data, dict):
                    template_root = manifest_data.get("template_root", template_root)
        except Exception:
            pass

    # 4. Hazmat CNN Classifier Model
    hazmat_mode = config.hazmat.mode
    classifier_backend = config.hazmat.classifier.backend
    is_dl_classifier = is_hazmat_active and hazmat_mode in {"classifier_only", "dl", "hybrid"} and classifier_backend in {"cnn", "prototype"}

    cls_model_raw = config.hazmat.classifier.model_path
    cls_model_configured = str(cls_model_raw) if cls_model_raw else "models/hazmat/hazmat_classifier.onnx"
    cls_model_resolved = resolve_runtime_model_path(cls_model_raw or "models/hazmat/hazmat_classifier.onnx")
    assets.append(
        RuntimeAsset(
            name="hazmat_classifier_model",
            configured_path=cls_model_configured,
            resolved_path=str(cls_model_resolved),
            exists=cls_model_resolved.exists() and cls_model_resolved.is_file(),
            required=is_dl_classifier,
            source="hazmat.classifier.model_path",
            note="Required for feature extraction in CNN/Prototype classifier",
        )
    )

    # 5. Hazmat Prototypes
    is_prototype_classifier = is_hazmat_active and hazmat_mode in {"classifier_only", "dl", "hybrid"} and classifier_backend == "prototype"

    prototypes_raw = config.hazmat.classifier.prototypes_path
    prototypes_configured = str(prototypes_raw) if prototypes_raw else "models/hazmat/hazmat_prototypes.npz"
    prototypes_resolved = resolve_runtime_model_path(prototypes_raw or "models/hazmat/hazmat_prototypes.npz")
    assets.append(
        RuntimeAsset(
            name="hazmat_prototypes",
            configured_path=prototypes_configured,
            resolved_path=str(prototypes_resolved),
            exists=prototypes_resolved.exists() and prototypes_resolved.is_file(),
            required=is_prototype_classifier,
            source="hazmat.classifier.prototypes_path",
            note="Required for prototype similarity gating in Prototype classifier",
        )
    )

    # 6. Optional Hazmat Color Profile
    color_profile_configured = str(config.hazmat.color.profile_path)
    color_profile_resolved = resolve_inspection_relative_path(config.hazmat.color.profile_path)
    assets.append(
        RuntimeAsset(
            name="hazmat_color_profile",
            configured_path=color_profile_configured,
            resolved_path=str(color_profile_resolved),
            exists=color_profile_resolved.exists() and color_profile_resolved.is_file(),
            required=False,
            source="hazmat.color.profile_path",
            note="Optional; missing profile disables low-weight color fusion without failing runtime",
        )
    )

    # 7. Hazmat Templates Directory
    is_template_required = is_hazmat_active and (
        hazmat_mode == "classical" or
        (hazmat_mode in {"classifier_only", "hybrid"} and classifier_backend == "template")
    )
    templates_resolved = resolve_inspection_relative_path(template_root)
    assets.append(
        RuntimeAsset(
            name="hazmat_templates_dir",
            configured_path=template_root,
            resolved_path=str(templates_resolved),
            exists=templates_resolved.exists() and templates_resolved.is_dir(),
            required=is_template_required,
            source="manifest.template_root",
            note="Directory containing templates for classical HSV template classifier",
        )
    )

    return assets


def validate_runtime_assets(config_path: str | Path) -> RuntimeAssetValidationResult:
    """Load configuration from config_path and perform existence check on all active/required assets."""

    resolved_config_path = Path(config_path).resolve()
    config = load_config(resolved_config_path)
    assets = collect_runtime_assets(config)

    missing_required = [asset.name for asset in assets if asset.required and not asset.exists]
    ok = len(missing_required) == 0

    model_root_env = os.environ.get("QRC_RUNTIME_MODEL_ROOT")
    output_root_env = os.environ.get("QRC_OUTPUT_ROOT")
    config_root_env = os.environ.get("QRC_CONFIG_ROOT")

    path_policy = {
        "QRC_RUNTIME_MODEL_ROOT": {
            "value": str(Path(model_root_env).resolve()) if model_root_env else str(resolve_inspection_root() / "models"),
            "source": "environment" if model_root_env else "default",
        },
        "QRC_OUTPUT_ROOT": {
            "value": str(Path(output_root_env).resolve()) if output_root_env else str(resolve_inspection_root() / "outputs"),
            "source": "environment" if output_root_env else "default",
        },
        "QRC_CONFIG_ROOT": {
            "value": str(Path(config_root_env).resolve()) if config_root_env else str(resolve_inspection_root() / "configs"),
            "source": "environment" if config_root_env else "default",
        },
    }

    return RuntimeAssetValidationResult(
        ok=ok,
        config_path=str(resolved_config_path),
        path_policy=path_policy,
        assets=assets,
        missing_required=missing_required,
    )
