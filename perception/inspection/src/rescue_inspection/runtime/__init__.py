"""Stable runtime facade for external inspection callers."""

from rescue_inspection.runtime.artifacts import SensingRunArtifactWriter
from rescue_inspection.runtime.assets import (
    RuntimeAsset,
    RuntimeAssetValidationResult,
    collect_runtime_assets,
    validate_runtime_assets,
)
from rescue_inspection.runtime.confirmed_capture import (
    ConfirmedEvent,
    LatestConfirmedRefresh,
    SensingConfirmedCapture,
    SensingConfirmedRecordResult,
)
from rescue_inspection.runtime.frame_metadata import SensingFrameMetadata
from rescue_inspection.runtime.paths import (
    resolve_repo_root,
    resolve_runtime_output_root,
)
from rescue_inspection.runtime.sensing_runtime import SensingRuntime
from rescue_inspection.runtime.sensing_schema import (
    QrcSensingResult,
    SensingFrameResult,
    result_to_json,
    result_to_payload,
)

__all__ = [
    "QrcSensingResult",
    "RuntimeAsset",
    "RuntimeAssetValidationResult",
    "ConfirmedEvent",
    "LatestConfirmedRefresh",
    "SensingFrameMetadata",
    "SensingFrameResult",
    "SensingConfirmedCapture",
    "SensingConfirmedRecordResult",
    "SensingRunArtifactWriter",
    "SensingRuntime",
    "collect_runtime_assets",
    "resolve_repo_root",
    "resolve_runtime_output_root",
    "result_to_json",
    "result_to_payload",
    "validate_runtime_assets",
]
