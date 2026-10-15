"""Compatibility wrapper for the inspection sensing result schema."""

from rescue_inspection.runtime import (
    QrcSensingResult,
    SensingFrameResult,
    result_to_json,
    result_to_payload,
)

__all__ = [
    "QrcSensingResult",
    "SensingFrameResult",
    "result_to_json",
    "result_to_payload",
]
