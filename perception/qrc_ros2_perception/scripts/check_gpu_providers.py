#!/usr/bin/env python3
"""Print actual ONNX Runtime providers used by the ROS2 sensing bridge."""

from __future__ import annotations

import argparse
import json

from qrc_ros2_perception.bridge import QrcSensingBridge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection-config", default="../inspection/configs/sensing_runtime_dl.yaml")
    args = parser.parse_args()

    bridge = QrcSensingBridge(args.inspection_config)
    processor = getattr(bridge.runtime, "_processor", None)
    if processor is None:
        raise RuntimeError("QrcSensingBridge runtime does not expose an inspection sensing processor.")

    qr_roi_detector = getattr(processor.qr_detector, "_get_roi_detector")()
    hazmat_recognizer = processor.hazmat_recognizer
    hazmat_dl_detector = getattr(hazmat_recognizer, "dl_detector", None) or getattr(
        hazmat_recognizer,
        "detector",
        None,
    )
    hazmat_target_detector = getattr(hazmat_dl_detector, "_target_detector", None)
    hazmat_classifier = getattr(hazmat_recognizer, "classifier", None)
    if hasattr(hazmat_classifier, "_ensure_backend_loaded"):
        hazmat_classifier._ensure_backend_loaded()

    payload = {
        "inspection_config": args.inspection_config,
        "target_detector_requested": bridge.config.target_detector.execution_providers,
        "qr_roi_detector_actual": getattr(qr_roi_detector, "runtime_providers", None),
        "hazmat_roi_detector_actual": getattr(hazmat_target_detector, "runtime_providers", None),
        "hazmat_classifier_requested": bridge.config.hazmat.classifier.execution_providers,
        "hazmat_classifier_backend": getattr(hazmat_classifier, "runtime_backend", None),
        "hazmat_classifier_actual": getattr(hazmat_classifier, "runtime_providers", None),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
