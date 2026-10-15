# Perception

A self-contained source group for rescue marker sensing, with the original two-package layout preserved.

| Component | Responsibility |
| --- | --- |
| [inspection](inspection/README.md) | Standalone QR, AprilTag and Hazmat processing, temporal confirmation and saved results |
| [qrc_ros2_perception](qrc_ros2_perception/README.md) | ROS 2 image input, detection/result publication and a result-panel node |
| [compose.sensing-release.yaml](compose.sensing-release.yaml) | Imported CPU/GPU container service descriptions |

The inspection source, representative configurations, label metadata, two ONNX models, Hazmat prototypes and one QR sample image are retained. Training workflows and field-operation instructions are not part of this edition.

The standalone inspection package declares Python 3.11+. The ROS 2 Humble bridge uses its own container dependency setup and source import path; this edition has not reconciled or validated every standalone/container environment combination.

The module contributes the combined sensing pipeline and stable-result handling. It does not claim new QR decoding, a new neural-network architecture, or measured recognition accuracy. The arm execution stack is not included.

See [source origins](../docs/source_origins.md) and [license information](../LICENSES.md), including the unresolved distribution metadata for model weights.

