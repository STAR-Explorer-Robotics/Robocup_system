# Inspection runtime

The standalone Python sensing package. Its command entry point is inspection-run, implemented by rescue_inspection.cli.

## Main source

- [config.py](src/rescue_inspection/config.py): typed runtime configuration.
- [tasks/sensing/pipeline.py](src/rescue_inspection/tasks/sensing/pipeline.py): combined task pipeline.
- [tasks/qr/detector.py](src/rescue_inspection/tasks/qr/detector.py): QR decoding.
- [tasks/apriltag/detector.py](src/rescue_inspection/tasks/apriltag/detector.py): AprilTag detection.
- [tasks/hazmat/pipeline.py](src/rescue_inspection/tasks/hazmat/pipeline.py): Hazmat recognition and evidence fusion.
- [runtime/confirmed_capture.py](src/rescue_inspection/runtime/confirmed_capture.py): stable-result transitions and evidence capture.
- [configs/sensing_runtime_dl.yaml](configs/sensing_runtime_dl.yaml): representative model/runtime settings.

## Inputs and outputs

Inputs are image, video or configured camera frames. Outputs include structured detections, temporally confirmed results, overlays and optional result files. Model, label and output roots are configurable.

The provided models/ directory contains the imported ONNX weights and Hazmat prototypes; samples/ contains a small QR image. Their hashes are recorded in the repository manifest.

## Dependencies and limits

See [pyproject.toml](pyproject.toml). The standalone package declares Python 3.11+, NumPy, OpenCV, pupil-apriltags, Pydantic and PyYAML; ONNX Runtime and RealSense are optional dependency groups. Container definitions are retained as references, not as a validated release deployment.

The model training process, full evaluation data and field operation scripts are outside this snapshot. See the [module overview](../README.md) and [license inventory](../../LICENSES.md).

