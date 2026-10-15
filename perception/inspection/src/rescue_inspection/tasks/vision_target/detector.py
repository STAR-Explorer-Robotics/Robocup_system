"""ONNX-backed generic visual target ROI detector."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

from rescue_inspection.config import TargetDetectorConfig
from rescue_inspection.models import FramePacket
from rescue_inspection.runtime.paths import resolve_runtime_model_path
from rescue_inspection.tasks.vision_target.geometry import (
    crop_box,
    letterbox_image,
    map_box_from_letterbox,
    nms,
    polygon_from_box,
    square_padded_crop,
    xywh_to_xyxy,
)
from rescue_inspection.tasks.vision_target.metadata import (
    TargetDetectorMetadata,
    load_target_detector_metadata,
    validate_target_detector_metadata,
)
from rescue_inspection.tasks.vision_target.types import TargetROIDetection, TargetROIDetectorDisabled
from rescue_inspection.utils.onnx_runtime import create_onnx_session, get_session_providers


class InferenceSessionLike(Protocol):
    """Minimal session protocol used by onnxruntime and tests."""

    def get_inputs(self) -> list[Any]: ...

    def run(self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]: ...


class TargetROIDetector:
    """Run a deployed ONNX ROI detector and return original-image ROI candidates."""

    def __init__(
        self,
        config: TargetDetectorConfig,
        *,
        logger: logging.Logger | None = None,
        session: InferenceSessionLike | None = None,
        metadata: TargetDetectorMetadata | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model_path = resolve_runtime_model_path(config.model_path)
        self.metadata_path = resolve_runtime_model_path(config.metadata_path)
        self._session = session
        self._input_name: str | None = None
        self.metadata = metadata
        self.runtime_providers: list[str] = []

        if not config.enabled and session is None:
            raise TargetROIDetectorDisabled("target_detector.enabled is false")

        self._validate_or_load_metadata()
        if self._session is None:
            self._load_session()
        if self._session is not None:
            self._input_name = self._session.get_inputs()[0].name
            self.runtime_providers = get_session_providers(self._session, self.runtime_providers)

    def detect(self, packet: FramePacket, *, class_names: set[str] | None = None) -> list[TargetROIDetection]:
        """Detect target ROIs in one frame."""

        model_input, info = letterbox_image(packet.image, self.config.input_size)
        tensor = self._image_to_tensor(model_input)
        outputs = self._session.run(None, {self._input_name: tensor})
        decoded = self._decode_outputs(outputs[0])
        if class_names is not None:
            decoded = [item for item in decoded if item["class_name"] in class_names]

        keep = _class_aware_nms(decoded, self.config.nms_iou_threshold)
        detections: list[TargetROIDetection] = []
        for index in keep[: self.config.max_detections]:
            item = decoded[index]
            mapped_box = map_box_from_letterbox(item["bbox_xyxy"], info)
            if not _has_minimum_box_area(mapped_box):
                continue
            crop = crop_box(packet.image, mapped_box)
            square_crop = square_padded_crop(packet.image, mapped_box)
            detections.append(
                TargetROIDetection(
                    class_name=str(item["class_name"]),
                    class_id=int(item["class_id"]),
                    bbox_xyxy=mapped_box,
                    score=float(item["score"]),
                    polygon=polygon_from_box(mapped_box),
                    crop=crop,
                    square_crop=square_crop,
                    diagnostics={
                        "model_path": str(self.model_path),
                        "input_size": self.config.input_size,
                        "source": "target_detector_onnx",
                        "execution_providers": self.runtime_providers,
                        "class_name": str(item["class_name"]),
                        "class_id": int(item["class_id"]),
                        "score": float(item["score"]),
                        "bbox_xyxy": mapped_box,
                    },
                )
            )
        return detections

    def _validate_or_load_metadata(self) -> None:
        if self.metadata is None:
            if not self.metadata_path.exists():
                self._handle_policy(
                    self.config.missing_model_policy,
                    f"target detector metadata does not exist: {self.metadata_path}",
                )
            self.metadata = load_target_detector_metadata(self.metadata_path)
        try:
            validate_target_detector_metadata(self.metadata, self.config)
        except ValueError as exc:
            self._handle_policy(self.config.metadata_mismatch_policy, str(exc))

    def _load_session(self) -> None:
        if not self.model_path.exists():
            self._handle_policy(
                self.config.missing_model_policy,
                f"target detector model does not exist: {self.model_path}",
            )
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            self._handle_policy(
                self.config.runtime_unavailable_policy,
                "onnxruntime is unavailable for target detector runtime",
            )
            raise TargetROIDetectorDisabled(str(exc)) from exc
        self._session, self.runtime_providers = create_onnx_session(
            ort,
            str(self.model_path),
            self.config.execution_providers,
            logger=self.logger,
            context="target detector",
        )

    def _handle_policy(self, policy: str, message: str) -> None:
        if policy == "fail_fast":
            raise RuntimeError(message)
        self.logger.warning("%s; disabling target ROI detector.", message)
        raise TargetROIDetectorDisabled(message)

    def _image_to_tensor(self, image: np.ndarray) -> np.ndarray:
        rgb = image[:, :, ::-1].astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)

    def _decode_outputs(self, output: np.ndarray) -> list[dict[str, object]]:
        """Decode YOLO-style xywh objectness/class-score outputs."""

        return _decode_yolo_output(output, class_names=self.config.classes, score_threshold=self.config.score_threshold)


def _decode_yolo_output(
    output: np.ndarray,
    *,
    class_names: list[str],
    score_threshold: float,
) -> list[dict[str, object]]:
    """Decode YOLO-style rows into one candidate per class above threshold."""

    rows = _normalize_yolo_rows(output, class_count=len(class_names))
    class_count = len(class_names)
    xywh_objectness_class_cols = 5 + class_count

    decoded: list[dict[str, object]] = []
    for row in rows:
        if rows.shape[1] == xywh_objectness_class_cols:
            objectness = float(row[4])
            class_scores = row[5:5 + class_count]
        else:
            objectness = 1.0
            class_scores = row[4:4 + class_count]

        for class_id, class_score_value in enumerate(class_scores):
            class_score = float(class_score_value)
            score = objectness * class_score
            if score < score_threshold:
                continue
            decoded.append(
                {
                    "bbox_xyxy": xywh_to_xyxy(row),
                    "score": score,
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "objectness": objectness,
                    "class_score": class_score,
                }
            )
    return decoded


def _normalize_yolo_rows(output: np.ndarray, *, class_count: int) -> np.ndarray:
    """Return model output as row-major xywh[/objectness]/class-score rows."""

    rows = np.asarray(output, dtype=np.float32)
    if rows.ndim == 3:
        rows = rows[0]
    if rows.ndim != 2:
        raise RuntimeError(f"Unsupported target detector output rank: {rows.ndim}")

    xywh_class_cols = 4 + class_count
    xywh_objectness_class_cols = 5 + class_count
    if rows.shape[0] in {xywh_class_cols, xywh_objectness_class_cols}:
        rows = rows.T
    if rows.shape[1] not in {xywh_class_cols, xywh_objectness_class_cols}:
        raise RuntimeError(
            "Unsupported target detector output shape: "
            f"{rows.shape}; expected {xywh_class_cols} or {xywh_objectness_class_cols} columns"
        )
    return rows


def _class_aware_nms(decoded: list[dict[str, object]], iou_threshold: float) -> list[int]:
    """Run NMS within each class, then return kept indices globally sorted by score."""

    kept: list[int] = []
    class_ids = sorted({int(item["class_id"]) for item in decoded})
    for class_id in class_ids:
        indices = [index for index, item in enumerate(decoded) if int(item["class_id"]) == class_id]
        boxes = [decoded[index]["bbox_xyxy"] for index in indices]
        scores = [float(decoded[index]["score"]) for index in indices]
        kept.extend(indices[index] for index in nms(boxes, scores, iou_threshold))
    return sorted(kept, key=lambda index: float(decoded[index]["score"]), reverse=True)


def _has_minimum_box_area(bbox_xyxy: tuple[float, float, float, float]) -> bool:
    """Return true when a mapped box is large enough to crop and debug safely."""

    x1, y1, x2, y2 = bbox_xyxy
    return (x2 - x1) >= 1.0 and (y2 - y1) >= 1.0
