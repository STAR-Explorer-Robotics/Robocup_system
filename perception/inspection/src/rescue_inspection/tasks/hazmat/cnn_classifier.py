"""CNN-backed Hazmat classifier runtime."""

from __future__ import annotations

import logging

import numpy as np

from rescue_inspection.config import HazmatClassifierConfig
from rescue_inspection.models import HazmatClassificationDebug
from rescue_inspection.runtime.paths import resolve_runtime_model_path
from rescue_inspection.tasks.hazmat.preprocessing import preprocess_hazmat_patch
from rescue_inspection.tasks.hazmat.label_registry import LabelRegistry, UNKNOWN_LABEL
from rescue_inspection.utils.onnx_runtime import create_onnx_session


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.clip(exp_values.sum(), 1e-8, None)


class CNNHazmatClassifier:
    """Run a CNN classifier through ONNX Runtime."""

    def __init__(
        self,
        config: HazmatClassifierConfig,
        registry: LabelRegistry,
        *,
        active_label_set: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.active_label_set = active_label_set
        self.logger = logger or logging.getLogger(__name__)
        self.model_path = resolve_runtime_model_path(config.model_path or "models/hazmat/hazmat_classifier.onnx")
        self._onnx_session = None
        self._onnx_input_name: str | None = None
        self.runtime_backend = "uninitialized"
        self.runtime_providers: list[str] = []

    def classify(self, patch: np.ndarray) -> tuple[str, float]:
        """Return the predicted label or UNKNOWN after gating."""

        debug = self.classify_with_debug(patch)
        return (debug.top1_label if debug.gated_reason is None else UNKNOWN_LABEL, debug.top1_score)

    def classify_with_debug(self, patch: np.ndarray) -> HazmatClassificationDebug:
        """Return classification scores and gating details for a patch."""

        logits, _ = self._infer(patch)
        debug, _ = self._build_debug_bundle(logits, None)
        return debug

    def _label_for_index(self, index: int) -> str:
        if index >= len(self.registry.labels):
            return UNKNOWN_LABEL
        return self.registry.id_to_key(index)

    def build_preprocess_debug(self, patch: np.ndarray) -> dict[str, object]:
        """Return the exact runtime preprocessing payload used by the classifier."""

        payload = preprocess_hazmat_patch(patch, input_size=self.config.input_size)
        meta = dict(payload["meta"])
        meta.update(
            {
                "classifier_backend": self.config.backend,
                "active_label_set": self.active_label_set,
                "model_path": str(self.model_path),
                "execution_providers": self.runtime_providers,
            }
        )
        payload["meta"] = meta
        return payload

    def diagnose_patch(self, patch: np.ndarray, *, topk: int = 5) -> dict[str, object]:
        """Run runtime-equivalent preprocessing and inference for one rectified crop."""

        self._ensure_backend_loaded()
        preprocess = self.build_preprocess_debug(patch)
        logits, embeddings = self._infer_from_tensor(np.asarray(preprocess["tensor"], dtype=np.float32))
        debug, probs = self._build_debug_bundle(logits, embeddings)
        label_keys = [self._label_for_index(int(index)) for index in np.argsort(probs)[::-1][:topk]]
        topk_softmax = [
            {"label": label, "score": float(probs[self._index_for_label(label)])}
            for label in label_keys
        ]
        return {
            "runtime_backend": self.runtime_backend,
            "active_label_set": self.active_label_set,
            "active_labels": self.registry.active_label_keys(self.active_label_set),
            "topk_softmax": topk_softmax,
            "topk_prototype": [],
            "final_decision": "accept" if debug.gated_reason is None else "reject",
            "rejected": debug.gated_reason is not None,
            "reject_reason": debug.gated_reason,
            "classifier_debug": debug.model_dump(mode="json"),
            "preprocess_meta": preprocess["meta"],
            "classifier_input_bgr": preprocess["classifier_input_bgr"],
        }

    def _infer(self, patch: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        self._ensure_backend_loaded()
        preprocess = self.build_preprocess_debug(patch)
        return self._infer_from_tensor(np.asarray(preprocess["tensor"], dtype=np.float32))

    def _infer_from_tensor(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        self._ensure_backend_loaded()
        if self._onnx_session is None or self._onnx_input_name is None:
            raise RuntimeError("Hazmat classifier ONNX Runtime session is not loaded")
        outputs = self._onnx_session.run(None, {self._onnx_input_name: tensor})
        logits = np.asarray(outputs[0][0], dtype=np.float32)
        embeddings = np.asarray(outputs[1][0], dtype=np.float32) if len(outputs) > 1 else logits.astype(np.float32)
        return logits, embeddings

    def _build_debug_bundle(
        self,
        logits: np.ndarray,
        embeddings: np.ndarray | None,
    ) -> tuple[HazmatClassificationDebug, np.ndarray]:
        probs = _softmax(logits / self.config.temperature)
        top_indices = np.argsort(probs)[::-1]
        top1_index = int(top_indices[0])
        top2_index = int(top_indices[1]) if len(top_indices) > 1 else top1_index

        top1_label = self._label_for_index(top1_index)
        top2_label = self._label_for_index(top2_index)
        top1_score = float(probs[top1_index])
        top2_score = float(probs[top2_index]) if len(top_indices) > 1 else 0.0
        margin = top1_score - top2_score if len(top_indices) > 1 else top1_score
        gated_reason = None

        if top1_label == UNKNOWN_LABEL:
            gated_reason = "predicted_unknown"
        elif top1_score < self.config.top1_min_score:
            gated_reason = "top1_below_min_score"
        elif len(top_indices) > 1 and margin < self.config.top1_top2_min_margin:
            gated_reason = "top1_top2_margin_too_small"
        elif not self.registry.is_active(top1_label, self.active_label_set):
            gated_reason = "inactive_label"

        return (
            HazmatClassificationDebug(
                top1_label=top1_label,
                top1_score=top1_score,
                top2_label=top2_label,
                top2_score=top2_score,
                margin=margin,
                raw_label=top1_label,
                backend=self.config.backend,
                gated_reason=gated_reason,
            ),
            probs,
        )

    def _index_for_label(self, label: str) -> int:
        if label == UNKNOWN_LABEL:
            return len(self.registry.labels)
        return self.registry.key_to_id(label)

    def _ensure_backend_loaded(self) -> None:
        if self.model_path.suffix.lower() != ".onnx":
            raise RuntimeError(
                "Hazmat classifier runtime requires an ONNX model_path; "
                f"got {self.model_path}. Export a training checkpoint with make export-hazmat-classifier."
            )
        if not self.model_path.exists():
            raise RuntimeError(
                f"Hazmat classifier ONNX model_path does not exist: {self.model_path}. "
                "Provide the runtime asset or run make export-hazmat-classifier to create an ONNX model."
            )
        if self._onnx_session is not None:
            return

        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for Hazmat classifier runtime. "
                "Install inspection[runtime] for CPU inference or inspection[runtime-gpu] for GPU inference."
            ) from exc

        self._onnx_session, self.runtime_providers = create_onnx_session(
            ort,
            str(self.model_path),
            self.config.execution_providers,
            logger=self.logger,
            context="hazmat classifier",
        )
        self._onnx_input_name = self._onnx_session.get_inputs()[0].name
        self.runtime_backend = "onnxruntime"
