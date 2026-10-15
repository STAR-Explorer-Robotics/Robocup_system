"""Prototype-gated Hazmat classifier runtime."""

from __future__ import annotations

import logging

import numpy as np

from rescue_inspection.config import HazmatClassifierConfig
from rescue_inspection.runtime.paths import resolve_runtime_model_path
from rescue_inspection.tasks.hazmat.cnn_classifier import CNNHazmatClassifier
from rescue_inspection.tasks.hazmat.label_registry import LabelRegistry, UNKNOWN_LABEL
from rescue_inspection.models import HazmatClassificationDebug


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_norm = np.linalg.norm(lhs)
    rhs_norm = np.linalg.norm(rhs)
    if lhs_norm <= 1e-8 or rhs_norm <= 1e-8:
        return 0.0
    return float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))


class PrototypeHazmatClassifier(CNNHazmatClassifier):
    """CNN classifier with prototype-similarity gating."""

    def __init__(
        self,
        config: HazmatClassifierConfig,
        registry: LabelRegistry,
        *,
        active_label_set: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(config, registry, active_label_set=active_label_set, logger=logger)
        self.prototypes_path = resolve_runtime_model_path(config.prototypes_path or "models/hazmat/hazmat_prototypes.npz")
        self._prototype_map: dict[str, np.ndarray] | None = None

    def classify_with_debug(self, patch: np.ndarray) -> HazmatClassificationDebug:
        logits, embeddings = self._infer(patch)
        debug, _ = self._build_debug_bundle(logits, embeddings)
        return debug

    def diagnose_patch(self, patch: np.ndarray, *, topk: int = 5) -> dict[str, object]:
        payload = super().diagnose_patch(patch, topk=topk)
        _, embeddings = self._infer(patch)
        if embeddings is None:
            payload["topk_prototype"] = []
            return payload
        similarities = []
        for label in self.registry.labels:
            similarity = self._prototype_similarity(label.key, embeddings)
            similarities.append({"label": label.key, "similarity": float(similarity)})
        similarities.sort(key=lambda item: item["similarity"], reverse=True)
        payload["topk_prototype"] = similarities[:topk]
        return payload

    def _build_debug_bundle(
        self,
        logits: np.ndarray,
        embeddings: np.ndarray | None,
    ) -> tuple[HazmatClassificationDebug, np.ndarray]:
        probs = np.exp(logits / self.config.temperature - np.max(logits / self.config.temperature))
        probs = probs / np.clip(probs.sum(), 1e-8, None)
        top_indices = np.argsort(probs)[::-1]
        top1_index = int(top_indices[0])
        top2_index = int(top_indices[1]) if len(top_indices) > 1 else top1_index
        top1_label = self._label_for_index(top1_index)
        top2_label = self._label_for_index(top2_index)
        top1_score = float(probs[top1_index])
        top2_score = float(probs[top2_index]) if len(top_indices) > 1 else 0.0
        margin = top1_score - top2_score if len(top_indices) > 1 else top1_score
        prototype_similarity = self._prototype_similarity(top1_label, embeddings if embeddings is not None else logits)

        gated_reason = None
        if top1_label == UNKNOWN_LABEL:
            gated_reason = "predicted_unknown"
        elif top1_score < self.config.top1_min_score:
            gated_reason = "top1_below_min_score"
        elif len(top_indices) > 1 and margin < self.config.top1_top2_min_margin:
            gated_reason = "top1_top2_margin_too_small"
        elif prototype_similarity < self.config.prototype_min_similarity:
            gated_reason = "prototype_similarity_too_low"
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
                backend="prototype",
                prototype_similarity=prototype_similarity,
                gated_reason=gated_reason,
            ),
            probs,
        )

    def _prototype_similarity(self, label: str, embedding: np.ndarray) -> float:
        if label == UNKNOWN_LABEL:
            return 0.0
        if not self.prototypes_path.exists():
            raise RuntimeError(
                f"prototypes_path does not exist; run make train-hazmat-classifier and make export-hazmat-classifier "
                f"or switch hazmat.mode=classical ({self.prototypes_path})"
            )
        if self._prototype_map is None:
            payload = np.load(self.prototypes_path, allow_pickle=True)
            labels = [str(value) for value in payload["labels"].tolist()]
            vectors = np.asarray(payload["prototypes"], dtype=np.float32)
            self._prototype_map = {key: vectors[index] for index, key in enumerate(labels)}
        vector = self._prototype_map.get(label)
        if vector is None:
            return 0.0
        return _cosine_similarity(embedding.astype(np.float32), vector.astype(np.float32))
