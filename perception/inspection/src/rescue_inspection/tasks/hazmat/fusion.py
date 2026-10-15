"""Low-weight Hazmat classifier/color fusion."""

from __future__ import annotations

from dataclasses import dataclass, field

from rescue_inspection.config import HazmatFusionConfig
from rescue_inspection.models import HazmatClassificationDebug
from rescue_inspection.tasks.hazmat.color_evidence import HazmatColorMatch
from rescue_inspection.tasks.hazmat.label_registry import UNKNOWN_LABEL
from rescue_inspection.tasks.hazmat.ocr_evidence import HazmatOcrEvidence


@dataclass(frozen=True)
class HazmatFusionResult:
    """Final fused decision for one candidate."""

    label: str
    confidence: float
    accepted: bool
    fused_top1_label: str
    fused_top1_score: float
    fused_top2_label: str = ""
    fused_top2_score: float = 0.0
    fused_margin: float = 0.0
    fusion_method: str = ""
    fusion_reason: str = ""
    quality_penalty: float = 0.0
    color_weight_used: float = 0.0
    ocr_weight_used: float = 0.0
    fused_scores_by_label: dict[str, float] = field(default_factory=dict)


def fuse_classifier_and_color(
    *,
    classifier_debug: HazmatClassificationDebug,
    color_match: HazmatColorMatch | None,
    config: HazmatFusionConfig,
    ocr_evidence: HazmatOcrEvidence | None = None,
) -> HazmatFusionResult:
    """Fuse classifier top candidates with low-weight color/OCR evidence."""

    classifier_label = classifier_debug.top1_label or UNKNOWN_LABEL
    classifier_score = float(classifier_debug.top1_score)
    classifier_accepted = classifier_debug.gated_reason is None and classifier_label != UNKNOWN_LABEL

    if not config.enabled or not config.color_enabled or color_match is None or color_match.unavailable_reason:
        base_result = HazmatFusionResult(
            label=classifier_label if classifier_accepted else UNKNOWN_LABEL,
            confidence=classifier_score,
            accepted=classifier_accepted,
            fused_top1_label=classifier_label,
            fused_top1_score=classifier_score,
            fused_top2_label=classifier_debug.top2_label,
            fused_top2_score=float(classifier_debug.top2_score),
            fused_margin=float(classifier_debug.margin),
            fusion_method="classifier_only",
            fusion_reason="color_unavailable" if color_match is None else color_match.unavailable_reason or "fusion_disabled",
        )
        return _apply_ocr_bonus(
            result=base_result,
            classifier_debug=classifier_debug,
            classifier_accepted=classifier_accepted,
            config=config,
            ocr_evidence=ocr_evidence,
            high_confidence=False,
        )

    quality_penalty = _quality_penalty(color_match, config)
    candidates = _classifier_candidate_scores(classifier_debug)
    if not candidates:
        return HazmatFusionResult(
            label=UNKNOWN_LABEL,
            confidence=classifier_score,
            accepted=False,
            fused_top1_label=classifier_label,
            fused_top1_score=classifier_score,
            fusion_method="classifier_only",
            fusion_reason="no_classifier_candidates",
            quality_penalty=quality_penalty,
        )

    high_confidence = (
        classifier_debug.gated_reason is None
        and classifier_score >= config.high_confidence_classifier_score
        and classifier_debug.margin >= config.high_confidence_classifier_margin
    )

    fused_scores = {
        label: _fused_score(classifier_value, color_match.scores_by_label.get(label, 0.0), quality_penalty, config)
        for label, classifier_value in candidates.items()
    }
    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    fused_top1_label, fused_top1_score = ranked[0]
    fused_top2_label, fused_top2_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
    fused_margin = float(fused_top1_score - fused_top2_score)

    if high_confidence:
        accepted = classifier_accepted
        base_result = HazmatFusionResult(
            label=classifier_label if accepted else UNKNOWN_LABEL,
            confidence=max(0.0, min(1.0, fused_scores.get(classifier_label, classifier_score))),
            accepted=accepted,
            fused_top1_label=classifier_label,
            fused_top1_score=max(0.0, min(1.0, fused_scores.get(classifier_label, classifier_score))),
            fused_top2_label=fused_top2_label,
            fused_top2_score=fused_top2_score,
            fused_margin=float(classifier_debug.margin),
            fusion_method="color_fusion",
            fusion_reason="classifier_high_confidence",
            quality_penalty=quality_penalty,
            color_weight_used=config.color_weight,
            fused_scores_by_label=fused_scores,
        )
        return _apply_ocr_bonus(
            result=base_result,
            classifier_debug=classifier_debug,
            classifier_accepted=classifier_accepted,
            config=config,
            ocr_evidence=ocr_evidence,
            high_confidence=True,
        )

    reason = "color_supports_classifier_top1"
    selected_label = fused_top1_label
    selected_score = fused_top1_score
    if selected_label != classifier_label:
        if selected_label == classifier_debug.top2_label and config.allow_color_top2_flip:
            reason = "color_top2_flip"
        else:
            selected_label = classifier_label
            selected_score = fused_scores.get(classifier_label, classifier_score)
            reason = "color_flip_blocked"
    elif classifier_debug.gated_reason is not None:
        reason = "color_supported_gated_classifier"

    selected_margin = fused_margin if selected_label == fused_top1_label else float(selected_score - fused_top2_score)
    accepted = classifier_debug.gated_reason is None
    if not accepted and config.allow_color_accept_gated_classifier:
        accepted = selected_score >= config.min_fused_score and selected_margin >= config.min_fused_margin
    if selected_score < config.min_fused_score:
        accepted = False
        reason = "fused_score_below_min"
    if selected_margin < config.min_fused_margin:
        accepted = False
        reason = "fused_margin_below_min"

    base_result = HazmatFusionResult(
        label=selected_label if accepted else UNKNOWN_LABEL,
        confidence=max(0.0, min(1.0, selected_score)),
        accepted=accepted,
        fused_top1_label=fused_top1_label,
        fused_top1_score=max(0.0, min(1.0, fused_top1_score)),
        fused_top2_label=fused_top2_label,
        fused_top2_score=max(0.0, min(1.0, fused_top2_score)),
        fused_margin=max(0.0, selected_margin),
        fusion_method="color_fusion",
        fusion_reason=reason,
        quality_penalty=quality_penalty,
        color_weight_used=config.color_weight,
        fused_scores_by_label=fused_scores,
    )
    return _apply_ocr_bonus(
        result=base_result,
        classifier_debug=classifier_debug,
        classifier_accepted=classifier_accepted,
        config=config,
        ocr_evidence=ocr_evidence,
        high_confidence=False,
    )


def _classifier_candidate_scores(debug: HazmatClassificationDebug) -> dict[str, float]:
    candidates = {}
    if debug.top1_label and debug.top1_label != UNKNOWN_LABEL:
        candidates[debug.top1_label] = float(debug.top1_score)
    if debug.top2_label and debug.top2_label != UNKNOWN_LABEL:
        candidates[debug.top2_label] = float(debug.top2_score)
    return candidates


def _fused_score(
    classifier_score: float,
    color_score: float,
    quality_penalty: float,
    config: HazmatFusionConfig,
) -> float:
    color_delta = min(config.color_weight * color_score, config.max_color_delta)
    score = config.classifier_weight * classifier_score + color_delta - quality_penalty
    return float(max(0.0, min(1.0, score)))


def _apply_ocr_bonus(
    *,
    result: HazmatFusionResult,
    classifier_debug: HazmatClassificationDebug,
    classifier_accepted: bool,
    config: HazmatFusionConfig,
    ocr_evidence: HazmatOcrEvidence | None,
    high_confidence: bool,
) -> HazmatFusionResult:
    """Apply capped OCR bonus to the existing classifier/color fused scores."""

    if not _ocr_usable(ocr_evidence, config):
        return result
    if high_confidence and not config.high_confidence_ocr_override:
        return result

    candidates = _classifier_candidate_scores(classifier_debug)
    if not candidates:
        return result
    base_scores = dict(result.fused_scores_by_label) if result.fused_scores_by_label else dict(candidates)
    scores = {label: float(base_scores.get(label, candidates.get(label, 0.0))) for label in candidates}
    supported = False
    for label in list(scores):
        ocr_score = float(ocr_evidence.scores_by_label.get(label, 0.0)) if ocr_evidence else 0.0
        if ocr_score <= 0.0:
            continue
        supported = True
        bonus = min(config.ocr_weight * ocr_score, config.max_ocr_bonus)
        scores[label] = max(0.0, min(1.0, scores[label] + bonus))

    if not supported:
        return HazmatFusionResult(
            **{
                **result.__dict__,
                "fusion_reason": _append_reason(result.fusion_reason, "ocr_outside_classifier_candidates"),
                "ocr_weight_used": config.ocr_weight,
            }
        )

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused_top1_label, fused_top1_score = ranked[0]
    fused_top2_label, fused_top2_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
    fused_margin = float(fused_top1_score - fused_top2_score)

    selected_label = fused_top1_label
    selected_score = fused_top1_score
    reason = _append_reason(result.fusion_reason, "ocr_supports_classifier_top1")
    if selected_label != result.label and selected_label != classifier_debug.top1_label:
        if selected_label == classifier_debug.top2_label and config.allow_ocr_top2_flip:
            reason = _append_reason(result.fusion_reason, "ocr_top2_flip")
        else:
            selected_label = result.label if result.label != UNKNOWN_LABEL else classifier_debug.top1_label
            selected_score = scores.get(selected_label, result.confidence)
            reason = _append_reason(result.fusion_reason, "ocr_flip_blocked")

    selected_margin = fused_margin if selected_label == fused_top1_label else float(selected_score - fused_top2_score)
    accepted = result.accepted
    if not accepted and classifier_accepted:
        accepted = selected_score >= config.min_fused_score and selected_margin >= config.min_fused_margin
    if selected_score < config.min_fused_score:
        accepted = False
        reason = _append_reason(result.fusion_reason, "ocr_fused_score_below_min")
    if selected_margin < config.min_fused_margin:
        accepted = False
        reason = _append_reason(result.fusion_reason, "ocr_fused_margin_below_min")

    return HazmatFusionResult(
        label=selected_label if accepted else UNKNOWN_LABEL,
        confidence=max(0.0, min(1.0, selected_score)),
        accepted=accepted,
        fused_top1_label=fused_top1_label,
        fused_top1_score=max(0.0, min(1.0, fused_top1_score)),
        fused_top2_label=fused_top2_label,
        fused_top2_score=max(0.0, min(1.0, fused_top2_score)),
        fused_margin=max(0.0, selected_margin),
        fusion_method="ocr_fusion" if result.fusion_method == "classifier_only" else result.fusion_method,
        fusion_reason=reason,
        quality_penalty=result.quality_penalty,
        color_weight_used=result.color_weight_used,
        ocr_weight_used=config.ocr_weight,
        fused_scores_by_label=scores,
    )


def _ocr_usable(ocr_evidence: HazmatOcrEvidence | None, config: HazmatFusionConfig) -> bool:
    if not config.enabled or not config.ocr_enabled or ocr_evidence is None:
        return False
    if not ocr_evidence.ocr_available or ocr_evidence.ocr_disabled_reason:
        return False
    return bool(ocr_evidence.scores_by_label)


def _append_reason(current: str, addition: str) -> str:
    if not current:
        return addition
    if addition in current.split("+"):
        return current
    return f"{current}+{addition}"


def _quality_penalty(color_match: HazmatColorMatch, config: HazmatFusionConfig) -> float:
    if color_match.evidence is None:
        return 0.0
    flags = set(color_match.evidence.quality_flags)
    penalty_units = 0.0
    if "low_valid_pixel_ratio" in flags:
        penalty_units += 1.0
    if "likely_background_or_pipe" in flags:
        penalty_units += 0.8
    if "high_white_ratio" in flags:
        penalty_units += 0.5
    if "low_saturation" in flags:
        penalty_units += 0.4
    if "shadow_heavy" in flags:
        penalty_units += 0.4
    return float(min(config.quality_penalty_weight * penalty_units, config.max_color_delta))
