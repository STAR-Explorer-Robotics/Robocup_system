"""Optional OCR evidence and known-set Hazmat keyword matching."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Any

import cv2
import numpy as np

from rescue_inspection.config import HazmatDebugConfig, HazmatOcrConfig
from rescue_inspection.tasks.hazmat.label_registry import LabelRegistry


_LOGGER = logging.getLogger(__name__)
_WARNED_BACKENDS: set[str] = set()


@dataclass(frozen=True)
class HazmatOcrEvidence:
    """OCR keyword evidence for one Hazmat crop."""

    raw_text: str = ""
    normalized_text: str = ""
    tokens: list[str] = field(default_factory=list)
    matched_numbers: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    scores_by_label: dict[str, float] = field(default_factory=dict)
    top1_label: str | None = None
    top1_score: float = 0.0
    top2_label: str | None = None
    top2_score: float = 0.0
    margin: float = 0.0
    quality_flags: list[str] = field(default_factory=list)
    backend: str = "none"
    ocr_available: bool = False
    ocr_disabled_reason: str | None = None
    ocr_backend: str = "none"
    ocr_backend_error: str | None = None
    ocr_reused: bool = False


@dataclass(frozen=True)
class NormalizedOcrText:
    """Normalized OCR text and token stream."""

    text: str
    tokens: list[str]


@dataclass(frozen=True)
class OcrCacheContext:
    """Conservative cache context for reusing OCR evidence."""

    frame_id: int
    candidate_index: int
    bbox_xyxy: tuple[float, float, float, float]
    classifier_top1_label: str
    classifier_top2_label: str


@dataclass(frozen=True)
class CachedOcrEvidence:
    """Cached OCR evidence with the context that produced it."""

    context: OcrCacheContext
    evidence: HazmatOcrEvidence


@dataclass(frozen=True)
class KeywordCandidate:
    """One possible keyword match before span priority filtering."""

    label: str
    keyword: str
    strength: str
    span: tuple[int, int]
    fuzzy: bool
    score: float
    priority: tuple[int, int, int]


KEYWORD_RULES: dict[str, dict[str, list[str]]] = {
    "1.1-explosives": {
        "strong_numbers": ["1.1"],
        "strong_keywords": ["EXPLOSIVE", "EXPLOSIVES"],
    },
    "1.5-blasting-agents": {
        "strong_numbers": ["1.5"],
        "strong_keywords": ["BLASTING", "AGENT", "BLASTING AGENT"],
    },
    "2-flammable-gas": {
        "weak_numbers": ["2"],
        "weak_keywords": ["FLAMMABLE", "GAS"],
        "strong_keywords": ["FLAMMABLE GAS"],
    },
    "2-non-flammable-gas": {
        "weak_numbers": ["2"],
        "strong_keywords": [
            "NON FLAMMABLE GAS",
            "NON-FLAMMABLE GAS",
            "NONFLAMMABLE GAS",
            "NOT FLAMMABLE GAS",
            "NON FLAMMABLE",
            "NON-FLAMMABLE",
            "NONFLAMMABLE",
            "NOT FLAMMABLE",
        ],
    },
    "2-oxygen": {
        "weak_numbers": ["2"],
        "strong_keywords": ["OXYGEN"],
    },
    "3-fuel-oil": {
        "weak_numbers": ["3"],
        "weak_keywords": ["FUEL", "OIL"],
        "strong_keywords": ["FUEL OIL"],
    },
    "4-dangerous-when-wet": {
        "weak_numbers": ["4"],
        "strong_keywords": ["DANGEROUS", "WET", "DANGEROUS WET", "DANGEROUS WHEN WET"],
    },
    "4-flammable-solid": {
        "weak_numbers": ["4"],
        "weak_keywords": ["FLAMMABLE"],
        "strong_keywords": ["FLAMMABLE SOLID", "SOLID"],
    },
    "4-spontaneously-combustible": {
        "weak_numbers": ["4"],
        "strong_keywords": ["SPONTANEOUSLY", "COMBUSTIBLE", "SPONTANEOUSLY COMBUSTIBLE"],
    },
    "5.1-oxidizer": {
        "strong_numbers": ["5.1"],
        "strong_keywords": ["OXIDIZER", "OXIDISER", "OXIDIZING"],
    },
    "5.2-organic-peroxide": {
        "strong_numbers": ["5.2"],
        "strong_keywords": ["ORGANIC", "PEROXIDE", "ORGANIC PEROXIDE"],
    },
    "6-inhalation-hazard": {
        "weak_numbers": ["6"],
        "strong_keywords": ["INHALATION", "HAZARD", "INHALATION HAZARD"],
    },
    "6-poison": {
        "weak_numbers": ["6"],
        "strong_keywords": ["POISON"],
    },
    "7-radioactive": {
        "medium_numbers": ["7"],
        "strong_keywords": ["RADIOACTIVE"],
    },
    "8-corrosive": {
        "medium_numbers": ["8"],
        "strong_keywords": ["CORROSIVE"],
    },
}


def extract_ocr_text(
    image_bgr: np.ndarray,
    config: HazmatOcrConfig,
    *,
    logger: logging.Logger | None = None,
) -> HazmatOcrEvidence:
    """Run the configured optional OCR backend and return raw text status."""

    logger = logger or _LOGGER
    backend = config.backend
    if not config.enabled:
        return _disabled_evidence(backend=backend, reason="ocr_disabled")
    if backend == "none":
        return _disabled_evidence(backend=backend, reason="backend_none")
    if image_bgr.size == 0 or min(image_bgr.shape[:2]) < config.min_crop_size:
        return _disabled_evidence(backend=backend, reason="crop_too_small")
    if backend == "external":
        _log_once(logger, "external", "Hazmat OCR external backend is not implemented; OCR disabled.")
        return _disabled_evidence(backend=backend, reason="external_backend_unimplemented")
    if backend != "tesseract":
        return _disabled_evidence(backend=backend, reason="unsupported_backend")

    try:
        import pytesseract  # type: ignore[import-not-found]
    except ImportError as exc:
        _log_once(logger, "pytesseract", "Hazmat OCR requested tesseract but pytesseract is unavailable.")
        return _disabled_evidence(
            backend=backend,
            reason="pytesseract_unavailable",
            error=str(exc),
        )
    if shutil.which("tesseract") is None:
        _log_once(logger, "tesseract_binary", "Hazmat OCR requested tesseract but binary is unavailable.")
        return _disabled_evidence(backend=backend, reason="tesseract_binary_unavailable")

    try:
        preprocessed = preprocess_for_tesseract(image_bgr)
        raw_text = str(pytesseract.image_to_string(preprocessed, config="--psm 6"))
    except Exception as exc:  # pragma: no cover - depends on optional backend
        _log_once(logger, "tesseract_runtime", f"Hazmat OCR tesseract backend failed: {exc}")
        return _disabled_evidence(backend=backend, reason="backend_error", error=str(exc))
    return HazmatOcrEvidence(
        raw_text=raw_text,
        backend=backend,
        ocr_backend=backend,
        ocr_available=bool(raw_text.strip()),
        ocr_disabled_reason=None if raw_text.strip() else "no_text",
    )


def preprocess_for_tesseract(image_bgr: np.ndarray) -> np.ndarray:
    """Return an in-memory OCR-friendly grayscale image."""

    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(image_bgr[..., :3], cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    scale = 2 if min(height, width) < 160 else 1
    if scale > 1:
        gray = cv2.resize(gray, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def normalize_ocr_text(raw_text: str) -> NormalizedOcrText:
    """Normalize OCR text without global character substitutions."""

    text = raw_text.upper()
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = _normalize_numeric_context(text)
    text = re.sub(r"[^A-Z0-9.\-\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"[A-Z0-9]+(?:[.\-][A-Z0-9]+)*", text)
    return NormalizedOcrText(text=text, tokens=tokens)


def match_hazmat_ocr_keywords(
    tokens: list[str] | NormalizedOcrText,
    label_registry: LabelRegistry,
    config: HazmatOcrConfig,
    *,
    raw_text: str = "",
    backend: str | None = None,
    base_evidence: HazmatOcrEvidence | None = None,
) -> HazmatOcrEvidence:
    """Score OCR tokens against the known Hazmat label keyword registry."""

    normalized = tokens if isinstance(tokens, NormalizedOcrText) else NormalizedOcrText(" ".join(tokens), list(tokens))
    text = normalized.text
    scores: dict[str, float] = {}
    matched_keywords: set[str] = set()
    matched_numbers: set[str] = set()
    quality_flags: set[str] = set(base_evidence.quality_flags if base_evidence else [])

    active_labels = set(label_registry.active_label_keys())
    keyword_scores, keyword_matches, strong_semantic_labels = _score_keywords(
        normalized,
        active_labels=active_labels,
        config=config,
    )
    for label, rules in KEYWORD_RULES.items():
        if active_labels and label not in active_labels:
            continue
        score = 0.0
        strong_semantic = label in strong_semantic_labels
        for number in rules.get("strong_numbers", []):
            if _number_present(number, normalized.tokens):
                score += config.strong_number_bonus
                matched_numbers.add(number)
        for number in rules.get("medium_numbers", []):
            if _number_present(number, normalized.tokens):
                score += config.medium_number_bonus
                matched_numbers.add(number)
        for number in rules.get("weak_numbers", []):
            if _number_present(number, normalized.tokens):
                score += config.weak_number_bonus
                matched_numbers.add(number)
        score += keyword_scores.get(label, 0.0)
        if label in {"7-radioactive", "8-corrosive"} and score > 0 and not strong_semantic:
            quality_flags.add("number_without_semantic_keyword")
        if score > 0:
            scores[label] = float(min(score, config.max_ocr_bonus))
    matched_keywords.update(keyword_matches)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top1_label, top1_score = ranked[0] if ranked else (None, 0.0)
    top2_label, top2_score = ranked[1] if len(ranked) > 1 else (None, 0.0)
    available = bool(raw_text.strip() or text or scores)
    disabled_reason = None if available else "no_text"
    if base_evidence and base_evidence.ocr_disabled_reason and not raw_text.strip():
        disabled_reason = base_evidence.ocr_disabled_reason
    return HazmatOcrEvidence(
        raw_text=raw_text or (base_evidence.raw_text if base_evidence else ""),
        normalized_text=text,
        tokens=list(normalized.tokens),
        matched_numbers=sorted(matched_numbers),
        matched_keywords=sorted(matched_keywords),
        scores_by_label=scores,
        top1_label=top1_label,
        top1_score=float(top1_score),
        top2_label=top2_label,
        top2_score=float(top2_score),
        margin=float(top1_score - top2_score),
        quality_flags=sorted(quality_flags),
        backend=backend or (base_evidence.backend if base_evidence else config.backend),
        ocr_available=available and bool(scores),
        ocr_disabled_reason=disabled_reason,
        ocr_backend=backend or (base_evidence.ocr_backend if base_evidence else config.backend),
        ocr_backend_error=base_evidence.ocr_backend_error if base_evidence else None,
        ocr_reused=base_evidence.ocr_reused if base_evidence else False,
    )


def reused_ocr_evidence(evidence: HazmatOcrEvidence) -> HazmatOcrEvidence:
    """Return a copy marked as reused from cache."""

    return replace(evidence, ocr_reused=True)


def can_reuse_ocr_cache(
    *,
    cached: CachedOcrEvidence,
    context: OcrCacheContext,
    config: HazmatOcrConfig,
) -> bool:
    """Return whether cached OCR evidence is conservative enough to reuse."""

    if cached.context.candidate_index != context.candidate_index:
        return False
    if cached.context.classifier_top1_label != context.classifier_top1_label:
        return False
    if cached.context.classifier_top2_label != context.classifier_top2_label:
        return False
    if _bbox_iou(cached.context.bbox_xyxy, context.bbox_xyxy) < config.cache_min_iou:
        return False
    if _bbox_size_ratio(cached.context.bbox_xyxy, context.bbox_xyxy) < config.cache_min_size_ratio:
        return False
    return True


def serialize_ocr_evidence_for_debug(
    evidence: HazmatOcrEvidence,
    *,
    ocr_config: HazmatOcrConfig,
    debug_config: HazmatDebugConfig,
) -> dict[str, object]:
    """Serialize OCR evidence according to debug verbosity settings."""

    payload: dict[str, object] = {
        "ocr_raw_text": _truncate(evidence.raw_text, ocr_config.debug_text_max_chars),
        "ocr_normalized_text": _truncate(evidence.normalized_text, ocr_config.debug_text_max_chars),
        "ocr_top1_label": evidence.top1_label or "",
        "ocr_top1_score": evidence.top1_score,
        "ocr_top2_label": evidence.top2_label or "",
        "ocr_top2_score": evidence.top2_score,
        "ocr_margin": evidence.margin,
        "ocr_matched_keywords": list(evidence.matched_keywords),
        "ocr_matched_numbers": list(evidence.matched_numbers),
        "ocr_quality_flags": list(evidence.quality_flags),
        "ocr_available": evidence.ocr_available,
        "ocr_disabled_reason": evidence.ocr_disabled_reason,
        "ocr_backend": evidence.ocr_backend,
        "ocr_backend_error": evidence.ocr_backend_error,
        "ocr_reused": evidence.ocr_reused,
    }
    if debug_config.include_full_ocr_scores:
        payload["ocr_scores_by_label"] = dict(evidence.scores_by_label)
    return payload


def full_ocr_evidence_payload(evidence: HazmatOcrEvidence) -> dict[str, Any]:
    """Return full OCR evidence for explicit debug files or CLI reports."""

    return {
        "ocr_raw_text": evidence.raw_text,
        "ocr_normalized_text": evidence.normalized_text,
        "ocr_tokens": list(evidence.tokens),
        "ocr_matched_keywords": list(evidence.matched_keywords),
        "ocr_matched_numbers": list(evidence.matched_numbers),
        "ocr_scores_by_label": dict(evidence.scores_by_label),
        "ocr_top1_label": evidence.top1_label or "",
        "ocr_top1_score": evidence.top1_score,
        "ocr_top2_label": evidence.top2_label or "",
        "ocr_top2_score": evidence.top2_score,
        "ocr_margin": evidence.margin,
        "ocr_quality_flags": list(evidence.quality_flags),
        "ocr_available": evidence.ocr_available,
        "ocr_disabled_reason": evidence.ocr_disabled_reason,
        "ocr_backend": evidence.ocr_backend,
        "ocr_backend_error": evidence.ocr_backend_error,
        "ocr_reused": evidence.ocr_reused,
    }


def _disabled_evidence(
    *,
    backend: str,
    reason: str,
    error: str | None = None,
) -> HazmatOcrEvidence:
    return HazmatOcrEvidence(
        backend=backend,
        ocr_backend=backend,
        ocr_available=False,
        ocr_disabled_reason=reason,
        ocr_backend_error=error,
        quality_flags=[reason],
    )


def _normalize_numeric_context(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        value = match.group(0)
        return value.replace("O", "0").replace("I", "1").replace("S", "5")

    return re.sub(r"(?<![A-Z])(?:[0-9OIS][.\-\s]*){1,3}[0-9OIS]?(?![A-Z])", repl, text)


def _number_present(number: str, tokens: list[str]) -> bool:
    normalized = {token.replace("-", ".") for token in tokens}
    if number in normalized:
        return True
    return number in {token.strip(".") for token in normalized}


def _score_keywords(
    normalized: NormalizedOcrText,
    *,
    active_labels: set[str],
    config: HazmatOcrConfig,
) -> tuple[dict[str, float], set[str], set[str]]:
    """Score keywords after global phrase-priority span consumption."""

    words = _keyword_words(normalized)
    candidates: list[KeywordCandidate] = []
    for label, rules in KEYWORD_RULES.items():
        if active_labels and label not in active_labels:
            continue
        for keyword in rules.get("strong_keywords", []):
            candidates.extend(
                _keyword_candidates(
                    label=label,
                    keyword=keyword,
                    strength="strong",
                    words=words,
                    config=config,
                )
            )
        for keyword in rules.get("weak_keywords", []):
            candidates.extend(
                _keyword_candidates(
                    label=label,
                    keyword=keyword,
                    strength="weak",
                    words=words,
                    config=config,
                )
            )

    consumed: set[int] = set()
    selected: list[KeywordCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.priority, reverse=True):
        if _blocked_by_negative_modifier(candidate, words):
            continue
        span_indices = set(range(candidate.span[0], candidate.span[1]))
        if span_indices & consumed:
            continue
        selected.append(candidate)
        consumed.update(span_indices)

    scores: dict[str, float] = {}
    matched_keywords: set[str] = set()
    strong_semantic_labels: set[str] = set()
    for candidate in selected:
        scores[candidate.label] = scores.get(candidate.label, 0.0) + candidate.score
        matched_keywords.add(candidate.keyword)
        if candidate.strength == "strong":
            strong_semantic_labels.add(candidate.label)
    scores = _apply_class2_conflict_group(scores, selected)
    return scores, matched_keywords, strong_semantic_labels


def _keyword_candidates(
    *,
    label: str,
    keyword: str,
    strength: str,
    words: list[str],
    config: HazmatOcrConfig,
) -> list[KeywordCandidate]:
    normalized_keyword = keyword.replace("-", " ")
    parts = normalized_keyword.split()
    if not parts:
        return []
    score = (
        config.strong_keyword_bonus * _phrase_multiplier(keyword)
        if strength == "strong"
        else config.weak_keyword_bonus * _phrase_multiplier(keyword)
    )
    strength_priority = 3 if strength == "strong" else 1
    candidates = []
    for span, fuzzy in _find_keyword_spans(parts, words, config):
        candidates.append(
            KeywordCandidate(
                label=label,
                keyword=keyword,
                strength=strength,
                span=span,
                fuzzy=fuzzy,
                score=score,
                priority=(strength_priority, span[1] - span[0], 0 if fuzzy else 1),
            )
        )
    return candidates


def _find_keyword_spans(
    parts: list[str],
    words: list[str],
    config: HazmatOcrConfig,
) -> list[tuple[tuple[int, int], bool]]:
    phrase_len = len(parts)
    spans: list[tuple[tuple[int, int], bool]] = []
    for index in range(max(len(words) - phrase_len + 1, 0)):
        window = words[index : index + phrase_len]
        if window == parts:
            spans.append(((index, index + phrase_len), False))
        elif (
            config.fuzzy_match_enabled
            and phrase_len == 1
            and not _is_negative_modifier_keyword(parts)
            and _similarity(" ".join(parts), " ".join(window)) >= config.fuzzy_match_threshold
        ):
            spans.append(((index, index + phrase_len), True))
    return spans


def _keyword_words(normalized: NormalizedOcrText) -> list[str]:
    """Return keyword-matching words, splitting hyphenated OCR tokens."""

    return [part for token in normalized.tokens for part in token.replace("-", " ").split()]


def _is_negative_modifier_keyword(parts: list[str]) -> bool:
    text = " ".join(parts)
    return text.startswith("NON ") or text.startswith("NOT ") or text.startswith("NO ") or "NONFLAMMABLE" in parts


def _blocked_by_negative_modifier(candidate: KeywordCandidate, words: list[str]) -> bool:
    if candidate.label != "2-flammable-gas":
        return False
    keyword = candidate.keyword.replace("-", " ")
    if keyword not in {"FLAMMABLE GAS", "FLAMMABLE", "GAS"}:
        return False
    start, end = candidate.span
    previous = words[start - 1] if start > 0 else ""
    previous_two = words[start - 2 : start] if start >= 2 else []
    if previous in {"NON", "NOT"}:
        return True
    if previous == "NO":
        return True
    if previous_two == ["NON", ""] or previous_two == ["NOT", ""]:
        return True
    if start == 0 and words[:1] == ["NONFLAMMABLE"]:
        return True
    return end <= len(words) and any(word == "NONFLAMMABLE" for word in words[max(0, start - 1) : end])


def _apply_class2_conflict_group(
    scores: dict[str, float],
    selected: list[KeywordCandidate],
) -> dict[str, float]:
    class2_specific = {
        candidate.label
        for candidate in selected
        if candidate.label in {"2-non-flammable-gas", "2-oxygen"}
        and candidate.strength == "strong"
    }
    if not class2_specific:
        return scores
    adjusted = dict(scores)
    for label in {"2-flammable-gas", "2-non-flammable-gas", "2-oxygen"} - class2_specific:
        if label in adjusted:
            # Keep number-only support but remove generic keyword support from conflicting class-2 labels.
            adjusted[label] = min(adjusted[label], 0.05)
    return adjusted


def _phrase_multiplier(keyword: str) -> float:
    return 1.35 if len(keyword.replace("-", " ").split()) > 1 else 1.0


def _similarity(lhs: str, rhs: str) -> float:
    return SequenceMatcher(None, lhs, rhs).ratio()


def _truncate(value: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _bbox_iou(lhs: tuple[float, float, float, float], rhs: tuple[float, float, float, float]) -> float:
    lx1, ly1, lx2, ly2 = lhs
    rx1, ry1, rx2, ry2 = rhs
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    lhs_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    rhs_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    denom = lhs_area + rhs_area - inter
    return inter / denom if denom > 0 else 0.0


def _bbox_size_ratio(lhs: tuple[float, float, float, float], rhs: tuple[float, float, float, float]) -> float:
    lhs_w, lhs_h = max(0.0, lhs[2] - lhs[0]), max(0.0, lhs[3] - lhs[1])
    rhs_w, rhs_h = max(0.0, rhs[2] - rhs[0]), max(0.0, rhs[3] - rhs[1])
    if lhs_w <= 0 or lhs_h <= 0 or rhs_w <= 0 or rhs_h <= 0:
        return 0.0
    return min(lhs_w / rhs_w, rhs_w / lhs_w, lhs_h / rhs_h, rhs_h / lhs_h)


def _log_once(logger: logging.Logger, key: str, message: str) -> None:
    if key in _WARNED_BACKENDS:
        return
    _WARNED_BACKENDS.add(key)
    logger.warning(message)
