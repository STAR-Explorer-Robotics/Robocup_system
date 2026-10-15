"""Color evidence extraction and Hazmat color profile utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rescue_inspection.config import HazmatColorConfig, HazmatDebugConfig
from rescue_inspection.project_paths import relative_to_root, resolve_project_path
from rescue_inspection.tasks.hazmat.label_registry import LabelRegistry
from rescue_inspection.utils.io import ensure_dir


COLOR_RATIO_KEYS = [
    "red_ratio",
    "orange_ratio",
    "yellow_ratio",
    "green_ratio",
    "blue_ratio",
    "black_ratio",
    "white_ratio",
    "gray_ratio",
    "colored_ratio",
    "valid_pixel_ratio",
]

PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HazmatColorEvidence:
    """Color ratios and quality diagnostics for a candidate patch."""

    ratios: dict[str, float]
    hue_histogram: list[float]
    quality: dict[str, float]
    quality_flags: list[str]

    @property
    def histogram(self) -> list[float]:
        """Backward-compatible alias for the hue-only histogram."""

        return self.hue_histogram

    def lightweight_payload(self) -> dict[str, object]:
        """Return the small payload safe for default runtime debug JSON."""

        return {"quality_flags": list(self.quality_flags)}

    def full_payload(self, *, include_histogram: bool = False) -> dict[str, object]:
        """Return full color evidence for explicit debug/report commands."""

        payload: dict[str, object] = {
            "ratios": dict(self.ratios),
            "quality": dict(self.quality),
            "quality_flags": list(self.quality_flags),
        }
        if include_histogram:
            payload["hue_histogram"] = list(self.hue_histogram)
        return payload


@dataclass(frozen=True)
class HazmatColorProfile:
    """Per-label color profile."""

    label: str
    source: str
    ratios: dict[str, float]
    hue_histogram: list[float]
    sample_count: int
    quality: dict[str, float] = field(default_factory=dict)

    @property
    def histogram(self) -> list[float]:
        """Backward-compatible alias for the hue-only histogram."""

        return self.hue_histogram


@dataclass(frozen=True)
class HazmatColorMatch:
    """Color profile match details for one crop."""

    top1_label: str = ""
    top1_score: float = 0.0
    top2_label: str = ""
    top2_score: float = 0.0
    margin: float = 0.0
    scores_by_label: dict[str, float] = field(default_factory=dict)
    evidence: HazmatColorEvidence | None = None
    unavailable_reason: str | None = None


class HazmatColorProfileStore:
    """Loaded color profile collection used by runtime fusion."""

    def __init__(self, profiles: dict[str, HazmatColorProfile], *, payload: dict[str, Any] | None = None) -> None:
        self.profiles = profiles
        self.payload = payload or {}

    @classmethod
    def load(cls, path: Path | str) -> "HazmatColorProfileStore":
        """Load a color profile JSON file."""

        resolved = resolve_project_path(path)
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        profiles = {}
        for label, profile_payload in (payload.get("profiles") or {}).items():
            profiles[label] = HazmatColorProfile(
                label=str(profile_payload.get("label", label)),
                source=str(profile_payload.get("source", payload.get("source", "standard"))),
                ratios={str(key): float(value) for key, value in (profile_payload.get("ratios") or {}).items()},
                hue_histogram=[
                    float(value)
                    for value in (profile_payload.get("hue_histogram") or profile_payload.get("histogram") or [])
                ],
                sample_count=int(profile_payload.get("sample_count", 0)),
                quality={str(key): float(value) for key, value in (profile_payload.get("quality") or {}).items()},
            )
        return cls(profiles, payload=payload)

    def match(self, evidence: HazmatColorEvidence) -> HazmatColorMatch:
        """Score one color evidence object against all loaded profiles."""

        scores = {
            label: _color_profile_similarity(evidence, profile)
            for label, profile in self.profiles.items()
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return HazmatColorMatch(evidence=evidence, unavailable_reason="no_color_profiles")
        top1_label, top1_score = ranked[0]
        top2_label, top2_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
        return HazmatColorMatch(
            top1_label=top1_label,
            top1_score=float(top1_score),
            top2_label=top2_label,
            top2_score=float(top2_score),
            margin=float(top1_score - top2_score),
            scores_by_label={label: float(score) for label, score in scores.items()},
            evidence=evidence,
        )


def extract_color_evidence(image_bgr: np.ndarray, config: HazmatColorConfig) -> HazmatColorEvidence:
    """Extract conservative HSV/Lab color evidence from a BGR patch."""

    if image_bgr.size == 0:
        return HazmatColorEvidence(
            ratios={key: 0.0 for key in COLOR_RATIO_KEYS},
            hue_histogram=[0.0 for _ in range(config.hue_bins)],
            quality={
                "pixel_count": 0.0,
                "foreground_pixel_count": 0.0,
                "valid_pixel_count": 0.0,
                "mean_saturation": 0.0,
                "mean_value": 0.0,
            },
            quality_flags=["low_valid_pixel_ratio"],
        )

    bgr, foreground_mask = _split_bgr_and_foreground_mask(image_bgr, config)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    lightness = lab[..., 0]

    pixel_count = float(foreground_mask.size)
    foreground_count = float(np.count_nonzero(foreground_mask))
    valid_ratio = foreground_count / max(pixel_count, 1.0)
    sat_float = saturation.astype(np.float32) / 255.0
    val_float = value.astype(np.float32) / 255.0
    low_saturation_threshold = _saturation_threshold_uint8(config)

    black_mask = foreground_mask & (value < config.black_value_threshold)
    white_mask = (
        foreground_mask
        & ~black_mask
        & (saturation < low_saturation_threshold)
        & (value > config.white_value_threshold)
    )
    gray_mask = foreground_mask & ~black_mask & ~white_mask & (saturation < low_saturation_threshold)
    colored_mask = (
        foreground_mask
        & ~black_mask
        & ~white_mask
        & ~gray_mask
        & (saturation >= low_saturation_threshold)
        & (value >= config.min_value)
    )

    red_mask = colored_mask & ((hue <= 10) | (hue >= 165))
    orange_mask = colored_mask & (hue > 10) & (hue <= 20)
    yellow_mask = colored_mask & (hue > 20) & (hue <= 38)
    green_mask = colored_mask & (hue > 38) & (hue <= 85)
    blue_mask = colored_mask & (hue > 85) & (hue <= 135)
    shadow_mask = foreground_mask & ~black_mask & (value < config.shadow_value_threshold) & (saturation < 120)

    denom = max(foreground_count, 1.0)
    ratios = {
        "red_ratio": _mask_ratio(red_mask, denom),
        "orange_ratio": _mask_ratio(orange_mask, denom),
        "yellow_ratio": _mask_ratio(yellow_mask, denom),
        "green_ratio": _mask_ratio(green_mask, denom),
        "blue_ratio": _mask_ratio(blue_mask, denom),
        "black_ratio": _mask_ratio(black_mask, denom),
        "white_ratio": _mask_ratio(white_mask, denom),
        "gray_ratio": _mask_ratio(gray_mask, denom),
        "colored_ratio": _mask_ratio(colored_mask, denom),
        "valid_pixel_ratio": valid_ratio,
    }

    hue_histogram = _normalized_hue_histogram(hue, colored_mask, bins=config.hue_bins)
    mean_saturation = float(np.mean(sat_float[foreground_mask])) if foreground_count else 0.0
    mean_value = float(np.mean(val_float[foreground_mask])) if foreground_count else 0.0
    shadow_ratio = _mask_ratio(shadow_mask, denom)
    mean_lightness = float(np.mean(lightness[foreground_mask])) / 255.0 if foreground_count else 0.0
    quality = {
        "pixel_count": pixel_count,
        "foreground_pixel_count": foreground_count,
        "valid_pixel_count": foreground_count,
        "mean_saturation": mean_saturation,
        "mean_value": mean_value,
        "mean_lightness": mean_lightness,
        "shadow_ratio": shadow_ratio,
    }
    quality_flags = _quality_flags(ratios, quality, config)
    return HazmatColorEvidence(
        ratios=ratios,
        hue_histogram=hue_histogram,
        quality=quality,
        quality_flags=quality_flags,
    )


def build_standard_color_profile_payload(
    *,
    config: HazmatColorConfig,
    registry: LabelRegistry,
    manifest_path: Path,
) -> dict[str, object]:
    """Generate the standard color profile JSON payload from the manifest templates."""

    source_root = registry.template_root
    profiles: dict[str, object] = {}
    source_files = []
    labels = []

    for entry in registry.active_labels(registry.manifest.default_label_set):
        image_path = source_root / entry.filename
        labels.append(entry.key)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        evidence = extract_color_evidence(image, config)
        profiles[entry.key] = asdict(
            HazmatColorProfile(
                label=entry.key,
                source="standard",
                ratios=evidence.ratios,
                hue_histogram=evidence.hue_histogram,
                sample_count=1,
                quality=evidence.quality,
            )
        )
        source_files.append(
            {
                "label": entry.key,
                "path": relative_to_root(image_path),
                "sha256": _sha256_file(image_path),
            }
        )

    extraction_config = color_extraction_config_payload(config)
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "standard",
        "source_root": relative_to_root(source_root),
        "manifest_path": relative_to_root(manifest_path),
        "labels": labels,
        "profiles": profiles,
        "source_files": source_files,
        "color_extraction_config": extraction_config,
        "color_extraction_config_hash": _stable_hash(extraction_config),
    }


def write_standard_color_profile(
    *,
    output_path: Path,
    config: HazmatColorConfig,
    registry: LabelRegistry,
    manifest_path: Path,
) -> dict[str, object]:
    """Generate and write a standard color profile JSON file."""

    payload = build_standard_color_profile_payload(config=config, registry=registry, manifest_path=manifest_path)
    resolved_output = resolve_project_path(output_path)
    ensure_dir(resolved_output.parent)
    resolved_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def color_extraction_config_payload(config: HazmatColorConfig) -> dict[str, object]:
    """Return the profile-affecting color extraction config."""

    return {
        "hue_bins": config.hue_bins,
        "alpha_threshold": config.alpha_threshold,
        "min_value": config.min_value,
        "black_value_threshold": config.black_value_threshold,
        "white_value_threshold": config.white_value_threshold,
        "low_valid_pixel_ratio_threshold": config.low_valid_pixel_ratio_threshold,
        "high_white_ratio_threshold": config.high_white_ratio_threshold,
        "low_saturation_threshold": config.low_saturation_threshold,
        "shadow_value_threshold": config.shadow_value_threshold,
    }


def serialize_color_match_for_debug(
    match: HazmatColorMatch,
    *,
    debug_config: HazmatDebugConfig,
) -> dict[str, object]:
    """Serialize color match details according to debug verbosity settings."""

    payload: dict[str, object] = {
        "color_top1_label": match.top1_label,
        "color_top1_score": match.top1_score,
        "color_top2_label": match.top2_label,
        "color_top2_score": match.top2_score,
        "color_margin": match.margin,
        "quality_flags": list(match.evidence.quality_flags if match.evidence else []),
    }
    if debug_config.include_full_color_scores:
        payload["color_scores_by_label"] = dict(match.scores_by_label)
        if match.evidence is not None:
            payload["color_evidence"] = match.evidence.full_payload(
                include_histogram=debug_config.include_color_histograms
            )
    return payload


def _color_profile_similarity(evidence: HazmatColorEvidence, profile: HazmatColorProfile) -> float:
    hue_similarity = _histogram_intersection(evidence.hue_histogram, profile.hue_histogram)
    chromatic_similarity = _ratio_cosine_similarity(
        evidence.ratios,
        profile.ratios,
        keys=["red_ratio", "orange_ratio", "yellow_ratio", "green_ratio", "blue_ratio"],
    )
    neutral_similarity = _ratio_cosine_similarity(
        evidence.ratios,
        profile.ratios,
        keys=["black_ratio", "white_ratio", "gray_ratio"],
    )
    colored_similarity = 1.0 - abs(evidence.ratios.get("colored_ratio", 0.0) - profile.ratios.get("colored_ratio", 0.0))
    return float(
        np.clip(
            0.35 * hue_similarity
            + 0.35 * chromatic_similarity
            + 0.20 * neutral_similarity
            + 0.10 * colored_similarity,
            0.0,
            1.0,
        )
    )


def _ratio_cosine_similarity(lhs: dict[str, float], rhs: dict[str, float], *, keys: list[str]) -> float:
    lhs_vector = np.asarray([lhs.get(key, 0.0) for key in keys], dtype=np.float32)
    rhs_vector = np.asarray([rhs.get(key, 0.0) for key in keys], dtype=np.float32)
    denom = float(np.linalg.norm(lhs_vector) * np.linalg.norm(rhs_vector))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(lhs_vector, rhs_vector) / denom)


def _histogram_intersection(lhs: list[float], rhs: list[float]) -> float:
    if not lhs or not rhs:
        return 0.0
    length = min(len(lhs), len(rhs))
    return float(np.sum(np.minimum(np.asarray(lhs[:length]), np.asarray(rhs[:length]))))


def _normalized_hue_histogram(hue: np.ndarray, mask: np.ndarray, *, bins: int) -> list[float]:
    values = hue[mask]
    if values.size == 0:
        return [0.0 for _ in range(bins)]
    hist, _ = np.histogram(values, bins=bins, range=(0, 180))
    hist = hist.astype(np.float32)
    hist = hist / max(float(hist.sum()), 1.0)
    return [float(value) for value in hist]


def _split_bgr_and_foreground_mask(
    image: np.ndarray,
    config: HazmatColorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a BGR image and the foreground mask used for all color ratios."""

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return bgr, np.ones(image.shape[:2], dtype=bool)
    if image.shape[2] == 4:
        return image[..., :3].copy(), image[..., 3] > config.alpha_threshold
    return image[..., :3].copy(), np.ones(image.shape[:2], dtype=bool)


def _saturation_threshold_uint8(config: HazmatColorConfig) -> int:
    return int(round(config.low_saturation_threshold * 255.0))


def _quality_flags(
    ratios: dict[str, float],
    quality: dict[str, float],
    config: HazmatColorConfig,
) -> list[str]:
    flags = []
    if ratios["valid_pixel_ratio"] < config.low_valid_pixel_ratio_threshold:
        flags.append("low_valid_pixel_ratio")
    if ratios["white_ratio"] >= config.high_white_ratio_threshold:
        flags.append("high_white_ratio")
    if quality["mean_saturation"] < config.low_saturation_threshold:
        flags.append("low_saturation")
    if ratios["white_ratio"] >= config.high_white_ratio_threshold and ratios["colored_ratio"] < 0.08:
        flags.append("likely_background_or_pipe")
    if quality["shadow_ratio"] > 0.35:
        flags.append("shadow_heavy")
    return flags


def _mask_ratio(mask: np.ndarray, denom: float) -> float:
    return float(np.count_nonzero(mask) / denom)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
