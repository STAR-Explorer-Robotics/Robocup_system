"""Temporal stabilizer for Hazmat classifications."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from rescue_inspection.config import HazmatStabilizerConfig
from rescue_inspection.models import HazmatDetectionItem, HazmatDetectionResult


@dataclass
class StableHazmatVoteResult:
    """State of the current sliding-window vote."""

    is_stable: bool
    stable_label: str
    stability_score: float
    representative_items: list[HazmatDetectionItem]
    representative_result: HazmatDetectionResult | None = None


class HazmatDetectionStabilizer:
    """Vote across recent frames to suppress classification jitter."""

    def __init__(self, config: HazmatStabilizerConfig) -> None:
        self.config = config
        self._history: deque[HazmatDetectionResult] = deque(maxlen=config.window_size)

    def update(self, result: HazmatDetectionResult) -> StableHazmatVoteResult:
        """Push a new frame result and return the latest stability state."""

        self._history.append(result)

        valid_labels = []
        for r in self._history:
            for item in r.items:
                if item.label != "UNKNOWN":
                    valid_labels.append(item.label)

        if not valid_labels:
            return StableHazmatVoteResult(
                is_stable=False,
                stable_label="UNKNOWN",
                stability_score=0.0,
                representative_items=[],
                representative_result=None,
            )

        counts = Counter(valid_labels)
        top_label, top_count = counts.most_common(1)[0]
        score = top_count / float(self.config.window_size)

        is_stable = top_count >= self.config.min_votes and score >= self.config.stability_threshold

        rep_items = []
        representative_result = None
        if is_stable:
            for r in reversed(self._history):
                candidates = [i for i in r.items if i.label == top_label]
                if candidates:
                    rep_items = candidates
                    representative_result = r
                    break

        return StableHazmatVoteResult(
            is_stable=is_stable,
            stable_label=top_label if is_stable else "UNKNOWN",
            stability_score=score,
            representative_items=rep_items,
            representative_result=representative_result,
        )
