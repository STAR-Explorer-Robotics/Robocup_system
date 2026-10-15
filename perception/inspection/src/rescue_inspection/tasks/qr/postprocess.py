"""Post-processing helpers for QR outputs."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from rescue_inspection.config import StabilizerConfig
from rescue_inspection.models import QRDetectionResult


@dataclass
class StableVoteResult:
    """State of the current sliding-window vote."""

    is_stable: bool
    stable_text: str
    stability_score: float
    representative_result: QRDetectionResult | None


class QRDetectionStabilizer:
    """Vote across recent frames to suppress decode jitter."""

    def __init__(self, config: StabilizerConfig) -> None:
        self.config = config
        self._history: deque[QRDetectionResult] = deque(maxlen=config.window_size)

    def update(self, result: QRDetectionResult) -> StableVoteResult:
        """Push a new frame result and return the latest stability state."""

        self._history.append(result)
        texts = [item.decoded_text.strip() for item in self._history if item.decoded_text.strip()]

        if not texts:
            return StableVoteResult(
                is_stable=False,
                stable_text="",
                stability_score=0.0,
                representative_result=None,
            )

        counts = Counter(texts)
        top_text, top_count = counts.most_common(1)[0]
        score = top_count / float(self.config.window_size)

        representative = next(
            (
                item
                for item in reversed(self._history)
                if item.detected and item.decoded_text.strip() == top_text
            ),
            None,
        )
        is_stable = top_count >= self.config.min_votes and score >= self.config.stability_threshold

        return StableVoteResult(
            is_stable=is_stable,
            stable_text=top_text if is_stable else "",
            stability_score=score,
            representative_result=representative,
        )

