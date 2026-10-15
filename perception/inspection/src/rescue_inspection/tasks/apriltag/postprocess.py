"""Post-processing helpers for AprilTag outputs."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from rescue_inspection.config import StabilizerConfig
from rescue_inspection.models import AprilTagDetectionResult


@dataclass
class StableAprilTagVoteResult:
    """State of the current AprilTag sliding-window vote."""

    is_stable: bool
    stable_tag_id: str
    stability_score: float
    representative_result: AprilTagDetectionResult | None


class AprilTagDetectionStabilizer:
    """Vote across recent frames to suppress AprilTag jitter."""

    def __init__(self, config: StabilizerConfig) -> None:
        self.config = config
        self._history: deque[AprilTagDetectionResult] = deque(maxlen=config.window_size)

    def update(self, result: AprilTagDetectionResult) -> StableAprilTagVoteResult:
        """Push a new frame result and return the latest stability state."""

        self._history.append(result)
        tag_ids = [item.tag_id.strip() for item in self._history if item.tag_id.strip()]

        if not tag_ids:
            return StableAprilTagVoteResult(
                is_stable=False,
                stable_tag_id="",
                stability_score=0.0,
                representative_result=None,
            )

        counts = Counter(tag_ids)
        top_tag_id, top_count = counts.most_common(1)[0]
        score = top_count / float(self.config.window_size)
        representative = next(
            (
                item
                for item in reversed(self._history)
                if item.detected and item.tag_id.strip() == top_tag_id
            ),
            None,
        )
        is_stable = top_count >= self.config.min_votes and score >= self.config.stability_threshold

        return StableAprilTagVoteResult(
            is_stable=is_stable,
            stable_tag_id=top_tag_id if is_stable else "",
            stability_score=score,
            representative_result=representative,
        )
