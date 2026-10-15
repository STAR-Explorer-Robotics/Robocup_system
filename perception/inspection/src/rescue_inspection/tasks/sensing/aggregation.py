"""Run-level aggregation state for combined sensing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rescue_inspection.models import (
    AprilTagDetectionItem,
    AprilTagDetectionResult,
    FramePacket,
    HazmatDetectionItem,
    HazmatDetectionResult,
    QRDetectionResult,
)
from rescue_inspection.tasks.apriltag.postprocess import StableAprilTagVoteResult
from rescue_inspection.tasks.hazmat.temporal import StableHazmatVoteResult
from rescue_inspection.tasks.qr.postprocess import StableVoteResult


@dataclass
class SensingRunCounters:
    """Counters accumulated across one sensing run."""

    total_frames: int = 0
    frames_with_detection: int = 0
    frames_with_any_detection: int = 0
    qr_frames_with_detection: int = 0
    hazmat_frames_with_detection: int = 0
    frames_with_both_qr_and_hazmat: int = 0

    def record_frame_seen(self) -> None:
        """Record that a frame was read from the source."""

        self.total_frames += 1

    def record_detection_result(
        self,
        *,
        qr_result: QRDetectionResult,
        hazmat_result: HazmatDetectionResult,
    ) -> None:
        """Record per-frame QR/Hazmat detection counters."""

        qr_detected = qr_result.detected
        hazmat_detected = hazmat_result.detected
        if qr_detected:
            self.qr_frames_with_detection += 1
        if hazmat_detected:
            self.hazmat_frames_with_detection += 1
        if qr_detected and hazmat_detected:
            self.frames_with_both_qr_and_hazmat += 1
        if qr_detected or hazmat_detected:
            self.frames_with_any_detection += 1
            self.frames_with_detection += 1


@dataclass
class SensingSubmissionCandidate:
    """Current best frame and per-task results for submission output."""

    score: float = -1.0
    frame_id: int | None = None
    frame_image: np.ndarray | None = None
    qr_result: QRDetectionResult | None = None
    apriltag_result: AprilTagDetectionResult | None = None
    hazmat_result: HazmatDetectionResult | None = None

    def consider(
        self,
        *,
        packet: FramePacket,
        qr_result: QRDetectionResult,
        qr_vote: StableVoteResult,
        apriltag_result: AprilTagDetectionResult | None = None,
        apriltag_vote: StableAprilTagVoteResult | None = None,
        hazmat_result: HazmatDetectionResult,
        hazmat_vote: StableHazmatVoteResult,
    ) -> None:
        """Update the submission candidate when this frame ties or beats the best score."""

        code_detected = apriltag_result.detected if apriltag_result is not None else qr_result.detected
        code_score = apriltag_vote.stability_score if apriltag_vote is not None else qr_vote.stability_score
        frame_score = float(int(code_detected) + int(hazmat_result.detected))
        frame_score += code_score + hazmat_vote.stability_score
        if frame_score >= self.score:
            self.score = frame_score
            self.frame_id = packet.frame_id
            self.frame_image = packet.image.copy()
            self.qr_result = qr_result
            self.apriltag_result = apriltag_result
            self.hazmat_result = hazmat_result


@dataclass
class SensingRunAggregationResult:
    """Final run-level state consumed by SensingPipeline finalization."""

    counters: SensingRunCounters
    final_qr_vote: StableVoteResult
    final_apriltag_vote: StableAprilTagVoteResult | None
    final_hazmat_vote: StableHazmatVoteResult
    selected_frame_id: int | None
    submission_qr_result: QRDetectionResult | None
    submission_apriltag_result: AprilTagDetectionResult | None
    submission_hazmat_result: HazmatDetectionResult | None
    submission_candidate: SensingSubmissionCandidate


@dataclass
class SensingRunAggregator:
    """Accumulate cross-frame sensing state without owning IO or overlay output."""

    counters: SensingRunCounters = field(default_factory=SensingRunCounters)
    submission_candidate: SensingSubmissionCandidate = field(default_factory=SensingSubmissionCandidate)
    qr_best_candidate_vote: StableVoteResult = field(
        default_factory=lambda: StableVoteResult(False, "", 0.0, None)
    )
    qr_best_stable_vote: StableVoteResult | None = None
    apriltag_best_candidate_vote: StableAprilTagVoteResult | None = None
    apriltag_best_stable_vote: StableAprilTagVoteResult | None = None
    hazmat_best_candidate_vote: StableHazmatVoteResult = field(
        default_factory=lambda: StableHazmatVoteResult(False, "UNKNOWN", 0.0, [], None)
    )
    hazmat_best_stable_vote: StableHazmatVoteResult | None = None

    def record_frame_seen(self) -> None:
        """Record that the source yielded a frame."""

        self.counters.record_frame_seen()

    def record_frame_result(
        self,
        *,
        packet: FramePacket,
        qr_result: QRDetectionResult,
        qr_vote: StableVoteResult,
        apriltag_result: AprilTagDetectionResult | None = None,
        apriltag_vote: StableAprilTagVoteResult | None = None,
        hazmat_result: HazmatDetectionResult,
        hazmat_vote: StableHazmatVoteResult,
    ) -> None:
        """Record one processed frame's votes, counters, and submission candidate."""

        if qr_vote.stability_score >= self.qr_best_candidate_vote.stability_score:
            self.qr_best_candidate_vote = qr_vote

        if qr_vote.is_stable and (
            self.qr_best_stable_vote is None
            or qr_vote.stability_score >= self.qr_best_stable_vote.stability_score
        ):
            self.qr_best_stable_vote = qr_vote

        if apriltag_vote is not None:
            if (
                self.apriltag_best_candidate_vote is None
                or apriltag_vote.stability_score >= self.apriltag_best_candidate_vote.stability_score
            ):
                self.apriltag_best_candidate_vote = apriltag_vote
            if apriltag_vote.is_stable and (
                self.apriltag_best_stable_vote is None
                or apriltag_vote.stability_score >= self.apriltag_best_stable_vote.stability_score
            ):
                self.apriltag_best_stable_vote = apriltag_vote

        if hazmat_vote.stability_score >= self.hazmat_best_candidate_vote.stability_score:
            self.hazmat_best_candidate_vote = hazmat_vote

        if hazmat_vote.is_stable and (
            self.hazmat_best_stable_vote is None
            or hazmat_vote.stability_score >= self.hazmat_best_stable_vote.stability_score
        ):
            self.hazmat_best_stable_vote = hazmat_vote

        self.counters.record_detection_result(qr_result=qr_result, hazmat_result=hazmat_result)
        self.submission_candidate.consider(
            packet=packet,
            qr_result=qr_result,
            qr_vote=qr_vote,
            apriltag_result=apriltag_result,
            apriltag_vote=apriltag_vote,
            hazmat_result=hazmat_result,
            hazmat_vote=hazmat_vote,
        )

    def finalize(self) -> SensingRunAggregationResult:
        """Return final votes and aligned submission results using the legacy selection rules."""

        final_qr_vote = self.qr_best_stable_vote or StableVoteResult(
            False,
            self.qr_best_candidate_vote.stable_text,
            self.qr_best_candidate_vote.stability_score,
            self.qr_best_candidate_vote.representative_result,
        )
        final_hazmat_vote = self.hazmat_best_stable_vote or StableHazmatVoteResult(
            False,
            self.hazmat_best_candidate_vote.stable_label,
            self.hazmat_best_candidate_vote.stability_score,
            self.hazmat_best_candidate_vote.representative_items,
            self.hazmat_best_candidate_vote.representative_result,
        )
        final_apriltag_vote = self.apriltag_best_stable_vote or self.apriltag_best_candidate_vote
        selected_frame_id = (
            self.submission_candidate.frame_id
            or (
                final_qr_vote.representative_result.frame_id
                if final_qr_vote.representative_result is not None
                else None
            )
            or (
                final_hazmat_vote.representative_result.frame_id
                if final_hazmat_vote.representative_result is not None
                else None
            )
        )
        submission_qr_result = _align_qr_submission_result(
            self.submission_candidate.qr_result,
            final_qr_vote,
            selected_frame_id,
        )
        submission_apriltag_result = _align_apriltag_submission_result(
            self.submission_candidate.apriltag_result,
            final_apriltag_vote,
            selected_frame_id,
        )
        submission_hazmat_result = _align_hazmat_submission_result(
            self.submission_candidate.hazmat_result,
            final_hazmat_vote,
            selected_frame_id,
        )
        return SensingRunAggregationResult(
            counters=self.counters,
            final_qr_vote=final_qr_vote,
            final_apriltag_vote=final_apriltag_vote,
            final_hazmat_vote=final_hazmat_vote,
            selected_frame_id=selected_frame_id,
            submission_qr_result=submission_qr_result,
            submission_apriltag_result=submission_apriltag_result,
            submission_hazmat_result=submission_hazmat_result,
            submission_candidate=self.submission_candidate,
        )


def _align_qr_submission_result(
    result: QRDetectionResult | None,
    vote: StableVoteResult,
    selected_frame_id: int | None,
) -> QRDetectionResult | None:
    if result is None:
        return None
    final_text = vote.stable_text or result.decoded_text
    updated_items = [
        item.model_copy(update={"decoded_text": final_text or item.decoded_text})
        for item in result.items
    ]
    return result.model_copy(
        update={
            "frame_id": selected_frame_id if selected_frame_id is not None else result.frame_id,
            "decoded_text": final_text,
            "items": updated_items,
            "stability_score": vote.stability_score,
        }
    )


def _align_hazmat_submission_result(
    result: HazmatDetectionResult | None,
    vote: StableHazmatVoteResult,
    selected_frame_id: int | None,
) -> HazmatDetectionResult | None:
    if result is None:
        return None
    final_label = vote.stable_label if vote.is_stable and vote.stable_label else result.best_label
    updated_items: list[HazmatDetectionItem] = []
    for item in result.items:
        if item.label == "UNKNOWN":
            updated_items.append(item)
            continue
        if final_label and final_label != "UNKNOWN":
            updated_items.append(
                item.model_copy(
                    update={
                        "label": final_label,
                        "raw_label": final_label,
                    }
                )
            )
        else:
            updated_items.append(item)

    return result.model_copy(
        update={
            "frame_id": selected_frame_id if selected_frame_id is not None else result.frame_id,
            "items": updated_items,
            "best_label": final_label or result.best_label,
            "stability_score": vote.stability_score,
        }
    )


def _align_apriltag_submission_result(
    result: AprilTagDetectionResult | None,
    vote: StableAprilTagVoteResult | None,
    selected_frame_id: int | None,
) -> AprilTagDetectionResult | None:
    if result is None:
        return None
    stable_tag_id = vote.stable_tag_id if vote is not None and vote.stable_tag_id else result.tag_id
    updated_items: list[AprilTagDetectionItem] = []
    for item in result.items:
        if item.tag_id == result.tag_id and stable_tag_id:
            updated_items.append(item.model_copy(update={"tag_id": stable_tag_id}))
        else:
            updated_items.append(item)
    return result.model_copy(
        update={
            "frame_id": selected_frame_id if selected_frame_id is not None else result.frame_id,
            "tag_id": stable_tag_id,
            "items": updated_items,
            "stability_score": vote.stability_score if vote is not None else result.stability_score,
        }
    )
