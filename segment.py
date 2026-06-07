from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .features import FrameFeatures


@dataclass(slots=True)
class SquatSegment:
    segment_id: int
    start_index: int
    bottom_index: int
    end_index: int
    start_timestamp_ms: int
    bottom_timestamp_ms: int
    end_timestamp_ms: int
    duration_ms: int
    min_hip_height: float
    start_hip_height: float
    end_hip_height: float
    squat_amplitude: float
    start_end_height_gap: float
    start_avg_knee_angle: float
    bottom_avg_knee_angle: float
    end_avg_knee_angle: float

    def to_row(self) -> dict[str, float]:
        return {
            "segment_id": float(self.segment_id),
            "start_index": float(self.start_index),
            "bottom_index": float(self.bottom_index),
            "end_index": float(self.end_index),
            "start_timestamp_ms": float(self.start_timestamp_ms),
            "bottom_timestamp_ms": float(self.bottom_timestamp_ms),
            "end_timestamp_ms": float(self.end_timestamp_ms),
            "duration_ms": float(self.duration_ms),
            "min_hip_height": self.min_hip_height,
            "start_hip_height": self.start_hip_height,
            "end_hip_height": self.end_hip_height,
            "squat_amplitude": self.squat_amplitude,
            "start_end_height_gap": self.start_end_height_gap,
            "start_avg_knee_angle": self.start_avg_knee_angle,
            "bottom_avg_knee_angle": self.bottom_avg_knee_angle,
            "end_avg_knee_angle": self.end_avg_knee_angle,
        }


@dataclass(slots=True)
class SquatSegmentCandidate:
    candidate_id: int
    status: Literal["accepted", "rejected"]
    invalid_reason: str
    start_index: int
    bottom_index: int
    end_index: int
    start_timestamp_ms: int
    bottom_timestamp_ms: int
    end_timestamp_ms: int
    duration_ms: int
    squat_amplitude: float
    start_end_height_gap: float
    start_avg_knee_angle: float
    bottom_avg_knee_angle: float
    end_avg_knee_angle: float

    def to_row(self) -> dict[str, float | str]:
        return {
            "candidate_id": float(self.candidate_id),
            "status": self.status,
            "invalid_reason": self.invalid_reason,
            "start_index": float(self.start_index),
            "bottom_index": float(self.bottom_index),
            "end_index": float(self.end_index),
            "start_timestamp_ms": float(self.start_timestamp_ms),
            "bottom_timestamp_ms": float(self.bottom_timestamp_ms),
            "end_timestamp_ms": float(self.end_timestamp_ms),
            "duration_ms": float(self.duration_ms),
            "squat_amplitude": self.squat_amplitude,
            "start_end_height_gap": self.start_end_height_gap,
            "start_avg_knee_angle": self.start_avg_knee_angle,
            "bottom_avg_knee_angle": self.bottom_avg_knee_angle,
            "end_avg_knee_angle": self.end_avg_knee_angle,
        }


class SquatSegmenter:
    def __init__(
        self,
        min_distance_frames: int = 12,
        min_prominence: float = 0.015,
        smoothing_window: int = 11,
        smoothing_polyorder: int = 2,
        min_duration_ms: int = 700,
        max_duration_ms: int = 3500,
        min_squat_amplitude: float = 0.025,
        max_start_end_height_gap: float = 0.08,
        min_standing_knee_angle: float = 110.0,
        max_bottom_knee_angle: float = 170.0,
        min_knee_angle_drop: float = 8.0,
    ) -> None:
        self.min_distance_frames = min_distance_frames
        self.min_prominence = min_prominence
        self.smoothing_window = smoothing_window
        self.smoothing_polyorder = smoothing_polyorder
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.min_squat_amplitude = min_squat_amplitude
        self.max_start_end_height_gap = max_start_end_height_gap
        self.min_standing_knee_angle = min_standing_knee_angle
        self.max_bottom_knee_angle = max_bottom_knee_angle
        self.min_knee_angle_drop = min_knee_angle_drop
        self.latest_candidates: list[SquatSegmentCandidate] = []

    def detect_segments(self, features: list[FrameFeatures]) -> list[SquatSegment]:
        self.latest_candidates = []
        if len(features) < max(self.min_distance_frames * 2, 8):
            return []

        hip_series = np.array([frame.values["hip_height"] for frame in features], dtype=np.float32)
        smoothed = self._smooth_signal(hip_series)

        # MediaPipe image-space y grows downward, so a deeper squat has a larger hip_height.
        # We therefore detect squat bottoms from local maxima of hip_height and standing tops
        # from local minima.
        top_indices, _ = find_peaks(-smoothed, distance=self.min_distance_frames)
        bottom_indices, _ = find_peaks(
            smoothed,
            distance=self.min_distance_frames,
            prominence=self.min_prominence,
        )

        if len(bottom_indices) == 0:
            return []

        segments: list[SquatSegment] = []
        used_ranges: list[tuple[int, int]] = []
        accepted_segment_id = 1
        for candidate_id, bottom_index in enumerate(bottom_indices, start=1):
            start_index = self._find_previous_top(top_indices, bottom_index)
            end_index = self._find_next_top(top_indices, bottom_index, len(features) - 1)

            if start_index is None or end_index is None:
                self.latest_candidates.append(
                    SquatSegmentCandidate(
                        candidate_id=candidate_id,
                        status="rejected",
                        invalid_reason="missing_boundary_peak",
                        start_index=-1 if start_index is None else start_index,
                        bottom_index=int(bottom_index),
                        end_index=-1 if end_index is None else end_index,
                        start_timestamp_ms=-1,
                        bottom_timestamp_ms=features[int(bottom_index)].timestamp_ms,
                        end_timestamp_ms=-1,
                        duration_ms=-1,
                        squat_amplitude=0.0,
                        start_end_height_gap=0.0,
                        start_avg_knee_angle=0.0,
                        bottom_avg_knee_angle=self._avg_knee_angle(features[int(bottom_index)]),
                        end_avg_knee_angle=0.0,
                    )
                )
                continue
            if end_index - start_index < self.min_distance_frames:
                self.latest_candidates.append(
                    SquatSegmentCandidate(
                        candidate_id=candidate_id,
                        status="rejected",
                        invalid_reason="too_short_frame_span",
                        start_index=start_index,
                        bottom_index=int(bottom_index),
                        end_index=end_index,
                        start_timestamp_ms=features[start_index].timestamp_ms,
                        bottom_timestamp_ms=features[int(bottom_index)].timestamp_ms,
                        end_timestamp_ms=features[end_index].timestamp_ms,
                        duration_ms=features[end_index].timestamp_ms - features[start_index].timestamp_ms,
                        squat_amplitude=0.0,
                        start_end_height_gap=0.0,
                        start_avg_knee_angle=self._avg_knee_angle(features[start_index]),
                        bottom_avg_knee_angle=self._avg_knee_angle(features[int(bottom_index)]),
                        end_avg_knee_angle=self._avg_knee_angle(features[end_index]),
                    )
                )
                continue
            # Adjacent segments are allowed to share a boundary frame.
            if any(not (end_index <= used_start or start_index >= used_end) for used_start, used_end in used_ranges):
                self.latest_candidates.append(
                    SquatSegmentCandidate(
                        candidate_id=candidate_id,
                        status="rejected",
                        invalid_reason="overlap_with_accepted_segment",
                        start_index=start_index,
                        bottom_index=int(bottom_index),
                        end_index=end_index,
                        start_timestamp_ms=features[start_index].timestamp_ms,
                        bottom_timestamp_ms=features[int(bottom_index)].timestamp_ms,
                        end_timestamp_ms=features[end_index].timestamp_ms,
                        duration_ms=features[end_index].timestamp_ms - features[start_index].timestamp_ms,
                        squat_amplitude=0.0,
                        start_end_height_gap=0.0,
                        start_avg_knee_angle=self._avg_knee_angle(features[start_index]),
                        bottom_avg_knee_angle=self._avg_knee_angle(features[int(bottom_index)]),
                        end_avg_knee_angle=self._avg_knee_angle(features[end_index]),
                    )
                )
                continue

            start_frame = features[start_index]
            bottom_frame = features[bottom_index]
            end_frame = features[end_index]
            duration_ms = end_frame.timestamp_ms - start_frame.timestamp_ms
            start_avg_knee_angle = self._avg_knee_angle(start_frame)
            bottom_avg_knee_angle = self._avg_knee_angle(bottom_frame)
            end_avg_knee_angle = self._avg_knee_angle(end_frame)
            start_end_height_gap = abs(float(smoothed[start_index]) - float(smoothed[end_index]))
            squat_amplitude = float(smoothed[bottom_index] - min(smoothed[start_index], smoothed[end_index]))

            invalid_reason = self._invalid_reason(
                duration_ms=duration_ms,
                squat_amplitude=squat_amplitude,
                start_end_height_gap=start_end_height_gap,
                start_avg_knee_angle=start_avg_knee_angle,
                bottom_avg_knee_angle=bottom_avg_knee_angle,
                end_avg_knee_angle=end_avg_knee_angle,
            )
            if invalid_reason is not None:
                self.latest_candidates.append(
                    SquatSegmentCandidate(
                        candidate_id=candidate_id,
                        status="rejected",
                        invalid_reason=invalid_reason,
                        start_index=start_index,
                        bottom_index=int(bottom_index),
                        end_index=end_index,
                        start_timestamp_ms=start_frame.timestamp_ms,
                        bottom_timestamp_ms=bottom_frame.timestamp_ms,
                        end_timestamp_ms=end_frame.timestamp_ms,
                        duration_ms=duration_ms,
                        squat_amplitude=squat_amplitude,
                        start_end_height_gap=start_end_height_gap,
                        start_avg_knee_angle=start_avg_knee_angle,
                        bottom_avg_knee_angle=bottom_avg_knee_angle,
                        end_avg_knee_angle=end_avg_knee_angle,
                    )
                )
                continue

            segment = SquatSegment(
                segment_id=accepted_segment_id,
                start_index=start_index,
                bottom_index=bottom_index,
                end_index=end_index,
                start_timestamp_ms=start_frame.timestamp_ms,
                bottom_timestamp_ms=bottom_frame.timestamp_ms,
                end_timestamp_ms=end_frame.timestamp_ms,
                duration_ms=duration_ms,
                min_hip_height=float(smoothed[bottom_index]),
                start_hip_height=float(smoothed[start_index]),
                end_hip_height=float(smoothed[end_index]),
                squat_amplitude=squat_amplitude,
                start_end_height_gap=start_end_height_gap,
                start_avg_knee_angle=start_avg_knee_angle,
                bottom_avg_knee_angle=bottom_avg_knee_angle,
                end_avg_knee_angle=end_avg_knee_angle,
            )
            segments.append(segment)
            used_ranges.append((start_index, end_index))
            self.latest_candidates.append(
                SquatSegmentCandidate(
                    candidate_id=candidate_id,
                    status="accepted",
                    invalid_reason="",
                    start_index=start_index,
                    bottom_index=int(bottom_index),
                    end_index=end_index,
                    start_timestamp_ms=start_frame.timestamp_ms,
                    bottom_timestamp_ms=bottom_frame.timestamp_ms,
                    end_timestamp_ms=end_frame.timestamp_ms,
                    duration_ms=duration_ms,
                    squat_amplitude=squat_amplitude,
                    start_end_height_gap=start_end_height_gap,
                    start_avg_knee_angle=start_avg_knee_angle,
                    bottom_avg_knee_angle=bottom_avg_knee_angle,
                    end_avg_knee_angle=end_avg_knee_angle,
                )
            )
            accepted_segment_id += 1
        return segments

    def _smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        if len(signal) < 5:
            return signal
        window = min(self.smoothing_window, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
        if window < 5:
            return signal
        polyorder = min(self.smoothing_polyorder, window - 1)
        return savgol_filter(signal, window_length=window, polyorder=polyorder, mode="interp")

    @staticmethod
    def _find_previous_top(top_indices: np.ndarray, bottom_index: int) -> int | None:
        previous = top_indices[top_indices < bottom_index]
        if len(previous) == 0:
            return None
        return int(previous[-1])

    @staticmethod
    def _find_next_top(top_indices: np.ndarray, bottom_index: int, fallback_end: int) -> int | None:
        following = top_indices[top_indices > bottom_index]
        if len(following) == 0:
            return fallback_end
        return int(following[0])

    @staticmethod
    def _avg_knee_angle(frame: FrameFeatures) -> float:
        return (
            float(frame.values["left_knee_angle"]) + float(frame.values["right_knee_angle"])
        ) / 2.0

    def _invalid_reason(
        self,
        duration_ms: int,
        squat_amplitude: float,
        start_end_height_gap: float,
        start_avg_knee_angle: float,
        bottom_avg_knee_angle: float,
        end_avg_knee_angle: float,
    ) -> str | None:
        if duration_ms < self.min_duration_ms or duration_ms > self.max_duration_ms:
            return "duration_out_of_range"
        if squat_amplitude < self.min_squat_amplitude:
            return "amplitude_too_small"
        if start_end_height_gap > self.max_start_end_height_gap:
            return "start_end_gap_too_large"
        if start_avg_knee_angle < self.min_standing_knee_angle:
            return "start_not_standing_enough"
        if end_avg_knee_angle < self.min_standing_knee_angle:
            return "end_not_standing_enough"
        if bottom_avg_knee_angle > self.max_bottom_knee_angle:
            return "bottom_not_deep_enough"
        if (start_avg_knee_angle - bottom_avg_knee_angle) < self.min_knee_angle_drop:
            return "start_bottom_knee_drop_too_small"
        if (end_avg_knee_angle - bottom_avg_knee_angle) < self.min_knee_angle_drop:
            return "end_bottom_knee_drop_too_small"
        return None
