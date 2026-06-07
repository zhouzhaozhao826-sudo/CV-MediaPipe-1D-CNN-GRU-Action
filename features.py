from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


@dataclass(slots=True)
class FrameFeatures:
    timestamp_ms: int
    values: dict[str, float]

    def to_row(self) -> dict[str, float]:
        return {"timestamp_ms": float(self.timestamp_ms), **self.values}


def _landmark_xyz(landmarks: Iterable, index: int) -> np.ndarray:
    landmark = list(landmarks)[index]
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def resample_features(feature_matrix: np.ndarray, target_length: int) -> np.ndarray:
    """线性插值重采样特征矩阵到固定帧数。"""
    original_length, feature_count = feature_matrix.shape
    if original_length == target_length:
        return feature_matrix
    old_pos = np.linspace(0.0, 1.0, num=original_length, dtype=np.float32)
    new_pos = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    resampled = np.zeros((target_length, feature_count), dtype=np.float32)
    for c in range(feature_count):
        resampled[:, c] = np.interp(new_pos, old_pos, feature_matrix[:, c])
    return resampled


FEATURE_COLUMNS = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "torso_tilt_angle",
    "hip_height",
    "shoulder_height",
    "ankle_height",
    "shoulder_width",
    "hip_width",
    "torso_length",
    "knee_gap_ratio",
    "ankle_gap_ratio",
]


def build_frame_features(landmarks: list, timestamp_ms: int) -> FrameFeatures:
    left_shoulder = _landmark_xyz(landmarks, LEFT_SHOULDER)
    right_shoulder = _landmark_xyz(landmarks, RIGHT_SHOULDER)
    left_elbow = _landmark_xyz(landmarks, LEFT_ELBOW)
    right_elbow = _landmark_xyz(landmarks, RIGHT_ELBOW)
    left_wrist = _landmark_xyz(landmarks, LEFT_WRIST)
    right_wrist = _landmark_xyz(landmarks, RIGHT_WRIST)
    left_hip = _landmark_xyz(landmarks, LEFT_HIP)
    right_hip = _landmark_xyz(landmarks, RIGHT_HIP)
    left_knee = _landmark_xyz(landmarks, LEFT_KNEE)
    right_knee = _landmark_xyz(landmarks, RIGHT_KNEE)
    left_ankle = _landmark_xyz(landmarks, LEFT_ANKLE)
    right_ankle = _landmark_xyz(landmarks, RIGHT_ANKLE)

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0
    ankle_center = (left_ankle + right_ankle) / 2.0

    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
    hip_width = float(np.linalg.norm(left_hip - right_hip))
    torso_length = float(np.linalg.norm(shoulder_center - hip_center))

    left_knee_angle = _angle_degrees(left_hip, left_knee, left_ankle)
    right_knee_angle = _angle_degrees(right_hip, right_knee, right_ankle)
    left_hip_angle = _angle_degrees(left_shoulder, left_hip, left_knee)
    right_hip_angle = _angle_degrees(right_shoulder, right_hip, right_knee)
    left_elbow_angle = _angle_degrees(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = _angle_degrees(right_shoulder, right_elbow, right_wrist)
    torso_tilt_angle = _angle_degrees(shoulder_center, hip_center, ankle_center)

    hip_height = float(hip_center[1])
    shoulder_height = float(shoulder_center[1])
    ankle_height = float(ankle_center[1])
    knee_gap_ratio = _safe_ratio(abs(left_knee[0] - right_knee[0]), shoulder_width)
    ankle_gap_ratio = _safe_ratio(abs(left_ankle[0] - right_ankle[0]), shoulder_width)

    values = {
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "torso_tilt_angle": torso_tilt_angle,
        "hip_height": hip_height,
        "shoulder_height": shoulder_height,
        "ankle_height": ankle_height,
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "torso_length": torso_length,
        "knee_gap_ratio": knee_gap_ratio,
        "ankle_gap_ratio": ankle_gap_ratio,
    }
    return FrameFeatures(timestamp_ms=timestamp_ms, values=values)
