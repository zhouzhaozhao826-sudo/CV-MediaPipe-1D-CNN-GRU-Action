from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.vision import drawing_utils

from .config import ProjectConfig


@dataclass(slots=True)
class PoseFrame:
    rgb_frame: np.ndarray
    annotated_bgr_frame: np.ndarray
    timestamp_ms: int
    landmarks: list[Any]


class PoseExtractor:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        base_options = python.BaseOptions(model_asset_path=str(config.model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=config.max_num_poses,
            min_pose_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            min_pose_detection_confidence=config.min_pose_detection_confidence,
            output_segmentation_masks=False,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.landmark_style = drawing_styles.get_default_pose_landmarks_style()
        self.connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    def close(self) -> None:
        self.detector.close()

    def process_bgr_frame(self, frame: np.ndarray, timestamp_ms: int) -> PoseFrame | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        if not detection_result.pose_landmarks:
            return None

        landmarks = detection_result.pose_landmarks[0]
        annotated_rgb = np.copy(rgb_frame)
        drawing_utils.draw_landmarks(
            image=annotated_rgb,
            landmark_list=landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=self.landmark_style,
            connection_drawing_spec=self.connection_style,
        )
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        return PoseFrame(
            rgb_frame=rgb_frame,
            annotated_bgr_frame=annotated_bgr,
            timestamp_ms=timestamp_ms,
            landmarks=landmarks,
        )
