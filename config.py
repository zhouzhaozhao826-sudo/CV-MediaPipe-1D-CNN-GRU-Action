from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ProjectConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    model_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "text" / "pose_landmarker.task")
    export_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "exports")
    camera_index: int = 0
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_pose_detection_confidence: float = 0.5
    max_num_poses: int = 1
    smoothing_alpha: float = 0.25
    display_width: int = 960
    display_height: int = 540

    def ensure_directories(self) -> None:
        self.export_dir.mkdir(parents=True, exist_ok=True)
