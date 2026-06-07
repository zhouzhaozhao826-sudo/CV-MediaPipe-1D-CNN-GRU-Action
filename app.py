from __future__ import annotations

import argparse
from pathlib import Path

from pose_action import PoseActionPipeline, ProjectConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["camera", "video"], default="camera",
                        help="camera=实时规则法; video=视频规则法（提取特征+切片）")
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--export-name", type=str, default=None)
    parser.add_argument("--preview-video", action="store_true")
    parser.add_argument("--log-interval", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ProjectConfig()
    pipeline = PoseActionPipeline(config)

    if args.source == "camera":
        pipeline.run_camera(export_name=args.export_name or "camera_session.csv")
    else:
        if args.video_path is None:
            raise ValueError("使用视频模式时必须提供 --video-path。")
        pipeline.run_video(
            video_path=args.video_path,
            export_name=args.export_name,
            preview=args.preview_video,
            log_interval=args.log_interval,
        )

    if pipeline.latest_segment_export_path is not None:
        print(f"动作切片结果已导出到: {pipeline.latest_segment_export_path}")
        print(f"检测到的动作段数量: {len(pipeline.latest_segments)}")


if __name__ == "__main__":
    main()
