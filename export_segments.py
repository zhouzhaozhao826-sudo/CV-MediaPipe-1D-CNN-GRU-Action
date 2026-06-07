from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--segments-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-video", action="store_true")
    parser.add_argument("--export-frames", action="store_true")
    parser.add_argument("--padding-frames", type=int, default=5)
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--end-id", type=int, default=None)
    return parser


def ensure_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    return capture


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def write_segment_video(
    capture: cv2.VideoCapture,
    output_path: Path,
    start_index: int,
    end_index: int,
    fps: float,
    width: int,
    height: int,
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    for _ in range(start_index, end_index + 1):
        success, frame = capture.read()
        if not success:
            break
        writer.write(frame)
    writer.release()


def save_key_frame(capture: cv2.VideoCapture, frame_index: int, output_path: Path) -> None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = capture.read()
    if not success:
        return
    cv2.imwrite(str(output_path), frame)


def main() -> None:
    args = build_parser().parse_args()
    if not args.export_video and not args.export_frames:
        raise ValueError("至少需要开启 --export-video 或 --export-frames 之一。")

    segments_df = pd.read_csv(args.segments_path)
    if segments_df.empty:
        raise ValueError("切片文件为空，无法导出 segment 可视化内容。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture = ensure_capture(args.video_path)

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        for _, segment in segments_df.iterrows():
            segment_id = int(segment["segment_id"])
            if args.start_id is not None and segment_id < args.start_id:
                continue
            if args.end_id is not None and segment_id > args.end_id:
                continue

            start_index = clamp(int(segment["start_index"]) - args.padding_frames, 0, max(total_frames - 1, 0))
            bottom_index = clamp(int(segment["bottom_index"]), 0, max(total_frames - 1, 0))
            end_index = clamp(int(segment["end_index"]) + args.padding_frames, 0, max(total_frames - 1, 0))

            segment_dir = args.output_dir / f"segment_{segment_id:03d}"
            segment_dir.mkdir(parents=True, exist_ok=True)

            if args.export_video:
                video_output = segment_dir / f"segment_{segment_id:03d}.mp4"
                write_segment_video(
                    capture=capture,
                    output_path=video_output,
                    start_index=start_index,
                    end_index=end_index,
                    fps=fps,
                    width=width,
                    height=height,
                )

            if args.export_frames:
                save_key_frame(capture, start_index, segment_dir / f"segment_{segment_id:03d}_start.jpg")
                save_key_frame(capture, bottom_index, segment_dir / f"segment_{segment_id:03d}_bottom.jpg")
                save_key_frame(capture, end_index, segment_dir / f"segment_{segment_id:03d}_end.jpg")

            print(
                f"已导出 segment_id={segment_id} | "
                f"start={start_index}, bottom={bottom_index}, end={end_index}, "
                f"duration_ms={int(segment.get('duration_ms', 0))}, "
                f"amplitude={float(segment.get('squat_amplitude', 0.0)):.3f}, "
                f"bottom_avg_knee={float(segment.get('bottom_avg_knee_angle', 0.0)):.1f}"
            )
    finally:
        capture.release()


if __name__ == "__main__":
    main()
