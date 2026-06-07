from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import ProjectConfig
from .dataset import LABEL_SCHEMA, RULE_LABEL_TO_ID
from .features import FEATURE_COLUMNS, FrameFeatures, build_frame_features, resample_features
from .landmarks import PoseExtractor
from .model import SquatCNNGRU
from .repetition import SquatRepCounter
from .rules import RuleAssessment, SquatRuleEngine
from .segment import SquatSegment, SquatSegmentCandidate, SquatSegmenter


class PoseActionPipeline:
    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config or ProjectConfig()
        self.config.ensure_directories()
        self.extractor = PoseExtractor(self.config)
        self.counter = SquatRepCounter(alpha=self.config.smoothing_alpha)
        self.rule_engine = SquatRuleEngine()
        self.segmenter = SquatSegmenter()
        self.feature_rows: list[FrameFeatures] = []
        self.latest_segments: list[SquatSegment] = []
        self.latest_segment_candidates: list[SquatSegmentCandidate] = []
        self.latest_segment_export_path: Path | None = None
        self.latest_segment_candidate_export_path: Path | None = None

    def close(self) -> None:
        self.extractor.close()

    def run_camera(self, export_name: str = "camera_session.csv") -> Path | None:
        self.feature_rows = []
        self.counter = SquatRepCounter(alpha=self.config.smoothing_alpha)
        capture = cv2.VideoCapture(self.config.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.display_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.display_height)
        if not capture.isOpened():
            raise RuntimeError("无法打开摄像头。")

        session_start = time.time()
        try:
            while capture.isOpened():
                success, frame = capture.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                timestamp_ms = int((time.time() - session_start) * 1000)
                pose_frame = self.extractor.process_bgr_frame(frame, timestamp_ms)
                if pose_frame is None:
                    cv2.imshow("try_action - pose capture", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                features = build_frame_features(pose_frame.landmarks, timestamp_ms)
                self.feature_rows.append(features)
                rep_count = self.counter.update(features.values["hip_height"])
                assessment = self.rule_engine.assess(features, self.counter.state)
                output_frame = self._overlay_metrics(
                    pose_frame.annotated_bgr_frame,
                    pose_frame.landmarks,
                    features,
                    rep_count,
                    assessment,
                )
                cv2.imshow("try_action - pose capture", output_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
            self.close()

        if not self.feature_rows:
            return None
        export_path = self.config.export_dir / export_name
        feature_path = export_feature_rows(self.feature_rows, export_path)
        self._finalize_segments(export_path.stem)
        return feature_path

    def run_video(
        self,
        video_path: Path,
        export_name: str | None = None,
        preview: bool = False,
        log_interval: int = 30,
    ) -> Path | None:
        self.feature_rows = []
        self.counter = SquatRepCounter(alpha=self.config.smoothing_alpha)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        preview_window_name = "try_action - video processing"
        print(f"开始处理视频: {video_path}")
        if total_frames > 0:
            print(f"总帧数: {total_frames}")

        try:
            while capture.isOpened():
                success, frame = capture.read()
                if not success:
                    break
                processed_frames += 1
                timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                pose_frame = self.extractor.process_bgr_frame(frame, timestamp_ms)
                if processed_frames == 1 or processed_frames % max(log_interval, 1) == 0:
                    self._print_video_progress(processed_frames, total_frames)
                if pose_frame is None:
                    if preview:
                        cv2.imshow(preview_window_name, frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue
                features = build_frame_features(pose_frame.landmarks, timestamp_ms)
                self.feature_rows.append(features)
                rep_count = self.counter.update(features.values["hip_height"])
                assessment = self.rule_engine.assess(features, self.counter.state)
                if preview:
                    output_frame = self._overlay_metrics(
                        pose_frame.annotated_bgr_frame,
                        pose_frame.landmarks,
                        features,
                        rep_count,
                        assessment,
                    )
                    cv2.imshow(preview_window_name, output_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            capture.release()
            if preview:
                cv2.destroyWindow(preview_window_name)
            self.close()

        if not self.feature_rows:
            return None

        file_name = export_name or f"{video_path.stem}_features.csv"
        export_path = self.config.export_dir / file_name
        feature_path = export_feature_rows(self.feature_rows, export_path)
        self._finalize_segments(export_path.stem)
        self._print_video_progress(processed_frames, total_frames, done=True)
        return feature_path

    def _overlay_metrics(
        self,
        frame,
        landmarks,
        features: FrameFeatures,
        rep_count: int,
        assessment: RuleAssessment,
    ):
        output = frame.copy()
        h, w = output.shape[:2]

        # ── 左上角精简信息 ──
        feedback_color = self._feedback_color(assessment.label)
        cv2.putText(output, f"Squat Reps: {rep_count}  |  Phase: {assessment.phase}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output, f"Label: {assessment.label}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2)
        cv2.putText(output, assessment.suggestion,
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, feedback_color, 2)

        # ── 骨架上标注角度 ──
        values = features.values
        landmark_list = list(landmarks)
        # 左膝角 — 标注在左膝(25)旁
        self._draw_angle_label(output, landmark_list, 25,
                               f"{values['left_knee_angle']:.0f}", w, h)
        # 右膝角 — 标注在右膝(26)旁
        self._draw_angle_label(output, landmark_list, 26,
                               f"{values['right_knee_angle']:.0f}", w, h)
        # 躯干倾角 — 标注在髋中点旁
        hip_cx = int((landmark_list[23].x + landmark_list[24].x) / 2 * w)
        hip_cy = int((landmark_list[23].y + landmark_list[24].y) / 2 * h)
        cv2.putText(output, f"torso:{values['torso_tilt_angle']:.0f}",
                    (hip_cx + 15, hip_cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        return output

    @staticmethod
    def _draw_angle_label(frame, landmarks, idx: int, text: str, w: int, h: int):
        lm = list(landmarks)[idx]
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.putText(frame, text, (px + 12, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def _finalize_segments(self, export_stem: str) -> None:
        self.latest_segments = self.segmenter.detect_segments(self.feature_rows)
        self.latest_segment_candidates = list(self.segmenter.latest_candidates)
        segment_path = self.config.export_dir / f"{export_stem}_segments.csv"
        candidate_path = self.config.export_dir / f"{export_stem}_segment_candidates.csv"
        self.latest_segment_export_path = export_segments(self.latest_segments, segment_path)
        self.latest_segment_candidate_export_path = export_segment_candidates(self.latest_segment_candidates, candidate_path)
        accepted_count = len([item for item in self.latest_segment_candidates if item.status == "accepted"])
        rejected_count = len(self.latest_segment_candidates) - accepted_count
        print(f"切片候选统计: accepted={accepted_count}, rejected={rejected_count}")

    @staticmethod
    def _print_video_progress(processed_frames: int, total_frames: int, done: bool = False) -> None:
        if total_frames > 0:
            percentage = processed_frames / total_frames * 100.0
            status = "处理完成" if done else "处理中"
            print(f"{status}: {processed_frames}/{total_frames} 帧 ({percentage:.1f}%)")
            return
        status = "处理完成" if done else "已处理"
        print(f"{status}: {processed_frames} 帧")

    @staticmethod
    def _feedback_color(label: str) -> tuple[int, int, int]:
        if label in {"good_form", "ready"}:
            return (0, 255, 0)
        if label == "depth_insufficient":
            return (0, 255, 255)
        return (0, 0, 255)

    # ── 深度法 + 融合法 视频推理与可视化 ──────────────────────────

    def run_video_with_model(
        self,
        video_path: Path,
        checkpoint_path: Path,
        output_video_path: Path | None = None,
        method: str = "fusion",
        sequence_length: int = 50,
        preview: bool = False,
        log_interval: int = 30,
    ) -> dict:
        """处理视频，对每个动作段用规则法+深度法+融合法判定，生成标注视频。

        method: "rule" | "dl" | "fusion"
        返回: {"output_video": Path, "segments": list[dict], "summary": dict}
        """
        if method not in {"rule", "dl", "fusion"}:
            raise ValueError(f"不支持的 method: {method}，可选 rule/dl/fusion")

        # ── 第1遍：提取特征 + 切片 ──
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.feature_rows = []
        raw_frames: list[np.ndarray] = []
        print(f"第1遍: 提取姿态特征... 视频: {video_path}")
        if total_frames > 0:
            print(f"总帧数: {total_frames}")

        processed = 0
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break
            processed += 1
            timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
            pose_frame = self.extractor.process_bgr_frame(frame, timestamp_ms)
            if processed == 1 or processed % max(log_interval, 1) == 0:
                self._print_video_progress(processed, total_frames)
            if pose_frame is None:
                raw_frames.append(frame)
                self.feature_rows.append(None)
                continue
            features = build_frame_features(pose_frame.landmarks, timestamp_ms)
            self.feature_rows.append(features)
            self.counter.update(features.values["hip_height"])
            raw_frames.append(pose_frame.annotated_bgr_frame)
        capture.release()
        self._print_video_progress(processed, total_frames, done=True)

        valid_features: list[FrameFeatures] = []
        feature_to_raw: list[int] = []
        for raw_idx, feat in enumerate(self.feature_rows):
            if feat is not None:
                valid_features.append(feat)
                feature_to_raw.append(raw_idx)
        if not valid_features:
            raise RuntimeError("未检测到任何有效姿态帧。")

        segments = self.segmenter.detect_segments(valid_features)
        if not segments:
            raise RuntimeError("未检测到任何动作段，无法进行推理。")

        # ── 加载模型 ──
        model = None
        label_mapping = None
        if method in {"dl", "fusion"}:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            label_mapping = {int(k): int(v) for k, v in checkpoint["label_mapping"].items()}
            model = SquatCNNGRU(
                input_dim=int(checkpoint["input_dim"]),
                num_classes=int(checkpoint["class_count"]),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            print(f"模型已加载: {checkpoint_path}")
            print(f"标签映射: {label_mapping}")

        # ── 对每个 segment 做推理 ──
        segment_results: list[dict] = []

        for seg in segments:
            seg_dict = seg.to_row()
            seg_dict["segment_id"] = seg.segment_id

            # 找到该 segment 在 valid_features 中的 bottom 帧
            bottom_feature = valid_features[seg.bottom_index]
            rule_assessment = self.rule_engine.assess(bottom_feature, "bottom")
            rule_label_name = rule_assessment.label
            rule_label_id = RULE_LABEL_TO_ID.get(rule_label_name, -1)
            seg_dict["rule_label"] = rule_label_id
            seg_dict["rule_label_name"] = rule_label_name
            seg_dict["rule_score"] = rule_assessment.score
            seg_dict["rule_issues"] = rule_assessment.issues

            # 深度法推理
            if model is not None and label_mapping is not None:
                inverse_mapping = {v: k for k, v in label_mapping.items()}
                dl_encoded_idx, dl_encoded_probs = self._predict_segment_dl(
                    valid_features, seg, model, sequence_length
                )
                dl_label_id = inverse_mapping.get(dl_encoded_idx, dl_encoded_idx)
                dl_probs = {
                    inverse_mapping.get(i, i): float(p)
                    for i, p in dl_encoded_probs.items()
                }
                seg_dict["dl_label"] = dl_label_id
                seg_dict["dl_label_name"] = LABEL_SCHEMA.get(dl_label_id, f"class_{dl_label_id}")
                seg_dict["dl_probs"] = {str(k): float(v) for k, v in dl_probs.items()}
                seg_dict["dl_confidence"] = float(dl_probs.get(dl_label_id, 0.0))

                # 融合法
                fusion_label_id, fusion_confidence = self._fuse_prediction(
                    rule_label_id, rule_assessment.score,
                    dl_label_id, dl_probs,
                )
                seg_dict["fusion_label"] = fusion_label_id
                seg_dict["fusion_label_name"] = LABEL_SCHEMA.get(fusion_label_id, f"class_{fusion_label_id}")
                seg_dict["fusion_confidence"] = fusion_confidence
            else:
                seg_dict["dl_label"] = -1
                seg_dict["dl_label_name"] = "N/A"
                seg_dict["dl_probs"] = {}
                seg_dict["dl_confidence"] = 0.0
                seg_dict["fusion_label"] = rule_label_id
                seg_dict["fusion_label_name"] = rule_label_name
                seg_dict["fusion_confidence"] = rule_assessment.score / 100.0

            segment_results.append(seg_dict)

        # ── 第2遍：生成标注视频 ──
        if output_video_path is None:
            output_video_path = self.config.export_dir / f"{video_path.stem}_{method}_annotated.mp4"
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        # 建立 raw_frame_index → segment_result 的映射
        frame_to_result: dict[int, dict] = {}
        for seg_result, seg in zip(segment_results, segments):
            feat_start = int(seg_result["start_index"])
            feat_end = int(seg_result["end_index"])
            for fi in range(feat_start, feat_end + 1):
                raw_idx = feature_to_raw[fi]
                frame_to_result[raw_idx] = seg_result

        for raw_idx in range(len(raw_frames)):
            frame = raw_frames[raw_idx].copy()
            feature = self.feature_rows[raw_idx]

            if feature is not None and raw_idx in frame_to_result:
                seg_info = frame_to_result[raw_idx]
                frame = self._overlay_prediction_overlay(
                    frame, feature, seg_info, method,
                )

            writer.write(frame)

            if preview:
                cv2.imshow("try_action - model inference", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        writer.release()
        if preview:
            cv2.destroyWindow("try_action - model inference")
        self.close()

        # ── 汇总 ──
        summary = {
            "method": method,
            "video_path": str(video_path),
            "output_video": str(output_video_path),
            "segment_count": len(segment_results),
            "per_segment": [
                {
                    "segment_id": r["segment_id"],
                    "rule": r["rule_label_name"],
                    "dl": r.get("dl_label_name", "N/A"),
                    "fusion": r.get("fusion_label_name", "N/A"),
                }
                for r in segment_results
            ],
        }

        print(f"\n推理完成，共 {len(segment_results)} 个动作段:")
        print(f"{'Seg':>4} | {'规则法':<20} | {'深度法':<20} | {'融合法':<20}")
        print("-" * 70)
        for r in segment_results:
            print(
                f"{r['segment_id']:>4} | "
                f"{r['rule_label_name']:<20} | "
                f"{r.get('dl_label_name', 'N/A'):<20} | "
                f"{r.get('fusion_label_name', 'N/A'):<20}"
            )
        print(f"\n标注视频已保存: {output_video_path}")

        return {"output_video": output_video_path, "segments": segment_results, "summary": summary}

    def _predict_segment_dl(
        self,
        features: list[FrameFeatures],
        segment: SquatSegment,
        model: SquatCNNGRU,
        sequence_length: int,
    ) -> tuple[int, dict[int, float]]:
        """对单个 segment 做深度模型推理，返回 (predicted_label_id, {label_id: probability})"""
        start = segment.start_index
        end = segment.end_index
        frame_slice = features[start : end + 1]
        feature_matrix = np.array(
            [[f.values[col] for col in FEATURE_COLUMNS] for f in frame_slice],
            dtype=np.float32,
        )

        feature_matrix = resample_features(feature_matrix, sequence_length)

        tensor = torch.tensor(feature_matrix, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_idx = int(np.argmax(probs))
        # 映射回原始 label id（因为训练时做了 label→index 编码）
        # 这里假设 label_mapping 是 {original_label: encoded_index}，需要反转
        # 实际从 checkpoint 加载的 label_mapping 已经在 run_video_with_model 中处理好
        return predicted_idx, {i: float(probs[i]) for i in range(len(probs))}

    def _fuse_prediction(
        self,
        rule_label: int,
        rule_score: float,
        dl_label: int,
        dl_probs: dict[int, float],
        alpha: float = 0.4,
    ) -> tuple[int, float]:
        """融合规则法和深度法结果。"""
        rule_conf = rule_score / 100.0
        dl_conf = dl_probs.get(dl_label, 0.0)

        # 如果规则法极度自信，直接采纳
        if rule_conf > 0.85:
            return rule_label, rule_conf

        # 如果深度法极度自信，直接采纳
        if dl_conf > 0.9:
            return dl_label, dl_conf

        # 两者一致，返回一致结果
        if rule_label == dl_label:
            return rule_label, max(rule_conf, dl_conf)

        # 不一致时用加权融合
        # 对每个可能的标签计算融合得分
        all_labels = set(dl_probs.keys()) | {rule_label}
        best_label = -1
        best_score = -1.0
        for label in all_labels:
            dl_p = dl_probs.get(label, 0.0)
            rule_p = 1.0 if label == rule_label else 0.0
            fused = alpha * rule_p + (1.0 - alpha) * dl_p
            if fused > best_score:
                best_score = fused
                best_label = label

        return best_label, best_score

    def _overlay_prediction_overlay(
        self,
        frame: np.ndarray,
        features: FrameFeatures,
        seg_info: dict,
        method: str,
    ) -> np.ndarray:
        """在帧顶部叠加三种方法的判定结果。"""
        output = frame.copy()
        h, w = output.shape[:2]

        panel_h = 130
        panel_y = 0
        x_start = 15

        # ── 顶部半透明面板背景 ──
        overlay = output.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, panel_h), (30, 30, 30), -1)
        output = cv2.addWeighted(overlay, 0.75, output, 0.25, 0)

        # ── 第1行：段号 ──
        y = 22
        cv2.putText(output, f"Seg #{seg_info['segment_id']}", (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # ── 第2行：规则法 + 深度法 并排 ──
        y += 26
        rule_label = seg_info["rule_label_name"]
        rule_score = seg_info["rule_score"]
        rule_color = self._label_color(rule_label)
        cv2.putText(output, f"Rule: {rule_label} ({rule_score:.0f}%)", (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, rule_color, 2)

        dl_label = seg_info.get("dl_label_name", "N/A")
        dl_conf = seg_info.get("dl_confidence", 0.0)
        col2_x = x_start + 280
        if dl_label != "N/A":
            dl_color = self._label_color(dl_label)
            cv2.putText(output, f"DL: {dl_label} ({dl_conf:.0%})", (col2_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, dl_color, 2)

        # ── 第3行：融合法结果 ──
        y += 24
        fusion_label = seg_info.get("fusion_label_name", rule_label)
        fusion_conf = seg_info.get("fusion_confidence", 0.0)
        fusion_color = self._label_color(fusion_label)
        cv2.putText(output, f"Fusion: {fusion_label} ({fusion_conf:.0%})", (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fusion_color, 2)

        # ── 第4行：概率条（深度法4类概率，水平紧凑排列）──
        y += 22
        probs = seg_info.get("dl_probs", {})
        if probs:
            bar_x = x_start
            bar_w = 150
            bar_h = 12
            sorted_probs = sorted(probs.items(), key=lambda x: -float(x[1]))
            for i, (label_id_str, prob_val) in enumerate(sorted_probs):
                label_id = int(float(label_id_str))
                label_name = LABEL_SCHEMA.get(label_id, f"cls_{label_id}")
                bx = bar_x + i * (bar_w + 10)
                cv2.rectangle(output, (bx, y), (bx + int(bar_w * prob_val), y + bar_h),
                              self._label_color(label_name), -1)
                cv2.putText(output, f"{label_name}:{prob_val:.0%}", (bx + 3, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # ── 第5行：实时特征值（紧凑单行）──
        y += 22
        vals = features.values
        feature_text = (
            f"LK:{vals['left_knee_angle']:.0f}  RK:{vals['right_knee_angle']:.0f}  "
            f"Torso:{vals['torso_tilt_angle']:.0f}  HipH:{vals['hip_height']:.3f}  "
            f"KGap:{vals['knee_gap_ratio']:.3f}"
        )
        cv2.putText(output, feature_text, (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        return output

    @staticmethod
    def _label_color(label_name: str) -> tuple[int, int, int]:
        """BGR color for cv2."""
        if label_name in {"standard", "good_form", "ready"}:
            return (0, 255, 0)       # 绿色
        if label_name == "depth_insufficient":
            return (0, 255, 255)     # 黄色
        if label_name == "knee_valgus":
            return (0, 0, 255)       # 红色
        if label_name == "torso_lean":
            return (0, 140, 255)     # 橙色
        return (180, 180, 180)       # 灰色


def export_feature_rows(features: list[FrameFeatures], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not features:
        output_path.write_text("", encoding="utf-8")
        return output_path
    fieldnames = list(features[0].to_row().keys())
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for feature in features:
            writer.writerow(feature.to_row())
    return output_path


def export_segments(segments: list[SquatSegment], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not segments:
        output_path.write_text("", encoding="utf-8")
        return output_path
    fieldnames = list(segments[0].to_row().keys())
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for segment in segments:
            writer.writerow(segment.to_row())
    return output_path


def export_segment_candidates(candidates: list[SquatSegmentCandidate], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not candidates:
        output_path.write_text("", encoding="utf-8")
        return output_path
    fieldnames = list(candidates[0].to_row().keys())
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate.to_row())
    return output_path
