"""三方法对比脚本：规则法 vs 深度法 vs 融合法

对视频运行三种方法，输出每段对比表、准确率、混淆矩阵，导出 JSON 报告。

用法:
    python try_action/compare.py \
        --video-path datasets/round3/videos_round3/test.mp4 \
        --checkpoint-path exports_round3/_merged/.../best_cnn_gru_model.pth \
        --labels-path exports_round3/test_labels.csv   # 可选，提供则计算准确率
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pose_action import LABEL_SCHEMA, PoseActionPipeline, ProjectConfig
from pose_action.training import compute_metrics


def _resolve_workspace() -> Path:
    return Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="规则法 vs 深度法 vs 融合法 对比评估")
    parser.add_argument("--video-path", type=Path, required=True, help="待评估视频路径")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="训练好的模型 .pth 路径")
    parser.add_argument("--labels-path", type=Path, default=None, help="真实标签 CSV（可选，segment_id + label 列）")
    parser.add_argument("--output-dir", type=Path, default=None, help="报告输出目录（默认视频同目录）")
    parser.add_argument("--output-json", type=Path, default=None, help="JSON 报告路径")
    parser.add_argument("--model-path", type=Path, default=None, help="MediaPipe .task 文件路径")
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--preview", action="store_true", help="处理时显示画面")
    return parser


def load_ground_truth(labels_path: Path) -> dict[int, int]:
    df = pd.read_csv(labels_path)
    if "segment_id" not in df.columns or "label" not in df.columns:
        raise ValueError("标签文件必须包含 segment_id 和 label 列")
    gt: dict[int, int] = {}
    for _, row in df.iterrows():
        label_val = row["label"]
        if pd.isna(label_val):
            continue
        gt[int(row["segment_id"])] = int(label_val)
    return gt


def main() -> None:
    args = build_parser().parse_args()
    workspace = _resolve_workspace()

    video_path = args.video_path
    checkpoint_path = args.checkpoint_path
    model_path = args.model_path or workspace / "text" / "pose_landmarker.task"

    for p, desc in [(video_path, "视频文件"), (checkpoint_path, "模型文件"), (model_path, "姿态模型")]:
        if not p.exists():
            raise FileNotFoundError(f"{desc}不存在: {p}")
    if args.labels_path is not None and not args.labels_path.exists():
        raise FileNotFoundError(f"标签文件不存在: {args.labels_path}")

    ground_truth = load_ground_truth(args.labels_path) if args.labels_path else {}

    # ── 运行 pipeline（内部加载模型、执行三方法推理）──
    config = ProjectConfig(model_path=model_path)
    pipeline = PoseActionPipeline(config)
    output_dir = args.output_dir or video_path.parent

    result = pipeline.run_video_with_model(
        video_path=video_path,
        checkpoint_path=checkpoint_path,
        output_video_path=output_dir / f"{video_path.stem}_compare_annotated.mp4",
        method="fusion",
        sequence_length=args.sequence_length,
        preview=args.preview,
    )

    segments_info = result["segments"]

    # ── 汇总三方法结果 ──
    rows: list[dict] = []
    rule_preds: list[int] = []
    dl_preds: list[int] = []
    fusion_preds: list[int] = []
    gt_list: list[int] = []

    for seg in segments_info:
        seg_id = seg["segment_id"]
        rule_label = seg.get("rule_label", -1)
        dl_label = seg.get("dl_label", -1)
        fusion_label = seg.get("fusion_label", -1)

        row = {
            "segment_id": seg_id,
            "rule_label": rule_label,
            "rule_label_name": LABEL_SCHEMA.get(rule_label, "N/A"),
            "dl_label": dl_label,
            "dl_label_name": LABEL_SCHEMA.get(dl_label, "N/A"),
            "fusion_label": fusion_label,
            "fusion_label_name": LABEL_SCHEMA.get(fusion_label, "N/A"),
            "rule_score": seg.get("rule_score", 0),
            "dl_confidence": seg.get("dl_confidence", 0),
            "fusion_confidence": seg.get("fusion_confidence", 0),
        }

        gt_label = ground_truth.get(seg_id)
        if gt_label is not None:
            row["ground_truth"] = gt_label
            row["ground_truth_name"] = LABEL_SCHEMA.get(gt_label, "N/A")
            row["rule_correct"] = int(rule_label == gt_label)
            row["dl_correct"] = int(dl_label == gt_label)
            row["fusion_correct"] = int(fusion_label == gt_label)
            rule_preds.append(rule_label)
            dl_preds.append(dl_label)
            fusion_preds.append(fusion_label)
            gt_list.append(gt_label)

        rows.append(row)

    # ── 终端输出 ──
    has_gt = len(gt_list) > 0
    header = f"{'Seg':>4} | {'GT':<20} | {'规则法':<20} | {'深度法':<20} | {'融合法':<20}"
    if has_gt:
        header += " | R/D/F正确?"
    print(f"\n{'=' * 90}")
    print("三方法对比结果")
    print(f"{'=' * 90}")
    print(header)
    print("-" * 90)

    for row in rows:
        base = (
            f"{row['segment_id']:>4} | "
            f"{row.get('ground_truth_name', 'N/A'):<20} | "
            f"{row['rule_label_name']:<20} | "
            f"{row['dl_label_name']:<20} | "
            f"{row['fusion_label_name']:<20}"
        )
        if has_gt:
            r_ok = row.get("rule_correct", "?")
            d_ok = row.get("dl_correct", "?")
            f_ok = row.get("fusion_correct", "?")
            base += f" | {r_ok}/{d_ok}/{f_ok}"
        print(base)

    print("-" * 90)

    # ── 准确率统计 ──
    if has_gt:
        rule_metrics = compute_metrics(gt_list, rule_preds)
        dl_metrics = compute_metrics(gt_list, dl_preds)
        fusion_metrics = compute_metrics(gt_list, fusion_preds)

        print(f"\n总段数: {len(gt_list)}")
        print(f"{'方法':<12} {'准确率':<10} {'备注'}")
        print("-" * 40)
        print(f"{'规则法':<12} {rule_metrics['accuracy']:<10.2%} 生物力学阈值")
        print(f"{'深度法':<12} {dl_metrics['accuracy']:<10.2%} 1D-CNN+GRU")
        print(f"{'融合法':<12} {fusion_metrics['accuracy']:<10.2%} 规则+DL加权融合")

        for method_name, metrics in [("规则法", rule_metrics), ("深度法", dl_metrics), ("融合法", fusion_metrics)]:
            cm = metrics["confusion_matrix"]
            print(f"\n{method_name} 混淆矩阵:")
            labels_names = [LABEL_SCHEMA[i] for i in range(len(cm)) if i in LABEL_SCHEMA]
            header_cm = "GT\\Pred  " + " ".join(f"{n:>20}" for n in labels_names)
            print(header_cm)
            for i, label_name in enumerate(labels_names):
                row_cm = f"{label_name:<9}" + " ".join(f"{cm[i][j]:>20}" for j in range(len(labels_names)))
                print(row_cm)

    # ── 一致性分析 ──
    if len(rows) > 1:
        agreement_rd = sum(1 for r in rows if r["rule_label"] == r["dl_label"])
        agreement_rf = sum(1 for r in rows if r["rule_label"] == r["fusion_label"])
        agreement_df = sum(1 for r in rows if r["dl_label"] == r["fusion_label"])
        total = len(rows)
        print(f"\n方法间一致性:")
        print(f"  规则法 ↔ 深度法: {agreement_rd}/{total} ({agreement_rd/total:.1%})")
        print(f"  规则法 ↔ 融合法: {agreement_rf}/{total} ({agreement_rf/total:.1%})")
        print(f"  深度法 ↔ 融合法: {agreement_df}/{total} ({agreement_df/total:.1%})")

    # ── 导出 JSON ──
    json_path = args.output_json or output_dir / f"{video_path.stem}_compare_report.json"
    report = {
        "video_path": str(video_path),
        "checkpoint_path": str(checkpoint_path),
        "segment_count": len(rows),
        "ground_truth_available": has_gt,
        "segments": rows,
    }
    if has_gt:
        report["metrics"] = {
            "rule_based": rule_metrics,
            "deep_learning": dl_metrics,
            "fusion": fusion_metrics,
        }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存: {json_path}")


if __name__ == "__main__":
    main()
