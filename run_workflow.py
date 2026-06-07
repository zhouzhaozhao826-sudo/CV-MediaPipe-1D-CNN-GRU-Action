from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from pose_action import (
    FEATURE_COLUMNS,
    LABEL_SCHEMA,
    PoseActionPipeline,
    ProjectConfig,
    SquatDatasetBuilder,
    TrainingConfig,
    auto_label_segments,
    create_label_template,
    evaluate_model,
    inspect_dataset_readiness,
    resample_features,
    summarize_labels,
    train_model,
    update_label_record,
)


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
ROUND3_VIDEO_ROOT = PROJECT_DIR / "datasets" / "round3" / "videos_round3"
ROUND3_EXPORT_ROOT = PROJECT_DIR / "datasets" / "round3" / "exports_round3"


# =========================
# 用户修改区: 主要改这里的路径
# =========================
VIDEO_SOURCE = Path("C:/Users/zjisj/Desktop/valgus_01.mp4")  # 当前处理: 标准侧面
POSE_MODEL_PATH = WORKSPACE_DIR / "text" / "pose_landmarker.task"
EXPORT_ROOT = ROUND3_EXPORT_ROOT
EXPORT_DIR_OVERRIDE: Path | None = None
EXPORT_PREFIX_OVERRIDE: str | None = None
TRAIN_OUTPUT_DIRNAME = "train_run_manual"


# =========================
# 步骤开关: 按需开启/关闭
# =========================
RUN_VIDEO_ANALYSIS = False
CREATE_LABEL_TEMPLATE = False
AUTO_LABEL = False          # 规则法自动预标注（减少人工标注工作量）
APPLY_LABEL_UPDATES = False
BUILD_DATASET = False
INSPECT_DATASET = False
RUN_TRAINING = False
RUN_EVALUATION = False
RUN_MODEL_INFERENCE = True  # 用训练好的模型对视频做推理+可视化


# =========================
# 可选运行参数
# =========================
PREVIEW_VIDEO = False
LOG_INTERVAL = 30
SEQUENCE_LENGTH = 50
DEFAULT_LABEL = -1
TRAIN_EPOCHS = 80
TRAIN_BATCH_SIZE = 16
TRAIN_LEARNING_RATE = 1e-3
TRAIN_VAL_RATIO = 0.2
TRAIN_SEED = 42
TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_EARLY_STOPPING_PATIENCE = 20
TRAIN_LR_PATIENCE = 10
TRAIN_LR_FACTOR = 0.5
TRAIN_USE_CLASS_WEIGHTS = True
EVAL_DEVICE = "cpu"
INFERENCE_METHOD = "fusion"  # "rule" | "dl" | "fusion"
INFERENCE_CHECKPOINT_PATH = ROUND3_EXPORT_ROOT / "_merged" / "round3_merged_4class" / "train_run_manual" / "best_cnn_gru_model.pth"

# =========================
# 训练数据模式
# single: 使用当前 VIDEO_SOURCE 对应的单视频数据集
# merged: 合并多个已生成的数据集后再训练/评估
# short_video_batch: 短视频批量处理（每视频一次深蹲，无需切片，目录名即标签）
# =========================
TRAIN_DATA_MODE = "merged"
MERGED_DATASET_NAME = "round3_merged_4class"
# 注意：先逐个视频处理（TRAIN_DATA_MODE="single"），全部处理完后再切回 "merged"
MERGED_DATASET_SOURCES = [
    ROUND3_EXPORT_ROOT / "standard" / "r3_standard_C" / "r3_standard_C_dataset.npz",
    ROUND3_EXPORT_ROOT / "standard" / "r3_standard_Z" / "r3_standard_Z_dataset.npz",
    ROUND3_EXPORT_ROOT / "depth_insufficient" / "r3_depth_C" / "r3_depth_C_dataset.npz",
    ROUND3_EXPORT_ROOT / "knee_valgus" / "r3_valgus_Z" / "r3_valgus_Z_dataset.npz",
    ROUND3_EXPORT_ROOT / "torso_lean" / "r3_lean_C" / "r3_lean_C_dataset.npz",
]

# 短视频批量处理配置（仅 TRAIN_DATA_MODE = "short_video_batch" 时生效）
SHORT_VIDEO_ROOT: Path | None = None  # 短视频根目录，如 Path("datasets/round3/videos_round3")
SHORT_VIDEO_CATEGORIES = {             # 子目录名 → 标签值
    "standard": 0,
    "depth_insufficient": 1,
    "knee_valgus": 2,
    "torso_lean": 3,
}


# =========================
# 可选自动标注区
# 仅当 APPLY_LABEL_UPDATES = True 时生效
# label 可选值见 LABEL_SCHEMA
# =========================
LABEL_UPDATES = [
    # {"segment_id": 1, "label": 0, "notes": "标准动作"},
    # {"segment_id": 2, "label": 1, "notes": "下蹲深度不足"},
]


@dataclass(slots=True)
class WorkflowPaths:
    video_path: Path
    export_dir: Path
    export_prefix: str
    train_output_dir: Path
    feature_path: Path
    segments_path: Path
    labels_path: Path
    dataset_path: Path
    checkpoint_path: Path


@dataclass(slots=True)
class DatasetTargetPaths:
    dataset_mode: str
    dataset_path: Path
    train_output_dir: Path
    checkpoint_path: Path
    metadata_path: Path | None = None
    summary_path: Path | None = None


def resolve_video_source(source: Path) -> Path:
    if source.exists() and source.is_file():
        return source

    if source.exists() and source.is_dir():
        candidates = []
        for pattern in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
            candidates.extend(sorted(source.glob(pattern)))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            raise FileNotFoundError(f"视频目录中没有可用视频文件: {source}")
        candidate_text = "\n".join(f"  - {item}" for item in candidates)
        raise ValueError(f"视频目录中存在多个视频，请把 VIDEO_SOURCE 改成具体文件:\n{candidate_text}")

    matches = sorted(path for path in PROJECT_DIR.rglob(source.name) if path.is_file())
    if len(matches) == 1:
        print(f"提示: 未直接找到配置路径，已自动匹配到同名文件: {matches[0]}")
        return matches[0]
    if len(matches) > 1:
        candidate_text = "\n".join(f"  - {item}" for item in matches)
        raise FileNotFoundError(f"未找到配置的视频路径，且发现多个同名文件，请明确指定:\n{candidate_text}")
    raise FileNotFoundError(f"视频文件不存在: {source}")


def infer_export_dir(video_path: Path) -> Path:
    if EXPORT_DIR_OVERRIDE is not None:
        return EXPORT_DIR_OVERRIDE
    try:
        relative_path = video_path.relative_to(ROUND3_VIDEO_ROOT)
        if len(relative_path.parts) >= 2:
            category_name = relative_path.parts[0]
            return EXPORT_ROOT / category_name / video_path.stem
    except ValueError:
        pass
    return EXPORT_ROOT / video_path.stem


def infer_export_prefix(video_path: Path) -> str:
    if EXPORT_PREFIX_OVERRIDE is not None:
        return EXPORT_PREFIX_OVERRIDE
    return video_path.stem


def build_workflow_paths() -> WorkflowPaths:
    video_path = resolve_video_source(VIDEO_SOURCE)
    export_dir = infer_export_dir(video_path)
    export_prefix = infer_export_prefix(video_path)
    train_output_dir = export_dir / TRAIN_OUTPUT_DIRNAME
    feature_name = f"{export_prefix}_features.csv"
    feature_stem = Path(feature_name).stem
    return WorkflowPaths(
        video_path=video_path,
        export_dir=export_dir,
        export_prefix=export_prefix,
        train_output_dir=train_output_dir,
        feature_path=export_dir / feature_name,
        segments_path=export_dir / f"{feature_stem}_segments.csv",
        labels_path=export_dir / f"{export_prefix}_labels.csv",
        dataset_path=export_dir / f"{export_prefix}_dataset.npz",
        checkpoint_path=train_output_dir / "best_cnn_gru_model.pth",
    )


def build_dataset_target_paths(paths: WorkflowPaths) -> DatasetTargetPaths:
    if TRAIN_DATA_MODE == "single":
        return DatasetTargetPaths(
            dataset_mode="single",
            dataset_path=paths.dataset_path,
            train_output_dir=paths.train_output_dir,
            checkpoint_path=paths.checkpoint_path,
            metadata_path=paths.dataset_path.with_name(f"{paths.dataset_path.stem}_metadata.csv"),
            summary_path=paths.dataset_path.with_name(f"{paths.dataset_path.stem}_summary.json"),
        )

    if TRAIN_DATA_MODE == "merged":
        merged_root = EXPORT_ROOT / "_merged" / MERGED_DATASET_NAME
        dataset_path = merged_root / f"{MERGED_DATASET_NAME}_dataset.npz"
        train_output_dir = merged_root / TRAIN_OUTPUT_DIRNAME
        return DatasetTargetPaths(
            dataset_mode="merged",
            dataset_path=dataset_path,
            train_output_dir=train_output_dir,
            checkpoint_path=train_output_dir / "best_cnn_gru_model.pth",
            metadata_path=merged_root / f"{MERGED_DATASET_NAME}_dataset_metadata.csv",
            summary_path=merged_root / f"{MERGED_DATASET_NAME}_dataset_summary.json",
        )

    if TRAIN_DATA_MODE == "short_video_batch":
        merged_root = EXPORT_ROOT / "_merged" / MERGED_DATASET_NAME
        dataset_path = merged_root / f"{MERGED_DATASET_NAME}_dataset.npz"
        train_output_dir = merged_root / TRAIN_OUTPUT_DIRNAME
        return DatasetTargetPaths(
            dataset_mode="short_video_batch",
            dataset_path=dataset_path,
            train_output_dir=train_output_dir,
            checkpoint_path=train_output_dir / "best_cnn_gru_model.pth",
            metadata_path=merged_root / f"{MERGED_DATASET_NAME}_dataset_metadata.csv",
            summary_path=merged_root / f"{MERGED_DATASET_NAME}_dataset_summary.json",
        )

    raise ValueError(f"不支持的 TRAIN_DATA_MODE: {TRAIN_DATA_MODE}")


def ensure_file_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description}不存在: {path}")


def print_step(title: str) -> None:
    print(f"\n{'=' * 24} {title} {'=' * 24}")


def print_label_schema() -> None:
    print("当前标签体系:")
    for label, name in LABEL_SCHEMA.items():
        print(f"  {label}: {name}")


def merge_datasets(target: DatasetTargetPaths, dataset_sources: list[Path]) -> None:
    print_step("步骤4: 合并训练样本")
    if not dataset_sources:
        raise ValueError("MERGED_DATASET_SOURCES 为空，无法合并数据集。")

    feature_names_ref: list[str] | None = None
    sequence_length_ref: int | None = None
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    metadata_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for dataset_path in dataset_sources:
        ensure_file_exists(dataset_path, "待合并数据集文件")
        raw = np.load(dataset_path, allow_pickle=True)
        features = raw["X"]
        labels = raw["y"]
        feature_names = [str(item) for item in raw["feature_names"].tolist()]
        sequence_length = int(raw["sequence_length"][0])

        if feature_names_ref is None:
            feature_names_ref = feature_names
        elif feature_names != feature_names_ref:
            raise ValueError(f"特征列不一致，无法合并: {dataset_path}")

        if sequence_length_ref is None:
            sequence_length_ref = sequence_length
        elif sequence_length != sequence_length_ref:
            raise ValueError(f"序列长度不一致，无法合并: {dataset_path}")

        all_features.append(features.astype(np.float32))
        all_labels.append(labels.astype(np.int64))

        metadata_path = dataset_path.with_name(f"{dataset_path.stem}_metadata.csv")
        if metadata_path.exists():
            metadata_df = pd.read_csv(metadata_path)
            metadata_df.insert(0, "source_dataset", dataset_path.stem)
            metadata_frames.append(metadata_df)

        summary_rows.append(
            {
                "source_dataset": dataset_path.stem,
                "sample_count": int(len(labels)),
                "labeled_count": int((labels >= 0).sum()),
                "pending_count": int((labels < 0).sum()),
            }
        )

    merged_x = np.concatenate(all_features, axis=0)
    merged_y = np.concatenate(all_labels, axis=0)
    target.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target.dataset_path,
        X=merged_x,
        y=merged_y,
        feature_names=np.asarray(feature_names_ref, dtype=object),
        sequence_length=np.asarray([sequence_length_ref], dtype=np.int64),
    )

    if target.metadata_path is not None and metadata_frames:
        merged_metadata = pd.concat(metadata_frames, ignore_index=True)
        merged_metadata.to_csv(target.metadata_path, index=False, encoding="utf-8")

    if target.summary_path is not None:
        class_distribution = {
            int(label): int((merged_y[merged_y >= 0] == label).sum())
            for label in np.unique(merged_y[merged_y >= 0])
        }
        summary = {
            "dataset_mode": target.dataset_mode,
            "dataset_name": target.dataset_path.stem,
            "merged_sources": [str(path) for path in dataset_sources],
            "source_summaries": summary_rows,
            "sample_count": int(len(merged_y)),
            "labeled_count": int((merged_y >= 0).sum()),
            "pending_count": int((merged_y < 0).sum()),
            "class_distribution": class_distribution,
            "feature_count": int(len(feature_names_ref or [])),
            "sequence_length": int(sequence_length_ref or 0),
        }
        target.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"合并数据集: {target.dataset_path}")
    print(f"合并来源数量: {len(dataset_sources)}")
    print(f"总样本数: {len(merged_y)}")
    print(f"已标注样本: {(merged_y >= 0).sum()}")


def run_video_analysis(paths: WorkflowPaths) -> None:
    print_step("步骤1: 视频分析")
    ensure_file_exists(paths.video_path, "视频文件")
    ensure_file_exists(POSE_MODEL_PATH, "姿态模型文件")

    config = ProjectConfig(model_path=POSE_MODEL_PATH, export_dir=paths.export_dir)
    pipeline = PoseActionPipeline(config)
    output = pipeline.run_video(
        video_path=paths.video_path,
        export_name=paths.feature_path.name,
        preview=PREVIEW_VIDEO,
        log_interval=LOG_INTERVAL,
    )

    if output is None:
        raise RuntimeError("未检测到有效姿态数据，没有生成特征文件。")

    print(f"特征文件: {output}")
    if pipeline.latest_segment_export_path is None:
        raise RuntimeError("视频处理完成，但没有生成动作切片文件。")

    print(f"切片文件: {pipeline.latest_segment_export_path}")
    print(f"动作段数量: {len(pipeline.latest_segments)}")


def create_labels(paths: WorkflowPaths) -> None:
    print_step("步骤2: 生成标签模板")
    ensure_file_exists(paths.segments_path, "动作切片文件")
    result = create_label_template(
        segments_path=paths.segments_path,
        output_path=paths.labels_path,
        overwrite=True,
    )
    print(f"标签模板: {result.labels_path}")
    print(f"待标注数量: {result.pending_count}")
    print_label_schema()


def apply_label_updates(paths: WorkflowPaths, updates: Iterable[dict[str, object]]) -> None:
    print_step("步骤3: 写入标签")
    ensure_file_exists(paths.labels_path, "标签文件")
    applied_count = 0
    for item in updates:
        segment_id = int(item["segment_id"])
        label = int(item["label"])
        notes = str(item.get("notes", ""))
        result = update_label_record(
            labels_path=paths.labels_path,
            segment_id=segment_id,
            label=label,
            notes=notes,
        )
        applied_count += 1
        print(
            f"已写入 segment_id={segment_id}, label={label}({LABEL_SCHEMA[label]}), 剩余待标注={result.pending_count}"
        )

    if applied_count == 0:
        print("没有可写入的标签记录，已跳过。")
        return

    summary = summarize_labels(paths.labels_path)
    print(f"标注汇总: total={summary['total']}, labeled={summary['labeled']}, pending={summary['pending']}")


def auto_label_squats(paths: WorkflowPaths) -> None:
    print_step("步骤2b: 规则法自动预标注")
    ensure_file_exists(paths.feature_path, "特征文件")
    ensure_file_exists(paths.segments_path, "动作切片文件")
    result = auto_label_segments(
        features_path=paths.feature_path,
        segments_path=paths.segments_path,
        labels_path=paths.labels_path,
        overwrite=True,
    )
    print(f"标签文件: {result.labels_path}")
    print(f"自动标注段数: {result.record_count}")
    print(f"待人工复核: {result.pending_count}")
    summary = summarize_labels(paths.labels_path)
    print(f"标注汇总: total={summary['total']}, labeled={summary['labeled']}, "
          f"auto_labeled={summary.get('auto_labeled', 0)}, pending={summary['pending']}")


def process_short_video_batch(target: DatasetTargetPaths) -> None:
    """短视频批量处理：每视频=一次深蹲，无需切片，目录名即标签。"""
    print_step("步骤4: 短视频批量处理 → 构建数据集")

    video_root = SHORT_VIDEO_ROOT
    if video_root is None:
        video_root = PROJECT_DIR / "datasets" / "round3" / "videos_round3"
    if not video_root.exists():
        raise FileNotFoundError(f"短视频根目录不存在: {video_root}")

    ensure_file_exists(POSE_MODEL_PATH, "姿态模型文件")

    all_samples: list[np.ndarray] = []
    all_labels: list[int] = []
    metadata_rows: list[dict] = []
    sample_index = 0

    for category_name, label_id in SHORT_VIDEO_CATEGORIES.items():
        category_dir = video_root / category_name
        if not category_dir.is_dir():
            print(f"  跳过: 目录不存在 {category_dir}")
            continue

        videos = sorted(category_dir.glob("*.mp4")) + sorted(category_dir.glob("*.mov")) + sorted(category_dir.glob("*.avi"))
        if not videos:
            print(f"  跳过: {category_name} 下无视频文件")
            continue

        print(f"\n处理类别 [{category_name}] (label={label_id}), 共 {len(videos)} 个视频")
        for video_path in videos:
            print(f"  处理: {video_path.name}")

            export_dir = target.dataset_path.parent / "_temp" / category_name / video_path.stem
            export_dir.mkdir(parents=True, exist_ok=True)

            config = ProjectConfig(model_path=POSE_MODEL_PATH, export_dir=export_dir)
            pipeline = PoseActionPipeline(config)
            try:
                features_output = pipeline.run_video(
                    video_path=video_path,
                    export_name=f"{video_path.stem}_features.csv",
                    preview=False,
                    log_interval=60,
                )
            except Exception as exc:
                print(f"    警告: 特征提取失败 ({exc})，跳过")
                continue

            if features_output is None:
                print(f"    警告: 未检测到有效姿态，跳过")
                continue

            features_df = pd.read_csv(features_output)
            feature_cols = [c for c in FEATURE_COLUMNS if c in features_df.columns]
            if features_df.empty or len(features_df) < 2:
                print(f"    警告: 有效帧数不足，跳过")
                continue

            matrix = features_df[feature_cols].to_numpy(dtype=np.float32)

            all_samples.append(resample_features(matrix, SEQUENCE_LENGTH))
            all_labels.append(label_id)
            metadata_rows.append({
                "sample_index": sample_index,
                "source_video": str(video_path),
                "category": category_name,
                "label": label_id,
                "label_name": LABEL_SCHEMA[label_id],
                "original_frame_count": len(features_df),
            })
            sample_index += 1

    if not all_samples:
        raise RuntimeError("未生成任何有效样本，请检查短视频目录和视频文件。")

    X = np.stack(all_samples).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)

    target.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target.dataset_path,
        X=X,
        y=y,
        feature_names=np.array(feature_cols, dtype=object),
        sequence_length=np.array([SEQUENCE_LENGTH], dtype=np.int64),
    )

    if target.metadata_path:
        pd.DataFrame(metadata_rows).to_csv(target.metadata_path, index=False, encoding="utf-8")

    if target.summary_path:
        class_dist = {label: int((y == label).sum()) for label in sorted(set(y))}
        summary = {
            "dataset_mode": "short_video_batch",
            "dataset_name": target.dataset_path.stem,
            "sample_count": int(len(y)),
            "feature_count": int(len(feature_cols)),
            "sequence_length": int(SEQUENCE_LENGTH),
            "class_distribution": {LABEL_SCHEMA.get(k, f"cls_{k}"): v for k, v in class_dist.items()},
            "categories_processed": list(SHORT_VIDEO_CATEGORIES.keys()),
        }
        target.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n短视频批量处理完成:")
    print(f"  总样本数: {len(y)}")
    for label_id in sorted(set(y)):
        print(f"  {LABEL_SCHEMA[label_id]} (label={label_id}): {(y == label_id).sum()} 个")
    print(f"  数据集: {target.dataset_path}")


def build_dataset(paths: WorkflowPaths, target: DatasetTargetPaths) -> None:
    if target.dataset_mode == "merged":
        merge_datasets(target, MERGED_DATASET_SOURCES)
        return

    if target.dataset_mode == "short_video_batch":
        process_short_video_batch(target)
        return

    print_step("步骤4: 生成训练样本")
    ensure_file_exists(paths.feature_path, "特征文件")
    ensure_file_exists(paths.segments_path, "动作切片文件")

    labels_path = paths.labels_path if paths.labels_path.exists() else None
    builder = SquatDatasetBuilder(sequence_length=SEQUENCE_LENGTH)
    result = builder.build_from_csv(
        features_path=paths.feature_path,
        segments_path=paths.segments_path,
        output_path=target.dataset_path,
        labels_path=labels_path,
        default_label=DEFAULT_LABEL,
    )
    print(f"数据集文件: {result.dataset_path}")
    print(f"样本元数据: {result.metadata_path}")
    print(f"样本数量: {result.sample_count}")
    print(f"特征维度: {result.feature_count}")
    print(f"序列长度: {result.sequence_length}")


def inspect_dataset(target: DatasetTargetPaths) -> None:
    print_step("步骤5: 检查数据集")
    ensure_file_exists(target.dataset_path, "训练数据集文件")
    result = inspect_dataset_readiness(target.dataset_path)
    print(f"数据集路径: {result.dataset_path}")
    print(f"总样本数: {result.total_samples}")
    print(f"已标注样本: {result.labeled_samples}")
    print(f"待标注样本: {result.pending_samples}")
    print(f"类别分布: {result.class_distribution}")
    print(f"是否可训练: {result.is_trainable}")
    print(f"检查结论: {result.message}")


def run_training(target: DatasetTargetPaths) -> None:
    print_step("步骤6: 训练模型")
    ensure_file_exists(target.dataset_path, "训练数据集文件")
    config = TrainingConfig(
        dataset_path=target.dataset_path,
        output_dir=target.train_output_dir,
        batch_size=TRAIN_BATCH_SIZE,
        epochs=TRAIN_EPOCHS,
        learning_rate=TRAIN_LEARNING_RATE,
        val_ratio=TRAIN_VAL_RATIO,
        seed=TRAIN_SEED,
        device=TRAIN_DEVICE,
        weight_decay=TRAIN_WEIGHT_DECAY,
        early_stopping_patience=TRAIN_EARLY_STOPPING_PATIENCE,
        lr_patience=TRAIN_LR_PATIENCE,
        lr_factor=TRAIN_LR_FACTOR,
        use_class_weights=TRAIN_USE_CLASS_WEIGHTS,
    )
    artifacts = train_model(config)
    print(f"最佳模型: {artifacts.best_model_path}")
    print(f"训练历史: {artifacts.history_path}")
    print(f"训练指标: {artifacts.metrics_path}")
    print(f"训练样本数: {artifacts.train_samples}")
    print(f"验证样本数: {artifacts.val_samples}")
    print(f"类别数量: {artifacts.class_count}")


def run_evaluation(paths: WorkflowPaths) -> None:
    print_step("步骤7: 评估模型")
    target = build_dataset_target_paths(paths)
    ensure_file_exists(target.dataset_path, "训练数据集文件")
    ensure_file_exists(target.checkpoint_path, "模型权重文件")
    result = evaluate_model(
        dataset_path=target.dataset_path,
        checkpoint_path=target.checkpoint_path,
        device=EVAL_DEVICE,
    )
    print(f"评估样本数: {result['sample_count']}")
    print(f"准确率: {result['accuracy']:.4f}")
    print(f"混淆矩阵: {result['confusion_matrix']}")
    print(f"标签映射: {result['label_mapping']}")


def run_model_inference(paths: WorkflowPaths) -> None:
    print_step("步骤8: 模型推理 + 可视化")
    checkpoint_path = INFERENCE_CHECKPOINT_PATH
    if checkpoint_path is None:
        checkpoint_path = paths.checkpoint_path
    ensure_file_exists(paths.video_path, "视频文件")
    ensure_file_exists(POSE_MODEL_PATH, "姿态模型文件")
    ensure_file_exists(checkpoint_path, "模型权重文件")

    config = ProjectConfig(model_path=POSE_MODEL_PATH, export_dir=paths.export_dir)
    pipeline = PoseActionPipeline(config)
    output_video = paths.export_dir / f"{paths.export_prefix}_{INFERENCE_METHOD}_annotated.mp4"
    result = pipeline.run_video_with_model(
        video_path=paths.video_path,
        checkpoint_path=checkpoint_path,
        output_video_path=output_video,
        method=INFERENCE_METHOD,
    )
    print(f"标注视频: {result['output_video']}")
    print(f"动作段数量: {len(result['segments'])}")


def main() -> None:
    paths = build_workflow_paths()
    target = build_dataset_target_paths(paths)
    paths.export_dir.mkdir(parents=True, exist_ok=True)
    target.train_output_dir.mkdir(parents=True, exist_ok=True)

    print("当前运行配置:")
    print(f"TRAIN_DATA_MODE = {TRAIN_DATA_MODE}")
    print(f"VIDEO_SOURCE = {VIDEO_SOURCE}")
    print(f"RESOLVED_VIDEO_PATH = {paths.video_path}")
    print(f"POSE_MODEL_PATH = {POSE_MODEL_PATH}")
    print(f"EXPORT_DIR = {paths.export_dir}")
    print(f"EXPORT_PREFIX = {paths.export_prefix}")
    print(f"ACTIVE_DATASET_PATH = {target.dataset_path}")
    print(f"TRAIN_OUTPUT_DIR = {target.train_output_dir}")
    if TRAIN_DATA_MODE == "merged":
        print("MERGED_DATASET_SOURCES:")
        for dataset_path in MERGED_DATASET_SOURCES:
            print(f"  - {dataset_path}")

    if RUN_VIDEO_ANALYSIS and TRAIN_DATA_MODE != "short_video_batch":
        run_video_analysis(paths)

    if CREATE_LABEL_TEMPLATE and TRAIN_DATA_MODE != "short_video_batch":
        create_labels(paths)

    if AUTO_LABEL and TRAIN_DATA_MODE != "short_video_batch":
        auto_label_squats(paths)

    if APPLY_LABEL_UPDATES:
        apply_label_updates(paths, LABEL_UPDATES)

    if BUILD_DATASET:
        build_dataset(paths, target)

    if INSPECT_DATASET:
        inspect_dataset(target)

    if RUN_TRAINING:
        run_training(target)

    if RUN_EVALUATION:
        run_evaluation(paths)

    if RUN_MODEL_INFERENCE:
        run_model_inference(paths)

    print_step("流程结束")
    print("本次流程已执行完成。")


if __name__ == "__main__":
    main()
