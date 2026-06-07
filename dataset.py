from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, FrameFeatures, build_frame_features, resample_features
from .rules import SquatRuleEngine


LABEL_SCHEMA = {
    0: "standard",
    1: "depth_insufficient",
    2: "knee_valgus",
    3: "torso_lean",
}

RULE_LABEL_TO_ID = {
    "good_form": 0, "ready": 0,
    "depth_insufficient": 1,
    "knee_valgus": 2,
    "torso_lean": 3,
}


@dataclass(slots=True)
class SampleBuildResult:
    dataset_path: Path
    metadata_path: Path
    sample_count: int
    feature_count: int
    sequence_length: int


class SquatDatasetBuilder:
    def __init__(self, sequence_length: int = 50) -> None:
        self.sequence_length = sequence_length

    def build_from_csv(
        self,
        features_path: Path,
        segments_path: Path,
        output_path: Path,
        metadata_path: Path | None = None,
        labels_path: Path | None = None,
        default_label: int = -1,
    ) -> SampleBuildResult:
        features_df = pd.read_csv(features_path)
        segments_df = pd.read_csv(segments_path)

        if features_df.empty or segments_df.empty:
            raise ValueError("特征文件或切片文件为空，无法生成训练样本。")

        feature_columns = [column for column in features_df.columns if column != "timestamp_ms"]
        label_lookup = self._load_label_lookup(labels_path)

        samples: list[np.ndarray] = []
        labels: list[int] = []
        metadata_rows: list[dict[str, object]] = []

        for _, segment in segments_df.iterrows():
            start_index = int(segment["start_index"])
            end_index = int(segment["end_index"])
            segment_id = int(segment["segment_id"])

            frame_slice = features_df.iloc[start_index : end_index + 1]
            if frame_slice.empty or len(frame_slice) < 2:
                continue

            feature_matrix = frame_slice[feature_columns].to_numpy(dtype=np.float32)
            resampled_matrix = self._resample_feature_matrix(feature_matrix)

            label_value = label_lookup.get(segment_id, default_label)
            samples.append(resampled_matrix)
            labels.append(int(label_value))
            metadata_rows.append(
                {
                    "segment_id": segment_id,
                    "start_index": start_index,
                    "end_index": end_index,
                    "start_timestamp_ms": int(segment["start_timestamp_ms"]),
                    "end_timestamp_ms": int(segment["end_timestamp_ms"]),
                    "duration_ms": int(segment["duration_ms"]),
                    "label": int(label_value),
                    "label_status": "labeled" if segment_id in label_lookup else "pending",
                }
            )

        if not samples:
            raise ValueError("没有生成任何有效的单动作样本，请检查切片结果。")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_path or output_path.with_name(f"{output_path.stem}_metadata.csv")

        sample_array = np.stack(samples).astype(np.float32)
        label_array = np.asarray(labels, dtype=np.int64)
        np.savez_compressed(
            output_path,
            X=sample_array,
            y=label_array,
            feature_names=np.asarray(feature_columns, dtype=object),
            sequence_length=np.asarray([self.sequence_length], dtype=np.int64),
        )

        metadata_df = pd.DataFrame(metadata_rows)
        metadata_df.to_csv(metadata_path, index=False, encoding="utf-8")
        self._write_dataset_summary(output_path, feature_columns, metadata_df)

        return SampleBuildResult(
            dataset_path=output_path,
            metadata_path=metadata_path,
            sample_count=len(samples),
            feature_count=len(feature_columns),
            sequence_length=self.sequence_length,
        )

    def _resample_feature_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        return resample_features(feature_matrix, self.sequence_length)

    @staticmethod
    def _load_label_lookup(labels_path: Path | None) -> dict[int, int]:
        if labels_path is None or not labels_path.exists():
            return {}

        labels_df = pd.read_csv(labels_path)
        if "segment_id" not in labels_df.columns or "label" not in labels_df.columns:
            raise ValueError("标签文件必须包含 segment_id 和 label 两列。")

        return {
            int(row["segment_id"]): int(row["label"])
            for _, row in labels_df.iterrows()
            if not pd.isna(row["label"])
        }

    @staticmethod
    def _write_dataset_summary(output_path: Path, feature_columns: list[str], metadata_df: pd.DataFrame) -> None:
        summary = {
            "dataset_file": output_path.name,
            "sample_count": int(len(metadata_df)),
            "feature_count": int(len(feature_columns)),
            "feature_names": feature_columns,
            "labeled_count": int((metadata_df["label_status"] == "labeled").sum()),
            "pending_count": int((metadata_df["label_status"] == "pending").sum()),
        }
        summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(slots=True)
class LabelTemplateResult:
    labels_path: Path
    record_count: int
    pending_count: int


def create_label_template(
    segments_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> LabelTemplateResult:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"标签模板已存在: {output_path}")
    segments_df = pd.read_csv(segments_path)
    if segments_df.empty:
        raise ValueError("切片文件为空，无法生成标签模板。")
    label_df = pd.DataFrame(
        {
            "segment_id": segments_df["segment_id"].astype(int),
            "start_index": segments_df["start_index"].astype(int),
            "bottom_index": segments_df["bottom_index"].astype(int),
            "end_index": segments_df["end_index"].astype(int),
            "start_timestamp_ms": segments_df["start_timestamp_ms"].astype(int),
            "bottom_timestamp_ms": segments_df["bottom_timestamp_ms"].astype(int),
            "end_timestamp_ms": segments_df["end_timestamp_ms"].astype(int),
            "duration_ms": segments_df["duration_ms"].astype(int),
            "label": pd.Series([pd.NA] * len(segments_df), dtype="Int64"),
            "label_name": pd.Series([""] * len(segments_df), dtype="string"),
            "notes": pd.Series([""] * len(segments_df), dtype="string"),
            "review_status": pd.Series(["pending"] * len(segments_df), dtype="string"),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_df.to_csv(output_path, index=False, encoding="utf-8")
    return LabelTemplateResult(
        labels_path=output_path,
        record_count=len(label_df),
        pending_count=len(label_df),
    )


def update_label_record(
    labels_path: Path,
    segment_id: int,
    label: int,
    notes: str = "",
) -> LabelTemplateResult:
    if label not in LABEL_SCHEMA:
        raise ValueError(f"不支持的标签值: {label}，可选值为 {list(LABEL_SCHEMA)}")
    labels_df = pd.read_csv(labels_path)
    if "segment_id" not in labels_df.columns:
        raise ValueError("标签文件缺少 segment_id 列。")
    match_mask = labels_df["segment_id"].astype(int) == int(segment_id)
    if not match_mask.any():
        raise ValueError(f"未找到 segment_id={segment_id} 的记录。")
    labels_df.loc[match_mask, "label"] = int(label)
    labels_df.loc[match_mask, "label_name"] = LABEL_SCHEMA[label]
    labels_df.loc[match_mask, "notes"] = notes
    labels_df.loc[match_mask, "review_status"] = "labeled"
    labels_df.to_csv(labels_path, index=False, encoding="utf-8")
    pending_count = int((labels_df["review_status"] != "labeled").sum())
    return LabelTemplateResult(
        labels_path=labels_path,
        record_count=len(labels_df),
        pending_count=pending_count,
    )


def summarize_labels(labels_path: Path) -> dict[str, int]:
    labels_df = pd.read_csv(labels_path)
    if labels_df.empty:
        return {"total": 0, "labeled": 0, "pending": 0}
    labeled = int((labels_df["review_status"].isin(["labeled", "auto_labeled"])).sum())
    auto_labeled = int((labels_df["review_status"] == "auto_labeled").sum())
    total = int(len(labels_df))
    return {"total": total, "labeled": labeled, "pending": total - labeled, "auto_labeled": auto_labeled}


def auto_label_segments(
    features_path: Path,
    segments_path: Path,
    labels_path: Path,
    overwrite: bool = False,
) -> LabelTemplateResult:
    """用规则法自动预标注切片，减少人工标注工作量。

    对每个 segment 的 bottom 帧运行规则法判定，将结果写入 labels CSV。
    review_status 设为 "auto_labeled"，区别于人工确认的 "labeled"。
    """
    features_df = pd.read_csv(features_path)
    segments_df = pd.read_csv(segments_path)
    if segments_df.empty:
        raise ValueError("切片文件为空，无法自动标注。")

    # 创建或读取标签文件
    if not labels_path.exists() or overwrite:
        result = create_label_template(segments_path, labels_path, overwrite=True)
    else:
        pending = summarize_labels(labels_path)["pending"]
        result = LabelTemplateResult(labels_path=labels_path, record_count=0, pending_count=pending)

    labels_df = pd.read_csv(labels_path)
    engine = SquatRuleEngine()

    auto_count = 0
    for _, segment in segments_df.iterrows():
        segment_id = int(segment["segment_id"])
        bottom_index = int(segment["bottom_index"])

        if bottom_index < 0 or bottom_index >= len(features_df):
            continue

        row = features_df.iloc[bottom_index]
        timestamp_ms = int(row["timestamp_ms"])
        values = {col: float(row[col]) for col in FEATURE_COLUMNS if col in features_df.columns}
        if not values:
            continue

        features = FrameFeatures(timestamp_ms=timestamp_ms, values=values)
        assessment = engine.assess(features, "bottom")
        label_id = RULE_LABEL_TO_ID.get(assessment.label, -1)
        if label_id < 0:
            continue

        match_mask = labels_df["segment_id"].astype(int) == segment_id
        if not match_mask.any():
            continue

        labels_df.loc[match_mask, "label"] = label_id
        labels_df.loc[match_mask, "label_name"] = LABEL_SCHEMA[label_id]
        labels_df.loc[match_mask, "review_status"] = "auto_labeled"
        labels_df.loc[match_mask, "notes"] = f"auto: {assessment.label} score={assessment.score:.0f}"
        auto_count += 1

    labels_df.to_csv(labels_path, index=False, encoding="utf-8")
    pending_count = int((~labels_df["review_status"].isin(["labeled", "auto_labeled"])).sum())
    print(f"规则法自动预标注完成: {auto_count} 个段已标记，{pending_count} 个待人工复核")
    return LabelTemplateResult(
        labels_path=labels_path,
        record_count=len(labels_df),
        pending_count=pending_count,
    )
