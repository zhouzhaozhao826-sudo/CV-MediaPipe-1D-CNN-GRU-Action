from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import SquatCNNGRU


@dataclass(slots=True)
class TrainingConfig:
    dataset_path: Path
    output_dir: Path
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-3
    val_ratio: float = 0.2
    seed: int = 42
    device: str = "cpu"
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    lr_patience: int = 8
    lr_factor: float = 0.5
    use_class_weights: bool = True


class SquatSequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


@dataclass(slots=True)
class TrainingArtifacts:
    best_model_path: Path
    history_path: Path
    metrics_path: Path
    train_samples: int
    val_samples: int
    class_count: int


@dataclass(slots=True)
class DatasetReadiness:
    dataset_path: Path
    total_samples: int
    labeled_samples: int
    pending_samples: int
    feature_count: int
    sequence_length: int
    class_distribution: dict[int, int]
    is_trainable: bool
    message: str


def inspect_dataset_readiness(dataset_path: Path, min_labeled_samples: int = 4) -> DatasetReadiness:
    raw = np.load(dataset_path, allow_pickle=True)
    features = raw["X"]
    labels = raw["y"]
    total_samples = int(len(labels))
    labeled_mask = labels >= 0
    labeled_samples = int(labeled_mask.sum())
    pending_samples = total_samples - labeled_samples
    class_distribution = {
        int(label): int((labels[labeled_mask] == label).sum())
        for label in np.unique(labels[labeled_mask])
    }
    is_trainable = labeled_samples >= min_labeled_samples and len(class_distribution) >= 2
    if not is_trainable:
        message = "已标注样本数量不足或类别数不足，暂不建议启动训练。"
    else:
        message = "数据集满足基础训练条件，可以启动训练。"

    return DatasetReadiness(
        dataset_path=dataset_path,
        total_samples=total_samples,
        labeled_samples=labeled_samples,
        pending_samples=pending_samples,
        feature_count=int(features.shape[-1]),
        sequence_length=int(features.shape[1]),
        class_distribution=class_distribution,
        is_trainable=is_trainable,
        message=message,
    )


def train_model(config: TrainingConfig) -> TrainingArtifacts:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    readiness = inspect_dataset_readiness(config.dataset_path)
    if not readiness.is_trainable:
        raise ValueError(readiness.message)

    raw = np.load(config.dataset_path, allow_pickle=True)
    features = raw["X"]
    labels = raw["y"]
    valid_mask = labels >= 0
    features = features[valid_mask]
    labels = labels[valid_mask]

    unique_labels = np.unique(labels)
    label_to_index = {int(label): idx for idx, label in enumerate(unique_labels)}
    encoded_labels = np.asarray([label_to_index[int(label)] for label in labels], dtype=np.int64)
    train_indices, val_indices = _split_indices(encoded_labels, val_ratio=config.val_ratio, seed=config.seed)

    train_labels_encoded = encoded_labels[train_indices]
    val_labels_encoded = encoded_labels[val_indices]

    train_dataset = SquatSequenceDataset(features[train_indices], train_labels_encoded)
    val_dataset = SquatSequenceDataset(features[val_indices], val_labels_encoded)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = SquatCNNGRU(
        input_dim=features.shape[-1],
        num_classes=len(unique_labels),
    ).to(config.device)

    # class weights（处理样本不均衡）
    class_weights = None
    if config.use_class_weights:
        class_counts = np.bincount(train_labels_encoded, minlength=len(unique_labels))
        total = class_counts.sum()
        weights = total / (len(unique_labels) * class_counts + 1e-8)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=config.device)
        print(f"类别权重: {dict(zip(range(len(unique_labels)), weights.tolist()))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience, verbose=True,
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, float]] = []
    best_model_path = config.output_dir / "best_cnn_gru_model.pth"

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, config.device, training=True)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, config.device, training=False)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            }
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e} | patience={patience_counter}/{config.early_stopping_patience}"
        )

        # early stopping: monitor val_loss
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": int(features.shape[-1]),
                    "class_count": int(len(unique_labels)),
                    "label_mapping": {str(k): int(v) for k, v in label_to_index.items()},
                },
                best_model_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}, best epoch={best_epoch}, best_val_acc={best_val_acc:.4f}")
                break

    history_path = config.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_path = config.output_dir / "training_metrics.json"
    metrics = {
        "train_samples": int(len(train_indices)),
        "val_samples": int(len(val_indices)),
        "class_count": int(len(unique_labels)),
        "label_mapping": {str(k): int(v) for k, v in label_to_index.items()},
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "early_stopped": epoch < config.epochs,
        "final_epoch": epoch,
        "config": {
            **asdict(config),
            "dataset_path": str(config.dataset_path),
            "output_dir": str(config.output_dir),
        },
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return TrainingArtifacts(
        best_model_path=best_model_path,
        history_path=history_path,
        metrics_path=metrics_path,
        train_samples=len(train_indices),
        val_samples=len(val_indices),
        class_count=len(unique_labels),
    )


def _run_epoch(
    model: SquatCNNGRU,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    training: bool,
) -> tuple[float, float]:
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_features, batch_labels in data_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(batch_features)
        loss = criterion(logits, batch_labels)
        if training:
            loss.backward()
            optimizer.step()
        predictions = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * len(batch_labels)
        total_correct += int((predictions == batch_labels).sum().item())
        total_samples += int(len(batch_labels))

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def _split_indices(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)
        if len(label_indices) == 1:
            train_indices.extend(label_indices.tolist())
            continue
        val_count = max(1, int(round(len(label_indices) * val_ratio)))
        if val_count >= len(label_indices):
            val_count = len(label_indices) - 1
        val_indices.extend(label_indices[:val_count].tolist())
        train_indices.extend(label_indices[val_count:].tolist())

    if not val_indices and train_indices:
        val_indices.append(train_indices.pop())

    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def evaluate_model(
    dataset_path: Path,
    checkpoint_path: Path,
    device: str = "cpu",
) -> dict[str, object]:
    raw = np.load(dataset_path, allow_pickle=True)
    features = raw["X"]
    labels = raw["y"]
    labeled_mask = labels >= 0
    features = features[labeled_mask]
    labels = labels[labeled_mask]
    if len(labels) == 0:
        raise ValueError("数据集中没有已标注样本，无法评估。")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    label_mapping = {int(k): int(v) for k, v in checkpoint["label_mapping"].items()}
    valid_mask = np.asarray([int(label) in label_mapping for label in labels], dtype=bool)
    features = features[valid_mask]
    labels = labels[valid_mask]
    encoded_labels = np.asarray([label_mapping[int(label)] for label in labels], dtype=np.int64)

    model = SquatCNNGRU(
        input_dim=int(checkpoint["input_dim"]),
        num_classes=int(checkpoint["class_count"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        tensor_x = torch.tensor(features, dtype=torch.float32, device=device)
        logits = model(tensor_x)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    accuracy = float((predictions == encoded_labels).mean()) if len(encoded_labels) > 0 else 0.0
    confusion = _build_confusion_matrix(encoded_labels, predictions, int(checkpoint["class_count"]))
    inverse_mapping = {encoded: original for original, encoded in label_mapping.items()}

    return {
        "sample_count": int(len(encoded_labels)),
        "accuracy": accuracy,
        "confusion_matrix": confusion.tolist(),
        "label_mapping": {str(k): int(v) for k, v in inverse_mapping.items()},
    }


def compute_metrics(truth: list[int], predictions: list[int], num_classes: int = 4) -> dict:
    """计算准确率、混淆矩阵、per-class precision/recall/f1。"""
    if len(truth) != len(predictions) or len(truth) == 0:
        return {"accuracy": 0.0, "confusion_matrix": [], "per_class": {}}

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(truth, predictions):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    correct = int(np.trace(cm))
    total = int(cm.sum())
    accuracy = correct / total if total > 0 else 0.0

    per_class: dict[str, dict] = {}
    for label_id in range(num_classes):
        tp = int(cm[label_id, label_id])
        support = int(cm[label_id, :].sum())
        pred_total = int(cm[:, label_id].sum())
        precision = tp / pred_total if pred_total > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[str(label_id)] = {
            "label_id": label_id,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    return {
        "accuracy": round(accuracy, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def _build_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, class_count: int) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for truth, pred in zip(labels, predictions):
        matrix[int(truth), int(pred)] += 1
    return matrix
