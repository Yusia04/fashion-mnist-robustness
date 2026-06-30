from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().numpy()

        all_targets.append(targets.numpy())
        all_predictions.append(predictions)

    return np.concatenate(all_targets), np.concatenate(all_predictions)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 10,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def macro_f1_from_confusion_matrix(matrix: np.ndarray) -> float:
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_positive = matrix.sum(axis=0).astype(np.float64)
    actual_positive = matrix.sum(axis=1).astype(np.float64)

    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive),
        where=predicted_positive != 0,
    )
    recall = np.divide(
        true_positive,
        actual_positive,
        out=np.zeros_like(true_positive),
        where=actual_positive != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return float(f1.mean())


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> dict[str, float | np.ndarray]:
    y_true, y_pred = collect_predictions(model, loader, device)
    matrix = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    accuracy = float((y_true == y_pred).mean())
    macro_f1 = macro_f1_from_confusion_matrix(matrix)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": matrix,
    }

