from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

from .transforms import TEST_CONDITIONS, denormalize, get_eval_transform


PLOT_COLORS = ["#2E86AB", "#F18F01", "#6A994E", "#C73E1D", "#5E548E"]


def _unique_in_order(rows: list[dict[str, float | int | str]], key: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = str(row[key])
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _ordered_values(
    rows: list[dict[str, float | int | str]],
    model: str,
    conditions: list[str],
    value_col: str,
) -> list[float]:
    values = []
    for condition in conditions:
        match = [
            row
            for row in rows
            if str(row["model"]) == model and str(row["test_condition"]) == condition
        ]
        values.append(float(match[0][value_col]) if match else np.nan)
    return values


def plot_accuracy_comparison(
    metric_rows: list[dict[str, float | int | str]],
    output_path: str | Path,
) -> None:
    models = _unique_in_order(metric_rows, "model")
    available_conditions = {str(row["test_condition"]) for row in metric_rows}
    conditions = [c for c in TEST_CONDITIONS if c in available_conditions]
    x = np.arange(len(conditions))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, model in enumerate(models):
        offset = (idx - (len(models) - 1) / 2) * width
        values = _ordered_values(metric_rows, model, conditions, "accuracy")
        ax.bar(x + offset, values, width=width, label=model, color=PLOT_COLORS[idx % len(PLOT_COLORS)])

    ax.set_title("Accuracy by Model and Test Condition")
    ax.set_xlabel("Test condition")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_accuracy_drop(
    metric_rows: list[dict[str, float | int | str]],
    output_path: str | Path,
) -> None:
    models = _unique_in_order(metric_rows, "model")
    available_conditions = {str(row["test_condition"]) for row in metric_rows}
    corruptions = [c for c in TEST_CONDITIONS if c != "clean" and c in available_conditions]
    x = np.arange(len(corruptions))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, model in enumerate(models):
        clean_rows = [
            row
            for row in metric_rows
            if str(row["model"]) == model and str(row["test_condition"]) == "clean"
        ]
        if not clean_rows:
            continue
        clean_acc = float(clean_rows[0]["accuracy"])
        drops = []
        for condition in corruptions:
            condition_rows = [
                row
                for row in metric_rows
                if str(row["model"]) == model and str(row["test_condition"]) == condition
            ]
            drops.append(clean_acc - float(condition_rows[0]["accuracy"]) if condition_rows else np.nan)

        offset = (idx - (len(models) - 1) / 2) * width
        ax.bar(x + offset, drops, width=width, label=model, color=PLOT_COLORS[idx % len(PLOT_COLORS)])

    ax.set_title("Accuracy Drop from Clean Test")
    ax.set_xlabel("Corrupted test condition")
    ax.set_ylabel("Clean accuracy - corrupted accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(corruptions)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_training_curves(
    history_rows: list[dict[str, float | int | str]],
    figures_dir: str | Path,
) -> None:
    figures_dir = Path(figures_dir)
    for model in _unique_in_order(history_rows, "model"):
        model_history = sorted(
            [row for row in history_rows if str(row["model"]) == model],
            key=lambda row: int(row["epoch"]),
        )
        epochs = [int(row["epoch"]) for row in model_history]

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].plot(
            epochs,
            [float(row["train_loss"]) for row in model_history],
            marker="o",
            color=PLOT_COLORS[0],
        )
        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")

        axes[1].plot(
            epochs,
            [float(row["val_loss"]) for row in model_history],
            marker="o",
            color=PLOT_COLORS[1],
        )
        axes[1].set_title("Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")

        axes[2].plot(
            epochs,
            [float(row["val_accuracy"]) for row in model_history],
            marker="o",
            color=PLOT_COLORS[2],
        )
        axes[2].set_title("Validation Accuracy")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_ylim(0.0, 1.0)

        for ax in axes:
            ax.grid(alpha=0.25)

        fig.suptitle(f"Training Curves: {model}")
        fig.tight_layout()
        fig.savefig(figures_dir / f"training_curves_{model}.png", dpi=200)
        plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    title: str,
    normalize: bool = True,
) -> None:
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        plot_matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
        value_format = ".2f"
        colorbar_label = "Row-normalized ratio"
    else:
        plot_matrix = matrix
        value_format = "d"
        colorbar_label = "Count"

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(plot_matrix, cmap="Blues", vmin=0)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = format(plot_matrix[i, j], value_format)
            ax.text(j, i, value, ha="center", va="center", fontsize=7, color="black")

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_condition_samples(
    data_dir: str | Path,
    class_names: list[str],
    output_path: str | Path,
    sample_indices: list[int] | None = None,
) -> None:
    if sample_indices is None:
        sample_indices = [0, 1, 2, 3, 4, 5]

    datasets = {
        condition: FashionMNIST(
            root=str(data_dir),
            train=False,
            download=False,
            transform=get_eval_transform(condition),
        )
        for condition in TEST_CONDITIONS
    }

    fig, axes = plt.subplots(
        len(TEST_CONDITIONS),
        len(sample_indices),
        figsize=(1.8 * len(sample_indices), 1.9 * len(TEST_CONDITIONS)),
    )

    for row, condition in enumerate(TEST_CONDITIONS):
        for col, index in enumerate(sample_indices):
            image, label = datasets[condition][index]
            image = denormalize(image).squeeze(0).numpy()
            ax = axes[row, col]
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(class_names[label], fontsize=8)
            if col == 0:
                ax.set_ylabel(condition, fontsize=10)

    fig.suptitle("Clean and Corrupted Test Samples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def save_misclassified_examples(
    model: nn.Module,
    test_loaders: dict[str, DataLoader],
    device: torch.device,
    class_names: list[str],
    output_path: str | Path,
    max_examples: int = 8,
) -> None:
    model.eval()
    examples: list[tuple[torch.Tensor, int, int, str]] = []

    for condition, loader in test_loaders.items():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            predictions = logits.argmax(dim=1).cpu()
            misses = torch.nonzero(predictions != labels, as_tuple=False).flatten().tolist()

            for batch_index in misses:
                examples.append(
                    (
                        images[batch_index].cpu(),
                        int(labels[batch_index].item()),
                        int(predictions[batch_index].item()),
                        condition,
                    )
                )
                if len(examples) >= max_examples:
                    break
            if len(examples) >= max_examples:
                break
        if len(examples) >= max_examples:
            break

    if not examples:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No misclassified examples found.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    cols = min(4, len(examples))
    rows = math.ceil(len(examples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    axes = np.asarray(axes).reshape(rows, cols)

    for ax in axes.flatten():
        ax.axis("off")

    for ax, (image, true_label, predicted_label, condition) in zip(axes.flatten(), examples):
        image = denormalize(image).squeeze(0).numpy()
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"{condition}\ntrue: {class_names[true_label]}\npred: {class_names[predicted_label]}",
            fontsize=8,
        )

    fig.suptitle("Misclassified Examples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
