"""Grouped bar chart of per-class F1 for MLP / SmallCNN / SmallCNN_Aug.

Trains all three models over N seeds, then plots one bar chart per requested
test condition:
  - x axis: the 10 Fashion-MNIST classes
  - y axis: class-wise F1 on the given test condition
  - 3 bars per class (one per model)
  - Bars show the seed mean; error bars are standard deviation

Run with no arguments to produce clean and rotation charts. Pass
`--conditions clean noise rotation occlusion` to pick a subset.

Output:
  results/figures/per_class_f1_bar_{condition}.png

Self-contained: depends only on torch / torchvision / matplotlib / numpy.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import FashionMNIST
from torchvision.transforms import functional as TF


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530
BLACK_AFTER_NORMALIZE = (0.0 - FASHION_MNIST_MEAN) / FASHION_MNIST_STD

PLOT_COLORS = {
    "MLP": "#2E86AB",
    "SmallCNN": "#F18F01",
    "SmallCNN_Aug": "#6A994E",
}


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_transform() -> T.Normalize:
    return T.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,))


def get_train_transform(use_augmentation: bool) -> T.Compose:
    if not use_augmentation:
        return T.Compose([T.ToTensor(), normalize_transform()])
    return T.Compose(
        [
            T.RandomRotation(degrees=10, fill=0),
            T.RandomAffine(
                degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05), fill=0
            ),
            T.ToTensor(),
            normalize_transform(),
            T.RandomErasing(
                p=0.25,
                scale=(0.02, 0.12),
                ratio=(0.5, 2.0),
                value=BLACK_AFTER_NORMALIZE,
            ),
        ]
    )


class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.12) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


class SquareOcclusion:
    def __init__(
        self, mask_size: int = 8, position: str = "random", value: float = 0.0
    ) -> None:
        self.mask_size = mask_size
        self.position = position
        self.value = value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        _, height, width = tensor.shape
        size = min(self.mask_size, height, width)
        if self.position == "random":
            top = int(torch.randint(0, height - size + 1, (1,)).item())
            left = int(torch.randint(0, width - size + 1, (1,)).item())
        else:
            top = (height - size) // 2
            left = (width - size) // 2
        tensor = tensor.clone()
        tensor[:, top : top + size, left : left + size] = self.value
        return tensor


def _rotate_15(image):
    return TF.rotate(image, angle=15, fill=0)


def get_eval_transform(condition: str = "clean") -> T.Compose:
    condition = condition.lower()
    if condition == "clean":
        ops = [T.ToTensor()]
    elif condition == "noise":
        ops = [T.ToTensor(), AddGaussianNoise(std=0.12)]
    elif condition == "rotation":
        ops = [T.Lambda(_rotate_15), T.ToTensor()]
    elif condition == "occlusion":
        ops = [T.ToTensor(), SquareOcclusion(mask_size=8, position="random")]
    else:
        raise ValueError(f"Unknown test condition: {condition}")
    ops.append(normalize_transform())
    return T.Compose(ops)


def make_split_indices(
    dataset_size: int, train_size: int, val_size: int, seed: int
) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    return indices[:train_size], indices[train_size : train_size + val_size]


def build_loaders(
    data_dir: Path,
    train_transform: T.Compose,
    train_size: int,
    val_size: int,
    batch_size: int,
    seed: int,
    conditions: list[str],
) -> tuple[DataLoader, DataLoader, dict[str, DataLoader]]:
    base = FashionMNIST(root=str(data_dir), train=True, download=True)
    train_indices, val_indices = make_split_indices(
        len(base), train_size, val_size, seed
    )

    train_dataset = Subset(
        FashionMNIST(
            root=str(data_dir), train=True, download=False, transform=train_transform
        ),
        train_indices,
    )
    val_dataset = Subset(
        FashionMNIST(
            root=str(data_dir),
            train=True,
            download=False,
            transform=get_eval_transform("clean"),
        ),
        val_indices,
    )
    test_loaders: dict[str, DataLoader] = {}
    for condition in conditions:
        test_dataset = FashionMNIST(
            root=str(data_dir),
            train=False,
            download=True,
            transform=get_eval_transform(condition),
        )
        test_loaders[condition] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loaders


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    tag: str,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        print(f"  [{tag}] epoch {epoch}/{epochs} val_acc={correct / total:.4f}")


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def per_class_f1_from_confusion(matrix: np.ndarray) -> np.ndarray:
    m = matrix.astype(np.float64)
    tp = np.diag(m)
    predicted_positive = m.sum(axis=0)
    actual_positive = m.sum(axis=1)
    precision = np.divide(
        tp, predicted_positive, out=np.zeros_like(tp), where=predicted_positive != 0
    )
    recall = np.divide(
        tp, actual_positive, out=np.zeros_like(tp), where=actual_positive != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )
    return f1


@torch.no_grad()
def evaluate_per_class_f1(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    all_targets = []
    all_preds = []
    for images, targets in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    matrix = confusion_matrix(y_true, y_pred)
    return per_class_f1_from_confusion(matrix)


def build_model(model_name: str) -> nn.Module:
    if model_name == "MLP":
        return MLP()
    return SmallCNN()


def plot_bar(
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
    output_path: Path,
    condition: str,
) -> None:
    model_names = list(means.keys())
    n_models = len(model_names)
    n_classes = len(CLASS_NAMES)

    x = np.arange(n_classes)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for idx, name in enumerate(model_names):
        offset = (idx - (n_models - 1) / 2) * width
        ax.bar(
            x + offset,
            means[name],
            width=width,
            yerr=stds[name],
            capsize=3,
            label=name,
            color=PLOT_COLORS.get(name, None),
            error_kw={"elinewidth": 1.0, "ecolor": "black"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(f"F1 score ({condition} test)")
    ax.set_xlabel("Class")
    ax.set_title(
        f"Per-class F1: MLP / SmallCNN / SmallCNN_Aug ({condition} test, mean±std over 3 seeds)"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", ncol=n_models)

    # Highlight Shirt on the x-axis so the motivation is visible at a glance.
    shirt_idx = CLASS_NAMES.index("Shirt")
    ax.axvspan(shirt_idx - 0.4, shirt_idx + 0.4, color="red", alpha=0.07, zorder=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["clean", "rotation"],
        choices=["clean", "noise", "rotation", "occlusion"],
        help="Test condition(s) to evaluate F1 on. One chart is written per condition.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    candidates = [here] + list(here.parents)
    project_root = next(
        (p for p in candidates if (p / "data").exists() or (p / "results").exists()),
        Path.cwd(),
    )

    data_dir = project_root / "data"
    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    seeds = [42, 43, 44]
    epochs = 5
    train_size = 10_000
    val_size = 2_000
    batch_size = 128
    lr = 1e-3

    configs = [
        {"name": "MLP", "augmentation": False},
        {"name": "SmallCNN", "augmentation": False},
        {"name": "SmallCNN_Aug", "augmentation": True},
    ]
    conditions = list(dict.fromkeys(args.conditions))

    # per_seed_f1[condition][model] = list of per-class F1 arrays (one per seed)
    per_seed_f1: dict[str, dict[str, list[np.ndarray]]] = {
        cond: {c["name"]: [] for c in configs} for cond in conditions
    }

    for seed in seeds:
        print(f"\n=== seed = {seed} ===")
        for config in configs:
            name = config["name"]
            set_seed(seed)
            train_transform = get_train_transform(
                use_augmentation=bool(config["augmentation"])
            )
            train_loader, val_loader, test_loaders = build_loaders(
                data_dir=data_dir,
                train_transform=train_transform,
                train_size=train_size,
                val_size=val_size,
                batch_size=batch_size,
                seed=seed,
                conditions=conditions,
            )
            model = build_model(name).to(device)
            print(f"Training {name} (seed={seed}) ...")
            train_model(
                model,
                train_loader,
                val_loader,
                epochs=epochs,
                lr=lr,
                device=device,
                tag=name,
            )
            for condition in conditions:
                # Fix seed so stochastic corruptions (noise / occlusion) reproduce.
                set_seed(seed + 1000 + conditions.index(condition))
                f1 = evaluate_per_class_f1(model, test_loaders[condition], device)
                per_seed_f1[condition][name].append(f1)
                print(f"  {name} [{condition}] per-class F1: {np.round(f1, 3)}")

    for condition in conditions:
        means = {
            name: np.stack(values).mean(axis=0)
            for name, values in per_seed_f1[condition].items()
        }
        stds = {
            name: np.stack(values).std(axis=0, ddof=1)
            for name, values in per_seed_f1[condition].items()
        }
        output_path = figures_dir / f"per_class_f1_bar_{condition}.png"
        plot_bar(means, stds, output_path, condition=condition)
        print(f"\nWrote {output_path}")

        print(f"\n=== Summary (mean F1) — {condition} ===")
        header = f"{'class':14s} " + " ".join(
            f"{n:>14s}" for n in per_seed_f1[condition]
        )
        print(header)
        for k in range(len(CLASS_NAMES)):
            row = f"{CLASS_NAMES[k]:14s} " + " ".join(
                f"{means[n][k]:.3f}±{stds[n][k]:.3f}" for n in per_seed_f1[condition]
            )
            print(row)


if __name__ == "__main__":
    main()
