"""Visualize class prototypes and RandomAffine-induced Pullover -> Shirt flips.

Produces two figures:
  1. prototypes_upper_body.png
     Prototypes (pixel-wise mean of training images) for the four upper-body
     classes: T-shirt/top, Pullover, Shirt, Coat.
  2. pullover_flips_to_shirt_under_affine.png
     Real Pullover images that are originally closer to the Pullover prototype,
     but move closer to the Shirt prototype after a small RandomAffine
     transform. Numerical L2 distances are printed under each cell.

Self-contained: depends only on torch / torchvision / matplotlib / numpy.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.datasets import FashionMNIST


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_training_arrays(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    dataset = FashionMNIST(root=str(data_dir), train=True, download=True)
    images = dataset.data.numpy().astype(np.float32) / 255.0
    labels = dataset.targets.numpy()
    return images, labels


def compute_prototypes(
    images: np.ndarray, labels: np.ndarray, num_classes: int = 10
) -> np.ndarray:
    return np.stack([images[labels == k].mean(axis=0) for k in range(num_classes)])


def figure_prototypes(prototypes: np.ndarray, output_path: Path) -> None:
    upper = [
        ("T-shirt/top", 0),
        ("Pullover", 2),
        ("Shirt", 6),
        ("Coat", 4),
    ]
    fig, axes = plt.subplots(1, len(upper), figsize=(3.0 * len(upper), 3.6))
    for ax, (name, idx) in zip(axes, upper):
        ax.imshow(prototypes[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{name}\n(class {idx} prototype)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Class prototype = pixel-wise mean of all training images in that class",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def figure_affine_flips_class(
    images: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    output_path: Path,
    n_examples: int = 6,
    n_transforms_per_image: int = 4,
    seed: int = 42,
) -> None:
    """Find real Pullover images that get pulled closer to the Shirt prototype
    after RandomAffine, and lay them out with the numerical distance change."""
    set_seed(seed)
    affine = T.RandomAffine(
        degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05), fill=0
    )

    pullover = CLASS_NAMES.index("Pullover")
    shirt = CLASS_NAMES.index("Shirt")

    proto_pullover = prototypes[pullover].reshape(-1)
    proto_shirt = prototypes[shirt].reshape(-1)

    pullover_indices = np.flatnonzero(labels == pullover)
    to_pil = T.ToPILImage()

    selected: list[dict] = []
    for idx in pullover_indices:
        original = images[idx]
        d0_pull = float(np.linalg.norm(original.reshape(-1) - proto_pullover))
        d0_shirt = float(np.linalg.norm(original.reshape(-1) - proto_shirt))
        # Only consider originals that already sit on the Pullover side.
        if not (d0_pull < d0_shirt):
            continue

        pil = to_pil(torch.from_numpy(original).unsqueeze(0))
        variants = []
        variants_flipped = 0
        for _ in range(n_transforms_per_image):
            aug = np.array(affine(pil), dtype=np.float32) / 255.0
            d_pull = float(np.linalg.norm(aug.reshape(-1) - proto_pullover))
            d_shirt = float(np.linalg.norm(aug.reshape(-1) - proto_shirt))
            flipped = d_shirt < d_pull
            variants.append(
                {"image": aug, "d_pull": d_pull, "d_shirt": d_shirt, "flipped": flipped}
            )
            if flipped:
                variants_flipped += 1

        if variants_flipped >= 1:
            selected.append(
                {
                    "index": int(idx),
                    "original": original,
                    "d0_pull": d0_pull,
                    "d0_shirt": d0_shirt,
                    "variants": variants,
                }
            )
        if len(selected) >= n_examples:
            break

    if not selected:
        print("No qualifying Pullover images found.")
        return

    n_cols = (
        2 + n_transforms_per_image + 1
    )  # pullover proto | original | variants | shirt proto
    fig, axes = plt.subplots(
        len(selected), n_cols, figsize=(2.1 * n_cols, 2.5 * len(selected))
    )
    axes = np.atleast_2d(axes).reshape(len(selected), n_cols)

    for row, example in enumerate(selected):
        ax = axes[row, 0]
        ax.imshow(prototypes[pullover], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title("Pullover\nprototype", fontsize=9)

        ax = axes[row, 1]
        ax.imshow(example["original"], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title("original\n(labeled Pullover)", fontsize=9)
        ax.set_xlabel(
            f"d_pull={example['d0_pull']:.1f}\nd_shirt={example['d0_shirt']:.1f}",
            fontsize=8,
            color="gray",
        )

        for vi, variant in enumerate(example["variants"]):
            ax = axes[row, 2 + vi]
            ax.imshow(variant["image"], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"affine #{vi + 1}", fontsize=9)
            color = "red" if variant["flipped"] else "green"
            marker = "-> Shirt side!" if variant["flipped"] else "-> Pullover side"
            ax.set_xlabel(
                f"d_pull={variant['d_pull']:.1f}\nd_shirt={variant['d_shirt']:.1f}\n{marker}",
                fontsize=8,
                color=color,
            )

        ax = axes[row, -1]
        ax.imshow(prototypes[shirt], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title("Shirt\nprototype", fontsize=9)

    fig.suptitle(
        "Pullover images that move closer to the Shirt prototype under RandomAffine",
        fontsize=13,
    )
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=10,
            color="gray",
            label="gray: original image (reference)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=10,
            color="green",
            label="green: after affine, still on Pullover side (safe)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=10,
            color="red",
            label="red: after affine, moved to Shirt side (label noise)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    # Locate a project root — the current directory or an ancestor of this
    # script that already contains a data/ or results/ directory. Falls back
    # to cwd if nothing plausible is found.
    here = Path(__file__).resolve().parent
    candidates = [here] + list(here.parents)
    project_root = next(
        (p for p in candidates if (p / "data").exists() or (p / "results").exists()),
        Path.cwd(),
    )

    data_dir = project_root / "data"
    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print("Loading training data and computing prototypes...")
    images, labels = load_training_arrays(data_dir)
    prototypes = compute_prototypes(images, labels)

    print("Writing prototypes_upper_body.png ...")
    figure_prototypes(prototypes, figures_dir / "prototypes_upper_body.png")

    print("Searching for Pullover-to-Shirt affine flips...")
    figure_affine_flips_class(
        images,
        labels,
        prototypes,
        figures_dir / "pullover_flips_to_shirt_under_affine.png",
    )
    print("Done.")


if __name__ == "__main__":
    main()
