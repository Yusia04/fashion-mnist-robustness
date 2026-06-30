from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST

from .transforms import TEST_CONDITIONS, get_eval_transform, get_train_transform


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_split_indices(
    dataset_size: int,
    train_size: int,
    val_size: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if train_size <= 0 or val_size <= 0:
        raise ValueError("train_size and val_size must be positive.")
    if train_size + val_size > dataset_size:
        raise ValueError(
            f"train_size + val_size must be <= {dataset_size}, "
            f"but got {train_size + val_size}."
        )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    return train_indices, val_indices


def get_train_val_loaders(
    data_dir: str | Path,
    train_size: int,
    val_size: int,
    batch_size: int,
    use_augmentation: bool,
    seed: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    base_dataset = FashionMNIST(root=str(data_dir), train=True, download=True)
    train_indices, val_indices = make_split_indices(len(base_dataset), train_size, val_size, seed)

    train_dataset_full = FashionMNIST(
        root=str(data_dir),
        train=True,
        download=False,
        transform=get_train_transform(use_augmentation=use_augmentation),
    )
    val_dataset_full = FashionMNIST(
        root=str(data_dir),
        train=True,
        download=False,
        transform=get_eval_transform("clean"),
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    generator = torch.Generator().manual_seed(seed)

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker if num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_kwargs,
    )
    return train_loader, val_loader


def get_test_loaders(
    data_dir: str | Path,
    batch_size: int,
    test_size: int = 10_000,
    conditions: Iterable[str] = TEST_CONDITIONS,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    data_dir = Path(data_dir)
    loaders: dict[str, DataLoader] = {}

    for condition in conditions:
        dataset = FashionMNIST(
            root=str(data_dir),
            train=False,
            download=True,
            transform=get_eval_transform(condition),
        )
        if test_size <= 0:
            raise ValueError("test_size must be positive.")
        if test_size > len(dataset):
            raise ValueError(f"test_size must be <= {len(dataset)}, but got {test_size}.")
        if test_size < len(dataset):
            dataset = Subset(dataset, list(range(test_size)))

        loaders[condition] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_seed_worker if num_workers > 0 else None,
        )

    return loaders

