from __future__ import annotations

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF


FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530
BLACK_AFTER_NORMALIZE = (0.0 - FASHION_MNIST_MEAN) / FASHION_MNIST_STD
TEST_CONDITIONS = ["clean", "noise", "rotation", "occlusion"]


class AddGaussianNoise:
    """Add light Gaussian noise to a tensor image in [0, 1]."""

    def __init__(self, mean: float = 0.0, std: float = 0.12) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


class SquareOcclusion:
    """Mask a square region with black pixels before normalization."""

    def __init__(self, mask_size: int = 8, position: str = "center", value: float = 0.0) -> None:
        self.mask_size = mask_size
        self.position = position
        self.value = value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        _, height, width = tensor.shape
        size = min(self.mask_size, height, width)

        if self.position == "random":
            top = torch.randint(0, height - size + 1, (1,)).item()
            left = torch.randint(0, width - size + 1, (1,)).item()
        else:
            top = (height - size) // 2
            left = (width - size) // 2

        tensor = tensor.clone()
        tensor[:, top : top + size, left : left + size] = self.value
        return tensor


def _normalize() -> T.Normalize:
    return T.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,))


def _rotate_15_degrees(image):
    return TF.rotate(image, angle=15, fill=0)


def get_train_transform(use_augmentation: bool = False) -> T.Compose:
    if not use_augmentation:
        return T.Compose(
            [
                T.ToTensor(),
                _normalize(),
            ]
        )

    return T.Compose(
        [
            T.RandomRotation(degrees=10, fill=0),
            T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05), fill=0),
            T.ToTensor(),
            _normalize(),
            T.RandomErasing(
                p=0.25,
                scale=(0.02, 0.12),
                ratio=(0.5, 2.0),
                value=BLACK_AFTER_NORMALIZE,
            ),
        ]
    )


def get_eval_transform(condition: str = "clean") -> T.Compose:
    condition = condition.lower()

    if condition == "clean":
        ops = [T.ToTensor()]
    elif condition == "noise":
        ops = [T.ToTensor(), AddGaussianNoise(std=0.12)]
    elif condition == "rotation":
        ops = [T.Lambda(_rotate_15_degrees), T.ToTensor()]
    elif condition == "occlusion":
        ops = [T.ToTensor(), SquareOcclusion(mask_size=8, position="center")]
    else:
        raise ValueError(f"Unknown test condition: {condition}")

    ops.append(_normalize())
    return T.Compose(ops)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor * FASHION_MNIST_STD + FASHION_MNIST_MEAN, 0.0, 1.0)

