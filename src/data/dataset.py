"""Fashion-MNIST pipeline for an unconditional, from-scratch Mini DDPM.

Only the official dataset reader comes from torchvision. Splitting, preprocessing,
train-only augmentation and the image-only training interface are project code.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FashionMNIST as TorchvisionFashionMNIST
from torchvision.transforms.functional import pil_to_tensor

CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
SOURCE_SIZE = 28
IMAGE_SIZE = 32
PADDING = (IMAGE_SIZE - SOURCE_SIZE) // 2


class FashionMNIST(TorchvisionFashionMNIST):
    """Use the dataset authors' HTTPS mirror and torchvision's checksum checks.

    The class name preserves the conventional ``FashionMNIST/raw`` cache path.
    Existing manually uploaded raw files work with ``download=False`` as well.
    """

    mirrors = [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"
        "master/data/fashion/",
    ]


@dataclass(frozen=True)
class DataConfig:
    """Data hyperparameters. Relative roots are resolved from the working directory."""

    root: str = "data"
    val_size: int = 5000
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = False
    download: bool = True
    horizontal_flip_p: float = 0.5
    drop_last_train: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("val_size", "batch_size", "num_workers", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if self.val_size <= 0 or self.batch_size <= 0:
            raise ValueError("val_size and batch_size must be positive")
        if self.num_workers < 0 or not 0 <= self.seed < 2**32:
            raise ValueError("num_workers must be nonnegative; seed must be in [0, 2**32)")
        for name in ("pin_memory", "download", "drop_last_train"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a YAML boolean (true/false)")
        if not isinstance(self.horizontal_flip_p, (int, float)) or not (
            0.0 <= self.horizontal_flip_p <= 1.0
        ):
            raise ValueError("horizontal_flip_p must be in [0, 1]")


def load_data_config(path: str | Path) -> DataConfig:
    """Read only the shared seed and data section; leave other roles' config intact."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("Configuration must contain a 'data' mapping")
    values = dict(config["data"])
    if "seed" in values:
        raise ValueError("Put seed at the top level, not inside data")
    values["seed"] = config.get("seed", 42)
    return DataConfig(**values)


def seed_data(seed: int) -> None:
    """Seed main-process augmentation and Python/NumPy randomness before a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _seed_worker(worker_id: int) -> None:
    """Initialize extra RNGs from the per-worker seed assigned by DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _split_indices(targets: Tensor, val_size: int, seed: int) -> tuple[Tensor, Tensor]:
    """Seeded, stratified split of official training indices without sklearn.

    Proportional allocation plus largest remainders gives exactly val_size items.
    Labels are used for the split/diagnostics, never as DDPM model inputs.
    """
    targets = torch.as_tensor(targets, dtype=torch.long)
    if targets.ndim != 1 or not 0 < val_size < len(targets):
        raise ValueError("val_size must leave nonempty train/validation subsets")
    labels, counts = torch.unique(targets, sorted=True, return_counts=True)
    numerators = counts * val_size
    allocations = numerators // len(targets)
    remainders = (numerators % len(targets)).tolist()
    remaining = val_size - int(allocations.sum())
    for index in sorted(range(len(labels)), key=lambda i: (-remainders[i], i))[:remaining]:
        allocations[index] += 1

    generator = torch.Generator().manual_seed(seed)
    train_parts, val_parts = [], []
    for label, n_val in zip(labels, allocations):
        members = torch.where(targets == label)[0]
        members = members[torch.randperm(len(members), generator=generator)]
        val_parts.append(members[:int(n_val)])
        train_parts.append(members[int(n_val):])
    train, val = torch.cat(train_parts), torch.cat(val_parts)
    return (
        train[torch.randperm(len(train), generator=generator)],
        val[torch.randperm(len(val), generator=generator)],
    )


class FashionDataset(Dataset[Tensor]):
    """Independent transform view over raw Fashion-MNIST; returns images only.

    ``dataset[i]``: float32 [1, 32, 32], finite, in [-1, 1].
    ``targets`` remains available for diagnostics/a separate evaluator classifier.
    The underlying raw dataset must have no transform or target_transform.
    """

    def __init__(
        self,
        base_dataset: Any,
        indices: Sequence[int] | Tensor | None = None,
        *,
        training: bool = False,
        horizontal_flip_p: float = 0.5,
    ) -> None:
        if not 0 <= horizontal_flip_p <= 1:
            raise ValueError("horizontal_flip_p must be in [0, 1]")
        if getattr(base_dataset, "transform", None) is not None or getattr(
            base_dataset, "target_transform", None
        ) is not None:
            raise ValueError("base_dataset must be raw, with transforms disabled")
        self.base_dataset = base_dataset
        self.indices = tuple(range(len(base_dataset))) if indices is None else tuple(
            int(i) for i in indices
        )
        if not self.indices or min(self.indices) < 0 or max(self.indices) >= len(base_dataset):
            raise ValueError("indices must select a nonempty, in-bounds subset")
        self.training = training
        self.horizontal_flip_p = horizontal_flip_p

    @property
    def targets(self) -> Tensor:
        """Class IDs in this view's order; excluded from __getitem__ output."""
        return torch.as_tensor(self.base_dataset.targets)[list(self.indices)].clone()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tensor:
        image, _label = self.base_dataset[self.indices[index]]
        pixels = pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)  # [1, 28, 28]
        if tuple(pixels.shape) != (1, SOURCE_SIZE, SOURCE_SIZE):
            raise ValueError(f"Expected grayscale [1, 28, 28], got {tuple(pixels.shape)}")
        if self.training and torch.rand(()).item() < self.horizontal_flip_p:
            pixels = pixels.flip(-1)  # [1, 28, 28], horizontal flip only
        # Pad black in [0, 1] BEFORE normalization, so the border becomes -1.
        pixels = F.pad(pixels, (PADDING,) * 4, value=0.0)  # [1, 32, 32]
        return pixels.mul_(2.0).sub_(1.0)  # [1, 32, 32] in [-1, 1]


@dataclass(frozen=True)
class FashionDatasets:
    """Separate views: augmented train, deterministic validation and official test."""

    train: FashionDataset
    val: FashionDataset
    test: FashionDataset


@dataclass(frozen=True)
class FashionLoaders:
    """Each iterator yields a float32 [B, 1, 32, 32] image tensor, without labels."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_datasets(config: DataConfig) -> FashionDatasets:
    """Split only the official training set; keep official test images separate."""
    raw_train = FashionMNIST(root=config.root, train=True, download=config.download)
    train_ids, val_ids = _split_indices(raw_train.targets, config.val_size, config.seed)
    raw_test = FashionMNIST(root=config.root, train=False, download=config.download)
    return FashionDatasets(
        train=FashionDataset(
            raw_train, train_ids, training=True, horizontal_flip_p=config.horizontal_flip_p,
        ),
        val=FashionDataset(raw_train, val_ids),
        test=FashionDataset(raw_test),
    )


def build_dataloaders(config: DataConfig) -> FashionLoaders:
    """Build the data pipeline; call seed_data(config.seed) once before training.

    A separate generator per loader prevents validation iteration from changing
    the training shuffle. Windows callers must use an ``if __name__`` guard
    when num_workers > 0. Validation/test never shuffle or drop their last batch.
    """
    datasets = build_datasets(config)
    if config.drop_last_train and len(datasets.train) < config.batch_size:
        raise ValueError("drop_last_train would produce an empty training loader")

    def make_loader(dataset: FashionDataset, *, training: bool, offset: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=training,
            drop_last=config.drop_last_train if training else False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(config.seed + offset),
            # Avoid inheriting threaded PyTorch state via fork on Linux; also
            # exercise the same process-start behavior used by Windows.
            multiprocessing_context="spawn" if config.num_workers > 0 else None,
        )

    return FashionLoaders(
        train=make_loader(datasets.train, training=True, offset=0),
        val=make_loader(datasets.val, training=False, offset=1),
        test=make_loader(datasets.test, training=False, offset=2),
    )


def to_display(images: Tensor, *, crop: bool = True) -> Tensor:
    """Map normalized tensors to [0, 1], optionally removing the 2-pixel border.

    Supports [1, 32, 32] and [B, 1, 32, 32]. Does not mutate its input.
    """
    if images.ndim not in (3, 4) or tuple(images.shape[-3:]) != (1, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("Expected [1, 32, 32] or [B, 1, 32, 32]")
    result = ((images.detach().to(dtype=torch.float32) + 1.0) / 2.0).clamp(0.0, 1.0)
    if crop:
        result = result[..., PADDING:-PADDING, PADDING:-PADDING]
    return result
