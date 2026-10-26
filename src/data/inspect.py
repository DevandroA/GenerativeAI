"""Run ``python -m src.data.inspect`` to audit and visualize the real pipeline."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import Tensor

from .dataset import (
    CLASS_NAMES, DataConfig, FashionLoaders, build_dataloaders,
    load_data_config, seed_data, to_display,
)
from .dataset import FashionMNIST


def save_grid(images: Tensor, path: Path, title: str, *, crop: bool = False) -> None:
    """Write up to 16 images without changing their source tensor."""
    displayed = to_display(images[:16], crop=crop).cpu()
    fig, axes = plt.subplots(4, 4, figsize=(7, 7))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < len(displayed):
            axis.imshow(displayed[index, 0].numpy(), cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(loaders: FashionLoaders, config: DataConfig, output_dir: Path) -> dict[str, Any]:
    """Audit split membership, sample batches and retain reproducible split indices.

    Batch checks are smoke tests, not a claim that every sample was scanned.
    This function consumes one batch from each loader; use fresh loaders for training.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ids = np.asarray(loaders.train.dataset.indices)
    val_ids = np.asarray(loaders.val.dataset.indices)
    if np.intersect1d(train_ids, val_ids).size:
        raise AssertionError("Training and validation indices overlap")
    expected_size = len(loaders.train.dataset.base_dataset)
    if not np.array_equal(np.sort(np.concatenate((train_ids, val_ids))), np.arange(expected_size)):
        raise AssertionError("Train/validation must cover the official training set exactly once")

    summary: dict[str, Any] = {
        "dataset": "Fashion-MNIST",
        "source_repository": "https://github.com/zalandoresearch/fashion-mnist",
        "download_mirrors": list(getattr(FashionMNIST, "mirrors", [])),
        "config": asdict(config),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "train_val_overlap": 0,
        "test_source": "official train=False split; not included in train/validation",
        "batch_contract": "float32 [B, 1, 32, 32], image tensor only, range [-1, 1]",
        "verification_scope": "split membership and one batch per split",
        "splits": {},
    }
    counts_by_split = []
    for name in ("train", "val", "test"):
        loader = getattr(loaders, name)
        batch = next(iter(loader))
        if batch.dtype != torch.float32 or batch.ndim != 4 or tuple(batch.shape[1:]) != (1, 32, 32):
            raise AssertionError(f"Incorrect image contract in {name}")
        if not torch.isfinite(batch).all() or batch.min() < -1 or batch.max() > 1:
            raise AssertionError(f"Invalid numeric values in {name}")
        if not (batch[..., :2, :] == -1).all() or not (batch[..., -2:, :] == -1).all():
            raise AssertionError(f"Incorrect top/bottom padding in {name}")
        if not (batch[..., :2] == -1).all() or not (batch[..., -2:] == -1).all():
            raise AssertionError(f"Incorrect left/right padding in {name}")
        counts = torch.bincount(loader.dataset.targets.long(), minlength=len(CLASS_NAMES)).tolist()
        counts_by_split.append(counts)
        summary["splits"][name] = {
            "images": len(loader.dataset),
            "batches": len(loader),
            "class_counts": dict(zip(CLASS_NAMES, counts)),
            "checked_batch_shape": list(batch.shape),
            "checked_batch_min": float(batch.min()),
            "checked_batch_max": float(batch.max()),
            "augmentation": "horizontal flip" if loader.dataset.training and config.horizontal_flip_p > 0 else "none",
        }
        save_grid(batch, output_dir / f"{name}_preview.png", f"{name}: normalized pipeline, displayed in [0, 1]")

    positions = np.arange(len(CLASS_NAMES))
    fig, axis = plt.subplots(figsize=(11, 5))
    for offset, name, counts in zip((-0.26, 0.0, 0.26), ("train", "val", "test"), counts_by_split):
        axis.bar(positions + offset, counts, width=0.26, label=name)
    axis.set_xticks(positions, CLASS_NAMES, rotation=30, ha="right")
    axis.set_ylabel("Number of images")
    axis.set_title("Real Fashion-MNIST class distribution by split")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    np.savez_compressed(output_dir / "split_indices.npz", train=train_ids, val=val_ids)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """CLI entry point; guarded for multiprocessing compatibility on Windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/data"))
    parser.add_argument("--root", type=str, default=None, help="Override dataset cache directory")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    config = load_data_config(args.config)
    overrides = {
        key: value for key, value in {
            "root": args.root, "download": args.download, "num_workers": args.num_workers,
        }.items() if value is not None
    }
    config = replace(config, **overrides)
    seed_data(config.seed)
    try:
        loaders = build_dataloaders(config)
    except RuntimeError as error:
        raise SystemExit(
            f"Dataset could not be loaded: {error}\n"
            "Enable --download, or place the four extracted Fashion-MNIST IDX files "
            "inside ROOT/FashionMNIST/raw and use --no-download."
        ) from error
    summary = write_report(loaders, config, args.output_dir)
    print(json.dumps(summary, indent=2))
    print(f"Reports saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
