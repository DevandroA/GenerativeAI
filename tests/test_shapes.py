"""The implemented A2 portion of the project tensor contract; no model stubs."""

import torch

from src.data import DataConfig, build_dataloaders


def test_dataset_and_dataloader_image_only_contract(offline_source):
    loaders = build_dataloaders(DataConfig(val_size=20, batch_size=16))
    for loader in (loaders.train, loaders.val, loaders.test):
        image = loader.dataset[0]
        batch = next(iter(loader))
        assert isinstance(image, torch.Tensor) and tuple(image.shape) == (1, 32, 32)
        assert isinstance(batch, torch.Tensor) and tuple(batch.shape) == (16, 1, 32, 32)
        assert batch.dtype == torch.float32 and torch.isfinite(batch).all()
        assert batch.min() >= -1 and batch.max() <= 1
