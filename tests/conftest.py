"""Offline fixtures: synthetic pixels only, never presented as real Fashion-MNIST."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import dataset as data_module


class FakeFashionMNIST:
    """Small raw PIL dataset to exercise data semantics without network requests."""

    transform = None
    target_transform = None

    def __init__(self, root: str = "", train: bool = True, download: bool = False) -> None:
        self.train = train
        self.targets = torch.arange(10).repeat_interleave(12 if train else 4)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        # Asymmetric image detects unwanted flips; both intensity endpoints exist.
        pixels = np.zeros((28, 28), dtype=np.uint8)
        pixels[:, :7] = 255
        pixels[0, 10] = index % 256
        return Image.fromarray(pixels), int(self.targets[index])


@pytest.fixture
def raw_data() -> FakeFashionMNIST:
    return FakeFashionMNIST()


@pytest.fixture
def offline_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module, "FashionMNIST", FakeFashionMNIST)
