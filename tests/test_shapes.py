import pytest
import torch
from src.data.dataset import load_data_config, build_dataloaders

def test_tensor_contract_and_range():
    config = load_data_config()
    loaders = build_dataloaders(config)
    batch = next(iter(loaders.train))
    
    # Shape: [Batch, 1, 32, 32]
    assert batch.shape[1:] == (1, 32, 32), f"Bentuk tensor salah: {batch.shape}"
    
    # Rentang [-1, 1]
    assert batch.min() >= -1.0, "Nilai minimum di bawah -1.0"
    assert batch.max() <= 1.0, "Nilai maksimum di atas 1.0"
    
    # Padding harus -1 (pojok gambar)
    assert batch[0, 0, 0, 0].item() == -1.0, "Padding hitam gagal menjadi -1.0"
