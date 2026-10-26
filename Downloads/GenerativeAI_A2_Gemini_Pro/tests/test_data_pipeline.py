import pytest
import torch
from src.data.dataset import load_data_config, build_datasets, build_dataloaders

def test_dataset_split_lengths():
    config = load_data_config()
    train_ds, val_ds, test_ds = build_datasets(config)
    
    assert len(train_ds) == 55000, "Train dataset harus berukuran 55000"
    assert len(val_ds) == 5000, "Val dataset harus berukuran 5000"
    assert len(test_ds) == 10000, "Test dataset harus berukuran 10000"

def test_stratified_split():
    config = load_data_config()
    _, val_ds, _ = build_datasets(config)
    
    # Tiap kelas harus berjumlah 500 di validation
    class_counts = torch.bincount(val_ds.targets)
    assert (class_counts == 500).all(), "Split stratified gagal, distribusi tidak merata"