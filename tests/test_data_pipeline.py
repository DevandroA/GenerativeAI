"""A2 behavior checks, including split isolation and train-only augmentation."""

import numpy as np
import pytest
import torch
import socket

from src.data import (
    DataConfig, FashionDataset, build_dataloaders, build_datasets,
    load_data_config, seed_data, to_display,
)
from src.data.dataset import _split_indices
from src.data.inspect import write_report


def test_normalization_padding_and_round_trip(raw_data):
    dataset = FashionDataset(raw_data)
    image = dataset[0]
    original = torch.tensor(np.asarray(raw_data[0][0]).copy()).unsqueeze(0).float() / 255
    assert image.dtype == torch.float32
    assert tuple(image.shape) == (1, 32, 32)
    assert image.min() == -1 and image.max() == 1
    assert (image[:, :2] == -1).all() and (image[:, -2:] == -1).all()
    assert (image[:, :, :2] == -1).all() and (image[:, :, -2:] == -1).all()
    assert torch.allclose(to_display(image), original, atol=1e-7)
    assert to_display(image, crop=False).shape == image.shape


def test_flip_only_applies_to_training_and_does_not_modify_raw(raw_data):
    baseline = FashionDataset(raw_data, horizontal_flip_p=1.0)
    augmented = FashionDataset(raw_data, training=True, horizontal_flip_p=1.0)
    before = baseline[0].clone()
    assert torch.equal(augmented[0], before.flip(-1))
    assert torch.equal(baseline[0], before)
    assert torch.equal(baseline[0], baseline[0])


def test_stratified_split_covers_all_training_data_without_overlap():
    targets = torch.arange(10).repeat_interleave(6000)
    train, val = _split_indices(targets, val_size=5000, seed=42)
    assert len(train) == 55000 and len(val) == 5000
    assert not set(train.tolist()).intersection(val.tolist())
    assert sorted(train.tolist() + val.tolist()) == list(range(60000))
    assert torch.equal(torch.bincount(targets[val]), torch.full((10,), 500))
    again = _split_indices(targets, val_size=5000, seed=42)
    changed = _split_indices(targets, val_size=5000, seed=43)
    assert torch.equal(val, again[1])
    assert not torch.equal(val, changed[1])


def test_split_allocation_handles_nondivisible_validation_size():
    targets = torch.tensor([0] * 7 + [1] * 3 + [2] * 11)
    train, val = _split_indices(targets, val_size=8, seed=4)
    assert len(val) == 8 and len(train) == 13
    assert torch.bincount(targets[val]).tolist() == [3, 1, 4]


def test_views_have_independent_augmentation_and_official_test_is_separate(offline_source):
    datasets = build_datasets(DataConfig(val_size=20))
    assert len(datasets.train) == 100 and len(datasets.val) == 20 and len(datasets.test) == 40
    assert datasets.train.base_dataset is datasets.val.base_dataset
    assert datasets.test.base_dataset is not datasets.train.base_dataset
    assert datasets.train.training and not datasets.val.training and not datasets.test.training
    assert datasets.train.base_dataset.train and not datasets.test.base_dataset.train
    assert torch.equal(torch.bincount(datasets.val.targets), torch.full((10,), 2))


def test_loader_reproducibility_and_validation_order(offline_source):
    config = DataConfig(val_size=20, batch_size=16)
    first = build_dataloaders(config)
    second = build_dataloaders(config)
    seed_data(config.seed)
    first_batch = next(iter(first.train))
    seed_data(config.seed)
    second_batch = next(iter(second.train))
    assert torch.equal(first_batch, second_batch)
    assert torch.equal(next(iter(first.val)), next(iter(first.val)))
    assert sum(len(batch) for batch in first.val) == 20
    assert sum(len(batch) for batch in first.test) == 40
    assert sum(len(batch) for batch in first.train) == 100


def test_validation_iteration_does_not_consume_training_shuffle(offline_source):
    config = DataConfig(val_size=20, batch_size=16, horizontal_flip_p=0)
    first, second = build_dataloaders(config), build_dataloaders(config)
    next(iter(first.val))
    assert torch.equal(next(iter(first.train)), next(iter(second.train)))


def test_drop_last_changes_training_only(offline_source):
    loaders = build_dataloaders(DataConfig(val_size=20, batch_size=16, drop_last_train=True))
    assert sum(len(batch) for batch in loaders.train) == 96
    assert sum(len(batch) for batch in loaders.val) == 20
    with pytest.raises(ValueError, match="empty training loader"):
        build_dataloaders(DataConfig(val_size=20, batch_size=128, drop_last_train=True))


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0}, {"val_size": 0}, {"num_workers": -1},
    {"horizontal_flip_p": -0.1}, {"horizontal_flip_p": 1.1},
    {"horizontal_flip_p": float("nan")}, {"seed": -1},
    {"batch_size": 1.5}, {"download": "false"},
])
def test_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        DataConfig(**kwargs)


def test_configuration_keeps_other_roles_sections(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 7\ndata:\n  val_size: 20\nmodel:\n  channels: 64\n", encoding="utf-8")
    config = load_data_config(path)
    assert config.seed == 7 and config.val_size == 20
    assert config.batch_size == 128


def test_report_generation_on_explicitly_synthetic_fixture(offline_source, tmp_path):
    config = DataConfig(val_size=20, batch_size=16)
    summary = write_report(build_dataloaders(config), config, tmp_path)
    assert summary["train_val_overlap"] == 0
    assert summary["splits"]["train"]["images"] == 100
    saved = np.load(tmp_path / "split_indices.npz")
    assert len(saved["train"]) == 100 and len(saved["val"]) == 20
    for filename in ("summary.json", "train_preview.png", "val_preview.png", "test_preview.png", "class_distribution.png"):
        assert (tmp_path / filename).stat().st_size > 0


def test_multiworker_batches_are_reproducible(offline_source):
    # PyTorch's multiprocessing tensor transfer needs local IPC sockets.
    # Some sandboxed runtimes forbid these even though normal CPU loading works.
    try:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.close()
    except (PermissionError, AttributeError) as error:
        pytest.skip(f"Multiprocessing IPC unavailable in this runtime: {error}")
    config = DataConfig(val_size=20, batch_size=16, num_workers=2)
    first = next(iter(build_dataloaders(config).train))
    second = next(iter(build_dataloaders(config).train))
    assert torch.equal(first, second)
