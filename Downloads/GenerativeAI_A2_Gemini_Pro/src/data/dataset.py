import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import yaml

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

class DataConfig:
    def __init__(self, config_dict):
        self.seed = config_dict.get('seed', 42)
        data = config_dict.get('data', {})
        self.root = data.get('root', 'data')
        self.val_size = data.get('val_size', 5000)
        self.batch_size = data.get('batch_size', 128)
        self.num_workers = data.get('num_workers', 0)
        self.pin_memory = data.get('pin_memory', False)
        self.download = data.get('download', True)
        self.horizontal_flip_p = data.get('horizontal_flip_p', 0.5)
        self.drop_last_train = data.get('drop_last_train', False)

def load_data_config(path='configs/default.yaml'):
    with open(path, 'r') as f:
        return DataConfig(yaml.safe_load(f))

def seed_data(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class FashionDataset(Dataset):
    """
    Dataset wrapper untuk mengembalikan TENSOR SAJA [1, 32, 32] di rentang [-1, 1].
    Sesuai kontrak DDPM (tanpa kondisi teks/label).
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
        # Ekstrak target untuk kebutuhan stratifikasi dan inspeksi
        if hasattr(subset, 'dataset') and hasattr(subset, 'indices'):
            self.targets = torch.tensor([subset.dataset.targets[i] for i in subset.indices])
        else:
            self.targets = subset.targets.clone().detach() if isinstance(subset.targets, torch.Tensor) else torch.tensor(subset.targets)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, _ = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img  # Return ONLY image tensor

def get_transforms(is_train: bool, config: DataConfig) -> transforms.Compose:
    t_list = []
    if is_train and config.horizontal_flip_p > 0:
        t_list.append(transforms.RandomHorizontalFlip(p=config.horizontal_flip_p))
    
    t_list.extend([
        transforms.ToTensor(), # [0, 1]
        transforms.Pad(2, fill=0), # Pad hitam (0) -> 32x32
        transforms.Normalize([0.5], [0.5]) # [0, 1] -> [-1, 1]. Hitam jadi -1
    ])
    return transforms.Compose(t_list)

def build_datasets(config: DataConfig):
    seed_data(config.seed)
    
    # Download raw dataset
    base_train_val = datasets.FashionMNIST(root=config.root, train=True, download=config.download)
    test_raw = datasets.FashionMNIST(root=config.root, train=False, download=config.download)
    
    # Stratified Split 55k Train, 5k Val
    targets = base_train_val.targets.numpy()
    classes = np.unique(targets)
    val_per_class = config.val_size // len(classes)
    
    train_idx, val_idx = [], []
    for c in classes:
        c_idx = np.where(targets == c)[0]
        np.random.shuffle(c_idx)
        val_idx.extend(c_idx[:val_per_class])
        train_idx.extend(c_idx[val_per_class:])
        
    train_subset = Subset(base_train_val, train_idx)
    val_subset = Subset(base_train_val, val_idx)
    
    # Apply transforms
    train_ds = FashionDataset(train_subset, get_transforms(True, config))
    val_ds = FashionDataset(val_subset, get_transforms(False, config))
    test_ds = FashionDataset(test_raw, get_transforms(False, config))
    
    return train_ds, val_ds, test_ds

class DataLoaderWrapper:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test

def build_dataloaders(config: DataConfig):
    train_ds, val_ds, test_ds = build_datasets(config)
    
    # Generator independen untuk validasi agar tidak mengganggu RNG training
    g_val = torch.Generator()
    g_val.manual_seed(config.seed)
    
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, 
                          num_workers=config.num_workers, pin_memory=config.pin_memory, 
                          drop_last=config.drop_last_train)
                          
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, 
                        num_workers=config.num_workers, pin_memory=config.pin_memory,
                        drop_last=False, generator=g_val)
                        
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, 
                         num_workers=config.num_workers, pin_memory=config.pin_memory,
                         drop_last=False)
                         
    return DataLoaderWrapper(train_dl, val_dl, test_dl)

def to_display(images: torch.Tensor, crop=True):
    """
    Mengembalikan tensor gambar ke rentang [0, 1] dan menghapus padding.
    """
    imgs = (images * 0.5) + 0.5
    imgs = torch.clamp(imgs, 0, 1)
    if crop:
        imgs = imgs[..., 2:-2, 2:-2] # 32x32 -> 28x28
    return imgs
