import os
import json
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from src.data.dataset import load_data_config, build_dataloaders, to_display

def audit_loader(loader, name):
    print(f"Memproses {name} loader...")
    total_images = 0
    min_val = float('inf')
    max_val = float('-inf')
    has_nan_or_inf = False
    last_batch_size = 0
    
    for batch in loader:
        b_size = batch.size(0)
        total_images += b_size
        last_batch_size = b_size
        
        if not torch.isfinite(batch).all():
            has_nan_or_inf = True
            
        min_val = min(min_val, batch.min().item())
        max_val = max(max_val, batch.max().item())
        
    return {
        "total_images": total_images,
        "total_batches": len(loader),
        "last_batch_size": last_batch_size,
        "min_val": round(min_val, 4),
        "max_val": round(max_val, 4),
        "has_nan_or_inf": has_nan_or_inf
    }

def save_preview(loader, name):
    # Ambil satu batch untuk divisualisasikan
    batch = next(iter(loader))
    images = batch[:16] # Ambil 16 gambar pertama saja
    
    # Gunakan fungsi to_display untuk hapus padding (jadi 28x28) dan kembalikan ke rentang [0, 1]
    disp_images = to_display(images, crop=True)
    
    grid = make_grid(disp_images, nrow=4, padding=2, normalize=False)
    plt.figure(figsize=(5, 5))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.axis('off')
    plt.title(f"Preview {name.capitalize()} (28x28, Rentang [0,1])")
    plt.tight_layout()
    
    path = f"reports/data/{name}_preview.png"
    plt.savefig(path)
    plt.close()

def run_full_inspection():
    os.makedirs("reports/data", exist_ok=True)
    config = load_data_config()
    loaders = build_dataloaders(config)
    
    print("=== AUDIT KESELURUHAN DATASET NYATA ===")
    results = {}
    
    for name, loader in [("train", loaders.train), ("val", loaders.val), ("test", loaders.test)]:
        results[name] = audit_loader(loader, name)
        save_preview(loader, name)
        
    with open("reports/data/summary.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n=== RINGKASAN HASIL ===")
    print(json.dumps(results, indent=2))
    
    print("\n=== PEMERIKSAAN BATCH TERAKHIR ===")
    print(f"Target  -> Train: 88, Val: 8, Test: 16")
    print(f"Aktual  -> Train: {results['train']['last_batch_size']}, Val: {results['val']['last_batch_size']}, Test: {results['test']['last_batch_size']}")
    print("Gambar preview telah disimpan di folder reports/data/")

if __name__ == "__main__":
    run_full_inspection()