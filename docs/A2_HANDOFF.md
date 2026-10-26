# Serah Terima Modul (Handoff) - Role 2: Data Pipeline

## Spesifikasi Tensor
- **Ukuran Image**: `[B, 1, 32, 32]` (sudah dipadding).
- **Rentang Pixel**: `[-1.0, 1.0]`. Pixel pad hitam bernilai `-1.0`.
- **Output Iterator**: Mengembalikan tensor gambar saja, **BUKAN** `(image, label)`. Label dibuang agar iterasi training U-Net dan DDPM mulus tanpa unpack.

## API Yang Dapat Digunakan (Role 3 & 4)
- Gunakan `loaders = build_dataloaders(config)`
- Iterasi via `for images in loaders.train:`
- Gunakan `to_display(images)` sebelum log gambar ke TensorBoard / wandb untuk memulihkan rentang `[0, 1]` dan kembali ke `28x28`.
