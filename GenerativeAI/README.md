# Mini DDPM — Fashion-MNIST 32x32 (PyTorch murni)

Model difusi skala kecil (U-Net ~3,4M parameter) yang belajar memprediksi noise pada gambar Fashion-MNIST,
tanpa library `diffusers`. Tantangan utama: **noise schedule** (linear vs cosine) dan **timestep embedding**.

## Struktur
```
notebooks/Mini_DDPM_Full.ipynb   # seluruh kode: data, model, difusi, training, evaluasi, tes
configs/default.yaml             # konfigurasi data (batch size, split, augmentasi)
checkpoints/<run>/               # best_model.pth, training_log.csv, config.json   (tidak di-commit)
reports/<run>/                   # generated_grid.png, denoising.png, loss_curve.png
reports/compare_loss.png         # perbandingan kurva loss antar run
reports/tuning/                  # hasil tuning lr & batch size (results.csv, val_curves.png, samples.png)
reports/data/                    # ringkasan & pratinjau dataset
docs/A2_HANDOFF.md               # kontrak tensor data
docs/HASIL_PROYEK.md             # ringkasan hasil, letak grafik, arsitektur, tuning
logs/llm_interaction/            # log penggunaan LLM
data/                            # Fashion-MNIST (diunduh otomatis, tidak di-commit)
```

## Cara menjalankan
```
pip install -r requirements.txt
```
Buka `notebooks/Mini_DDPM_Full.ipynb`, pilih kernel Python yang punya PyTorch, lalu **Run All** (training default mati).
Untuk melatih, di bagian *4. Training* set `RUN_TRAINING = True` dan atur `EXPERIMENTS` (GPU CUDA, ~50 detik/epoch di RTX 3060 Ti).

## Hasil (50 epoch, DDIM 50 langkah)
| Run | val loss | Sampel |
|---|---|---|
| `linear` | – | jelas |
| `cosine` | 0,0459 | jelas |
| `linear_notime` (tanpa timestep embedding) | 0,0259 | gagal (noise) |

Loss antar schedule tidak sebanding; tanpa timestep embedding loss tampak bagus tetapi sampling gagal. Detail di notebook bagian 6.

## Tuning hyperparameter (15 epoch, schedule linear)
| Run | val loss |
|---|---|
| lr 8e-4, batch 128 | 0,02612 |
| lr 2e-4, batch 64 | 0,02613 |
| lr 4e-4, batch 128 | 0,02617 |
| lr 2e-4, batch 128 (baseline) | 0,02682 |
| lr 1e-4, batch 128 | 0,02794 |
| lr 2e-4, batch 256 | 0,02925 |

Tiga teratas berbeda sekitar 0,2% (satu seed), jadi belum ada pemenang pasti. Detail di notebook bagian 5 dan `docs/HASIL_PROYEK.md`.
