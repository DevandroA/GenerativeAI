# Ringkasan Hasil Mini DDPM (Fashion-MNIST 32x32)

Semua kode ada di satu notebook: `notebooks/Mini_DDPM_Full.ipynb`. Dokumen ini hanya menunjukkan letak hasil dan merangkum angkanya.

## 1. Arsitektur yang dipilih

**Mini Diffusion Model (DDPM)**, salah satu pilihan di guidebook, dengan dua tantangan yang diminta: noise schedule dan timestep embedding.

- Data: Fashion-MNIST dipad ke 32x32, rentang [-1, 1], split 55.000 train / 5.000 val / 10.000 test (stratified, seed 42).
- Difusi: T = 1000, loss noise-MSE `||eps - eps_theta(x_t, t)||^2`. Schedule yang dibandingkan: linear dan cosine.
- Timestep embedding: sinusoidal (dim 64), lalu MLP ke 256.
- Jaringan: U-Net kecil (channel dasar 64, multiplier [1, 2], ResBlock GroupNorm, self-attention di bottleneck), sekitar 3,39 juta parameter (3,05 juta tanpa time embedding).
- Training: AdamW lr 2e-4, batch 128, AMP, grad clip 1.0, EMA 0,999. Val loss dihitung dengan bobot EMA dan seed tetap, checkpoint terbaik dipilih dari val loss.
- Sampling: DDIM 50 langkah (eta = 0). Sampler DDPM ancestral belum diimplementasikan.

## 2. Letak hasil

| Yang dicari | Lokasi |
|---|---|
| Grafik loss train/val per run | `reports/<run>/loss_curve.png` (`<run>` = `linear`, `cosine`, `linear_notime`) |
| Perbandingan loss antar run | `reports/compare_loss.png` |
| Grid sampel | `reports/<run>/generated_grid.png` |
| Gambar proses denoising | `reports/<run>/denoising.png` |
| Log loss per epoch | `checkpoints/<run>/training_log.csv` (tidak di-commit) |
| Hasil tuning (tabel) | `reports/tuning/results.csv` |
| Hasil tuning (kurva val loss) | `reports/tuning/val_curves.png` |
| Hasil tuning (sampel per konfigurasi) | `reports/tuning/samples.png` |
| Kesimpulan tuning | notebook, bagian 5, sel "Kesimpulan tuning" |
| Pratinjau dan ringkasan data | `reports/data/` |

## 3. Loss akhir per run (50 epoch, epoch 50)

| Run | Schedule | Time emb | Train loss | Val loss (EMA) | Kualitas sampel |
|---|---|---|---|---|---|
| `linear` | linear | ya | 0,02672 | tidak dicatat | jelas |
| `cosine` | cosine | ya | 0,04852 | 0,04590 | jelas |
| `linear_notime` | linear | tidak | 0,02775 | 0,02587 | gagal (noise) |

Catatan cara membaca:
- Log run `linear` hanya punya kolom `epoch,loss` (versi lama), jadi tidak ada val loss. Untuk perbandingan val yang adil perlu dilatih ulang.
- Loss antar schedule **tidak sebanding**: cosine memberi lebih banyak bobot ke timestep yang sulit, sehingga angkanya lebih tinggi tanpa berarti lebih buruk.
- `linear_notime` punya loss yang rendah tetapi sampelnya gagal. Jadi loss saja tidak cukup untuk menilai kualitas, dan peran timestep embedding terlihat dari sini.
- Belum ada metrik kuantitatif kualitas sampel (FID/KID). Evaluasi saat ini loss dan inspeksi visual.

## 4. Tuning learning rate dan batch size

Schedule linear, time embedding aktif, 15 epoch, satu seed per konfigurasi. Metrik: val loss terbaik (EMA).

| Run | Val loss | Selisih dari baseline | Waktu (mnt) |
|---|---|---|---|
| lr 8e-4, batch 128 | 0,02612 | -2,6% | 13,3 |
| lr 2e-4, batch 64 | 0,02613 | -2,6% | 16,1 |
| lr 4e-4, batch 128 | 0,02617 | -2,4% | 13,4 |
| lr 2e-4, batch 128 (baseline) | 0,02682 | 0 | 13,5 |
| lr 1e-4, batch 128 | 0,02794 | +4,2% | 13,4 |
| lr 2e-4, batch 256 | 0,02925 | +9,1% | 12,2 |

Tiga teratas hanya berbeda sekitar 0,2%, jadi tidak ada pemenang tunggal. Pilihan untuk dipakai selanjutnya: lr 4e-4, batch 128. Semua run masih membaik di epoch 15, dan angka ini belum diulang dengan seed lain.

## 5. Repo KAIST (`Diffusion-Assignment1-DDPM`)

Folder itu adalah clone repo tugas KAIST CS492(D) Programming Assignment 1 dan **belum dikerjakan**: bagian `TODO` di `ddpm.py`, `network.py`, `scheduler.py`, dan `model.py` masih kosong, dan tidak ada jawaban di dalamnya.

Tugasnya berbeda dari proyek ini: Task 1 memakai Swiss-roll 2D dengan metrik Chamfer Distance, Task 2 memakai AFHQ 64x64 dengan FID dan training sekitar 14 jam. Itu bukan Fashion-MNIST dan bukan yang diminta guidebook mata kuliah. Karena itu tidak ada jawaban yang bisa dirangkum di sini tanpa mengerjakan tugas tersebut dari awal.
