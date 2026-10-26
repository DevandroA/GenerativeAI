# Mini DDPM Fashion-MNIST — Data Pipeline (A2)

Implementasi role 2 milik Hafid: penyiapan Fashion-MNIST untuk Mini DDPM
tanpa kondisi teks. Repository masih kosong saat pekerjaan A2 dimulai;
branch ini menyediakan data pipeline, bukan implementasi model atau training.

## Menjalankan

Gunakan Python 3.10+ dan jalankan perintah dari folder repository.
Pilih interpreter yang sama di terminal dan notebook VS Code.

```bash
python -m pip install -r requirements.txt
python -m src.data.inspect --config configs/default.yaml
python -m pytest -q
```

Untuk GPU, pasang pasangan PyTorch/torchvision yang sesuai dari
[petunjuk resmi PyTorch](https://pytorch.org/get-started/locally/).
Pipeline data sendiri dapat diperiksa sepenuhnya di CPU.

Notebook penjelasan dan visualisasi:
`notebooks/02_data_pipeline.ipynb`. Untuk menjalankannya, pasang kernel:

```bash
python -m pip install ipykernel
```

## Apa yang dikerjakan

| Tahap | Perilaku |
| --- | --- |
| Sumber | Fashion-MNIST resmi melalui pembaca dataset torchvision; tanpa pretrained model |
| Split | 55.000 train + 5.000 validasi dari 60.000 gambar training resmi |
| Stratifikasi | Dengan konfigurasi default: 5.500 train dan 500 validasi per kelas |
| Test | 10.000 gambar test resmi terpisah; tidak masuk split train/validasi |
| Augmentasi | Horizontal flip dengan probabilitas 0,5 pada train saja |
| Padding | Tambahkan 2 piksel hitam di setiap sisi: 28×28 menjadi 32×32 |
| Normalisasi | `x / 255 → [0,1]`, lalu `2x - 1 → [-1,1]` |
| Keluaran dataset | Tensor `float32 [1,32,32]` saja, tanpa tuple label |
| Keluaran DataLoader | Tensor `float32 [B,1,32,32]` |
| Visualisasi | Denormalisasi ke [0,1], dengan opsi crop kembali ke 28×28 |

Split menggunakan seed tetap di konfigurasi. Label hanya digunakan untuk
stratifikasi dan pemeriksaan distribusi, tidak diberikan sebagai kondisi DDPM.
Train dan validasi memiliki view transform terpisah: mengaktifkan augmentasi
train tidak mengubah validasi. Tidak ada rotasi besar atau vertical flip.

## Integrasi untuk anggota 1 dan 3

```python
from src.data import build_dataloaders, load_data_config, seed_data

config = load_data_config("configs/default.yaml")
seed_data(config.seed)  # Panggil sekali sebelum memulai run baru.
loaders = build_dataloaders(config)

for x0 in loaders.train:
    x0 = x0.to(device)  # [B,1,32,32], float32, [-1,1]
    # loss = diffusion.loss(model, x0)  # Diimplementasikan A1/A3.

for x0 in loaders.val:
    # Evaluasi tanpa update bobot; timestep/noise ditangani A3.
    pass
```

Jangan gunakan `for images, labels in loaders.train`: iterator mengembalikan
tensor gambar saja. Atribut `loaders.train.dataset.targets` tersedia untuk
diagnostik atau pembangunan classifier evaluator yang terpisah. Classifier
tersebut bukan bagian implementasi A2 ini.

`device`, model, loss, optimizer, serta validasi diffusion dikelola A1/A3.
Untuk membandingkan validation loss antar-epoch, A3 perlu mengendalikan
noise/timestep evaluasi; data validasi yang tetap saja belum cukup.

Seed awal membuat split dan run ulang dapat direproduksi pada konfigurasi
yang sama. Resume persis di tengah training membutuhkan penyimpanan state
RNG dan generator DataLoader di checkpoint A3; pengaturan seed awal saja
tidak menjamin urutan augmentasi lanjutan yang identik.

## Konfigurasi dan penggunaan offline

Semua pengaturan data ada di `configs/default.yaml`. Bagian model/training
dapat ditambahkan anggota lain tanpa mengubah pembaca konfigurasi data.
Jalankan dari root repository; `data.root` relatif terhadap working directory.
Default `num_workers: 0` sesuai untuk awal penggunaan Windows/notebook.
Jika dinaikkan pada Windows, panggil training di dalam guard
`if __name__ == "__main__":`. Script inspeksi sudah memiliki guard tersebut.
Saat worker diaktifkan, loader memakai metode `spawn` untuk menghindari
pewarisan state thread PyTorch melalui `fork`; startup worker sedikit lebih mahal.

Data diunduh dari mirror HTTPS milik pembuat Fashion-MNIST dan diperiksa
menggunakan checksum bawaan torchvision. File mentah tidak di-commit.
Untuk Kaggle/offline, letakkan empat file IDX yang sudah diekstrak di:

```text
data/FashionMNIST/raw/
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
```

```bash
python -m src.data.inspect --no-download
python -m src.data.inspect --root /path/to/cache --no-download
```

## Hasil pemeriksaan

Perintah inspeksi menghasilkan `reports/data/`:

- `summary.json`: konfigurasi, versi runtime, jumlah gambar, distribusi
  kelas, bentuk/rentang satu batch per split, dan pemeriksaan overlap.
- `train_preview.png`, `val_preview.png`, `test_preview.png`: contoh data
  nyata setelah pipeline, ditampilkan kembali dalam [0,1].
- `class_distribution.png`: sebaran kelas train/validasi/test.
- `download_manifest.json`: bukti checksum empat arsip sumber yang diperiksa
  pada sesi eksekusi ini; file ini merupakan arsip verifikasi unduhan.
- `tests.txt`: hasil pengujian otomatis dari sesi eksekusi ini.
- `crosscheck.json`: hasil pemeriksaan lanjutan seluruh 70.000 gambar melalui
  DataLoader default, termasuk batch terakhir, rentang, padding dan checksum.
- `split_indices.npz`: indeks split untuk arsip lokal, dibuat ulang dari seed
  dan tidak di-commit.

Inspeksi mengonsumsi satu batch dari tiap loader. Buat loader baru untuk run
training setelah inspeksi agar posisi iterator/RNG tidak terbawa.
Tes otomatis memakai gambar sintetis kecil yang ditandai sebagai fixture;
laporan `reports/data/` harus dibuat dari dataset Fashion-MNIST nyata.

Pada eksekusi yang disertakan: 20 tes lulus, 1 tes multi-worker dilewati karena
runtime melarang socket IPC. Tujuh sel kode notebook dijalankan pada dataset
nyata melalui IPython dalam satu proses dan outputnya disimpan. Pengujian
kernel Jupyter terpisah terhalang pembatasan socket lingkungan; ini tidak
menghalangi eksekusi pipeline CPU dengan `num_workers=0`.

## Jejak vibe coding dan pembagian tim

- Implementasi data: `src/data/dataset.py`.
- Inspeksi/visualisasi: `src/data/inspect.py`.
- Pengujian role A2: `tests/test_data_pipeline.py`, `tests/test_shapes.py`.
- Serah terima dan penjelasan: `docs/A2_HANDOFF.md`.
- Log sesi aktual: `logs/llm_interaction/2026-10-26_A2_data_pipeline.md`.

Kode ditulis modular melalui sesi LLM ini, tidak diambil dari tutorial atau
repository implementasi diffusion. Dokumentasi resmi digunakan untuk kontrak
API dasar: [FashionMNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)
dan [DataLoader](https://docs.pytorch.org/docs/stable/data.html).
Hafid tetap perlu menjalankan ulang dan memahami modul sebelum presentasi.
