# Serah terima A2 — Data Pipeline

Pemilik role: Hafid. Implementasi dibangun bersama LLM pada sesi permintaan
role data pipeline. Tidak ada klaim model DDPM telah dilatih pada branch ini.

## Penjelasan alur untuk presentasi

1. Fashion-MNIST dibaca sebagai gambar abu-abu 28×28 dan label kelas.
2. Dari training resmi, dipilih 5.000 gambar validasi secara stratified
   dengan seed 42; sisanya 55.000 untuk training. Tidak ada indeks yang sama
   pada kedua subset. Test resmi sebanyak 10.000 tetap terpisah.
3. Gambar dikonversi menjadi tensor float32 dan dibagi 255.
4. Hanya gambar training yang boleh di-flip horizontal secara acak.
5. Padding hitam 2 piksel per sisi menghasilkan ukuran 32×32, sehingga
   ukuran feature map dapat turun menjadi 16×16 dan 8×8 tanpa ukuran ganjil.
6. Normalisasi `2x - 1` menghasilkan rentang [-1,1]. Padding dilakukan
   sebelumnya agar batas hitam menjadi -1, sesuai latar belakang gambar.
7. DataLoader menggabungkan gambar menjadi batch [B,1,32,32]. Label tidak
   ikut dikirim karena DDPM yang dirancang belum menggunakan kondisi kelas.

## Keputusan yang perlu dapat dijelaskan Hafid

- Normalisasi bukan memakai mean/std ImageNet: model DDPM dilatih dari nol
  untuk data grayscale, dan kontrak proyek menetapkan rentang [-1,1].
- Stratifikasi menjaga proporsi kategori untuk train/validasi; penggunaan
  label dalam split tidak membuat DDPM menjadi class-conditional.
- Tidak ada augmentasi pada validasi/test supaya gambar evaluasi tetap.
- Tidak ada resize ke 32×32: padding mempertahankan bentuk piksel asli.
- Horizontal flip bisa dimatikan lewat konfigurasi untuk eksperimen tanpa
  augmentasi. Pipeline tidak menambahkan augmentasi lain secara tersembunyi.
- `to_display` hanya mengubah tampilan kembali ke [0,1] dan crop 28×28;
  jangan memberi hasil denormalisasi tersebut ke DDPM saat training.

## Untuk A1: kontrak model

Terima `x_t` [B,1,32,32] dan `t` [B] bertipe long. Output prediksi noise
harus berbentuk [B,1,32,32]. `q_sample` dan timestep bukan tanggung jawab A2.

## Untuk A3: kontrak training

Terima `x0` langsung dari `loaders.train`, lalu pindahkan ke device yang
digunakan model. Data masih di CPU. Batch terakhir bisa lebih kecil dari
128: gunakan `x0.shape[0]` saat membuat timestep/noise. Default tidak membuang
batch terakhir, dan validasi/test tidak pernah membuangnya.

Latih model pada train, gunakan validasi untuk tuning, dan simpan test untuk
evaluasi akhir yang sesuai. Tes/visualisasi data bukan hasil generasi model.
Classifier untuk mengevaluasi sebaran sampel generatif perlu dilatih dan
diuji secara terpisah; label mentah dapat diakses melalui atribut `targets`.

## Untuk A4: bukti yang dikumpulkan

- Log sesi, prompt aktual, serta ringkasan keputusan dan hasil eksekusi.
- Hasil `python -m pytest -q`.
- `reports/data/summary.json`, grid preview dan grafik distribusi.
- Commit branch `feat/data`, dengan tautan ke file log di badan commit.
- Share link percakapan diisi setelah pengguna membuatnya. Jangan membuat
  tautan, error, hasil training, atau aktivitas anggota yang tidak terjadi.

## Batas pengujian

Tes otomatis mencakup normalisasi, padding, augmentasi train-only,
stratifikasi, pemisahan test, reproducibility loader, batch terakhir,
multi-worker, konfigurasi, ekspor laporan, dan kontrak tensor A2.
Pemeriksaan dataset nyata melaporkan bentuk/rentang satu batch per split
serta membership seluruh indeks split. Pengujian U-Net, rumus diffusion,
overfit satu batch dan training GPU dilakukan setelah modul A1/A3 tersedia.
