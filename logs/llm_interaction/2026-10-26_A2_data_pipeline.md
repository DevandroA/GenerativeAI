# Log Interaksi LLM — A2 Data Pipeline

- Tanggal sesi: 26/10/2026, mengikuti tanggal percakapan pengguna.
- Pemilik role: Hafid, anggota 2.
- Target: data pipeline Fashion-MNIST untuk Mini DDPM sesuai rancangan PDF.
- Metode: implementasi kode oleh LLM, dilanjutkan eksekusi/pengujian pada sesi
  yang sama. Tidak mengklaim Hafid sudah menjalankan atau menjelaskan ulang kode.
- Share link sesi: BELUM TERSEDIA; pengguna perlu membuat dan memasukkannya.

## 1. Prompt pengguna yang benar-benar diterima

Permintaan awal:

> https://github.com/DevandroA/GenerativeAI.git
> ini github nya , dan saya role nya data pipeline nomor 2 tolong kerjakan

Arahan berikutnya:

> https://github.com/zalandoresearch/fashion-mnist
> untuk dataset nya pake yang ini , lanjutkan eksekusinya

Rancangan PDF sebelumnya menetapkan input grayscale 28×28, padding ke 32×32,
normalisasi [-1,1], split train/validasi, augmentasi aman dan DataLoader gambar
tanpa label untuk DDPM tanpa kondisi teks. Judul PDF memuat Text-to-Image,
tetapi modul A2 mengikuti isi model unconditional, bukan mengklaim conditioning.

## 2. Pemeriksaan awal dan keputusan implementasi

- Repository berhasil di-clone; `git ls-remote` tidak mengembalikan ref dan
  `git status` melaporkan belum ada commit. Tidak ada modul existing untuk diubah.
- Dibuat branch lokal `feat/data` sesuai pembagian pada rancangan.
- Implementasi dibatasi pada role A2 dan berkas pendukung integrasinya.
- PyTorch, torchvision dan pytest belum tersedia, lalu dipasang untuk eksekusi.
- Dokumentasi resmi FashionMNIST/DataLoader dipakai untuk kontrak API dasar,
  bukan menyalin tutorial atau implementasi diffusion dari repository orang lain.
- Pembaca FashionMNIST torchvision dipakai untuk format/unduhan resmi.
  Subclass hanya mengalihkan mirror HTTP default ke HTTPS milik Zalando;
  daftar resource dan checksum tetap berasal dari torchvision.
- Sumber aktual: folder `data/fashion` pada repository Zalando yang diminta pengguna.
- Normalisasi, split stratified, transform view, augmentasi, loader, inspeksi dan
  pengujian ditulis sebagai kode proyek pada sesi ini.

## 3. Kode dan alasan desain

| Berkas | Keputusan |
| --- | --- |
| `src/data/dataset.py` | Konfigurasi bertipe, split exact-size dengan seed, train-only flip, padding sebelum normalisasi, loader image-only |
| `src/data/inspect.py` | Audit membership split, satu batch per split, grafik distribusi, grid dan ekspor indeks |
| `configs/default.yaml` | Seed 42, validasi 5.000, batch 128, workers 0, flip 0,5 |
| `tests/test_data_pipeline.py` | Cek semantik data, kebocoran split, reproducibility, konfigurasi dan laporan |
| `tests/test_shapes.py` | Kontrak tensor A2; tidak membuat stub model yang pura-pura lolos |
| `notebooks/02_data_pipeline.ipynb` | Penjelasan 7 sel kode dengan output aktual dari Fashion-MNIST nyata |

Label dipertahankan sebagai metadata untuk stratifikasi dan diagnostik, tetapi
tidak ikut output iterator training. Train/validasi berbagi sumber raw, bukan
objek transform mutable. Official test berasal dari `train=False`.

## 4. Debugging dan keterbatasan yang benar-benar terjadi

### A. Tes multi-worker tidak dapat berkomunikasi di runtime ini

Pesan utama traceback aktual dari proses worker:

```text
File "multiprocessing/resource_sharer.py", line 126, in _start
    self._listener = Listener(authkey=process.current_process().authkey, backlog=128)
File "multiprocessing/connection.py", line 601, in __init__
    self._socket = socket.socket(getattr(socket, family))
File "socket.py", line 233, in __init__
    _socket.socket.__init__(self, family, type, proto, fileno)
PermissionError: [Errno 1] Operation not permitted
```

Analisis: pengiriman tensor multiprocessing membutuhkan IPC socket yang
dilarang oleh lingkungan eksekusi. Ini bukan error ukuran/rentang dataset.
Run yang menunggu IPC dihentikan. Default tetap `num_workers=0`, dan tes
multi-worker memeriksa kemampuan IPC sebelum berjalan, lalu skip dengan
alasan eksplisit bila dilarang. Worker menggunakan `spawn` demi perilaku
lintas platform; tidak diklaim telah lolos eksekusi multi-worker di sini.

### B. Peringatan interoperabilitas tensor dan NumPy saat visualisasi

Pesan aktual:

```text
DeprecationWarning: __array__ implementation doesn't accept a copy keyword,
so passing copy=False failed.
```

Perbaikan: konversi tensor preview CPU secara eksplisit dengan `.numpy()`
sebelum diberikan ke `imshow`. Run tes terakhir tidak lagi mengeluarkan warning ini.

### C. Kernel notebook terpisah tidak berhasil dimulai

Pesan akhir aktual:

```text
Operation not permitted (src/ip_resolver.cpp:542)
RuntimeError: Kernel died before replying to kernel_info
```

Analisis: transport socket kernel dibatasi runtime. Solusi eksekusi sesi:
jalankan seluruh sel Python notebook berurutan menggunakan IPython dalam
satu proses, tangkap stdout dan display data asli, lalu simpan output ke
notebook. Tidak ada hasil atau execution count yang diisi tanpa eksekusi.
Kernel interaktif VS Code perlu dijalankan ulang di komputer pengguna.

## 5. Hasil verifikasi aktual

- Dataset nyata: 55.000 train, 5.000 validasi, 10.000 official test.
- Per kelas: 5.500 train, 500 validasi, 1.000 test.
- Overlap indeks train/validasi: 0; gabungannya mencakup seluruh 60.000 indeks.
- Satu batch yang diperiksa per split: float32 [128,1,32,32], min -1, max 1.
- Padding empat sisi: -1 setelah normalisasi.
- Empat arsip `.gz` sumber cocok dengan checksum referensi; bukti pada
  `reports/data/download_manifest.json`.
- Perintah tes: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -rA`.
- Hasil terakhir: **20 passed, 1 skipped**. Rincian: `reports/data/tests.txt`.
- Tujuh sel kode notebook dieksekusi berurutan, **0 error**.
- Grid validasi dan grafik distribusi dirender dan diperiksa secara visual.
- Tidak menjalankan training DDPM, benchmark GPU atau classifier evaluator;
  modul anggota 1/3 belum tersedia pada repository yang di-clone.

## 6. Tindak lanjut yang belum dilakukan oleh Hafid

1. Jalankan ulang notebook dan tes di environment kelompok.
2. Pelajari alasan split stratified, padding sebelum normalisasi, dan label
   metadata yang tidak dimasukkan ke DDPM.
3. Masukkan share link percakapan yang nyata pada log ini.
4. Serahkan kontrak loader ke A1/A3 dan log/bukti tes ke A4.

Tidak ada prompt eksperimen tambahan, error model, hasil training, atau
aktivitas anggota lain yang dibuat-buat untuk melengkapi rubrik.

## 7. Sesi cross-check lanjutan dan permintaan push

Prompt aktual pengguna:

> crosscheck lagi sudah benar dan beres eksperimen yang role 2 lakukan ?
> jika sudah semua tlong push ke https://github.com/DevandroA/GenerativeAI.git

Pemeriksaan kode tidak menemukan perubahan implementasi yang diperlukan untuk
konfigurasi default. Tes dijalankan ulang: 20 passed, 1 skipped karena socket
IPC lingkungan tetap dibatasi. Hasil terbaru disalin ke `reports/data/tests.txt`.

Pemeriksaan tambahan benar-benar mengiterasi seluruh loader default: 55.000
train, 5.000 validasi dan 10.000 test. Semua 70.000 gambar memenuhi shape,
float32, nilai finite, rentang [-1,1] dan padding -1 di empat sisi. Batch
terakhir juga tidak hilang. Keempat checksum arsip sumber diperiksa ulang
dan cocok. Tujuh sel kode notebook tersimpan memiliki execution count dan
tidak mengandung output error. Bukti: `reports/data/crosscheck.json`.

Koneksi GitHub berhasil diverifikasi sebagai `hafidramadhan22-prog`.
Metadata repository `DevandroA/GenerativeAI` menunjukkan izin `pull: true`
dan `push: false`. Repository masih kosong. Tidak mencoba mengubah ref atau
menulis file melalui koneksi yang diketahui tidak memiliki akses tulis.
Push menunggu pemilik repository memberikan akun tersebut akses Write.

## 8. Publikasi melalui browser GitHub

Pengguna kemudian memberikan akses dan instruksi lanjutan:

> boleh lakukan apa aja untuk push eksperimen final saya

Setelah autentikasi browser selesai, instruksi pengguna:

> oke sudah gas push

Akses akun `hafidramadhan22-prog` ke repository telah berubah menjadi
`push: true`. Integrasi API tetap menolak penulisan dengan error `403:
Resource not accessible by integration`; kredensial Git CLI juga tidak
tersedia. Atas persetujuan pengguna, publikasi dilanjutkan melalui antarmuka
web GitHub menggunakan sesi browser yang terautentikasi. Tidak mencatat
password, kode verifikasi, cookie, atau token pada repository.

Repository kosong diinisialisasi dengan README petunjuk pada `main`, lalu
dibuat branch `feat/data`. Setup proyek, konfigurasi, paket Python, modul
data, tes, panduan serah terima, notebook beroutput dan laporan cross-check
telah disimpan sebagai commit web GitHub. Log ini diserahkan pada tahap
terakhir publikasi. Isi kode dan hasil eksperimen tidak diubah oleh proses
unggah; tidak ada training DDPM baru yang diklaim berjalan.

Lokasi hasil role A2:
https://github.com/DevandroA/GenerativeAI/tree/feat/data
