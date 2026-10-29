# Laporan Evaluasi Model v1

Model: **`v1-random_forest`** (`models/model.joblib`). Dibuat oleh `python -m training.train`. Rincian dan grafik ada di [notebooks/02_training.ipynb](../notebooks/02_training.ipynb), angka mentahnya di [models/metrics.json](../models/metrics.json).

## Ringkasan

| Target (project plan) | Hasil | Status |
|---|---|---|
| ROC-AUC ≥ 0,75 dan lebih baik dari baseline | RF 0,813 vs baseline 0,625 (batas atas 0,840) | Tercapai |
| NDCG@3 lebih baik dari "pilih termurah" | 0,982 vs 0,795 (preset Seimbang) | Tercapai |
| Top-1 hit lebih baik dari "pilih termurah" | 84,3% vs 36,0% (preset Seimbang) | Tercapai |

## Desain eksperimen

- **Tugas.** Klasifikasi biner P(berhasil) per pasangan (kebutuhan, vendor). `berhasil = 1` jika barang datang ≤ deadline dan cacat ≤ 3%.
- **Data.** 6.075 transaksi sintetis (lihat [asumsi_data.md](asumsi_data.md)), 16 fitur (kebutuhan, penawaran, rekam jejak vendor anti-leakage).
- **Split waktu.** Train bulan 1–18 (Okt 2024 – Mar 2026, 4.492 baris). Test bulan 19–24 (Apr – Sep 2026, 1.583 baris).
- **Preprocessing.** StandardScaler untuk fitur numerik; log1p + scaler untuk fitur berekor panjang (`qty`, `moq`, `rasio_qty`, `jumlah_transaksi`, `hari_sejak_pesanan_terakhir`); one-hot untuk `kategori`. Preprocessing berada di dalam pipeline, sehingga hanya di-fit pada data latih setiap lipatan.
- **Tuning.** GridSearchCV dengan TimeSeriesSplit 4 lipatan, **hanya di data train**.
- **Aturan pemilihan** (ditetapkan sebelum melihat test). ROC-AUC validasi silang tertinggi. Jika selisihnya < 0,005, dipilih model yang lebih sederhana (LR → RF → LightGBM).
- **Model akhir.** Konfigurasi terpilih dilatih ulang pada seluruh data bulan 1–24.

## Hasil validasi silang (train)

| Model | ROC-AUC CV | Std | Parameter terbaik |
|---|---|---|---|
| Baseline (rata-rata historis vendor) | 0,650 | 0,033 | – |
| Logistic Regression | 0,778 | 0,026 | C = 0,03 |
| **Random Forest** | **0,796** | **0,012** | min_samples_leaf = 20, max_features = 0,5 |
| LightGBM | 0,793 | 0,013 | num_leaves = 7, n_estimators = 200, min_child_samples = 50 |

LightGBM berada dalam toleransi 0,005 dari RF, sehingga RF (lebih sederhana) yang dipilih.

## Hasil test (Apr – Sep 2026)

| Model | ROC-AUC | Brier | Log loss |
|---|---|---|---|
| Baseline | 0,625 | 0,163 | 0,504 |
| Logistic Regression | 0,813 | 0,132 | 0,410 |
| **Random Forest** | **0,813** | **0,129** | **0,406** |
| LightGBM | 0,819 | 0,128 | 0,401 |
| Oracle (P sebenarnya, batas atas) | 0,840 | 0,122 | 0,381 |

- Hasil pengiriman di generator diundi secara acak, sehingga bahkan P sebenarnya hanya mencapai ROC-AUC 0,840. Model tertinggal 0,02–0,03 dari batas itu.
- Di test, LightGBM sedikit lebih baik dari RF. Model tetap dipilih lewat validasi silang, supaya data test tidak ikut dipakai memilih model. Selisihnya kecil dibanding jarak ke baseline.
- Kurva kalibrasi RF mengikuti diagonal. Ini penting karena P(berhasil) dipakai langsung di skor dan ditampilkan ke pengguna.

## Evaluasi ranking

Data historis hanya mencatat hasil vendor yang dipilih owner. Untuk mengevaluasi ranking, setiap kebutuhan di periode test (1.464 kebutuhan dengan ≥ 2 kandidat, 8.570 pasangan) diberi **P(berhasil) sebenarnya untuk semua kandidat** dengan `generator.generate.true_success_prob`. Fungsi ini memakai sifat tersembunyi vendor dan dihitung eksak dari mekanisme generator. Kalibrasinya dicek terhadap data historis: rata-rata 78,3% vs teramati 78,0%.

- **Relevansi** kandidat = skor berbobot preset dengan P sebenarnya. Strategi "Sempurna" adalah batas atas.
- **NDCG@3**: gain = relevansi − relevansi terendah dalam kebutuhan itu.
- **Top-1 hit**: pilihan #1 sama dengan kandidat terbaik sebenarnya.

| Preset | Strategi | NDCG@3 | Top-1 | P(berhasil) pilihan #1 | Harga pilihan #1 vs termurah |
|---|---|---|---|---|---|
| Seimbang | Pilih termurah | 0,795 | 36,0% | 74,7% | 1,000 |
| Seimbang | Baseline | 0,961 | 75,0% | 84,6% | 1,076 |
| Seimbang | **Random Forest** | **0,982** | **84,3%** | **87,5%** | 1,089 |
| Seimbang | Sempurna | 1,000 | 100% | 88,3% | 1,093 |
| Termurah | Pilih termurah | 0,976 | 77,1% | 74,7% | 1,000 |
| Termurah | **Random Forest** | **0,997** | **91,4%** | **78,8%** | 1,012 |
| Tercepat | Pilih termurah | 0,519 | 15,8% | 74,7% | 1,000 |
| Tercepat | **Random Forest** | **0,995** | **89,8%** | **89,3%** | 1,179 |

Tabel lengkap untuk semua model ada di notebook. Interpretasi untuk preset Seimbang: dibanding memilih yang termurah, rekomendasi model menaikkan peluang berhasil pilihan #1 dari 74,7% ke 87,5%, dengan harga rata-rata 8,9% lebih mahal.

## Fitur terpenting

Permutation importance RF di data test (penurunan ROC-AUC jika fitur diacak): `kelonggaran_waktu` (0,149) jauh paling dominan, lalu `musim_ramai` (0,033), `tingkat_cacat` (0,025), dan `rata_hari_telat` (0,018). `harga_relatif` dan `moq` hampir tidak berpengaruh pada P(berhasil). Harga tetap memengaruhi ranking lewat bobot `w_harga`.

## Keterbatasan

- **Data sintetis.** Ground truth dan pola data berasal dari asumsi generator. Angka di sini menunjukkan pipeline bekerja dengan benar, bukan performa pada data nyata Firaza.
- **Relevansi memakai rumus skor yang sama.** Perbandingan dengan "pilih termurah" mengukur seberapa baik sistem mencapai tujuan berbobot itu. Karena itu P(berhasil) pilihan #1 dan selisih harga juga dilaporkan sebagai ukuran yang lebih netral.
- **Bias seleksi.** Model dilatih hanya dari vendor yang pernah dipilih owner. Kandidat yang jarang dipilih punya rekam jejak lebih tipis; ini ditandai "data terbatas" jika < 5 transaksi.

## Reproduksi

```bash
python -m generator.generate   # data (deterministik, seed 42)
python -m training.train       # tuning, evaluasi, simpan models/model.joblib + metrics.json (±1,5 menit)
pytest                         # termasuk tes oracle, metrik ranking, dan artefak model
```
