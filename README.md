# Firaza Vendor Recommender

Decision-support pemilihan vendor untuk Firaza (Capstone Dicoding Bootcamp Batch 15, Use Case 3: Vendor Recommendation & Procurement Assistant).

Sistem memprediksi **P(berhasil)** setiap vendor kandidat untuk suatu kebutuhan (barang datang ≤ deadline **dan** cacat ≤ 3%), lalu meranking kandidat dengan skor berbobot (keberhasilan, harga, kecepatan). Gemini hanya **menjelaskan** hasil ranking; Gemini tidak menentukan ranking.

> Status (Minggu 2): data, fitur, model v1 (`v1-random_forest`), ranking, dan Streamlit versi awal sudah jalan. Modul `explain` (Gemini), `storage` (Supabase), dan API masih kerangka.

## Hasil model v1

| Metrik (data test Apr–Sep 2026) | Random Forest v1 | Pembanding |
|---|---|---|
| ROC-AUC | 0,813 | baseline 0,625; batas atas 0,840 |
| NDCG@3 (preset Seimbang) | 0,982 | pilih termurah 0,795 |
| Top-1 hit (preset Seimbang) | 84,3% | pilih termurah 36,0% |
| P(berhasil) pilihan #1 | 87,5% (+8,9% harga) | pilih termurah 74,7% |

Rincian: [docs/evaluasi_model.md](docs/evaluasi_model.md).

## Mencoba versi awal

Versi ini sudah bisa dicoba **tanpa API key dan tanpa database**. Data sintetis (`data/`) dan model v1 (`models/model.joblib`) sudah ikut di repo, jadi tidak perlu menjalankan generator atau training. Diuji dengan Python 3.14 di Windows.

### 1. Siapkan lingkungan

```powershell
git clone https://github.com/broBaagas/firaza-vendor-recommender.git
cd firaza-vendor-recommender
python -m venv .venv
.venv\Scripts\activate              # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`scikit-learn` dan `lightgbm` dikunci ke versi tertentu karena `model.joblib` hanya bisa dimuat dengan versi yang sama.

### 2. Coba lewat aplikasi web (Streamlit)

```powershell
streamlit run app/streamlit_app.py
```

Buka http://localhost:8501, lalu:

1. Di tab **Rekomendasi**, pilih material, isi jumlah dan hari tersedia sampai deadline (nilai awal sudah terisi), lalu klik **Cari rekomendasi vendor**.
2. Ganti **Prioritas** (Seimbang / Termurah / Tercepat / Kustom) untuk melihat ranking berubah. Pilih Kustom untuk mengatur bobot dengan slider.
3. Centang **Batasi total budget** untuk menyaring vendor yang total harganya melebihi budget.
4. Buka tab **Data Vendor** untuk melihat rekam jejak semua vendor dan daftar penawaran per material.

Yang ditampilkan: kartu 3 vendor teratas, ringkasan perbandingan dengan vendor termurah, grafik harga vs peluang berhasil, dan tabel semua kandidat. Vendor dengan < 5 transaksi ditandai ⚠️ data terbatas.

### 3. Coba lewat Python

```python
import pandas as pd
from core.ranking import recommend

r = recommend({"material_id": "M03", "qty": 150, "hari_tersedia": 12}, preset="Seimbang")
pd.DataFrame(r["ranking"])[["peringkat", "nama_vendor", "harga_satuan", "lead_time_janji", "p_berhasil", "skor"]]
```

Kebutuhan berisi `material_id`, `qty`, `hari_tersedia`, dan opsional `budget` (total, Rp). Preset: `"Seimbang"`, `"Termurah"`, `"Tercepat"`, atau bobot sendiri lewat `weights={"w_berhasil": .., "w_harga": .., "w_cepat": ..}`.

| Kode | Material | Lead time wajar (hari) | Kode | Material | Lead time wajar (hari) |
|---|---|---|---|---|---|
| M01 | Handuk mandi 70x140 cm | 7–14 | M09 | Tas spunbond | 4–8 |
| M02 | Handuk muka 30x30 cm | 5–10 | M10 | Bubble wrap 125 cm x 50 m (roll) | 2–4 |
| M03 | Sajadah polos dewasa | 7–14 | M11 | Plastik OPP | 2–4 |
| M04 | Sajadah travel lipat | 7–12 | M12 | Pita satin (roll) | 2–5 |
| M05 | Mukena katun dewasa | 10–18 | M13 | Kartu ucapan cetak | 3–6 |
| M06 | Kotak hampers rigid box | 5–10 | M14 | Jasa bordir nama | 4–9 |
| M07 | Kardus pengiriman | 2–5 | M15 | Jasa printing sajadah | 5–10 |
| M08 | Paper bag | 3–7 | | | |

### Cara membaca hasil

- **P(berhasil)**: peluang barang datang ≤ deadline **dan** cacat ≤ 3%, hasil prediksi model v1.
- **Skor**: gabungan P(berhasil), harga murah, dan lead time cepat sesuai bobot preset (lihat [Skor ranking](#skor-ranking)).
- **Data terbatas**: vendor dengan < 5 transaksi; prediksinya kurang pasti.
- Nama vendor ("Tekstil F", "Packaging C", …) dan semua transaksi **fiktif**, berasal dari generator data sintetis.

### Belum tersedia di versi ini

- Penjelasan rekomendasi oleh Gemini (Minggu 3). Ringkasan yang tampil saat ini masih teks template.
- Inference API `POST /api/v1/recommend` (Minggu 3). Endpoint sudah ada, tetapi masih mengembalikan 501.
- Penyimpanan riwayat rekomendasi ke Supabase dan live URL (Minggu 4).

## Struktur

```
├── data/                 # CSV hasil generator
├── docs/
│   ├── asumsi_data.md    # kamus data, mekanisme generator, kalibrasi harga, keterbatasan
│   └── evaluasi_model.md # laporan evaluasi model v1
├── generator/generate.py # generator dataset sintetis (seed tetap) + true_success_prob untuk evaluasi
├── training/
│   ├── models.py         # pipeline preprocessing + model, grid hyperparameter
│   ├── ranking_eval.py   # NDCG@3, Top-1 terhadap ground truth generator
│   └── train.py          # tuning, evaluasi, pemilihan, simpan model
├── core/
│   ├── seasons.py        # kalender musim ramai (Ramadan-Lebaran, musim nikah)
│   ├── data.py           # memuat CSV
│   ├── features.py       # label + fitur, rekam jejak vendor anti-leakage
│   ├── ranking.py        # saringan kandidat, skor berbobot, recommend()
│   ├── explain.py        # penjelasan Gemini (Minggu 3)
│   └── storage.py        # riwayat ke Supabase (Minggu 4)
├── notebooks/            # 01_eda.ipynb, 02_training.ipynb
├── models/               # model.joblib (v1) + metrics.json
├── app/streamlit_app.py  # antarmuka Streamlit
├── api/main.py           # FastAPI: POST /api/v1/recommend
└── tests/                # pytest
```

## Contoh pemakaian `core`

```python
from core.ranking import recommend

hasil = recommend({"material_id": "M01", "qty": 200, "hari_tersedia": 14, "budget": 9_000_000},
                  preset="Seimbang", top_k=3)
hasil["top"]   # 3 vendor teratas: harga, lead time, p_berhasil, skor, flag data_terbatas
```

## Pengembangan

```bash
pip install -r requirements-dev.txt   # + pytest, jupyter, matplotlib
copy .env.example .env                # isi GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY (belum dipakai di versi ini)

python -m generator.generate    # buat ulang dataset sintetis ke data/ (deterministik, ±8 detik)
python -m training.train        # latih ulang model v1 + metrics.json (±1,5 menit)
pytest                          # 30 tes: generator, fitur/anti-leakage, ranking, training, aplikasi
uvicorn api.main:app --reload   # kerangka API (/recommend belum aktif)
```

## Skor ranking

`skor = w_berhasil·P(berhasil) + w_harga·(1 − harga_norm) + w_cepat·(1 − leadtime_norm)`

| Preset | w_berhasil | w_harga | w_cepat |
|---|---|---|---|
| Seimbang | 0,50 | 0,30 | 0,20 |
| Termurah | 0,35 | 0,55 | 0,10 |
| Tercepat | 0,50 | 0,10 | 0,40 |

## Asumsi data

Dataset sepenuhnya **sintetis**, dibuat oleh `generator/generate.py` (numpy/pandas, seed tetap), bukan ditulis LLM: 30 vendor, 15 material, 109 penawaran, 6.075 transaksi (Okt 2024 – Sep 2026), dengan efek musim ramai dan trade-off harga vs keandalan. Kamus data, mekanisme, dan keterbatasannya ada di [docs/asumsi_data.md](docs/asumsi_data.md). Hasil eksplorasi ada di [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb).

## Lisensi dan SBOM

Proyek ini berlisensi [MIT](LICENSE). Semua dependensi langsung berlisensi permisif:

| Library | Lisensi |
|---|---|
| numpy, pandas, scikit-learn, joblib | BSD-3-Clause |
| lightgbm | MIT |
| fastapi, pydantic | MIT |
| uvicorn | BSD-3-Clause |
| streamlit | Apache-2.0 |
| plotly | MIT |
| google-genai | Apache-2.0 |
| pg8000 | BSD-3-Clause |
| python-dotenv | BSD-3-Clause |
| pytest (dev) | MIT |
| jupyter, nbformat, nbclient, ipykernel (dev) | BSD-3-Clause |
| matplotlib (dev) | PSF-based (permisif) |

## Sitasi

Akan dilengkapi: sumber kalibrasi harga marketplace, library, dan model yang dipakai.
