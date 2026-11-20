# Firaza Vendor Recommender

Decision-support pemilihan vendor untuk Firaza (Capstone Dicoding Bootcamp Batch 15, Use Case 3: Vendor Recommendation & Procurement Assistant).

Sistem memprediksi **P(berhasil)** setiap vendor kandidat untuk suatu kebutuhan (barang datang ≤ deadline **dan** cacat ≤ 3%), lalu meranking kandidat dengan skor berbobot (keberhasilan, harga, kecepatan). Gemini hanya **menjelaskan** hasil ranking; Gemini tidak menentukan ranking.

**Live demo:** https://firaza.streamlit.app/

> Status: data, fitur, model v1 (`v1-random_forest`), ranking, Streamlit, penjelasan Gemini (dengan validasi angka), inference API, dan riwayat rekomendasi di Supabase sudah jalan.

## Hasil model v1

| Metrik (data test Apr–Sep 2026) | Random Forest v1 | Pembanding |
|---|---|---|
| ROC-AUC | 0,813 | baseline 0,625; batas atas 0,840 |
| NDCG@3 (preset Seimbang) | 0,982 | pilih termurah 0,795 |
| Top-1 hit (preset Seimbang) | 84,3% | pilih termurah 36,0% |
| P(berhasil) pilihan #1 | 87,5% (+8,9% harga) | pilih termurah 74,7% |

Rincian: [docs/evaluasi_model.md](docs/evaluasi_model.md).

## Mencoba versi awal

Versi ini sudah bisa dicoba **tanpa API key dan tanpa database**; tanpa `GEMINI_API_KEY`, penjelasan memakai template cadangan. Data sintetis (`data/`) dan model v1 (`models/model.joblib`) sudah ikut di repo, jadi tidak perlu menjalankan generator atau training. Diuji dengan Python 3.14 di Windows.

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
2. Isi **Tanggal pesan** (sampai 31/12/2027). Bagian **📅 Kalender pesanan** di hasil menyebut apakah tanggal itu **hari besar** nasional (🎉, misalnya 17/08/2027 Hari Kemerdekaan) atau **hari biasa**, hari besar lain sampai deadline, dan **musim ramai** (🌙 Ramadan–Lebaran atau musim nikah Syawal/Dzulhijjah). Di waktu rawan ini muncul **saran vendor dengan peluang berhasil paling tinggi**; ranking utama tidak berubah. Musim ramai adalah fitur model (peluang berhasil turun, ranking bisa berubah, coba 15/02/2027); hari besar belum menjadi fitur model karena data historis tidak mencatatnya. Kalender ada di `core/seasons.py`; hari besar keagamaan 2027 masih perkiraan dan perlu dicocokkan dengan SKB libur nasional. Ganti juga **Prioritas** (Seimbang / Termurah / Tercepat / Kustom) untuk melihat ranking berubah. Pilih Kustom untuk mengatur bobot dengan slider.
3. Centang **Batasi total budget** untuk menyaring vendor yang total harganya melebihi budget.
4. Buka tab **Data Vendor** untuk melihat rekam jejak semua vendor, daftar penawaran per material, dan **histori transaksi** per vendor (bisa disaring per material: tanggal pesan, deadline, tanggal datang, telat, cacat, berhasil atau tidak).
5. Tab **Riwayat** menampilkan permintaan rekomendasi yang tersimpan di database (butuh `DATABASE_URL`). Panduan singkat juga tersedia di aplikasi lewat panel **ℹ️ Cara pakai**.

Yang ditampilkan: kartu 3 vendor teratas, penjelasan (rekomendasi, alasan, trade-off, saran), grafik harga vs peluang berhasil, dan tabel semua kandidat. Vendor dengan < 5 transaksi ditandai ⚠️ data terbatas.

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

### 4. Coba lewat API (FastAPI)

```powershell
uvicorn api.main:app --reload
```

Dokumentasi interaktif di http://localhost:8000/docs. Endpoint:

| Method | Path | Isi |
|---|---|---|
| GET | `/health` | status dan versi model |
| GET | `/api/v1/materials` | daftar kode material |
| POST | `/api/v1/recommend` | ranking vendor + penjelasan |

```powershell
curl -X POST http://localhost:8000/api/v1/recommend -H "Content-Type: application/json" `
  -d '{"material_id": "M03", "qty": 200, "hari_tersedia": 12, "preset": "Termurah"}'
```

Body: `material_id`, `qty`, `hari_tersedia`, opsional `budget`, `tanggal` (tanggal pesan `YYYY-MM-DD`, menentukan musim ramai dan hari besar; paling lambat 2027-12-31), `preset` (`Seimbang`/`Termurah`/`Tercepat`/`Kustom`), `bobot` (wajib jika `Kustom`), `top_k` (1–10, default 3), `dengan_penjelasan` (default `true`). Respons berisi `tanggal_pesan`, `musim_ramai` (periode, atau `null`), `kalender` (hari besar di tanggal pesan dan sampai deadline, `waktu_rawan`), `paling_aman` (kandidat dengan peluang berhasil tertinggi), `top`, `ranking` (semua kandidat), dan `penjelasan` (`sumber`: `gemini` atau `template`, beserta hasil `validasi` angka).

Alternatif lewat Docker (image yang sama siap dipakai di layanan hosting container mana pun):

```powershell
docker build -t firaza-api .
docker run -p 7860:7860 --env-file .env firaza-api   # buka http://localhost:7860/docs
```

`--env-file .env` opsional; tanpa itu API tetap jalan dengan penjelasan template dan tanpa menyimpan riwayat.

### Penjelasan Gemini

Isi `GEMINI_API_KEY` di `.env` (lokal) atau Secrets (Streamlit Cloud). Model default `gemini-3.5-flash-lite`, bisa diganti lewat env `GEMINI_MODEL`. Gemini menerima konteks JSON (kebutuhan, 3 vendor teratas, dan selisih yang sudah dihitung) dan hanya menulis penjelasan. Setiap angka di penjelasan dicek terhadap konteks dan rekomendasi wajib menyebut vendor peringkat 1; jika tidak lolos, kuota habis, atau API gagal, dipakai template cadangan.

### Riwayat rekomendasi (Supabase)

Isi `DATABASE_URL` (Supabase, connection string **Session pooler**) di `.env` atau Secrets. Setiap permintaan dari Streamlit dan API disimpan ke tabel `riwayat_rekomendasi` (dibuat otomatis, Row Level Security aktif sehingga tertutup dari Data API publik Supabase) dan bisa dilihat di tab **Riwayat**. Koneksi memakai TLS yang diverifikasi dengan sertifikat CA Supabase di `certs/supabase-ca.crt`. Tanpa `DATABASE_URL`, aplikasi tetap jalan tanpa menyimpan riwayat.

### Catatan hosting API

Live URL proyek ini adalah aplikasi Streamlit (https://firaza.streamlit.app/), yang memanggil modul `core` secara langsung sehingga tidak bergantung pada server API. Inference API **tidak dihosting publik**: rencana awal memakai Render (free tier), tetapi Render mensyaratkan kartu kredit, dan Hugging Face Spaces kini mensyaratkan langganan PRO untuk Space Docker. API dijalankan secara lokal atau lewat Docker seperti di atas. Konfigurasi siap pakai tetap disertakan: `render.yaml` (Render) dan `Dockerfile` + `scripts/deploy_hf_space.py` (Hugging Face Spaces).

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
│   ├── explain.py        # penjelasan Gemini + validasi angka + template cadangan
│   └── storage.py        # riwayat rekomendasi ke Supabase Postgres (pg8000)
├── certs/supabase-ca.crt # sertifikat CA publik Supabase untuk verifikasi TLS
├── notebooks/            # 01_eda.ipynb, 02_training.ipynb
├── models/               # model.joblib (v1) + metrics.json
├── app/streamlit_app.py  # antarmuka Streamlit
├── api/main.py           # FastAPI: POST /api/v1/recommend
├── scripts/              # deploy_hf_space.py (opsional)
├── Dockerfile            # image inference API
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
copy .env.example .env                # isi GEMINI_API_KEY dan DATABASE_URL (keduanya opsional)

python -m generator.generate    # buat ulang dataset sintetis ke data/ (deterministik, ±8 detik)
python -m training.train        # latih ulang model v1 + metrics.json (±1,5 menit)
pytest                          # Gemini dan database di-mock; RUN_DB_TESTS=1 untuk tes ke Supabase
uvicorn api.main:app --reload   # inference API di http://localhost:8000/docs
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
| huggingface_hub (hanya skrip deploy, opsional) | Apache-2.0 |
| playwright (hanya untuk screenshot saat pengembangan, tidak di requirements) | Apache-2.0 |
| Font Plus Jakarta Sans (dimuat dari Google Fonts lewat `.streamlit/config.toml`) | SIL Open Font License 1.1 |

**Dependensi transitif yang perlu dicatat:** `certifi` (MPL-2.0) terpasang otomatis sebagai dependensi `requests`/`httpx`, yang dipakai Streamlit dan google-genai. Paket ini hanya berisi bundel sertifikat CA, dipakai apa adanya tanpa modifikasi, dan tidak ikut didistribusikan di repo ini. Tidak ada dependensi GPL/AGPL/LGPL/EPL. Daftar lengkap paket terpasang beserta lisensinya bisa dicetak dengan `pip-licenses` atau `pip show <paket>`.

## Sitasi

**Data**
- Dataset vendor, penawaran, dan transaksi: **sintetis**, dibuat oleh `generator/generate.py` di repo ini (bukan data asli Firaza, bukan ditulis LLM).
- Rentang harga material dikalibrasi dari listing publik Tokopedia, Lazada, Ruparupa, dan Indotrading (diakses 29 Oktober 2026). Tautan per material ada di [docs/asumsi_data.md](docs/asumsi_data.md#kalibrasi-harga).
- Kalender musim ramai dan hari besar nasional (Ramadan, Idulfitri, Iduladha, dan libur nasional lain) memakai tanggal libur nasional Indonesia 2025–2026 dan perkiraan hisab/astronomis untuk hari besar keagamaan 2027 ([core/seasons.py](core/seasons.py)); perlu dicocokkan dengan SKB 3 Menteri tentang hari libur nasional dan cuti bersama 2027.
- Konteks masalah: dokumen use case *CP-03 Firaza: Vendor Recommendation & Procurement Assistant*, Dicoding Bootcamp Batch 15.

**Model dan layanan**
- Model prediksi P(berhasil): Random Forest yang dilatih sendiri (`models/model.joblib`, versi `v1-random_forest`); tidak memakai model pra-latih.
- Penjelasan: Google Gemini (`gemini-3.5-flash-lite`) lewat Gemini API dan SDK [google-genai](https://github.com/googleapis/python-genai). Gemini hanya menulis penjelasan dari angka hasil ranking.
- Database: [Supabase](https://supabase.com/) (PostgreSQL). Hosting aplikasi: [Streamlit Community Cloud](https://streamlit.io/cloud).

**Library**
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12, 2825–2830.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 30*.
- Harris, C. R. et al. (2020). Array programming with NumPy. *Nature* 585, 357–362.
- McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proc. 9th Python in Science Conf.*, 56–61. (pandas)
- Breiman, L. (2001). Random Forests. *Machine Learning* 45, 5–32.
- [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/), [Uvicorn](https://www.uvicorn.org/), [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/python/), [pg8000](https://github.com/tlocke/pg8000), [joblib](https://joblib.readthedocs.io/), [python-dotenv](https://github.com/theskumar/python-dotenv), [pytest](https://pytest.org/), [Jupyter](https://jupyter.org/), [Matplotlib](https://matplotlib.org/).
