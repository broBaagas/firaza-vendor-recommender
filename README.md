# Firaza Vendor Recommender

Decision-support pemilihan vendor untuk Firaza (Capstone Dicoding Bootcamp Batch 15, Use Case 3: Vendor Recommendation & Procurement Assistant).

Sistem memprediksi **P(berhasil)** setiap vendor kandidat untuk suatu kebutuhan (barang datang ≤ deadline **dan** cacat ≤ 3%), lalu meranking kandidat dengan skor berbobot (keberhasilan, harga, kecepatan). Gemini hanya **menjelaskan** hasil ranking; Gemini tidak menentukan ranking.

> Status: kerangka proyek (Minggu 1). Sebagian besar modul masih berupa stub.

## Struktur

```
├── data/                 # CSV hasil generator
├── generator/generate.py # generator dataset sintetis (seed tetap)
├── core/                 # logika bersama: features, ranking, explain (Gemini), storage (Supabase)
├── notebooks/            # 01_eda.ipynb, 02_training.ipynb
├── models/               # model.joblib hasil training
├── app/streamlit_app.py  # antarmuka Streamlit
├── api/main.py           # FastAPI: POST /api/v1/recommend
└── tests/                # pytest
```

## Menjalankan secara lokal

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt
copy .env.example .env          # lalu isi GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY

python -m generator.generate    # buat dataset sintetis ke data/
streamlit run app/streamlit_app.py
uvicorn api.main:app --reload
pytest
```

## Skor ranking

`skor = w_berhasil·P(berhasil) + w_harga·(1 − harga_norm) + w_cepat·(1 − leadtime_norm)`

| Preset | w_berhasil | w_harga | w_cepat |
|---|---|---|---|
| Seimbang | 0,50 | 0,30 | 0,20 |
| Termurah | 0,35 | 0,55 | 0,10 |
| Tercepat | 0,50 | 0,10 | 0,40 |

## Asumsi data

Dataset sepenuhnya **sintetis**, dibuat oleh `generator/generate.py` (numpy/pandas, seed tetap), bukan ditulis LLM. Rincian asumsi (jumlah vendor, efek musim ramai, kalibrasi harga) akan dilengkapi bersama generator.

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

## Sitasi

Akan dilengkapi: sumber kalibrasi harga marketplace, library, dan model yang dipakai.
