# Firaza Vendor Recommender

Decision-support pemilihan vendor untuk Firaza (Capstone Dicoding Bootcamp Batch 15, Use Case 3: Vendor Recommendation & Procurement Assistant).

Sistem memprediksi **P(berhasil)** setiap vendor kandidat untuk suatu kebutuhan (barang datang ≤ deadline **dan** cacat ≤ 3%), lalu meranking kandidat dengan skor berbobot (keberhasilan, harga, kecepatan). Gemini hanya **menjelaskan** hasil ranking; Gemini tidak menentukan ranking.

> Status (Minggu 1): generator data, EDA awal, modul fitur dan ranking sudah jalan. Selama model belum dilatih, ranking memakai baseline rata-rata historis. Modul `explain`, `storage`, API, dan Streamlit masih kerangka.

## Struktur

```
├── data/                 # CSV hasil generator
├── docs/asumsi_data.md   # kamus data, mekanisme generator, asumsi, keterbatasan
├── generator/generate.py # generator dataset sintetis (seed tetap)
├── core/
│   ├── seasons.py        # kalender musim ramai (Ramadan-Lebaran, musim nikah)
│   ├── data.py           # memuat CSV
│   ├── features.py       # label + fitur, rekam jejak vendor anti-leakage
│   ├── ranking.py        # saringan kandidat, skor berbobot, recommend()
│   ├── explain.py        # penjelasan Gemini (Minggu 3)
│   └── storage.py        # riwayat ke Supabase (Minggu 4)
├── notebooks/01_eda.ipynb
├── models/               # model.joblib hasil training
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
