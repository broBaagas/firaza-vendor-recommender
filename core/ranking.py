"""Penyaringan kandidat dan skor berbobot.

skor = w_berhasil * P(berhasil) + w_harga * (1 - harga_norm) + w_cepat * (1 - leadtime_norm)

harga_norm dan leadtime_norm adalah min-max di antara kandidat yang lolos saringan
(0 = termurah/tercepat). Jika semua kandidat sama, nilai normalisasinya 0.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core.data import load_data
from core.features import LIMITED_DATA_MIN_TX, build_candidate_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

PRESETS = {
    "Seimbang": {"w_berhasil": 0.50, "w_harga": 0.30, "w_cepat": 0.20},
    "Termurah": {"w_berhasil": 0.35, "w_harga": 0.55, "w_cepat": 0.10},
    "Tercepat": {"w_berhasil": 0.50, "w_harga": 0.10, "w_cepat": 0.40},
}

RESULT_COLUMNS = [
    "peringkat", "vendor_id", "nama_vendor", "kota", "offer_id", "harga_satuan", "total_harga",
    "lead_time_janji", "moq", "p_berhasil", "skor", "jumlah_transaksi", "data_terbatas",
    "tingkat_berhasil_10", "rata_hari_telat", "tingkat_cacat", "berhasil_musim_ramai",
]


def resolve_weights(preset: str = "Seimbang", weights: dict | None = None) -> dict:
    """Bobot dari preset, atau bobot kustom (dinormalisasi agar berjumlah 1)."""
    if weights is None:
        if preset not in PRESETS:
            raise ValueError(f"preset harus salah satu dari {list(PRESETS)}")
        return dict(PRESETS[preset])
    w = {k: float(weights.get(k, 0.0)) for k in PRESETS["Seimbang"]}
    total = sum(w.values())
    if total <= 0 or min(w.values()) < 0:
        raise ValueError("bobot harus non-negatif dan tidak semuanya 0")
    return {k: v / total for k, v in w.items()}


def validate_need(need: dict) -> None:
    for key in ("material_id", "qty", "hari_tersedia"):
        if key not in need or need[key] in (None, ""):
            raise ValueError(f"kebutuhan wajib punya '{key}'")
    if need["qty"] <= 0 or need["hari_tersedia"] <= 0:
        raise ValueError("qty dan hari_tersedia harus > 0")
    if need.get("budget") is not None and need["budget"] <= 0:
        raise ValueError("budget harus > 0")


def filter_candidates(offers: pd.DataFrame, need: dict, vendors: pd.DataFrame | None = None) -> pd.DataFrame:
    """Saring: material cocok, MOQ <= qty, lead time <= hari tersedia, harga total <= budget.

    Jika `vendors` diberikan dan need punya 'tanggal', vendor yang belum bergabung dibuang.
    """
    qty = need["qty"]
    c = offers[(offers["material_id"] == need["material_id"])
               & (offers["moq"] <= qty)
               & (offers["lead_time_janji"] <= need["hari_tersedia"])]
    if need.get("budget") is not None:
        c = c[c["harga_satuan"] * qty <= need["budget"]]
    if vendors is not None and need.get("tanggal"):
        joined = vendors.set_index("vendor_id")["tanggal_bergabung"]
        c = c[pd.to_datetime(c["vendor_id"].map(joined)) <= pd.Timestamp(need["tanggal"])]
    return c.reset_index(drop=True)


def _minmax(s: pd.Series) -> pd.Series:
    span = s.max() - s.min()
    return (s - s.min()) / span if span > 0 else pd.Series(0.0, index=s.index)


def score(candidates: pd.DataFrame, p_success, preset: str = "Seimbang",
          weights: dict | None = None) -> pd.DataFrame:
    """Hitung skor berbobot, urutkan menurun, dan beri kolom peringkat (1 = terbaik)."""
    w = resolve_weights(preset, weights)
    df = candidates.copy()
    df["p_berhasil"] = np.asarray(p_success, dtype=float)
    df["harga_norm"] = _minmax(df["harga_satuan"].astype(float))
    df["leadtime_norm"] = _minmax(df["lead_time_janji"].astype(float))
    df["skor"] = (w["w_berhasil"] * df["p_berhasil"]
                  + w["w_harga"] * (1 - df["harga_norm"])
                  + w["w_cepat"] * (1 - df["leadtime_norm"]))
    if "jumlah_transaksi" in df:
        df["data_terbatas"] = df["jumlah_transaksi"] < LIMITED_DATA_MIN_TX
    # Seri: utamakan P(berhasil) lebih tinggi, lalu harga lebih murah.
    df = df.sort_values(["skor", "p_berhasil", "harga_satuan"], ascending=[False, False, True])
    df = df.reset_index(drop=True)
    df.insert(0, "peringkat", np.arange(1, len(df) + 1))
    return df


def load_model(path: Path | str = MODEL_PATH):
    """Model hasil training (dict berisi 'pipeline', 'features', 'version'), atau None jika belum ada."""
    path = Path(path)
    return joblib.load(path) if path.exists() else None


def predict_success(features: pd.DataFrame, model=None) -> np.ndarray:
    """P(berhasil). Tanpa model: baseline rata-rata historis vendor (dihaluskan)."""
    if model is None:
        return features["tingkat_berhasil_semua"].to_numpy(dtype=float)
    return model["pipeline"].predict_proba(features[model["features"]])[:, 1]


def recommend(need: dict, preset: str = "Seimbang", top_k: int = 3, weights: dict | None = None,
              data: dict | None = None, model=None) -> dict:
    """Pipeline lengkap: saring -> fitur -> prediksi -> skor -> top-k.

    `data` default dari core.data.load_data(); `model` default dari models/model.joblib
    (jika belum ada, dipakai baseline).
    """
    validate_need(need)
    w = resolve_weights(preset, weights)
    data = data if data is not None else load_data()
    model = model if model is not None else load_model()

    result = {
        "kebutuhan": need,
        "preset": preset if weights is None else "Kustom",
        "bobot": w,
        "model": "baseline" if model is None else model.get("version", "model"),
        "jumlah_kandidat": 0,
        "ranking": [],
        "top": [],
    }
    candidates = filter_candidates(data["offers"], need, data["vendors"])
    if candidates.empty:
        result["pesan"] = "Tidak ada vendor yang memenuhi syarat (material, MOQ, lead time, budget)."
        return result

    feats = build_candidate_features(need, candidates, data["transactions"], data["materials"])
    feats = feats.merge(data["vendors"][["vendor_id", "nama_vendor", "kota"]], on="vendor_id", how="left")
    feats["total_harga"] = feats["harga_satuan"] * need["qty"]
    ranked = score(feats, predict_success(feats, model), weights=w)

    rows = ranked[RESULT_COLUMNS].round(4).to_dict(orient="records")
    result.update(jumlah_kandidat=len(rows), ranking=rows, top=rows[:top_k])
    return result
