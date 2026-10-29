"""Penyaringan kandidat dan skor berbobot.

skor = w_berhasil * P(berhasil) + w_harga * (1 - harga_norm) + w_cepat * (1 - leadtime_norm)
"""

import pandas as pd

PRESETS = {
    "Seimbang": {"w_berhasil": 0.50, "w_harga": 0.30, "w_cepat": 0.20},
    "Termurah": {"w_berhasil": 0.35, "w_harga": 0.55, "w_cepat": 0.10},
    "Tercepat": {"w_berhasil": 0.50, "w_harga": 0.10, "w_cepat": 0.40},
}


def filter_candidates(offers: pd.DataFrame, need: dict) -> pd.DataFrame:
    """Saring: material cocok, MOQ <= qty, lead time <= hari tersedia, harga total <= budget."""
    raise NotImplementedError


def score(candidates: pd.DataFrame, p_success: pd.Series, preset: str = "Seimbang") -> pd.DataFrame:
    """Hitung skor berbobot dan urutkan menurun."""
    raise NotImplementedError


def recommend(need: dict, preset: str = "Seimbang", top_k: int = 3) -> dict:
    """Pipeline lengkap: saring -> fitur -> prediksi -> skor -> top-k."""
    raise NotImplementedError
