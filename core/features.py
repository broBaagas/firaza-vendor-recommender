"""Feature engineering untuk model P(berhasil) per pasangan (kebutuhan, vendor).

Label: berhasil = 1 jika barang datang <= deadline dan cacat <= 3%.
Rekam jejak vendor dihitung HANYA dari transaksi sebelum tanggal pesanan (anti-leakage).
"""

import pandas as pd

DEFECT_THRESHOLD = 0.03
LIMITED_DATA_MIN_TX = 5

FEATURE_COLUMNS = [
    # kebutuhan
    "qty", "hari_tersedia", "kelonggaran_waktu", "bulan", "musim_ramai", "kategori",
    # penawaran
    "harga", "lead_time_janji", "moq", "rasio_qty",
    # rekam jejak vendor
    "tingkat_berhasil_10", "rata_hari_telat", "tingkat_cacat",
    "berhasil_musim_ramai", "jumlah_transaksi", "hari_sejak_pesanan_terakhir",
]


def make_label(transactions: pd.DataFrame) -> pd.Series:
    """berhasil = tanggal_datang <= deadline dan persen_cacat <= 3%."""
    raise NotImplementedError


def vendor_history(transactions: pd.DataFrame, vendor_id: str, as_of: pd.Timestamp) -> dict:
    """Rekam jejak vendor dari transaksi dengan tanggal_pesan < as_of."""
    raise NotImplementedError


def build_training_frame(transactions: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    """Bangun tabel fitur + label untuk training."""
    raise NotImplementedError


def build_candidate_features(need: dict, candidates: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Bangun fitur untuk kandidat vendor pada saat inferensi."""
    raise NotImplementedError
