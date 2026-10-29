"""Feature engineering untuk model P(berhasil) per pasangan (kebutuhan, vendor).

Label: berhasil = 1 jika barang datang <= deadline dan cacat <= 3%.

Anti-leakage: rekam jejak vendor untuk pesanan pada tanggal t hanya memakai transaksi
vendor itu yang SUDAH DATANG sebelum t (tanggal_datang < t). Pesanan yang sudah dibuat
tetapi barangnya belum datang belum diketahui hasilnya, jadi tidak ikut dihitung.

Rasio rekam jejak dihaluskan (Bayesian smoothing) dengan prior tetap supaya vendor
dengan sedikit transaksi tidak menghasilkan nilai ekstrem/NaN. Prior berupa konstanta,
bukan dihitung dari data, sehingga tidak membocorkan informasi dari masa depan.
"""

import numpy as np
import pandas as pd

from core.seasons import is_peak

DEFECT_THRESHOLD = 0.03
LIMITED_DATA_MIN_TX = 5
RECENT_WINDOW = 10
MAX_GAP_DAYS = 365

PRIOR_SUCCESS, PRIOR_LATE_DAYS, PRIOR_DEFECT = 0.7, 1.0, 0.02
PRIOR_STRENGTH = 2.0

NEED_COLUMNS = ["qty", "hari_tersedia", "kelonggaran_waktu", "bulan", "musim_ramai", "kategori"]
OFFER_COLUMNS = ["harga_relatif", "lead_time_janji", "moq", "rasio_qty"]
HISTORY_COLUMNS = [
    "tingkat_berhasil_10", "rata_hari_telat", "tingkat_cacat",
    "berhasil_musim_ramai", "jumlah_transaksi", "hari_sejak_pesanan_terakhir",
]
FEATURE_COLUMNS = NEED_COLUMNS + OFFER_COLUMNS + HISTORY_COLUMNS
CATEGORICAL_COLUMNS = ["kategori"]

_EPOCH = pd.Timestamp("1970-01-01")


def _days(s) -> np.ndarray:
    """Tanggal -> nomor hari (int) agar perbandingan tidak bergantung pada resolusi datetime."""
    return (pd.to_datetime(pd.Series(s)) - _EPOCH).dt.days.to_numpy()


def _smooth(total, n, prior):
    return (total + prior * PRIOR_STRENGTH) / (n + PRIOR_STRENGTH)


def make_label(transactions: pd.DataFrame) -> pd.Series:
    """berhasil = tanggal_datang <= deadline dan persen cacat <= 3%."""
    on_time = pd.to_datetime(transactions["tanggal_datang"]) <= pd.to_datetime(transactions["deadline"])
    good = transactions["qty_cacat"] / transactions["qty"] <= DEFECT_THRESHOLD
    return (on_time & good).astype(int).rename("berhasil")


def prepare_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Tambah kolom turunan: berhasil, hari_telat (terhadap janji), persen_cacat, musim_ramai."""
    tx = transactions.copy()
    for col in ["tanggal_pesan", "deadline", "tanggal_datang"]:
        tx[col] = pd.to_datetime(tx[col])
    tx["berhasil"] = make_label(tx)
    actual = (tx["tanggal_datang"] - tx["tanggal_pesan"]).dt.days
    tx["hari_telat"] = (actual - tx["lead_time_janji"]).clip(lower=0)
    tx["persen_cacat"] = tx["qty_cacat"] / tx["qty"]
    tx["musim_ramai"] = is_peak(tx["tanggal_pesan"]).astype(int)
    return tx


def _history_one_vendor(vtx: pd.DataFrame, t: np.ndarray) -> dict[str, np.ndarray]:
    done = vtx.sort_values("tanggal_datang")
    arrived = _days(done["tanggal_datang"])
    k = np.searchsorted(arrived, t, side="left")  # jumlah transaksi yang datang < t

    def cum(values):
        return np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype=float))])

    succ = cum(done["berhasil"])
    late = cum(done["hari_telat"])
    defect = cum(done["persen_cacat"])
    peak_n = cum(done["musim_ramai"])
    peak_succ = cum(done["berhasil"] * done["musim_ramai"])

    lo = np.maximum(k - RECENT_WINDOW, 0)
    ordered = np.sort(_days(vtx["tanggal_pesan"]))
    j = np.searchsorted(ordered, t, side="left")  # pesanan terakhir (hasil belum tentu diketahui)
    if len(ordered):
        last = np.where(j > 0, ordered[np.maximum(j - 1, 0)], t - MAX_GAP_DAYS)
    else:
        last = t - MAX_GAP_DAYS

    return {
        "tingkat_berhasil_10": _smooth(succ[k] - succ[lo], k - lo, PRIOR_SUCCESS),
        "tingkat_berhasil_semua": _smooth(succ[k], k, PRIOR_SUCCESS),
        "rata_hari_telat": _smooth(late[k], k, PRIOR_LATE_DAYS),
        "tingkat_cacat": _smooth(defect[k], k, PRIOR_DEFECT),
        "berhasil_musim_ramai": _smooth(peak_succ[k], peak_n[k], PRIOR_SUCCESS),
        "jumlah_transaksi": k,
        "hari_sejak_pesanan_terakhir": np.minimum(t - last, MAX_GAP_DAYS),
    }


def vendor_history(prepared_tx: pd.DataFrame, vendor_ids, as_of) -> pd.DataFrame:
    """Rekam jejak untuk setiap pasangan (vendor_ids[i], as_of[i]).

    `prepared_tx` adalah keluaran prepare_transactions. Mengembalikan DataFrame dengan
    indeks yang sama urutannya dengan input. Kolom tingkat_berhasil_semua dipakai baseline.
    """
    vendor_ids = np.asarray(vendor_ids)
    t = _days(as_of) if np.ndim(as_of) else np.full(len(vendor_ids), _days([as_of])[0])
    cols = HISTORY_COLUMNS + ["tingkat_berhasil_semua"]
    out = pd.DataFrame(index=range(len(vendor_ids)), columns=cols, dtype=float)
    groups = {v: g for v, g in prepared_tx.groupby("vendor_id")}
    for v in np.unique(vendor_ids):
        idx = np.flatnonzero(vendor_ids == v)
        if v in groups:
            hist = _history_one_vendor(groups[v], t[idx])
        else:  # vendor belum pernah dipakai
            hist = _history_one_vendor(prepared_tx.iloc[:0], t[idx])
        for c in cols:
            out.loc[idx, c] = hist[c]
    out["jumlah_transaksi"] = out["jumlah_transaksi"].astype(int)
    return out


def _need_offer_features(df: pd.DataFrame, materials: pd.DataFrame, order_dates) -> pd.DataFrame:
    """Fitur kebutuhan + penawaran. `df` butuh kolom material_id, qty, hari_tersedia,
    harga_satuan, lead_time_janji, moq."""
    mat = materials.set_index("material_id")
    ref_price = ((mat["harga_ref_min"] + mat["harga_ref_max"]) / 2).reindex(df["material_id"]).to_numpy()
    dates = pd.to_datetime(pd.Series(order_dates))
    return pd.DataFrame({
        "qty": df["qty"].to_numpy(),
        "hari_tersedia": df["hari_tersedia"].to_numpy(),
        "kelonggaran_waktu": (df["hari_tersedia"] - df["lead_time_janji"]).to_numpy(),
        "bulan": dates.dt.month.to_numpy(),
        "musim_ramai": is_peak(dates).astype(int),
        "kategori": mat["kategori"].reindex(df["material_id"]).to_numpy(),
        # Harga absolut antarmaterial tidak sebanding (Rp50 s.d. Rp120.000), jadi dipakai relatif
        # terhadap harga referensi material.
        "harga_relatif": df["harga_satuan"].to_numpy() / ref_price,
        "lead_time_janji": df["lead_time_janji"].to_numpy(),
        "moq": df["moq"].to_numpy(),
        "rasio_qty": (df["qty"] / df["moq"]).to_numpy(),
    })


def build_training_frame(transactions: pd.DataFrame, offers: pd.DataFrame,
                         materials: pd.DataFrame) -> pd.DataFrame:
    """Satu baris per transaksi historis: id, tanggal, fitur, dan label berhasil."""
    tx = prepare_transactions(transactions).sort_values("tanggal_pesan").reset_index(drop=True)
    tx["moq"] = tx["offer_id"].map(offers.set_index("offer_id")["moq"])
    base = _need_offer_features(tx, materials, tx["tanggal_pesan"])
    hist = vendor_history(tx, tx["vendor_id"], tx["tanggal_pesan"])
    ids = tx[["transaksi_id", "tanggal_pesan", "vendor_id", "material_id", "offer_id"]]
    return pd.concat([ids, base, hist, tx[["berhasil"]]], axis=1)


def build_candidate_features(need: dict, candidates: pd.DataFrame, transactions: pd.DataFrame,
                             materials: pd.DataFrame) -> pd.DataFrame:
    """Fitur untuk setiap penawaran kandidat pada saat inferensi.

    `need`: material_id, qty, hari_tersedia, dan opsional tanggal (default: sehari setelah
    pesanan terakhir di data). Mengembalikan kolom kandidat + fitur.
    """
    tx = prepare_transactions(transactions)
    as_of = pd.Timestamp(need.get("tanggal") or tx["tanggal_pesan"].max() + pd.Timedelta(days=1))
    c = candidates.reset_index(drop=True).copy()
    c["qty"], c["hari_tersedia"] = need["qty"], need["hari_tersedia"]
    base = _need_offer_features(c, materials, [as_of] * len(c))
    hist = vendor_history(tx, c["vendor_id"], as_of)
    keep = c.drop(columns=[col for col in base.columns if col in c.columns])
    return pd.concat([keep, base, hist], axis=1)
