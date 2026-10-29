"""Generator dataset sintetis Firaza (numpy/pandas, seed tetap).

Keluaran di folder data/:
- vendors.csv               30 vendor
- materials.csv             15 material
- offers.csv                ±120 penawaran (vendor x material: harga, lead_time_janji, MOQ)
- transactions.csv          ±6.000 transaksi selama 24 bulan (Okt 2024 - Sep 2026)
- vendors_hidden.csv        sifat tersembunyi vendor, HANYA untuk evaluasi (tidak dibaca app/model)

Asumsi dan kamus data: docs/asumsi_data.md.
Jalankan: python -m generator.generate
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.seasons import is_peak  # noqa: E402

SEED = 42
START_DATE = "2024-10-01"
END_DATE = "2026-09-30"  # 24 bulan
N_TRANSACTIONS = 6000
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# id, nama, kategori, satuan, harga_min, harga_max (Rp/satuan), lead_min, lead_max (hari), qty_median, popularitas
# Rentang harga dikalibrasi dari listing marketplace (diakses 29 Okt 2026); sumber di docs/asumsi_data.md.
MATERIALS = [
    ("M01", "Handuk mandi 70x140 cm", "tekstil", "pcs", 25_000, 70_000, 7, 14, 150, 1.4),
    ("M02", "Handuk muka 30x30 cm", "tekstil", "pcs", 7_000, 12_000, 5, 10, 300, 1.0),
    ("M03", "Sajadah polos dewasa", "tekstil", "pcs", 25_000, 65_000, 7, 14, 120, 1.3),
    ("M04", "Sajadah travel lipat", "tekstil", "pcs", 10_000, 30_000, 7, 12, 100, 0.8),
    ("M05", "Mukena katun dewasa", "tekstil", "pcs", 35_000, 120_000, 10, 18, 60, 0.6),
    ("M06", "Kotak hampers rigid box", "packaging", "pcs", 10_000, 30_000, 5, 10, 200, 1.3),
    ("M07", "Kardus pengiriman", "packaging", "pcs", 1_000, 5_500, 2, 5, 400, 1.2),
    ("M08", "Paper bag", "packaging", "pcs", 750, 2_000, 3, 7, 250, 0.9),
    ("M09", "Tas spunbond", "packaging", "pcs", 700, 2_500, 4, 8, 250, 0.7),
    ("M10", "Bubble wrap 125 cm x 50 m", "packaging", "roll", 85_000, 115_000, 2, 4, 10, 0.6),
    ("M11", "Plastik OPP", "packaging", "pcs", 50, 300, 2, 4, 1000, 0.7),
    ("M12", "Pita satin", "aksesoris", "roll", 10_000, 20_000, 2, 5, 40, 0.7),
    ("M13", "Kartu ucapan cetak", "aksesoris", "pcs", 500, 4_000, 3, 6, 300, 0.8),
    ("M14", "Jasa bordir nama", "jasa", "pcs", 2_000, 8_000, 4, 9, 120, 1.1),
    ("M15", "Jasa printing sajadah", "jasa", "pcs", 10_000, 25_000, 5, 10, 100, 0.9),
]

# kategori: (jumlah vendor, kota asal, rentang jumlah vendor per material)
VENDOR_TYPES = {
    "tekstil": (9, ["Pekalongan", "Bandung", "Solo", "Tasikmalaya"], (7, 9)),
    "packaging": (10, ["Jakarta", "Tangerang", "Bekasi", "Surabaya"], (7, 9)),
    "jasa": (6, ["Bandung", "Tasikmalaya", "Jakarta"], (5, 6)),
    "aksesoris": (5, ["Jakarta", "Bandung", "Surabaya"], (6, 9)),
}
# Material aksesoris juga dijual sebagian vendor packaging.
EXTRA_POOL = {"aksesoris": ["packaging"]}
# Vendor yang baru bergabung belakangan (bulan ke-): memunculkan kasus "data terbatas".
LATE_JOIN_MONTHS = [8, 14, 20, 23]

PEAK_DEMAND_MULT = 1.8
PEAK_QTY_MULT = 1.4
GROWTH_24M = 0.30
INFLATION_PER_MONTH = 0.004
ROLL_DEFECT_MULT = 0.25


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return np.log(p / (1.0 - p))


def _round_price(price: float) -> int:
    """Bulatkan ke kelipatan yang wajar untuk besaran harganya (Rp5 / Rp50 / Rp100)."""
    step = 5 if price < 1_000 else 50 if price < 5_000 else 100
    return int(round(price / step) * step)


def generate_materials() -> pd.DataFrame:
    cols = ["material_id", "nama_material", "kategori", "satuan", "harga_ref_min", "harga_ref_max",
            "lead_time_min", "lead_time_max", "qty_median", "popularitas"]
    return pd.DataFrame(MATERIALS, columns=cols)


def generate_vendors(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kembalikan (vendors, vendors_hidden)."""
    rows, hidden = [], []
    start = pd.Timestamp(START_DATE)
    for kategori, (n, cities, _) in VENDOR_TYPES.items():
        for i in range(n):
            rows.append({
                "vendor_id": None,
                "nama_vendor": f"{kategori.title()} {chr(ord('A') + i)}",
                "kategori": kategori,
                "kota": rng.choice(cities),
                "tanggal_bergabung": start - pd.Timedelta(days=int(rng.integers(30, 720))),
            })
    vendors = pd.DataFrame(rows)
    vendors["vendor_id"] = [f"V{i + 1:02d}" for i in range(len(vendors))]

    late = rng.choice(len(vendors), size=len(LATE_JOIN_MONTHS), replace=False)
    for idx, month in zip(late, LATE_JOIN_MONTHS):
        vendors.loc[idx, "tanggal_bergabung"] = start + pd.DateOffset(months=month - 1) + pd.Timedelta(
            days=int(rng.integers(0, 28)))

    # Kompetensi laten: vendor yang lebih andal cenderung lebih rapi kualitasnya dan sedikit lebih mahal,
    # sehingga "pilih termurah" tidak selalu terbaik (trade-off yang ingin dibantu sistem).
    n = len(vendors)
    z = rng.normal(0, 1, n)
    hidden = pd.DataFrame({
        "vendor_id": vendors["vendor_id"],
        "true_reliability": np.clip(_sigmoid(1.2 + 0.9 * z + 0.4 * rng.normal(0, 1, n)), 0.45, 0.98),
        "true_defect_rate": np.clip(np.exp(np.log(0.010) - 0.5 * z + 0.4 * rng.normal(0, 1, n)), 0.002, 0.08),
        "price_factor": np.clip(1 + 0.08 * z + 0.06 * rng.normal(0, 1, n), 0.8, 1.3),
        "peak_sensitivity": rng.uniform(0, 1, n),
        "capacity_factor": np.exp(rng.normal(np.log(2.0), 0.4, n)),
        # Sebagian vendor membaik/memburuk selama 24 bulan (logit keandalan berubah linear).
        "drift_logit": np.where(rng.uniform(0, 1, n) < 0.3, rng.normal(0, 0.8, n), 0.0),
        # Seberapa tepat owner "merasakan" keandalan vendor (dipakai kebijakan pemilihan historis).
        "owner_belief": z + rng.normal(0, 0.8, n),
    })
    vendors["tanggal_bergabung"] = pd.to_datetime(vendors["tanggal_bergabung"]).dt.date
    return vendors, hidden


def generate_offers(rng: np.random.Generator, vendors: pd.DataFrame, materials: pd.DataFrame,
                    hidden: pd.DataFrame) -> pd.DataFrame:
    pf = hidden.set_index("vendor_id")["price_factor"]
    by_cat = vendors.groupby("kategori")["vendor_id"].apply(list).to_dict()
    pairs = []
    for m in materials.itertuples():
        pool = list(by_cat[m.kategori])
        for extra in EXTRA_POOL.get(m.kategori, []):
            pool += list(rng.choice(by_cat[extra], size=3, replace=False))
        lo, hi = VENDOR_TYPES[m.kategori][2]
        k = min(len(pool), int(rng.integers(lo, hi + 1)))
        pairs += [(v, m.material_id) for v in rng.choice(pool, size=k, replace=False)]

    # Pastikan setiap vendor punya minimal 1 penawaran.
    covered = {v for v, _ in pairs}
    for v in vendors.itertuples():
        if v.vendor_id not in covered:
            mats = materials.loc[materials["kategori"] == v.kategori, "material_id"]
            pairs.append((v.vendor_id, rng.choice(mats.to_numpy())))

    mat = materials.set_index("material_id")
    rows = []
    for vendor_id, material_id in pairs:
        m = mat.loc[material_id]
        lead = int(rng.integers(m.lead_time_min, m.lead_time_max + 1))
        lead_mid = (m.lead_time_min + m.lead_time_max) / 2
        base = (m.harga_ref_min + m.harga_ref_max) / 2
        # Vendor yang menjanjikan lebih cepat sedikit lebih mahal (±2% per hari).
        price = base * pf[vendor_id] * (1 + 0.02 * (lead_mid - lead)) * np.exp(rng.normal(0, 0.08))
        moq_choices = [0.1, 0.25, 0.5, 1.0] if m.satuan == "pcs" else [0.1, 0.2, 0.5]
        moq = max(1, int(round(m.qty_median * rng.choice(moq_choices))))
        if moq >= 20:
            moq = int(round(moq / 10) * 10)
        rows.append({
            "vendor_id": vendor_id,
            "material_id": material_id,
            "harga_satuan": _round_price(price),
            "lead_time_janji": lead,
            "moq": moq,
        })
    offers = pd.DataFrame(rows).sort_values(["material_id", "vendor_id"]).reset_index(drop=True)
    offers.insert(0, "offer_id", [f"O{i + 1:03d}" for i in range(len(offers))])
    return offers


def _order_dates(rng: np.random.Generator) -> pd.DatetimeIndex:
    days = pd.date_range(START_DATE, END_DATE, freq="D")
    t = np.arange(len(days)) / (len(days) - 1)
    mult = (1 + GROWTH_24M * t) * np.where(is_peak(days), PEAK_DEMAND_MULT, 1.0)
    mult *= np.where(days.day_of_week == 6, 0.5, 1.0)  # Minggu lebih sepi
    lam = N_TRANSACTIONS * mult / mult.sum()
    counts = rng.poisson(lam)
    return days.repeat(counts)


def generate_transactions(rng: np.random.Generator, vendors: pd.DataFrame, materials: pd.DataFrame,
                          offers: pd.DataFrame, hidden: pd.DataFrame) -> pd.DataFrame:
    """Simulasi pesanan historis: owner memilih vendor, lalu hasil (telat/cacat) diundi dari sifat vendor."""
    dates = _order_dates(rng)
    peak = is_peak(dates)
    n = len(dates)
    start, end = pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)
    total_days = (end - start).days

    mat = materials.set_index("material_id")
    hid = hidden.set_index("vendor_id")
    joined = pd.to_datetime(vendors.set_index("vendor_id")["tanggal_bergabung"])
    offers = offers.copy()
    offers["joined"] = offers["vendor_id"].map(joined).to_numpy()
    offers["price_z"] = offers.groupby("material_id")["harga_satuan"].transform(lambda s: (s - s.mean()) / s.std())
    offers_by_mat = {k: g for k, g in offers.groupby("material_id")}

    pop = materials["popularitas"].to_numpy()
    mat_ids = rng.choice(materials["material_id"].to_numpy(), size=n, p=pop / pop.sum())

    rows = []
    for i in range(n):
        d, is_pk, mid = dates[i], bool(peak[i]), mat_ids[i]
        m = mat.loc[mid]
        qty = rng.lognormal(np.log(m.qty_median * (PEAK_QTY_MULT if is_pk else 1.0)), 0.6)
        qty = max(1, int(round(qty)))
        hari = int(m.lead_time_max + rng.integers(-3, 12) - (2 if is_pk else 0))
        hari = max(int(m.lead_time_min), hari)

        cand = offers_by_mat[mid]
        cand = cand[cand["joined"] <= d]
        feasible = cand[(cand["lead_time_janji"] <= hari) & (cand["moq"] <= qty)]
        if feasible.empty:
            # Tidak ada yang memenuhi: owner menyesuaikan qty ke MOQ dan mengambil yang tercepat.
            feasible = cand.nsmallest(2, "lead_time_janji")
            qty = max(qty, int(feasible["moq"].min()))
        # Kebijakan owner: utamakan murah, sedikit menimbang "rasa" keandalan, plus faktor acak.
        util = (-1.2 * feasible["price_z"].to_numpy()
                + 0.6 * hid.loc[feasible["vendor_id"], "owner_belief"].to_numpy()
                + rng.gumbel(0, 1.0, len(feasible)))
        o = feasible.iloc[int(np.argmax(util))]
        h = hid.loc[o.vendor_id]
        qty = max(qty, int(o.moq))

        # Hasil pengiriman.
        frac = (d - start).days / total_days
        logit = _logit(h.true_reliability) + h.drift_logit * (frac - 0.5)
        capacity = h.capacity_factor * m.qty_median
        logit -= 0.8 * max(0.0, qty / capacity - 1.0)
        if is_pk:
            logit -= 1.5 * h.peak_sensitivity
        if rng.uniform() < _sigmoid(logit):
            actual = int(o.lead_time_janji) - int(rng.binomial(2, 0.3))
        else:
            lam_late = 1.0 + 3.0 * (1 - h.true_reliability) + (1.5 if is_pk else 0.0)
            actual = int(o.lead_time_janji) + 1 + int(rng.poisson(lam_late))
        actual = max(1, actual)

        # Cacat: tingkat dasar vendor, lebih tinggi saat musim ramai, plus variasi per batch.
        # Barang per roll jarang cacat, dan qty-nya kecil (1 roll cacat dari 10 = 10%), jadi diturunkan.
        p_def = h.true_defect_rate * (1 + 0.5 * h.peak_sensitivity * is_pk) * rng.lognormal(0, 0.4)
        if m.satuan == "roll":
            p_def *= ROLL_DEFECT_MULT
        qty_cacat = int(rng.binomial(qty, min(p_def, 0.5)))

        # Harga transaksi: harga daftar saat ini disesuaikan inflasi mundur dan sedikit naik saat musim ramai.
        months_before_end = (end - d).days / 30.4
        price = o.harga_satuan * (1 - INFLATION_PER_MONTH) ** months_before_end
        price *= (1.03 if is_pk else 1.0) * rng.lognormal(0, 0.02)
        price = _round_price(price)

        rows.append({
            "tanggal_pesan": d.date(),
            "vendor_id": o.vendor_id,
            "material_id": mid,
            "offer_id": o.offer_id,
            "qty": qty,
            "harga_satuan": price,
            "total_harga": price * qty,
            "lead_time_janji": int(o.lead_time_janji),
            "hari_tersedia": hari,
            "deadline": (d + pd.Timedelta(days=hari)).date(),
            "tanggal_datang": (d + pd.Timedelta(days=actual)).date(),
            "qty_cacat": qty_cacat,
        })
    tx = pd.DataFrame(rows)
    tx.insert(0, "transaksi_id", [f"T{i + 1:05d}" for i in range(len(tx))])
    return tx


def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    materials = generate_materials()
    vendors, hidden = generate_vendors(rng)
    offers = generate_offers(rng, vendors, materials, hidden)
    transactions = generate_transactions(rng, vendors, materials, offers, hidden)
    return {
        "vendors": vendors,
        "vendors_hidden": hidden.drop(columns="owner_belief"),
        "materials": materials.drop(columns="popularitas"),
        "offers": offers,
        "transactions": transactions,
    }


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name, df in generate().items():
        df.to_csv(DATA_DIR / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(df)} baris")


if __name__ == "__main__":
    main()
