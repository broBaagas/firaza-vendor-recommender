"""Generator dataset sintetis Firaza (numpy/pandas, seed tetap).

Keluaran di folder data/:
- vendors.csv               ~30 vendor
- materials.csv             ~15 material
- offers.csv                ~120 penawaran (vendor x material: harga, lead_time_janji, MOQ)
- transactions.csv          ~6.000 transaksi selama 24 bulan (wajib ada kolom deadline)
- vendors_hidden.csv        sifat tersembunyi vendor (true reliability/quality), hanya untuk evaluasi

Jalankan: python -m generator.generate
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_VENDORS = 30
N_MATERIALS = 15
N_MONTHS = 24
N_TRANSACTIONS = 6000
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def generate_vendors(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kembalikan (vendors, vendors_hidden)."""
    raise NotImplementedError


def generate_materials(rng: np.random.Generator) -> pd.DataFrame:
    raise NotImplementedError


def generate_offers(rng: np.random.Generator, vendors: pd.DataFrame, materials: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def generate_transactions(
    rng: np.random.Generator, offers: pd.DataFrame, hidden: pd.DataFrame
) -> pd.DataFrame:
    """Simulasi transaksi dengan efek musim ramai (Ramadan, Lebaran, musim nikah)."""
    raise NotImplementedError


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(exist_ok=True)
    vendors, hidden = generate_vendors(rng)
    materials = generate_materials(rng)
    offers = generate_offers(rng, vendors, materials)
    transactions = generate_transactions(rng, offers, hidden)

    vendors.to_csv(DATA_DIR / "vendors.csv", index=False)
    hidden.to_csv(DATA_DIR / "vendors_hidden.csv", index=False)
    materials.to_csv(DATA_DIR / "materials.csv", index=False)
    offers.to_csv(DATA_DIR / "offers.csv", index=False)
    transactions.to_csv(DATA_DIR / "transactions.csv", index=False)


if __name__ == "__main__":
    main()
