"""Memuat CSV hasil generator. vendors_hidden.csv sengaja TIDAK dimuat di sini."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_DATE_COLUMNS = {
    "vendors": ["tanggal_bergabung"],
    "materials": [],
    "offers": [],
    "transactions": ["tanggal_pesan", "deadline", "tanggal_datang"],
}


def load_data(data_dir: Path | str = DATA_DIR) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    return {
        name: pd.read_csv(data_dir / f"{name}.csv", parse_dates=dates)
        for name, dates in _DATE_COLUMNS.items()
    }
