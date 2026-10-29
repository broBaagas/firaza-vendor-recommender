"""Kalender musim ramai Firaza: Ramadan-Lebaran (hampers) dan musim nikah.

Dipakai bersama oleh generator (efek musim pada permintaan dan kinerja vendor)
dan oleh core.features (fitur musim_ramai). Tanggal hijriah adalah perkiraan
kalender pemerintah; selisih 1 hari tidak berpengaruh berarti.
"""

import numpy as np
import pandas as pd

# (mulai, selesai, nama), inklusif.
PEAK_PERIODS = [
    # Order hampers mulai ±3 minggu sebelum Ramadan sampai Idulfitri.
    ("2025-02-08", "2025-03-31", "Ramadan-Lebaran 2025"),  # 1 Ramadan 1446 ≈ 1 Mar, Idulfitri 31 Mar
    ("2025-04-01", "2025-04-29", "Musim nikah Syawal 2025"),
    ("2025-05-28", "2025-06-25", "Musim nikah Dzulhijjah 2025"),  # Iduladha 6 Jun
    ("2026-01-28", "2026-03-20", "Ramadan-Lebaran 2026"),  # 1 Ramadan 1447 ≈ 18 Feb, Idulfitri 20 Mar
    ("2026-03-21", "2026-04-18", "Musim nikah Syawal 2026"),
    ("2026-05-18", "2026-06-15", "Musim nikah Dzulhijjah 2026"),  # Iduladha 27 Mei
]

_RANGES = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b, _ in PEAK_PERIODS]


def is_peak(dates) -> np.ndarray:
    """Array bool: True jika tanggal jatuh di salah satu periode musim ramai."""
    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    mask = np.zeros(len(d), dtype=bool)
    for start, end in _RANGES:
        mask |= ((d >= start) & (d <= end)).to_numpy()
    return mask
