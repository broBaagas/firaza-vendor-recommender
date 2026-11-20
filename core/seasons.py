"""Kalender musim ramai Firaza: Ramadan-Lebaran (hampers) dan musim nikah.

Dipakai bersama oleh generator (efek musim pada permintaan dan kinerja vendor)
dan oleh core.features (fitur musim_ramai). Tanggal hijriah adalah perkiraan
kalender pemerintah; selisih 1 hari tidak berpengaruh berarti.

Periode 2027 dipakai hanya saat inferensi (pesanan ke depan); data latih berakhir Sep 2026.
Perbarui tabel ini setiap tahun (misalnya dari SKB hari libur nasional) dan naikkan
CALENDAR_END agar rekomendasi untuk tanggal yang lebih jauh diizinkan.
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
    # Perkiraan hisab: 1 Ramadan 1448 ≈ 8 Feb 2027, Idulfitri ≈ 10 Mar 2027, Iduladha ≈ 16 Mei 2027.
    ("2027-01-18", "2027-03-10", "Ramadan-Lebaran 2027"),
    ("2027-03-11", "2027-04-08", "Musim nikah Syawal 2027"),
    ("2027-05-08", "2027-06-05", "Musim nikah Dzulhijjah 2027"),
    # Musim ramai berikutnya (Ramadan 1449 ≈ akhir Jan 2028) dimulai setelah CALENDAR_END.
]

# Tanggal terakhir yang tercakup kalender. Pesanan setelah ini ditolak karena musim ramai
# dan hari besarnya belum dicatat (model akan menganggapnya hari biasa).
CALENDAR_END = pd.Timestamp("2027-12-31")

# Hari besar / libur nasional untuk rentang tanggal pesan yang bisa dipilih (Des 2026 – Des 2027).
# Tanggal tetap pasti; hari besar keagamaan bertanda (p) adalah perkiraan hisab/astronomis dan
# perlu dicocokkan dengan SKB 3 Menteri tentang hari libur nasional 2027.
HOLIDAYS = [
    ("2026-12-25", "Hari Raya Natal"),
    ("2027-01-01", "Tahun Baru Masehi"),
    ("2027-01-05", "Isra Mikraj Nabi Muhammad SAW"),  # (p) 27 Rajab 1448
    ("2027-02-06", "Tahun Baru Imlek"),
    ("2027-03-09", "Hari Suci Nyepi"),  # (p)
    ("2027-03-10", "Hari Raya Idulfitri"),  # (p) 1 Syawal 1448
    ("2027-03-11", "Hari Raya Idulfitri (hari kedua)"),  # (p)
    ("2027-03-26", "Wafat Yesus Kristus (Jumat Agung)"),
    ("2027-03-28", "Hari Paskah"),
    ("2027-05-01", "Hari Buruh Internasional"),
    ("2027-05-06", "Kenaikan Yesus Kristus"),
    ("2027-05-16", "Hari Raya Iduladha"),  # (p) 10 Dzulhijjah 1448
    ("2027-05-20", "Hari Raya Waisak"),  # (p)
    ("2027-06-01", "Hari Lahir Pancasila"),
    ("2027-06-06", "Tahun Baru Islam"),  # (p) 1 Muharram 1449
    ("2027-08-15", "Maulid Nabi Muhammad SAW"),  # (p) 12 Rabiulawal 1449
    ("2027-08-17", "Hari Kemerdekaan Republik Indonesia"),
    ("2027-12-25", "Hari Raya Natal"),
]
_HOLIDAY_MAP = {pd.Timestamp(d): name for d, name in HOLIDAYS}

_RANGES = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b, _ in PEAK_PERIODS]


def is_peak(dates) -> np.ndarray:
    """Array bool: True jika tanggal jatuh di salah satu periode musim ramai."""
    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    mask = np.zeros(len(d), dtype=bool)
    for start, end in _RANGES:
        mask |= ((d >= start) & (d <= end)).to_numpy()
    return mask


def peak_period(date) -> dict | None:
    """Periode musim ramai yang memuat `date`, atau None.

    Hasil: {"nama": "Ramadan-Lebaran 2027", "jenis": "Ramadan-Lebaran", "mulai": "2027-01-18", "selesai": ...}.
    `jenis` adalah nama tanpa tahun (dipakai di konteks Gemini, yang dilarang menulis tahun).
    """
    d = pd.Timestamp(date).normalize()
    for (start, end), (a, b, name) in zip(_RANGES, PEAK_PERIODS):
        if start <= d <= end:
            return {"nama": name, "jenis": name.rsplit(" ", 1)[0], "mulai": a, "selesai": b}
    return None


def holiday_on(date) -> str | None:
    """Nama hari besar nasional pada `date`, atau None jika hari biasa."""
    return _HOLIDAY_MAP.get(pd.Timestamp(date).normalize())


def holidays_between(start, end) -> list[dict]:
    """Hari besar dalam rentang [start, end] inklusif: [{"tanggal": "YYYY-MM-DD", "nama": ...}]."""
    a, b = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    return [{"tanggal": f"{d:%Y-%m-%d}", "nama": n} for d, n in sorted(_HOLIDAY_MAP.items()) if a <= d <= b]


def order_calendar(order_date, days_available: int) -> dict:
    """Ringkasan kalender untuk satu pesanan: hari besar di tanggal pesan dan selama masa
    pengiriman (tanggal pesan s.d. deadline), serta musim ramai.

    `waktu_rawan` True jika ada hari besar di rentang itu atau tanggal pesan jatuh di musim ramai.
    """
    start = pd.Timestamp(order_date).normalize()
    deadline = start + pd.Timedelta(days=int(days_available))
    in_window = holidays_between(start, deadline)
    musim = peak_period(start)
    return {
        "tanggal_pesan": f"{start:%Y-%m-%d}",
        "deadline": f"{deadline:%Y-%m-%d}",
        "hari_besar_tanggal_pesan": holiday_on(start),
        "hari_besar_periode": in_window,
        "musim_ramai": musim,
        "waktu_rawan": bool(in_window or musim),
    }
