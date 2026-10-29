import pandas as pd
import pytest

from core.features import FEATURE_COLUMNS, make_label, prepare_transactions, vendor_history


def _tx(rows):
    cols = ["transaksi_id", "vendor_id", "tanggal_pesan", "deadline", "tanggal_datang",
            "lead_time_janji", "qty", "qty_cacat"]
    return pd.DataFrame(rows, columns=cols)


def test_label_requires_on_time_and_low_defect():
    tx = _tx([
        ("T1", "V1", "2025-01-01", "2025-01-10", "2025-01-10", 7, 100, 3),  # tepat, cacat 3% -> 1
        ("T2", "V1", "2025-01-01", "2025-01-10", "2025-01-11", 7, 100, 0),  # telat -> 0
        ("T3", "V1", "2025-01-01", "2025-01-10", "2025-01-05", 7, 100, 4),  # cacat 4% -> 0
    ])
    assert list(make_label(tx)) == [1, 0, 0]


def test_history_ignores_orders_not_yet_arrived():
    tx = prepare_transactions(_tx([
        ("T1", "V1", "2025-01-01", "2025-01-10", "2025-01-05", 4, 100, 0),  # datang 5 Jan, berhasil
        ("T2", "V1", "2025-01-03", "2025-01-06", "2025-01-20", 3, 100, 0),  # datang 20 Jan, gagal
    ]))
    h = vendor_history(tx, ["V1", "V1", "V1"], pd.to_datetime(["2025-01-05", "2025-01-10", "2025-01-21"]))
    # Pada 5 Jan belum ada yang datang (strictly before), 10 Jan baru T1, 21 Jan keduanya.
    assert list(h["jumlah_transaksi"]) == [0, 1, 2]
    # Pesanan T2 sudah dibuat sebelum 10 Jan, jadi hari sejak pesanan terakhir = 7.
    assert h.loc[1, "hari_sejak_pesanan_terakhir"] == 7
    assert h.loc[2, "tingkat_berhasil_10"] < h.loc[1, "tingkat_berhasil_10"]


def test_history_for_unknown_vendor_uses_prior():
    tx = prepare_transactions(_tx([("T1", "V1", "2025-01-01", "2025-01-10", "2025-01-05", 4, 100, 0)]))
    h = vendor_history(tx, ["V9"], pd.Timestamp("2025-02-01"))
    assert h.loc[0, "jumlah_transaksi"] == 0
    assert h.loc[0, "tingkat_berhasil_10"] == pytest.approx(0.7)


def test_training_frame_has_no_leakage(training_frame):
    tf = training_frame
    tx = prepare_transactions(pd.read_csv("data/transactions.csv"))
    # Hitung ulang secara naif untuk sampel baris dan bandingkan.
    for _, row in tf.sample(25, random_state=0).iterrows():
        past = tx[(tx["vendor_id"] == row["vendor_id"]) & (tx["tanggal_datang"] < row["tanggal_pesan"])]
        assert row["jumlah_transaksi"] == len(past)


def test_training_frame_complete(training_frame):
    assert training_frame[FEATURE_COLUMNS].notna().all().all()
    assert set(training_frame["berhasil"].unique()) <= {0, 1}
