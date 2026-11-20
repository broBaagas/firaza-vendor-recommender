"""Musim ramai: kalender, efek pada ranking, dan penjelasan."""

import pandas as pd
import pytest

from core.data import load_data
from core.explain import build_context, template_explanation, validate_explanation
from core.features import build_candidate_features, prepare_transactions
from core.ranking import filter_candidates, recommend, safest
from core.seasons import (CALENDAR_END, HOLIDAYS, PEAK_PERIODS, holiday_on, holidays_between, is_peak,
                          order_calendar, peak_period)

NEED = {"material_id": "M03", "qty": 200, "hari_tersedia": 12}
LEBARAN_2027 = "2027-02-15"
BIASA_2026 = "2026-11-20"


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_periods_sorted_and_not_overlapping():
    ranges = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b, _ in PEAK_PERIODS]
    for (a, b), (c, _) in zip(ranges, ranges[1:]):
        assert a <= b < c
    assert ranges[-1][1] <= CALENDAR_END


def test_peak_period_lookup():
    p = peak_period(LEBARAN_2027)
    assert p["nama"] == "Ramadan-Lebaran 2027" and p["jenis"] == "Ramadan-Lebaran"
    assert peak_period(BIASA_2026) is None
    assert is_peak([LEBARAN_2027, BIASA_2026]).tolist() == [True, False]


def test_future_date_uses_latest_history(data):
    """Tanggal ke depan hanya mengubah fitur musim/bulan; rekam jejak tetap per data terakhir."""
    cand = filter_candidates(data["offers"], NEED)
    default = build_candidate_features(NEED, cand, data["transactions"], data["materials"])
    future = build_candidate_features({**NEED, "tanggal": LEBARAN_2027}, cand, data["transactions"],
                                      data["materials"])
    assert future["musim_ramai"].eq(1).all() and default["musim_ramai"].eq(0).all()
    assert (future["bulan"] == 2).all()
    for col in ["jumlah_transaksi", "hari_sejak_pesanan_terakhir", "tingkat_berhasil_10"]:
        pd.testing.assert_series_equal(future[col], default[col])


def test_past_date_still_replays_history(data):
    """Tanggal di dalam rentang data tetap memakai rekam jejak per tanggal itu (perilaku lama)."""
    cand = filter_candidates(data["offers"], NEED)
    past = build_candidate_features({**NEED, "tanggal": "2025-06-01"}, cand, data["transactions"],
                                    data["materials"])
    tx = prepare_transactions(data["transactions"])
    v = past.iloc[0]["vendor_id"]
    n = ((tx["vendor_id"] == v) & (tx["tanggal_datang"] < "2025-06-01")).sum()
    assert past.iloc[0]["jumlah_transaksi"] == n


def test_peak_lowers_success_probability(data):
    biasa = recommend({**NEED, "tanggal": BIASA_2026}, data=data)
    ramai = recommend({**NEED, "tanggal": LEBARAN_2027}, data=data)
    assert biasa["musim_ramai"] is None
    assert ramai["musim_ramai"]["nama"] == "Ramadan-Lebaran 2027"
    assert ramai["tanggal_pesan"] == LEBARAN_2027
    p_biasa = {r["offer_id"]: r["p_berhasil"] for r in biasa["ranking"]}
    p_ramai = {r["offer_id"]: r["p_berhasil"] for r in ramai["ranking"]}
    assert sum(p_ramai.values()) / len(p_ramai) < sum(p_biasa.values()) / len(p_biasa) - 0.05


def test_date_after_calendar_rejected(data):
    with pytest.raises(ValueError, match="kalender"):
        recommend({**NEED, "tanggal": "2028-01-15"}, data=data)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        recommend({**NEED, "tanggal": "bukan-tanggal"}, data=data)


def test_template_mentions_peak_and_passes_validation(data):
    result = recommend({**NEED, "tanggal": LEBARAN_2027}, data=data)
    ctx = build_context(result, data["materials"])
    assert ctx["musim"] == {"musim_ramai": True, "nama": "Ramadan-Lebaran"}
    expl = template_explanation(ctx)
    text = " ".join(expl["alasan"] + expl["saran"])
    assert "musim ramai (Ramadan-Lebaran)" in text
    assert validate_explanation(expl, ctx)["valid"]


def test_holiday_lookup():
    assert holiday_on("2027-08-17") == "Hari Kemerdekaan Republik Indonesia"
    assert holiday_on("2027-08-18") is None
    assert [h["nama"] for h in holidays_between("2027-08-10", "2027-08-20")] == [
        "Maulid Nabi Muhammad SAW", "Hari Kemerdekaan Republik Indonesia"]
    assert all(pd.Timestamp(d) <= CALENDAR_END for d, _ in HOLIDAYS)


def test_order_calendar_holiday_day():
    k = order_calendar("2027-08-17", 12)
    assert k["hari_besar_tanggal_pesan"] == "Hari Kemerdekaan Republik Indonesia"
    assert k["deadline"] == "2027-08-29" and k["waktu_rawan"] and k["musim_ramai"] is None


def test_order_calendar_holiday_before_deadline():
    k = order_calendar("2027-12-20", 10)  # Natal 25 Des di masa pengiriman
    assert k["hari_besar_tanggal_pesan"] is None
    assert [h["nama"] for h in k["hari_besar_periode"]] == ["Hari Raya Natal"] and k["waktu_rawan"]


def test_order_calendar_normal_day():
    k = order_calendar(BIASA_2026, 12)
    assert k["hari_besar_tanggal_pesan"] is None and k["hari_besar_periode"] == []
    assert not k["waktu_rawan"]


def test_safest_is_highest_success(data):
    r = recommend({**NEED, "tanggal": "2027-08-17"}, data=data)
    assert r["kalender"]["waktu_rawan"]
    best = max(round(x["p_berhasil"], 2) for x in r["ranking"])
    assert round(r["paling_aman"]["p_berhasil"], 2) == best
    assert safest([]) is None


@pytest.mark.parametrize("tanggal, material", [("2027-08-17", "M03"), ("2027-08-10", "M07"),
                                               ("2027-12-20", "M01"), ("2027-02-15", "M03")])
def test_template_safest_advice_passes_validation(data, tanggal, material):
    r = recommend({"material_id": material, "qty": 500 if material == "M07" else 200, "hari_tersedia": 12,
                   "tanggal": tanggal}, data=data)
    ctx = build_context(r, data["materials"])
    assert ctx["kalender"]["waktu_rawan"] and "vendor_paling_aman" in ctx
    expl = template_explanation(ctx)
    assert any("paling aman" in s for s in expl["saran"])
    assert validate_explanation(expl, ctx)["valid"]


def test_no_safest_advice_on_normal_day(data):
    ctx = build_context(recommend({**NEED, "tanggal": BIASA_2026}, data=data), data["materials"])
    assert "vendor_paling_aman" not in ctx and ctx["kalender"] == {"waktu_rawan": False, "hari_besar": []}


def test_context_without_peak(data):
    ctx = build_context(recommend({**NEED, "tanggal": BIASA_2026}, data=data), data["materials"])
    assert ctx["musim"] == {"musim_ramai": False, "nama": None}
    assert "musim ramai" not in " ".join(template_explanation(ctx)["saran"])
