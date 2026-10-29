import pandas as pd
import pytest

from core.ranking import PRESETS, filter_candidates, recommend, resolve_weights, score


@pytest.fixture
def offers():
    return pd.DataFrame({
        "offer_id": ["O1", "O2", "O3", "O4"],
        "vendor_id": ["V1", "V2", "V3", "V4"],
        "material_id": ["M01", "M01", "M01", "M02"],
        "harga_satuan": [10_000, 12_000, 8_000, 5_000],
        "lead_time_janji": [5, 3, 10, 2],
        "moq": [50, 10, 100, 10],
    })


@pytest.mark.parametrize("name", list(PRESETS))
def test_preset_weights_sum_to_one(name):
    assert sum(PRESETS[name].values()) == pytest.approx(1.0)


def test_custom_weights_are_normalized():
    w = resolve_weights(weights={"w_berhasil": 2, "w_harga": 1, "w_cepat": 1})
    assert w == pytest.approx({"w_berhasil": 0.5, "w_harga": 0.25, "w_cepat": 0.25})


def test_unknown_preset_rejected():
    with pytest.raises(ValueError):
        resolve_weights("Asal")


def test_filter_material_moq_leadtime_budget(offers):
    need = {"material_id": "M01", "qty": 60, "hari_tersedia": 7}
    assert list(filter_candidates(offers, need)["offer_id"]) == ["O1", "O2"]  # O3: MOQ & lead time
    need["budget"] = 650_000  # O1 = 600rb, O2 = 720rb
    assert list(filter_candidates(offers, need)["offer_id"]) == ["O1"]


def test_score_formula_and_order(offers):
    c = offers[offers["material_id"] == "M01"].reset_index(drop=True)
    ranked = score(c, [0.9, 0.5, 0.6], preset="Seimbang")
    # O1: 0.5*0.9 + 0.3*(1-0.5) + 0.2*(1-2/7) = 0.7429
    o1 = ranked.set_index("offer_id").loc["O1"]
    assert o1["skor"] == pytest.approx(0.45 + 0.15 + 0.2 * (5 / 7))
    assert list(ranked["peringkat"]) == [1, 2, 3]
    assert ranked["skor"].is_monotonic_decreasing


def test_score_equal_candidates_normalize_to_zero(offers):
    c = offers.iloc[[0]].reset_index(drop=True)
    ranked = score(c, [0.8])
    assert ranked.loc[0, "harga_norm"] == 0 and ranked.loc[0, "leadtime_norm"] == 0


def test_recommend_end_to_end_on_generated_data():
    r = recommend({"material_id": "M01", "qty": 200, "hari_tersedia": 14}, top_k=3)
    assert r["jumlah_kandidat"] >= 3 and len(r["top"]) == 3
    assert all(0 <= row["p_berhasil"] <= 1 for row in r["ranking"])
    assert [row["peringkat"] for row in r["top"]] == [1, 2, 3]


def test_recommend_no_candidates_returns_message():
    r = recommend({"material_id": "M05", "qty": 5, "hari_tersedia": 1})
    assert r["ranking"] == [] and "pesan" in r


def test_recommend_validates_need():
    with pytest.raises(ValueError):
        recommend({"material_id": "M01", "qty": 0, "hari_tersedia": 5})
