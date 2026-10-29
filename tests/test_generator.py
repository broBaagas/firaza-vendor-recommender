import pandas as pd

from core.features import make_label
from generator.generate import generate


def test_generator_is_deterministic(generated):
    again = generate()
    for name in generated:
        pd.testing.assert_frame_equal(generated[name], again[name])


def test_committed_csv_matches_generator(generated):
    tx = pd.read_csv("data/transactions.csv")
    assert len(tx) == len(generated["transactions"])
    assert tx["total_harga"].sum() == generated["transactions"]["total_harga"].sum()


def test_generated_data_shape_and_integrity(generated):
    v, m, o, tx = (generated[k] for k in ["vendors", "materials", "offers", "transactions"])
    assert len(v) == 30 and len(m) == 15
    assert 100 <= len(o) <= 130
    assert 5_500 <= len(tx) <= 6_500
    assert set(tx["offer_id"]) <= set(o["offer_id"])
    assert set(o["vendor_id"]) == set(v["vendor_id"])  # setiap vendor punya penawaran
    assert tx["deadline"].notna().all()
    assert (pd.to_datetime(tx["tanggal_datang"]) > pd.to_datetime(tx["tanggal_pesan"])).all()
    assert (tx["qty_cacat"] <= tx["qty"]).all()


def test_label_rate_is_reasonable(generated):
    rate = make_label(generated["transactions"]).mean()
    assert 0.6 <= rate <= 0.9
