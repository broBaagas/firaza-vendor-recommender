import numpy as np
import pandas as pd
import pytest

from core.features import FEATURE_COLUMNS, make_label
from core.ranking import MODEL_PATH, load_model
from generator.generate import true_success_prob
from training.ranking_eval import _ndcg_at_k, evaluate_strategy
from training.train import select_model


@pytest.fixture(scope="module")
def hidden():
    return pd.read_csv("data/vendors_hidden.csv").set_index("vendor_id")


def _p(hidden, generated, **kw):
    args = {"vendor_id": "V01", "material_id": "M01", "qty": 150, "lead_time_janji": 10,
            "hari_tersedia": 14, "tanggal": "2025-09-10"} | kw
    return true_success_prob(**args, hidden=hidden, materials=generated["materials"])


def test_true_prob_responds_to_slack_and_season(hidden, generated):
    p_tight = _p(hidden, generated, hari_tersedia=10)
    p_loose = _p(hidden, generated, hari_tersedia=20)
    assert 0 <= p_tight < p_loose <= 1
    assert _p(hidden, generated, tanggal="2026-03-01") < _p(hidden, generated)  # musim ramai


def test_true_prob_matches_observed_rate(hidden, generated):
    tx = generated["transactions"].sample(600, random_state=1)
    p = [true_success_prob(r.vendor_id, r.material_id, r.qty, r.lead_time_janji, r.hari_tersedia,
                           r.tanggal_pesan, hidden, generated["materials"]) for r in tx.itertuples()]
    assert np.mean(p) == pytest.approx(make_label(tx).mean(), abs=0.04)


def test_ndcg_perfect_and_worse():
    ideal = np.array([0.5, 0.3, 0.0])
    assert _ndcg_at_k(ideal, ideal) == pytest.approx(1.0)
    assert _ndcg_at_k(ideal[::-1], ideal) < 1.0


def test_evaluate_strategy_perfect_vs_reverse():
    df = pd.DataFrame({
        "need_id": ["N1"] * 3,
        "harga_satuan": [100, 100, 100],
        "lead_time_janji": [5, 5, 5],
        "p_sebenarnya": [0.9, 0.6, 0.3],
    })
    good = evaluate_strategy(df, df["p_sebenarnya"], "Seimbang")
    bad = evaluate_strategy(df, -df["p_sebenarnya"], "Seimbang")
    assert good["top1_hit"] == 1.0 and good["ndcg@3"] == pytest.approx(1.0) and good["regret"] == 0
    # gain = 0.5 * (p - 0.3) = [0.3, 0.15, 0] dibalik: DCG = 0.15/log2(3) + 0.3/2, IDCG = 0.3 + 0.15/log2(3)
    assert bad["top1_hit"] == 0.0 and bad["ndcg@3"] == pytest.approx(0.6199, abs=1e-3)


def test_select_model_prefers_simpler_when_close():
    t = pd.DataFrame({"model": ["baseline", "logreg", "random_forest", "lightgbm"],
                      "cv_roc_auc": [0.6, 0.790, 0.793, 0.800]})
    assert select_model(t) == "lightgbm"  # unggul 0,007 > toleransi 0,005
    t.loc[t["model"] == "random_forest", "cv_roc_auc"] = 0.797
    assert select_model(t) == "random_forest"  # dalam toleransi -> yang lebih sederhana
    t.loc[t["model"] == "baseline", "cv_roc_auc"] = 0.9
    assert select_model(t) == "random_forest"  # baseline tidak ikut dipilih


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model belum dilatih (python -m training.train)")
def test_saved_model_predicts_probabilities(training_frame):
    model = load_model()
    assert model["version"].startswith("v1-") and model["features"] == FEATURE_COLUMNS
    p = model["pipeline"].predict_proba(training_frame[FEATURE_COLUMNS].head(200))[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
