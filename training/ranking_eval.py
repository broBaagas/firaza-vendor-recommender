"""Evaluasi ranking dengan ground truth dari generator.

Data historis hanya mencatat hasil vendor yang DIPILIH owner, sehingga NDCG/Top-1 tidak bisa
dihitung langsung. Untuk setiap kebutuhan uji, peluang berhasil SEBENARNYA semua kandidat
dihitung dengan generator.generate.true_success_prob (memakai sifat tersembunyi vendor).

Relevansi kandidat = skor berbobot yang sama dengan ranking, tetapi memakai P(berhasil)
sebenarnya. Strategi sempurna (P diketahui) mendapat NDCG = 1 dan Top-1 = 100%.
Gain NDCG = relevansi - relevansi terendah dalam kebutuhan itu, sehingga kandidat terburuk
bernilai 0 dan perbedaan antarstrategi tidak tertutup oleh skor dasar yang tinggi.
"""

import numpy as np
import pandas as pd

from core.features import need_offer_features, prepare_transactions, vendor_history
from core.ranking import PRESETS, filter_candidates
from generator.generate import true_success_prob

K = 3


def build_eval_candidates(needs: pd.DataFrame, data: dict, hidden: pd.DataFrame) -> pd.DataFrame:
    """Satu baris per (kebutuhan, kandidat lolos saringan), lengkap dengan fitur dan P sebenarnya.

    `needs` butuh kolom need_id, material_id, qty, hari_tersedia, tanggal. Kebutuhan dengan
    kurang dari 2 kandidat dibuang (tidak ada yang perlu diranking).
    """
    parts = []
    for n in needs.itertuples():
        need = {"material_id": n.material_id, "qty": n.qty, "hari_tersedia": n.hari_tersedia, "tanggal": n.tanggal}
        c = filter_candidates(data["offers"], need, data["vendors"])
        if len(c) < 2:
            continue
        c = c.assign(need_id=n.need_id, qty=n.qty, hari_tersedia=n.hari_tersedia, tanggal=pd.Timestamp(n.tanggal))
        parts.append(c)
    cand = pd.concat(parts, ignore_index=True)

    tx = prepare_transactions(data["transactions"])
    base = need_offer_features(cand, data["materials"], cand["tanggal"])
    hist = vendor_history(tx, cand["vendor_id"], cand["tanggal"])
    keep = cand[["need_id", "offer_id", "vendor_id", "material_id", "harga_satuan", "tanggal"]]
    out = pd.concat([keep, base, hist], axis=1)

    hid = hidden.set_index("vendor_id")
    out["p_sebenarnya"] = [
        true_success_prob(r.vendor_id, r.material_id, r.qty, r.lead_time_janji, r.hari_tersedia, r.tanggal,
                          hid, data["materials"])
        for r in out.itertuples()
    ]
    return out


def _minmax(s: pd.Series) -> pd.Series:
    span = s.max() - s.min()
    return (s - s.min()) / span if span > 0 else s * 0.0


def utility(df: pd.DataFrame, p: np.ndarray, preset: str) -> pd.Series:
    """Skor berbobot per kandidat, dinormalisasi per kebutuhan (sama seperti core.ranking.score)."""
    w = PRESETS[preset]
    g = df.groupby("need_id")
    harga_norm = g["harga_satuan"].transform(_minmax)
    lt_norm = g["lead_time_janji"].transform(_minmax)
    return w["w_berhasil"] * pd.Series(p, index=df.index) + w["w_harga"] * (1 - harga_norm) + w["w_cepat"] * (1 - lt_norm)


def _ndcg_at_k(order_gain: np.ndarray, ideal_gain: np.ndarray, k: int = K) -> float:
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (order_gain[:k] * disc[:len(order_gain[:k])]).sum()
    idcg = (ideal_gain[:k] * disc[:len(ideal_gain[:k])]).sum()
    return 1.0 if idcg == 0 else dcg / idcg


def evaluate_strategy(df: pd.DataFrame, strategy_score: pd.Series, preset: str) -> dict:
    """NDCG@3, Top-1 hit, regret, dan profil pilihan #1 untuk satu strategi pengurutan.

    `strategy_score`: makin besar makin diutamakan. Seri dipecah dengan harga termurah.
    """
    rel = utility(df, df["p_sebenarnya"].to_numpy(), preset)
    d = df.assign(_s=strategy_score.to_numpy(), _rel=rel.to_numpy())
    ndcg, hit, regret, top_p, top_price_ratio = [], [], [], [], []
    for _, g in d.groupby("need_id", sort=False):
        gain = (g["_rel"] - g["_rel"].min()).to_numpy()
        order = np.lexsort((g["harga_satuan"].to_numpy(), -g["_s"].to_numpy()))
        ndcg.append(_ndcg_at_k(gain[order], np.sort(gain)[::-1]))
        best = g["_rel"].max()
        top = g.iloc[order[0]]
        hit.append(np.isclose(top["_rel"], best))
        regret.append(best - top["_rel"])
        top_p.append(top["p_sebenarnya"])
        top_price_ratio.append(top["harga_satuan"] / g["harga_satuan"].min())
    return {
        "ndcg@3": float(np.mean(ndcg)),
        "top1_hit": float(np.mean(hit)),
        "regret": float(np.mean(regret)),
        "p_berhasil_pilihan1": float(np.mean(top_p)),
        "harga_pilihan1_vs_termurah": float(np.mean(top_price_ratio)),
    }


def compare_strategies(df: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Bandingkan strategi untuk setiap preset.

    `predictions`: nama -> P(berhasil) prediksi per baris df. Strategi "Pilih termurah" dan
    "Sempurna (P sebenarnya)" ditambahkan otomatis sebagai pembanding bawah/atas.
    """
    rows = []
    for preset in PRESETS:
        strategies = {"Pilih termurah": -df["harga_satuan"].astype(float)}
        for name, p in predictions.items():
            strategies[name] = utility(df, p, preset)
        strategies["Sempurna (P sebenarnya)"] = utility(df, df["p_sebenarnya"].to_numpy(), preset)
        for name, s in strategies.items():
            rows.append({"preset": preset, "strategi": name, **evaluate_strategy(df, s, preset)})
    return pd.DataFrame(rows)
