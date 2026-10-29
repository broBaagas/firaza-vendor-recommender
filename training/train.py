"""Training model P(berhasil): tuning, evaluasi, pemilihan, dan penyimpanan model v1.

Alur:
1. Split waktu: train = bulan 1-18 (s.d. Mar 2026), test = bulan 19-24 (Apr-Sep 2026).
2. Tuning tiap model dengan GridSearchCV + TimeSeriesSplit (4 lipatan) HANYA di data train.
3. Pemilihan: ROC-AUC validasi silang tertinggi. Jika selisih < 0,005, dipilih model yang lebih
   sederhana (urutan MODEL_ORDER), karena lebih mudah dijelaskan dan lebih stabil.
4. Data test dipakai sekali untuk melaporkan metrik (ROC-AUC, Brier, log loss) dan evaluasi
   ranking (NDCG@3, Top-1) terhadap ground truth generator.
5. Model terpilih dilatih ulang pada SELURUH data (bulan 1-24) lalu disimpan ke
   models/model.joblib untuk dipakai core.ranking.

Jalankan: python -m training.train
"""

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from core.data import DATA_DIR, load_data
from core.features import FEATURE_COLUMNS, build_training_frame
from core.ranking import MODEL_PATH
from generator.generate import true_success_prob
from training.models import MODEL_LABELS, MODEL_ORDER, PARAM_GRIDS, build_pipeline
from training.ranking_eval import build_eval_candidates, compare_strategies

TRAIN_END = "2026-04-01"
CV_SPLITS = 4
SIMPLER_TOLERANCE = 0.005
MODEL_VERSION = "v1"
METRICS_PATH = MODEL_PATH.parent / "metrics.json"
BASELINE_COLUMN = "tingkat_berhasil_semua"


def _prob_metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"roc_auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p)}


def _baseline_cv(train: pd.DataFrame, cv: TimeSeriesSplit) -> tuple[float, float]:
    scores = [roc_auc_score(train["berhasil"].iloc[va], train[BASELINE_COLUMN].iloc[va])
              for _, va in cv.split(train)]
    return float(np.mean(scores)), float(np.std(scores))


def select_model(cv_table: pd.DataFrame) -> str:
    models = cv_table[cv_table["model"].isin(MODEL_ORDER)]
    best = models["cv_roc_auc"].max()
    close = models[models["cv_roc_auc"] >= best - SIMPLER_TOLERANCE]["model"]
    return min(close, key=MODEL_ORDER.index)


def run(save: bool = True, data_dir: Path = DATA_DIR, verbose: bool = True) -> dict:
    data = load_data(data_dir)
    tf = build_training_frame(data["transactions"], data["offers"], data["materials"])
    train = tf[tf["tanggal_pesan"] < TRAIN_END].reset_index(drop=True)
    test = tf[tf["tanggal_pesan"] >= TRAIN_END].reset_index(drop=True)
    X_tr, y_tr, X_te, y_te = train[FEATURE_COLUMNS], train["berhasil"], test[FEATURE_COLUMNS], test["berhasil"]
    cv = TimeSeriesSplit(n_splits=CV_SPLITS)

    # 1-2. Tuning per model.
    cv_rows, fitted = [], {}
    mean, std = _baseline_cv(train, cv)
    cv_rows.append({"model": "baseline", "cv_roc_auc": mean, "cv_std": std, "best_params": {}})
    for name in MODEL_ORDER:
        gs = GridSearchCV(build_pipeline(name), PARAM_GRIDS[name], scoring="roc_auc", cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr)
        fitted[name] = gs.best_estimator_
        cv_rows.append({"model": name, "cv_roc_auc": gs.best_score_,
                        "cv_std": gs.cv_results_["std_test_score"][gs.best_index_],
                        "best_params": {k.removeprefix("clf__"): v for k, v in gs.best_params_.items()}})
        if verbose:
            print(f"{name}: CV ROC-AUC {gs.best_score_:.4f} {gs.best_params_}")
    cv_table = pd.DataFrame(cv_rows)
    selected = select_model(cv_table)

    # 4. Metrik probabilitas di test.
    test_pred = {"baseline": test[BASELINE_COLUMN].to_numpy()}
    test_pred |= {name: m.predict_proba(X_te)[:, 1] for name, m in fitted.items()}
    # Batas atas: P sebenarnya dari generator (tidak bisa dicapai model mana pun karena hasil diundi).
    hidden = pd.read_csv(Path(data_dir) / "vendors_hidden.csv")
    hid = hidden.set_index("vendor_id")
    test_pred["oracle"] = np.array([
        true_success_prob(r.vendor_id, r.material_id, r.qty, r.lead_time_janji, r.hari_tersedia, r.tanggal_pesan,
                          hid, data["materials"])
        for r in test.itertuples()
    ])
    test_table = pd.DataFrame([{"model": k, **_prob_metrics(y_te, p)} for k, p in test_pred.items()])

    # 4. Evaluasi ranking pada kebutuhan periode test.
    needs = test.rename(columns={"transaksi_id": "need_id", "tanggal_pesan": "tanggal"})
    cand = build_eval_candidates(needs[["need_id", "material_id", "qty", "hari_tersedia", "tanggal"]], data, hidden)
    rank_pred = {MODEL_LABELS["baseline"]: cand[BASELINE_COLUMN].to_numpy()}
    rank_pred |= {MODEL_LABELS[n]: m.predict_proba(cand[FEATURE_COLUMNS])[:, 1] for n, m in fitted.items()}
    ranking_table = compare_strategies(cand, rank_pred)

    # 5. Latih ulang model terpilih pada seluruh data.
    final = clone(fitted[selected]).fit(tf[FEATURE_COLUMNS], tf["berhasil"])
    sel_cv = cv_table.set_index("model").loc[selected]
    artifact = {
        "pipeline": final,
        "features": FEATURE_COLUMNS,
        "version": f"{MODEL_VERSION}-{selected}",
        "model_name": selected,
        "params": sel_cv["best_params"],
        "trained_until": str(tf["tanggal_pesan"].max().date()),
        "trained_at": str(date.today()),
        "n_rows": len(tf),
        "cv_roc_auc": float(sel_cv["cv_roc_auc"]),
        "test_metrics": test_table.set_index("model").loc[selected].to_dict(),
    }
    summary = {
        "version": artifact["version"],
        "selected": selected,
        "selection_rule": f"CV ROC-AUC tertinggi; selisih < {SIMPLER_TOLERANCE} -> pilih yang lebih sederhana",
        "split": {"train_until": TRAIN_END, "n_train": len(train), "n_test": len(test),
                  "n_eval_needs": int(cand["need_id"].nunique()), "n_eval_candidates": len(cand)},
        "cv": cv_table.to_dict(orient="records"),
        "test": test_table.to_dict(orient="records"),
        "ranking": ranking_table.to_dict(orient="records"),
    }
    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(artifact, MODEL_PATH, compress=3)
        METRICS_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        if verbose:
            print(f"Tersimpan: {MODEL_PATH.name} ({artifact['version']}), {METRICS_PATH.name}")

    return {"data": data, "frame": tf, "train": train, "test": test, "fitted": fitted, "selected": selected,
            "final": final, "artifact": artifact, "cv_table": cv_table, "test_table": test_table,
            "test_pred": test_pred, "candidates": cand, "ranking_table": ranking_table, "summary": summary}


if __name__ == "__main__":
    out = run()
    pd.set_option("display.width", 160)
    print(out["cv_table"][["model", "cv_roc_auc", "cv_std"]].round(4).to_string(index=False))
    print(out["test_table"].round(4).to_string(index=False))
    print(out["ranking_table"].round(3).to_string(index=False))
    print("Terpilih:", out["selected"])
