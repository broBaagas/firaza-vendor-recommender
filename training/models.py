"""Definisi pipeline (preprocessing + model) dan grid hyperparameter."""

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from core.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

# Fitur berekor panjang: log1p supaya Logistic Regression tidak didominasi nilai ekstrem.
# Untuk model pohon transformasi monoton ini tidak mengubah hasil.
SKEWED_COLUMNS = ["qty", "moq", "rasio_qty", "jumlah_transaksi", "hari_sejak_pesanan_terakhir"]
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS + SKEWED_COLUMNS]

SEED = 42

# Urutan = dari yang paling sederhana. Dipakai sebagai pemecah seri saat memilih model.
MODEL_ORDER = ["logreg", "random_forest", "lightgbm"]
MODEL_LABELS = {
    "baseline": "Baseline (rata-rata historis)",
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
}


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLUMNS),
        ("skew", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()),
         SKEWED_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ])


def build_pipeline(name: str) -> Pipeline:
    if name == "logreg":
        clf = LogisticRegression(max_iter=5000)
    elif name == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1)
    elif name == "lightgbm":
        clf = LGBMClassifier(learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             random_state=SEED, n_jobs=1, verbose=-1)
    else:
        raise ValueError(name)
    return Pipeline([("prep", preprocessor()), ("clf", clf)])


PARAM_GRIDS = {
    "logreg": {"clf__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
    "random_forest": {"clf__min_samples_leaf": [5, 20, 50], "clf__max_features": ["sqrt", 0.5]},
    "lightgbm": {"clf__n_estimators": [200, 500], "clf__num_leaves": [7, 15, 31],
                 "clf__min_child_samples": [20, 50]},
}
