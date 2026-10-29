"""Penyimpanan riwayat rekomendasi ke Supabase (Postgres) lewat pg8000.

Tabel: riwayat_rekomendasi (id, waktu, input, hasil, penjelasan, versi_model).
"""

import os

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS riwayat_rekomendasi (
    id          BIGSERIAL PRIMARY KEY,
    waktu       TIMESTAMPTZ NOT NULL DEFAULT now(),
    input       JSONB NOT NULL,
    hasil       JSONB NOT NULL,
    penjelasan  TEXT,
    versi_model TEXT NOT NULL
);
"""


def get_connection():
    url = os.environ.get("SUPABASE_URL")
    raise NotImplementedError


def save_recommendation(need: dict, result: dict, explanation: str, model_version: str) -> None:
    raise NotImplementedError


def list_history(limit: int = 20) -> list[dict]:
    raise NotImplementedError
