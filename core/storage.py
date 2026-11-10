"""Penyimpanan riwayat rekomendasi ke Supabase (Postgres) lewat pg8000 (BSD-3).

Satu tabel: riwayat_rekomendasi. Konfigurasi:

- DATABASE_URL: connection string Postgres (Supabase: Session pooler, karena IPv4),
  mis. postgresql://postgres.<project>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
- Sertifikat CA Supabase di certs/supabase-ca.crt (atau path di env DATABASE_SSL_CA).
  Koneksi SELALU memakai TLS dengan verifikasi sertifikat; tanpa file CA dipakai CA bawaan sistem.

Penyimpanan bersifat pelengkap: jika DATABASE_URL tidak ada atau database gagal, aplikasi
tetap jalan (`safe_save` mengembalikan None dan mencatat peringatan).
"""

from __future__ import annotations

import json
import logging
import os
import ssl
from pathlib import Path
from urllib.parse import unquote, urlparse

log = logging.getLogger(__name__)

DEFAULT_CA_PATH = Path(__file__).resolve().parent.parent / "certs" / "supabase-ca.crt"
CONNECT_TIMEOUT_S = 10

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS riwayat_rekomendasi (
    id                 BIGSERIAL PRIMARY KEY,
    waktu              TIMESTAMPTZ NOT NULL DEFAULT now(),
    kanal              TEXT NOT NULL,
    material_id        TEXT NOT NULL,
    qty                INTEGER NOT NULL,
    hari_tersedia      INTEGER NOT NULL,
    budget             NUMERIC,
    preset             TEXT NOT NULL,
    bobot              JSONB NOT NULL,
    jumlah_kandidat    INTEGER NOT NULL,
    vendor_teratas     TEXT,
    hasil              JSONB NOT NULL,
    penjelasan         JSONB,
    sumber_penjelasan  TEXT,
    versi_model        TEXT NOT NULL
)
"""
# RLS aktif tanpa policy: tabel tidak bisa dibaca/ditulis lewat Data API Supabase (anon key);
# hanya koneksi Postgres langsung (role pemilik) yang bisa mengakses.
ENABLE_RLS_SQL = "ALTER TABLE riwayat_rekomendasi ENABLE ROW LEVEL SECURITY"

INSERT_SQL = """
INSERT INTO riwayat_rekomendasi
    (kanal, material_id, qty, hari_tersedia, budget, preset, bobot, jumlah_kandidat, vendor_teratas,
     hasil, penjelasan, sumber_penjelasan, versi_model)
VALUES
    (:kanal, :material_id, :qty, :hari_tersedia, :budget, :preset, CAST(:bobot AS JSONB), :jumlah_kandidat,
     :vendor_teratas, CAST(:hasil AS JSONB), CAST(:penjelasan AS JSONB), :sumber_penjelasan, :versi_model)
RETURNING id
"""

LIST_SQL = """
SELECT id, waktu, kanal, material_id, qty, hari_tersedia, budget, preset, jumlah_kandidat, vendor_teratas,
       sumber_penjelasan, versi_model
FROM riwayat_rekomendasi
ORDER BY waktu DESC
LIMIT :limit
"""

_initialized = False


def is_configured() -> bool:
    return bool(os.environ.get("DATABASE_URL", "").strip())


def _ssl_context() -> ssl.SSLContext:
    ca = Path(os.environ.get("DATABASE_SSL_CA", "") or DEFAULT_CA_PATH)
    if ca.exists():
        return ssl.create_default_context(cafile=str(ca))
    return ssl.create_default_context()


def get_connection():
    """Koneksi pg8000.native dari DATABASE_URL (TLS terverifikasi)."""
    import pg8000.native

    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL belum diatur")
    u = urlparse(url)
    return pg8000.native.Connection(
        user=unquote(u.username or ""), password=unquote(u.password or ""), host=u.hostname,
        port=u.port or 5432, database=(u.path or "/postgres").lstrip("/") or "postgres",
        ssl_context=_ssl_context(), timeout=CONNECT_TIMEOUT_S,
    )


def init_db(conn=None) -> None:
    """Buat tabel jika belum ada (idempoten)."""
    global _initialized
    own = conn is None
    conn = conn or get_connection()
    try:
        conn.run(CREATE_TABLE_SQL)
        conn.run(ENABLE_RLS_SQL)
        _initialized = True
    finally:
        if own:
            conn.close()


def _row_params(result: dict, explanation: dict | None, kanal: str) -> dict:
    need = result["kebutuhan"]
    top = result.get("top") or []
    hasil = {
        "top": top,
        "pesan": result.get("pesan"),
    }
    return {
        "kanal": kanal,
        "material_id": need["material_id"],
        "qty": int(need["qty"]),
        "hari_tersedia": int(need["hari_tersedia"]),
        "budget": need.get("budget"),
        "preset": result["preset"],
        "bobot": json.dumps(result["bobot"]),
        "jumlah_kandidat": int(result["jumlah_kandidat"]),
        "vendor_teratas": top[0]["nama_vendor"] if top else None,
        "hasil": json.dumps(hasil, ensure_ascii=False),
        "penjelasan": json.dumps(explanation, ensure_ascii=False) if explanation else None,
        "sumber_penjelasan": explanation["sumber"] if explanation else None,
        "versi_model": result["model"],
    }


def save_recommendation(result: dict, explanation: dict | None, kanal: str, conn=None) -> int:
    """Simpan satu rekomendasi; kembalikan id baris. `kanal`: 'streamlit' atau 'api'."""
    own = conn is None
    conn = conn or get_connection()
    try:
        if not _initialized:
            init_db(conn)
        return conn.run(INSERT_SQL, **_row_params(result, explanation, kanal))[0][0]
    finally:
        if own:
            conn.close()


def safe_save(result: dict, explanation: dict | None, kanal: str) -> int | None:
    """Seperti save_recommendation, tetapi tidak pernah melempar error (None jika gagal/tidak dikonfigurasi)."""
    if not is_configured():
        return None
    try:
        return save_recommendation(result, explanation, kanal)
    except Exception as e:
        log.warning("Gagal menyimpan riwayat: %r", e)
        return None


def list_history(limit: int = 20, conn=None) -> list[dict]:
    """Riwayat terbaru (tanpa kolom JSON besar)."""
    own = conn is None
    conn = conn or get_connection()
    try:
        if not _initialized:
            init_db(conn)
        rows = conn.run(LIST_SQL, limit=int(limit))
        cols = [c["name"] for c in conn.columns]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        if own:
            conn.close()
