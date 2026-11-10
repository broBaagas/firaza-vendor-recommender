"""Tes core.storage. Memakai koneksi tiruan; tes integrasi ke Supabase hanya jalan jika RUN_DB_TESTS=1."""

import json
import os

import pytest

from core import storage
from core.data import load_data
from core.explain import explain
from core.ranking import recommend


class FakeConn:
    def __init__(self):
        self.calls, self.closed = [], False
        self.columns = [{"name": "id"}, {"name": "kanal"}]

    def run(self, sql, **params):
        self.calls.append((" ".join(sql.split()), params))
        if sql.lstrip().startswith("INSERT"):
            return [[42]]
        if sql.lstrip().startswith("SELECT"):
            return [[2, "api"], [1, "streamlit"]]
        return []

    def close(self):
        self.closed = True


@pytest.fixture(scope="module")
def result():
    data = load_data()
    r = recommend({"material_id": "M01", "qty": 500, "hari_tersedia": 14, "budget": 30_000_000}, data=data)
    return r, explain(r, data["materials"], use_gemini=False)


@pytest.fixture(autouse=True)
def reset_init(monkeypatch):
    monkeypatch.setattr(storage, "_initialized", False)


def test_save_creates_table_and_inserts(result):
    r, e = result
    conn = FakeConn()
    assert storage.save_recommendation(r, e, "streamlit", conn=conn) == 42
    sqls = [s for s, _ in conn.calls]
    assert sqls[0].startswith("CREATE TABLE IF NOT EXISTS riwayat_rekomendasi")
    assert "ENABLE ROW LEVEL SECURITY" in sqls[1]
    params = conn.calls[2][1]
    assert params["kanal"] == "streamlit" and params["material_id"] == "M01" and params["budget"] == 30_000_000
    assert params["vendor_teratas"] == r["top"][0]["nama_vendor"] and params["sumber_penjelasan"] == "template"
    assert json.loads(params["hasil"])["top"] == r["top"]
    assert json.loads(params["penjelasan"])["rekomendasi"] == e["rekomendasi"]
    assert not conn.closed  # koneksi dari pemanggil tidak ditutup


def test_init_only_once(result):
    r, e = result
    conn = FakeConn()
    storage.save_recommendation(r, e, "api", conn=conn)
    storage.save_recommendation(r, e, "api", conn=conn)
    assert sum(s.startswith("CREATE TABLE") for s, _ in conn.calls) == 1


def test_save_without_candidates():
    r = recommend({"material_id": "M05", "qty": 3, "hari_tersedia": 1}, data=load_data())
    conn = FakeConn()
    storage.save_recommendation(r, None, "api", conn=conn)
    params = conn.calls[-1][1]
    assert params["vendor_teratas"] is None and params["penjelasan"] is None and params["jumlah_kandidat"] == 0


def test_list_history():
    rows = storage.list_history(limit=5, conn=FakeConn())
    assert rows == [{"id": 2, "kanal": "api"}, {"id": 1, "kanal": "streamlit"}]


def test_safe_save_not_configured(result, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    assert storage.safe_save(*result, "api") is None


def test_safe_save_swallows_errors(result, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:1/db")

    def boom():
        raise ConnectionError("tidak bisa terhubung")

    monkeypatch.setattr(storage, "get_connection", boom)
    assert storage.safe_save(*result, "api") is None


@pytest.mark.skipif(os.environ.get("RUN_DB_TESTS") != "1", reason="set RUN_DB_TESTS=1 untuk tes ke Supabase")
def test_supabase_roundtrip(result):
    from dotenv import load_dotenv

    load_dotenv()
    new_id = storage.save_recommendation(*result, "tes")
    try:
        assert any(row["id"] == new_id for row in storage.list_history(limit=20))
    finally:
        conn = storage.get_connection()
        conn.run("DELETE FROM riwayat_rekomendasi WHERE id = :id AND kanal = 'tes'", id=new_id)
        conn.close()
