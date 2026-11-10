"""Tes FastAPI. Gemini tidak dipanggil sungguhan: key dikosongkan atau klien di-mock."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import core.explain  # noqa: E402
from api.main import app  # noqa: E402
from tests.test_explain import FakeClient, good_payload  # noqa: E402

NEED = {"material_id": "M01", "qty": 500, "hari_tersedia": 14}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model_version"] != "baseline"


def test_materials(client):
    r = client.get("/api/v1/materials")
    assert r.status_code == 200
    assert len(r.json()) == 15 and r.json()[0]["material_id"] == "M01"


def test_recommend_default(client):
    r = client.post("/api/v1/recommend", json=NEED)
    assert r.status_code == 200
    body = r.json()
    assert body["preset"] == "Seimbang" and body["jumlah_kandidat"] == len(body["ranking"]) > 3
    assert [v["peringkat"] for v in body["top"]] == [1, 2, 3]
    assert body["top"] == body["ranking"][:3]
    skor = [v["skor"] for v in body["ranking"]]
    assert skor == sorted(skor, reverse=True)
    p = body["penjelasan"]
    assert p["sumber"] == "template" and p["validasi"]["valid"]
    assert body["top"][0]["nama_vendor"] in p["rekomendasi"]


def test_recommend_same_as_core(client):
    """API harus memberi ranking yang sama dengan core (yang dipakai Streamlit)."""
    from api.main import get_data, get_model
    from core.ranking import recommend

    expected = recommend(dict(NEED, budget=None), preset="Tercepat", data=get_data(), model=get_model())
    body = client.post("/api/v1/recommend", json=dict(NEED, preset="Tercepat")).json()
    assert [v["offer_id"] for v in body["ranking"]] == [v["offer_id"] for v in expected["ranking"]]


def test_recommend_custom_weights_and_top_k(client):
    r = client.post("/api/v1/recommend", json=dict(
        NEED, preset="Kustom", bobot={"w_berhasil": 2, "w_harga": 1, "w_cepat": 1}, top_k=5,
        dengan_penjelasan=False))
    assert r.status_code == 200
    body = r.json()
    assert body["preset"] == "Kustom" and len(body["top"]) == 5 and body["penjelasan"] is None
    assert body["bobot"] == pytest.approx({"w_berhasil": 0.5, "w_harga": 0.25, "w_cepat": 0.25})


def test_recommend_no_candidates(client):
    r = client.post("/api/v1/recommend", json={"material_id": "M05", "qty": 3, "hari_tersedia": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["ranking"] == [] and body["penjelasan"] is None and "Tidak ada vendor" in body["pesan"]


@pytest.mark.parametrize("payload, status", [
    ({"material_id": "M99", "qty": 10, "hari_tersedia": 10}, 404),
    (dict(NEED, qty=0), 422),
    (dict(NEED, preset="Paling Murah"), 422),
    (dict(NEED, preset="Kustom"), 422),  # bobot wajib
    (dict(NEED, bobot={"w_berhasil": 1, "w_harga": 0, "w_cepat": 0}), 422),  # bobot tanpa Kustom
    (dict(NEED, preset="Kustom", bobot={"w_berhasil": 0, "w_harga": 0, "w_cepat": 0}), 422),
    (dict(NEED, top_k=11), 422),
])
def test_recommend_invalid_input(client, payload, status):
    assert client.post("/api/v1/recommend", json=payload).status_code == status


def test_recommend_with_gemini(client, monkeypatch):
    from api.main import get_data
    from core.explain import build_context
    from core.ranking import recommend

    ctx = build_context(recommend(dict(NEED, budget=None), data=get_data()), get_data()["materials"])
    fake = FakeClient(good_payload(ctx))
    monkeypatch.setattr(core.explain, "get_client", lambda: fake)
    body = client.post("/api/v1/recommend", json=NEED).json()
    assert body["penjelasan"]["sumber"] == "gemini" and len(fake.calls) == 1
