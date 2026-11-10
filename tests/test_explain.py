"""Tes core.explain. Gemini di-mock (FakeClient), jadi tidak memakai kuota atau jaringan."""

import json
from types import SimpleNamespace

import pytest

from core.data import load_data
from core.explain import (build_context, explain, extract_numbers, template_explanation, to_markdown,
                          validate_explanation)
from core.ranking import recommend


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def result(data):
    return recommend({"material_id": "M01", "qty": 500, "hari_tersedia": 14, "budget": None}, data=data)


@pytest.fixture(scope="module")
def ctx(result, data):
    return build_context(result, data["materials"])


class FakeClient:
    """Meniru client.models.generate_content dari google-genai."""

    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error:
            raise self.error
        return SimpleNamespace(text=json.dumps(self.payload) if isinstance(self.payload, dict) else self.payload)


def good_payload(ctx):
    v1, v2 = ctx["vendor_teratas"][:2]
    c = ctx["perbandingan"][0]
    return {
        "rekomendasi": f"Pilih {v1['nama_vendor']} dengan perkiraan peluang berhasil {v1['peluang_berhasil_persen']}%.",
        "alasan": [f"Harga Rp{v1['harga_satuan_rp']:,} per satuan dan lead time {v1['lead_time_hari']} hari.".replace(",", ".")],
        "trade_off": [f"Dibanding {v2['nama_vendor']}, lebih mahal {str(abs(c['selisih_harga_persen'])).replace('.', ',')}% "
                      f"tetapi peluangnya {abs(c['selisih_peluang_poin'])} poin lebih tinggi."],
        "saran": ["Cek kualitas saat barang datang (batas cacat 3%)."],
    }


# --- parsing angka -----------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Rp25.350.000", 25_350_000),
    ("8,9%", 8.9),
    ("Rp1,2 juta", 1_200_000),
    ("lead time 7 hari", 7),
    ("0.92", 0.92),
])
def test_extract_numbers(text, expected):
    [(_, interp)] = extract_numbers(text)
    assert any(abs(v - expected) < 1e-6 for v, _ in interp)


def test_extract_ignores_names_and_ids():
    assert extract_numbers("Handuk mandi 70x140 cm dari V08", ignore=["Handuk mandi 70x140 cm"]) == []


# --- konteks dan validasi ---------------------------------------------------

def test_context_shape(ctx, result):
    assert ctx["kebutuhan"]["material"] == "Handuk mandi 70x140 cm"
    assert len(ctx["vendor_teratas"]) == len(result["top"])
    assert ctx["vendor_teratas"][0]["peringkat"] == 1
    assert len(ctx["perbandingan"]) >= len(result["top"]) - 1
    json.dumps(ctx)  # harus bisa diserialisasi


def test_template_passes_validation(ctx):
    check = validate_explanation(template_explanation(ctx), ctx)
    assert check["valid"] and check["jumlah_angka"] > 5


def test_validation_catches_invented_number(ctx):
    expl = template_explanation(ctx)
    expl["alasan"] = expl["alasan"] + ["Vendor ini sudah dipakai 987 kali."]
    check = validate_explanation(expl, ctx)
    assert not check["valid"] and check["tidak_cocok"] == ["987"]


def test_validation_requires_top_vendor(ctx):
    expl = template_explanation(ctx)
    expl["rekomendasi"] = f"Pilih {ctx['vendor_teratas'][1]['nama_vendor']}."
    assert not validate_explanation(expl, ctx)["menyebut_peringkat_1"]


# --- explain() dengan Gemini palsu --------------------------------------------

def test_explain_uses_gemini_when_valid(result, data, ctx):
    client = FakeClient(good_payload(ctx))
    e = explain(result, data["materials"], client=client)
    assert e["sumber"] == "gemini" and e["validasi"]["valid"]
    assert len(client.calls) == 1
    assert "Konteks" in client.calls[0]["contents"]


def test_explain_falls_back_on_invented_number(result, data, ctx):
    payload = good_payload(ctx)
    payload["saran"] = ["Vendor ini terbukti 99,9% andal."]
    e = explain(result, data["materials"], client=FakeClient(payload))
    assert e["sumber"] == "template" and "99,9" in e["catatan"]


@pytest.mark.parametrize("client", [FakeClient(error=RuntimeError("429 RESOURCE_EXHAUSTED")),
                                    FakeClient(payload="bukan json")])
def test_explain_falls_back_on_error(result, data, client):
    e = explain(result, data["materials"], client=client)
    assert e["sumber"] == "template" and "gagal" in e["catatan"]


def test_explain_without_key(result, data, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    e = explain(result, data["materials"])
    assert e["sumber"] == "template" and "GEMINI_API_KEY" in e["catatan"]
    assert to_markdown(e).startswith("Pilih ")


def test_explain_no_candidates(data):
    r = recommend({"material_id": "M05", "qty": 1, "hari_tersedia": 1, "budget": None}, data=data)
    assert explain(r, data["materials"], use_gemini=False) is None
