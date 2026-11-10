"""Penjelasan rekomendasi dengan Gemini (google-genai).

Gemini hanya MENJELASKAN, tidak menentukan ranking. Alurnya:

1. `build_context` menyusun konteks JSON dari hasil `core.ranking.recommend`: kebutuhan,
   3 vendor teratas, dan selisih antar-vendor yang SUDAH DIHITUNG di Python. Semua angka
   sudah dalam satuan tampilan (persen bulat, rupiah), jadi Gemini tidak perlu menghitung.
2. `explain_with_gemini` meminta Gemini menulis rekomendasi, alasan, trade-off, dan saran
   (output JSON terstruktur).
3. `validate_explanation` memeriksa setiap angka di teks terhadap angka di konteks, dan
   memastikan rekomendasi menyebut vendor peringkat 1.
4. Jika API key tidak ada, panggilan gagal, kuota habis, atau validasi gagal, dipakai
   `template_explanation` (deterministik, angkanya pasti dari konteks).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from pydantic import BaseModel, Field

from core.features import LIMITED_DATA_MIN_TX

log = logging.getLogger(__name__)

# gemini-2.5-flash tidak lagi tersedia untuk key baru (Nov 2026); gemini-3.8-flash melewati
# batas waktu 30 detik. Flash-Lite cepat (±5 detik) dan cukup untuk menjelaskan.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
TIMEOUT_MS = 20_000
BATAS_CACAT_PERSEN = 3
JENDELA_REKAM_JEJAK = 10
AMBANG_PELUANG_RENDAH = 80  # persen; di bawah ini template menyarankan vendor/waktu cadangan

SECTIONS = ("rekomendasi", "alasan", "trade_off", "saran")


class Penjelasan(BaseModel):
    """Skema output Gemini (dipakai sebagai response_schema)."""

    rekomendasi: str = Field(description="1-2 kalimat: vendor yang direkomendasikan dan alasan utamanya.")
    alasan: list[str] = Field(description="2-3 butir alasan berbasis angka di konteks.")
    trade_off: list[str] = Field(description="1-3 butir trade-off dibanding vendor lain di konteks.")
    saran: list[str] = Field(description="1-3 butir saran tindakan praktis untuk tim pengadaan.")


SYSTEM_INSTRUCTION = f"""\
Anda asisten pengadaan untuk Firaza, UMKM souvenir dan hampers. Anda menerima konteks JSON
berisi kebutuhan barang dan vendor teratas hasil model machine learning. Tugas Anda hanya
MENJELASKAN hasil tersebut kepada tim pengadaan dalam bahasa Indonesia yang ringkas dan jelas.

Aturan wajib:
1. Urutan vendor sudah final. Rekomendasikan vendor peringkat 1 dan sebut namanya persis.
   Jangan mengubah urutan dan jangan menyarankan vendor yang tidak ada di konteks.
2. Hanya gunakan angka yang tertulis di konteks, dengan pembulatan yang sama. Jangan
   menghitung angka baru (selisih, rata-rata, total). Selisih yang boleh disebut sudah ada
   di bagian "perbandingan". Jangan menulis tanggal atau tahun.
3. Format angka Indonesia: Rp50.700, Rp25.350.000, 92%, 8,9%, 1,3 hari.
4. "peluang_berhasil_persen" adalah prediksi model bahwa barang datang tepat deadline dan
   cacat paling banyak {BATAS_CACAT_PERSEN}%; sebut sebagai perkiraan, bukan jaminan.
5. Jika "data_terbatas" bernilai true, sebutkan bahwa rekam jejak vendor itu masih sedikit.
6. Jangan memakai heading atau format markdown; tulis kalimat biasa.
"""


# ---------------------------------------------------------------------------
# Konteks
# ---------------------------------------------------------------------------

def _pct(x: float) -> int:
    return int(round(float(x) * 100))


def _vendor_ctx(r: dict) -> dict:
    return {
        "peringkat": int(r["peringkat"]),
        "nama_vendor": r["nama_vendor"],
        "kota": r["kota"],
        "harga_satuan_rp": int(round(r["harga_satuan"])),
        "total_harga_rp": int(round(r["total_harga"])),
        "lead_time_hari": int(r["lead_time_janji"]),
        "moq": int(r["moq"]),
        "peluang_berhasil_persen": _pct(r["p_berhasil"]),
        "skor": round(float(r["skor"]), 3),
        "jumlah_transaksi": int(r["jumlah_transaksi"]),
        "data_terbatas": bool(r["data_terbatas"]),
        "berhasil_10_terakhir_persen": _pct(r["tingkat_berhasil_10"]),
        "rata_hari_telat": round(float(r["rata_hari_telat"]), 1),
        "tingkat_cacat_persen": round(float(r["tingkat_cacat"]) * 100, 1),
        "berhasil_musim_ramai_persen": _pct(r["berhasil_musim_ramai"]),
    }


def _compare(a: dict, b: dict, label: str) -> dict:
    """Selisih vendor a terhadap b (positif = a lebih tinggi). Dihitung dari angka yang sudah dibulatkan."""
    return {
        "antara": label,
        "selisih_harga_persen": round((a["harga_satuan_rp"] / b["harga_satuan_rp"] - 1) * 100, 1),
        "selisih_total_rp": a["total_harga_rp"] - b["total_harga_rp"],
        "selisih_peluang_poin": a["peluang_berhasil_persen"] - b["peluang_berhasil_persen"],
        "selisih_lead_time_hari": a["lead_time_hari"] - b["lead_time_hari"],
    }


def build_context(result: dict, materials=None) -> dict:
    """Konteks JSON dari hasil `recommend()`: kebutuhan + 3 teratas + perbandingan.

    `materials` (DataFrame) dipakai untuk nama dan satuan material; opsional.
    """
    need = result["kebutuhan"]
    nama, satuan = need["material_id"], "unit"
    if materials is not None:
        m = materials.set_index("material_id").loc[need["material_id"]]
        nama, satuan = m["nama_material"], m["satuan"]
    bobot = result["bobot"]

    top = [_vendor_ctx(r) for r in result["top"]]
    ctx: dict[str, Any] = {
        "kebutuhan": {
            "material": nama,
            "satuan": satuan,
            "qty": int(need["qty"]),
            "hari_tersedia": int(need["hari_tersedia"]),
            "budget_rp": None if need.get("budget") is None else int(round(need["budget"])),
        },
        "prioritas": {
            "preset": result["preset"],
            "bobot_berhasil_persen": _pct(bobot["w_berhasil"]),
            "bobot_harga_persen": _pct(bobot["w_harga"]),
            "bobot_cepat_persen": _pct(bobot["w_cepat"]),
        },
        "jumlah_kandidat": int(result["jumlah_kandidat"]),
        "definisi": {
            "berhasil": f"barang datang paling lambat deadline dan cacat paling banyak {BATAS_CACAT_PERSEN}%",
            "batas_cacat_persen": BATAS_CACAT_PERSEN,
            "jendela_rekam_jejak_transaksi": JENDELA_REKAM_JEJAK,
            "ambang_data_terbatas_transaksi": LIMITED_DATA_MIN_TX,
            "ambang_peluang_rendah_persen": AMBANG_PELUANG_RENDAH,
        },
        "vendor_teratas": top,
        "perbandingan": [_compare(top[0], v, f"peringkat 1 vs peringkat {v['peringkat']}") for v in top[1:]],
    }

    ranking = result.get("ranking") or result["top"]
    cheapest = min(ranking, key=lambda r: (r["harga_satuan"], r["peringkat"]))
    if cheapest["peringkat"] == 1:
        ctx["vendor_termurah"] = "peringkat 1 sekaligus vendor termurah"
    else:
        c = _vendor_ctx(cheapest)
        ctx["vendor_termurah"] = c
        if c["peringkat"] > len(top):
            ctx["perbandingan"].append(_compare(top[0], c, "peringkat 1 vs vendor termurah"))
    return ctx


# ---------------------------------------------------------------------------
# Validasi angka
# ---------------------------------------------------------------------------

_MULTIPLIER = {"ribu": 1e3, "rb": 1e3, "juta": 1e6, "jt": 1e6, "miliar": 1e9, "m": 1e9}
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:[.,]\d+)*)(?:\s*(ribu|rb|juta|jt|miliar)\b)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _interpretations(token: str) -> list[tuple[float, float]]:
    """Kemungkinan nilai (nilai, toleransi) dari satu token angka.

    Format Indonesia diutamakan (titik = ribuan, koma = desimal), tetapi bentuk Inggris juga
    dicoba karena LLM kadang memakainya. Toleransi = setengah satuan digit terakhir yang ditulis.
    """
    out = []
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", token) or re.fullmatch(r"\d+(,\d+)?", token):
        whole, _, dec = token.replace(".", "").partition(",")
        out.append((float(f"{whole}.{dec}" if dec else whole), 0.5 * 10 ** -len(dec)))
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", token) or re.fullmatch(r"\d+(\.\d+)?", token):
        whole, _, dec = token.replace(",", "").partition(".")
        out.append((float(f"{whole}.{dec}" if dec else whole), 0.5 * 10 ** -len(dec)))
    return out


def extract_numbers(text: str, ignore: list[str] = ()) -> list[tuple[str, list[tuple[float, float]]]]:
    """Ambil angka dari teks. String di `ignore` (nama vendor/material) dihapus dulu."""
    for s in sorted(ignore, key=len, reverse=True):
        if s:
            text = text.replace(s, " ")
    text = re.sub(r"Rp\.?\s*", " ", text)
    found = []
    for m in _NUMBER_RE.finditer(text):
        mult = _MULTIPLIER.get((m.group(2) or "").lower(), 1.0)
        interp = [(v * mult, tol * mult) for v, tol in _interpretations(m.group(1))]
        if interp:
            found.append((m.group(0).strip(), interp))
    return found


def _context_values(obj) -> set[float]:
    vals: set[float] = set()
    if isinstance(obj, bool) or obj is None:
        return vals
    if isinstance(obj, (int, float)):
        vals.update({float(obj), abs(float(obj))})
    elif isinstance(obj, dict):
        for v in obj.values():
            vals |= _context_values(v)
    elif isinstance(obj, list):
        for v in obj:
            vals |= _context_values(v)
    elif isinstance(obj, str):
        for _, interp in extract_numbers(obj):
            vals.update(v for v, _ in interp)
    return vals


def _context_names(ctx: dict) -> list[str]:
    names = [ctx["kebutuhan"]["material"], ctx["kebutuhan"]["satuan"]]
    vendors = list(ctx["vendor_teratas"])
    if isinstance(ctx.get("vendor_termurah"), dict):
        vendors.append(ctx["vendor_termurah"])
    for v in vendors:
        names += [v["nama_vendor"], v["kota"]]
    return names


def explanation_text(expl: dict) -> str:
    parts = []
    for key in SECTIONS:
        val = expl.get(key) or []
        parts += [val] if isinstance(val, str) else list(val)
    return "\n".join(parts)


def validate_explanation(expl: dict, ctx: dict) -> dict:
    """Cek setiap angka di penjelasan ada di konteks (dalam toleransi pembulatan).

    Hasil: {"valid", "jumlah_angka", "cocok", "tidak_cocok", "menyebut_peringkat_1"}.
    """
    allowed = _context_values(ctx)
    numbers = extract_numbers(explanation_text(expl), ignore=_context_names(ctx))
    mismatched = [
        raw for raw, interp in numbers
        if not any(abs(v - a) <= tol + 1e-9 for v, tol in interp for a in allowed)
    ]
    top_name = ctx["vendor_teratas"][0]["nama_vendor"]
    mentions_top = top_name.lower() in str(expl.get("rekomendasi", "")).lower()
    return {
        "valid": not mismatched and mentions_top,
        "jumlah_angka": len(numbers),
        "cocok": len(numbers) - len(mismatched),
        "tidak_cocok": mismatched,
        "menyebut_peringkat_1": mentions_top,
    }


# ---------------------------------------------------------------------------
# Template cadangan
# ---------------------------------------------------------------------------

def _rp(x: int) -> str:
    return "Rp" + f"{abs(int(x)):,}".replace(",", ".")


def _dec(x: float) -> str:
    return f"{abs(x):.1f}".replace(".", ",")


def template_explanation(ctx: dict) -> dict:
    """Penjelasan deterministik dari konteks (dipakai jika Gemini tidak tersedia atau tidak lolos validasi)."""
    top = ctx["vendor_teratas"]
    v1 = top[0]
    k = ctx["kebutuhan"]
    pr = ctx["prioritas"]

    rekomendasi = (
        f"Pilih {v1['nama_vendor']} ({v1['kota']}) untuk {k['qty']} {k['satuan']} {k['material']}: "
        f"perkiraan peluang berhasil {v1['peluang_berhasil_persen']}%, harga {_rp(v1['harga_satuan_rp'])} per "
        f"{k['satuan']} (total {_rp(v1['total_harga_rp'])}), dan lead time {v1['lead_time_hari']} hari."
    )

    bobot = (f"prioritas {pr['preset']} (bobot berhasil {pr['bobot_berhasil_persen']}%, harga "
             f"{pr['bobot_harga_persen']}%, cepat {pr['bobot_cepat_persen']}%)")
    alasan = [
        f"Skor tertinggi ({_dec_score(v1['skor'])}) dari {ctx['jumlah_kandidat']} vendor yang lolos saringan, "
        f"dengan {bobot}." if ctx["jumlah_kandidat"] > 1 else
        "Satu-satunya vendor yang lolos saringan (material, MOQ, lead time, dan budget).",
        f"Rekam jejak: berhasil {v1['berhasil_10_terakhir_persen']}% pada {JENDELA_REKAM_JEJAK} transaksi terakhir, "
        f"rata-rata telat {_dec(v1['rata_hari_telat'])} hari, dan tingkat cacat {_dec(v1['tingkat_cacat_persen'])}% "
        f"dari {v1['jumlah_transaksi']} transaksi.",
    ]
    if v1["lead_time_hari"] < k["hari_tersedia"]:
        alasan.append(f"Lead time {v1['lead_time_hari']} hari masih di bawah {k['hari_tersedia']} hari yang tersedia.")

    trade_off = []
    for c in ctx["perbandingan"]:
        d_harga, d_p = c["selisih_harga_persen"], c["selisih_peluang_poin"]
        harga = (f"lebih mahal {_dec(d_harga)}%" if d_harga > 0 else
                 f"lebih murah {_dec(d_harga)}%" if d_harga < 0 else "sama harganya")
        peluang = (f"peluang berhasilnya {abs(d_p)} poin lebih tinggi" if d_p > 0 else
                   f"peluang berhasilnya {abs(d_p)} poin lebih rendah" if d_p < 0 else "peluang berhasilnya sama")
        trade_off.append(f"Dibanding {c['antara'].split(' vs ')[1]}, pilihan ini {harga} dan {peluang}.")
    if isinstance(ctx.get("vendor_termurah"), dict):
        t = ctx["vendor_termurah"]
        trade_off.append(f"Vendor termurah adalah {t['nama_vendor']} ({_rp(t['harga_satuan_rp'])}), dengan perkiraan "
                         f"peluang berhasil {t['peluang_berhasil_persen']}%.")
    elif ctx["jumlah_kandidat"] == 1:
        trade_off.append("Tidak ada vendor pembanding; tambah hari tersedia atau naikkan budget untuk melihat pilihan lain.")
    else:
        trade_off.append("Vendor peringkat 1 sekaligus yang termurah di antara kandidat.")

    saran = []
    if v1["data_terbatas"]:
        saran.append(f"Rekam jejak {v1['nama_vendor']} masih sedikit ({v1['jumlah_transaksi']} transaksi); "
                     "minta sampel atau pesan bertahap dulu.")
    if v1["peluang_berhasil_persen"] < AMBANG_PELUANG_RENDAH:
        saran.append(f"Peluang berhasil di bawah {AMBANG_PELUANG_RENDAH}%; pertimbangkan menambah waktu pemesanan atau menyiapkan vendor cadangan.")
    if len(top) > 1:
        saran.append(f"Simpan {top[1]['nama_vendor']} sebagai vendor cadangan jika pilihan utama tidak bisa memenuhi pesanan.")
    saran.append(f"Lakukan pengecekan kualitas saat barang datang (batas cacat {BATAS_CACAT_PERSEN}%).")

    return {"rekomendasi": rekomendasi, "alasan": alasan, "trade_off": trade_off, "saran": saran}


def _dec_score(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def get_client():
    """Klien Gemini dari env var GEMINI_API_KEY, atau None jika key tidak ada."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    from google import genai
    from google.genai import types

    # Coba ulang sekali untuk gangguan server sementara (503 sibuk, 504 timeout).
    # 429 (kuota habis) tidak diulang: langsung pakai template.
    retry = types.HttpRetryOptions(attempts=2, initial_delay=1.0, max_delay=3.0, http_status_codes=[500, 503, 504])
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS, retry_options=retry))


def build_prompt(ctx: dict) -> str:
    return ("Jelaskan hasil rekomendasi vendor berikut untuk tim pengadaan.\n\nKonteks:\n"
            + json.dumps(ctx, ensure_ascii=False, indent=2))


def explain_with_gemini(ctx: dict, client, model: str = GEMINI_MODEL) -> dict:
    """Panggil Gemini dan kembalikan dict dengan kunci SECTIONS. Melempar exception jika gagal."""
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=build_prompt(ctx),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=Penjelasan,
        ),
    )
    if not response.text:
        raise ValueError("respons Gemini kosong")
    return Penjelasan.model_validate(json.loads(response.text)).model_dump()


def explain(result: dict, materials=None, client=None, model: str = GEMINI_MODEL,
            use_gemini: bool = True) -> dict | None:
    """Penjelasan untuk hasil `recommend()`. None jika tidak ada vendor teratas.

    Hasil: dict dengan kunci SECTIONS ditambah "sumber" ("gemini" atau "template"), "model",
    "catatan" (alasan memakai template, jika ada), dan "validasi".
    """
    if not result.get("top"):
        return None
    ctx = build_context(result, materials)
    catatan = None

    if use_gemini:
        client = client if client is not None else get_client()
        if client is None:
            catatan = "GEMINI_API_KEY belum diatur"
        else:
            try:
                expl = explain_with_gemini(ctx, client, model)
                check = validate_explanation(expl, ctx)
                if check["valid"]:
                    return {**expl, "sumber": "gemini", "model": model, "catatan": None, "validasi": check}
                catatan = ("penjelasan Gemini tidak lolos validasi angka: " + ", ".join(check["tidak_cocok"])
                           if check["tidak_cocok"] else "penjelasan Gemini tidak menyebut vendor peringkat 1")
                log.warning("%s | %s", catatan, json.dumps(expl, ensure_ascii=False))
            except Exception as e:  # kuota habis, jaringan, JSON rusak, dsb.
                catatan = f"Gemini gagal dipanggil ({type(e).__name__})"
                log.warning("Gemini gagal: %r", e)
    else:
        catatan = "Gemini dinonaktifkan"

    expl = template_explanation(ctx)
    return {**expl, "sumber": "template", "model": None, "catatan": catatan,
            "validasi": validate_explanation(expl, ctx)}


def to_markdown(expl: dict) -> str:
    lines = [expl["rekomendasi"], ""]
    for key, title in (("alasan", "Alasan"), ("trade_off", "Trade-off"), ("saran", "Saran")):
        if expl.get(key):
            lines += [f"**{title}**", *[f"- {s}" for s in expl[key]], ""]
    return "\n".join(lines).strip()
