"""Evaluasi batch penjelasan Gemini (Research Question 3: >= 95% angka sesuai konteks).

Pemakaian (dari root repo, butuh GEMINI_API_KEY di .env):
    python -m scripts.eval_gemini [--n 30] [--jeda 4]

Mengambil sampel kebutuhan dari periode test (Apr-Sep 2026) ditambah beberapa tanggal waktu rawan
(hari besar / musim ramai 2027), memanggil Gemini untuk tiap kebutuhan, lalu mengukur output MENTAH
Gemini (sebelum fallback ke template) dengan validator yang sama dengan aplikasi:
- akurasi angka = angka yang cocok dengan konteks / semua angka yang ditulis Gemini;
- tingkat valid = penjelasan yang semua angkanya cocok dan menyebut vendor peringkat 1;
- tingkat gagal = panggilan error (kuota, jaringan, JSON rusak).
Hasil: docs/evaluasi_gemini.md (ringkasan) dan docs/evaluasi_gemini_hasil.jsonl (per kasus).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from core.data import load_data  # noqa: E402
from core.explain import GEMINI_MODEL, build_context, explain_with_gemini, get_client, validate_explanation  # noqa: E402
from core.ranking import PRESETS, recommend  # noqa: E402

TANGGAL_RAWAN = ["2027-02-15", "2027-03-05", "2027-08-17", "2027-08-10", "2027-12-20", "2027-05-12"]


def sample_needs(data: dict, n: int, seed: int = 7) -> list[dict]:
    tx = data["transactions"].copy()
    tx["tanggal_pesan"] = pd.to_datetime(tx["tanggal_pesan"])
    test = tx[tx["tanggal_pesan"] >= "2026-04-01"]
    n_rawan = min(len(TANGGAL_RAWAN), max(1, n // 5))
    rows = test.sample(n - n_rawan, random_state=seed)
    needs = [{"material_id": r.material_id, "qty": int(r.qty), "hari_tersedia": int(r.hari_tersedia),
              "tanggal": r.tanggal_pesan.strftime("%Y-%m-%d")} for r in rows.itertuples()]
    rng = np.random.default_rng(seed)
    mats = data["materials"]
    for tgl in TANGGAL_RAWAN[:n_rawan]:
        m = mats.iloc[int(rng.integers(len(mats)))]
        needs.append({"material_id": m["material_id"], "qty": int(m["qty_median"]),
                      "hari_tersedia": int(m["lead_time_max"]), "tanggal": tgl})
    return needs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--jeda", type=float, default=4.0, help="detik antarpanggilan (batas RPM kuota gratis)")
    args = ap.parse_args()

    client = get_client()
    if client is None:
        raise SystemExit("GEMINI_API_KEY belum diatur di .env")
    data = load_data()
    presets = [*PRESETS]
    hasil = []
    for i, need in enumerate(sample_needs(data, args.n)):
        preset = presets[i % len(presets)]
        res = recommend(need, preset=preset, data=data)
        if not res["top"]:
            continue
        ctx = build_context(res, data["materials"])
        row = {"no": i + 1, "kebutuhan": need, "preset": preset, "waktu_rawan": ctx["kalender"]["waktu_rawan"]}
        t0 = time.perf_counter()
        try:
            expl = explain_with_gemini(ctx, client)
            v = validate_explanation(expl, ctx)
            row.update(status="ok", detik=round(time.perf_counter() - t0, 1), validasi=v, penjelasan=expl)
        except Exception as e:  # noqa: BLE001
            row.update(status="gagal", detik=round(time.perf_counter() - t0, 1), error=type(e).__name__)
        hasil.append(row)
        ok = row["status"] == "ok"
        print(f"{i + 1:>2} {need['material_id']} {need['tanggal']} {preset:<9} "
              + (f"angka {row['validasi']['cocok']}/{row['validasi']['jumlah_angka']} valid={row['validasi']['valid']}"
                 if ok else f"GAGAL {row['error']}"), flush=True)
        time.sleep(args.jeda)

    ok = [r for r in hasil if r["status"] == "ok"]
    total_angka = sum(r["validasi"]["jumlah_angka"] for r in ok)
    cocok = sum(r["validasi"]["cocok"] for r in ok)
    valid = sum(r["validasi"]["valid"] for r in ok)
    sebut = sum(r["validasi"]["menyebut_peringkat_1"] for r in ok)
    rawan = [r for r in ok if r["waktu_rawan"]]
    salah = [(r["no"], x) for r in ok for x in r["validasi"]["tidak_cocok"]]
    detik = [r["detik"] for r in ok]

    docs = ROOT / "docs"
    with open(docs / "evaluasi_gemini_hasil.jsonl", "w", encoding="utf-8") as f:
        for r in hasil:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    akurasi = cocok / total_angka if total_angka else 0.0
    md = [
        "# Evaluasi Penjelasan Gemini",
        "",
        f"Dibuat oleh `python -m scripts.eval_gemini --n {args.n}` pada {time.strftime('%d-%m-%Y %H:%M')}. "
        f"Model `{GEMINI_MODEL}`, temperature 0,2. Data per kasus: "
        "[evaluasi_gemini_hasil.jsonl](evaluasi_gemini_hasil.jsonl).",
        "",
        "Yang diukur adalah **output mentah Gemini sebelum fallback**: di aplikasi, penjelasan yang tidak lolos "
        "validasi otomatis diganti template, sehingga pengguna tidak pernah melihat angka yang salah.",
        "",
        "| Ukuran | Hasil | Target |",
        "|---|---|---|",
        f"| Akurasi angka (cocok / semua angka yang ditulis) | **{akurasi:.1%}** ({cocok}/{total_angka}) | ≥ 95% |",
        f"| Penjelasan valid penuh (semua angka cocok + menyebut peringkat 1) | {valid}/{len(ok)} ({valid / max(len(ok), 1):.0%}) | – |",
        f"| Menyebut vendor peringkat 1 | {sebut}/{len(ok)} | – |",
        f"| Panggilan gagal (kuota/jaringan/JSON) | {len(hasil) - len(ok)}/{len(hasil)} | – |",
        f"| Kasus waktu rawan (hari besar/musim ramai) yang valid | {sum(r['validasi']['valid'] for r in rawan)}/{len(rawan)} | – |",
        f"| Waktu respons median | {np.median(detik):.1f} detik | – |" if detik else "| Waktu respons | – | – |",
        "",
        "**Sampel.** Kebutuhan diambil acak (seed 7) dari transaksi periode test (Apr–Sep 2026), ditambah "
        "beberapa tanggal waktu rawan 2027; preset diputar Seimbang/Termurah/Tercepat.",
        "",
        "**Angka yang tidak cocok:** " + (", ".join(f"kasus {n}: `{x}`" for n, x in salah) if salah else "tidak ada."),
        "",
    ]
    (docs / "evaluasi_gemini.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nAkurasi angka {akurasi:.1%} ({cocok}/{total_angka}); valid {valid}/{len(ok)}; gagal {len(hasil) - len(ok)}")


if __name__ == "__main__":
    main()
