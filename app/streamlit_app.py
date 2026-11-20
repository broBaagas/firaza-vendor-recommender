"""Antarmuka Streamlit: tab Rekomendasi, Data Vendor (rekam jejak, penawaran, histori transaksi), dan Riwayat.

Memanggil modul core langsung, bukan lewat API.
Jalankan: streamlit run app/streamlit_app.py
"""

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env")  # lokal; di Streamlit Cloud key diisi lewat Secrets

from core.data import load_data  # noqa: E402
from core.explain import explain, to_markdown  # noqa: E402
from core.features import prepare_transactions  # noqa: E402
from core.ranking import PRESETS, load_model, recommend  # noqa: E402
from core.seasons import CALENDAR_END, peak_period  # noqa: E402

TOP_COLOR, OTHER_COLOR = "#2a78d6", "#a3a29c"

st.set_page_config(page_title="Firaza Vendor Recommender", page_icon="📦", layout="wide")

for _key in ("GEMINI_API_KEY", "DATABASE_URL"):
    if _key not in os.environ:
        try:
            os.environ[_key] = st.secrets.get(_key, "")
        except Exception:  # tidak ada secrets.toml
            pass

from core import storage  # noqa: E402


@st.cache_data
def get_data() -> dict[str, pd.DataFrame]:
    return load_data()


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def prepared_transactions(_data: dict) -> pd.DataFrame:
    tx = prepare_transactions(_data["transactions"])
    tx["telat_deadline"] = (tx["tanggal_datang"] - tx["deadline"]).dt.days.clip(lower=0)
    return tx


@st.cache_data
def vendor_stats(_data: dict) -> pd.DataFrame:
    tx = prepared_transactions(_data)
    stats = tx.groupby("vendor_id").agg(
        transaksi=("transaksi_id", "size"),
        tingkat_berhasil=("berhasil", "mean"),
        rata_hari_telat=("hari_telat", "mean"),
        rata_cacat=("persen_cacat", "mean"),
        pesanan_terakhir=("tanggal_pesan", "max"),
    )
    offers = _data["offers"].merge(_data["materials"][["material_id", "nama_material"]], on="material_id")
    stats["material"] = offers.groupby("vendor_id")["nama_material"].apply(", ".join)
    out = _data["vendors"].set_index("vendor_id").join(stats)
    out["transaksi"] = out["transaksi"].fillna(0).astype(int)
    return out.reset_index()


def rupiah(x: float) -> str:
    return "Rp" + f"{x:,.0f}".replace(",", ".")


def persen(x: float) -> str:
    return f"{x * 100:.0f}%"


BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def tgl_id(x) -> str:
    d = pd.Timestamp(x)
    return f"{d.day} {BULAN[d.month - 1]} {d.year}"


def tradeoff_chart(ranking: pd.DataFrame, top_k: int) -> go.Figure:
    fig = go.Figure()
    for is_top, name, color in [(False, "Kandidat lain", OTHER_COLOR), (True, f"{top_k} teratas", TOP_COLOR)]:
        d = ranking[(ranking["peringkat"] <= top_k) == is_top]
        fig.add_trace(go.Scatter(
            x=d["harga_satuan"], y=d["p_berhasil"], mode="markers+text", name=name,
            text=d["nama_vendor"], textposition="top center",
            marker={"size": 12, "color": color, "line": {"width": 2, "color": "rgba(255,255,255,0.9)"}},
            customdata=d[["peringkat", "lead_time_janji", "skor"]],
            hovertemplate=("<b>%{text}</b> (peringkat %{customdata[0]})<br>Harga: Rp%{x:,.0f}<br>"
                           "P(berhasil): %{y:.0%}<br>Lead time: %{customdata[1]} hari<br>"
                           "Skor: %{customdata[2]:.3f}<extra></extra>"),
        ))
    fig.update_layout(
        height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title="Harga satuan (Rp)", yaxis_title="P(berhasil)",
        yaxis={"tickformat": ".0%", "range": [0, 1.08]}, xaxis={"tickformat": ",.0f"},
        legend={"orientation": "h", "y": 1.12, "x": 0}, hoverlabel={"align": "left"},
    )
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def get_explanation(hasil: dict) -> dict | None:
    """Di-cache agar kebutuhan yang sama tidak memanggil Gemini berulang kali (hemat kuota)."""
    return explain(hasil, get_data()["materials"])


def explanation_caption(e: dict) -> str:
    v = e["validasi"]
    if e["sumber"] == "gemini":
        return (f"✨ Ditulis oleh Gemini ({e['model']}). Ranking tetap dari model ML; "
                f"{v['cocok']}/{v['jumlah_angka']} angka sudah dicek cocok dengan hasil model.")
    return f"Penjelasan template (cadangan): {e['catatan']}."


data = get_data()
model = get_model()
materials = data["materials"].set_index("material_id")

st.title("📦 Firaza Vendor Recommender")
st.caption("Decision-support pemilihan vendor: prediksi peluang berhasil (tepat deadline dan cacat ≤ 3%) per vendor, "
           "lalu ranking berdasarkan peluang berhasil, harga, dan kecepatan. Keputusan akhir tetap di tangan Anda. "
           "**Data vendor dan transaksi sintetis.**")

REPO_URL = "https://github.com/broBaagas/firaza-vendor-recommender"

with st.expander("ℹ️ Cara pakai"):
    st.markdown(
        "1. **Rekomendasi**: pilih material atau jasa, isi jumlah, sisa hari sampai deadline, dan tanggal pesan "
        "(opsional: batasi total budget), pilih prioritas, lalu klik **Cari rekomendasi vendor**. "
        "Tanggal pesan di musim ramai (🌙 Ramadan–Lebaran, musim nikah) menurunkan peluang berhasil vendor "
        "yang sering telat di periode itu.\n"
        "2. Baca kartu **3 vendor teratas**, **penjelasan** (rekomendasi, alasan, trade-off, saran), grafik "
        "harga vs peluang berhasil, dan tabel semua kandidat.\n"
        "3. Ganti **Prioritas** untuk melihat ranking berubah: *Seimbang*, *Termurah*, *Tercepat*, "
        "atau *Kustom* (atur bobot sendiri).\n"
        "4. **Data Vendor**: rekam jejak semua vendor, penawaran per material, dan histori transaksi per vendor.\n"
        "5. **Riwayat**: permintaan rekomendasi yang pernah dibuat (tersimpan di database).\n\n"
        "**Istilah.** *P(berhasil)* = peluang barang datang paling lambat di deadline **dan** cacat ≤ 3%, "
        "diprediksi model machine learning dari rekam jejak vendor dan kondisi pesanan. "
        "*Skor* = gabungan P(berhasil), harga murah, dan lead time cepat sesuai bobot prioritas. "
        "⚠️ *Data terbatas* = vendor dengan kurang dari 5 transaksi, prediksinya kurang pasti. "
        "Ranking ditentukan model, bukan Gemini; Gemini hanya menulis penjelasannya.\n\n"
        f"Kode sumber, dokumentasi, dan evaluasi model: [GitHub]({REPO_URL}) · "
        f"[Laporan evaluasi]({REPO_URL}/blob/main/docs/evaluasi_model.md) · "
        f"[Asumsi data]({REPO_URL}/blob/main/docs/asumsi_data.md)"
    )

tab_rekomendasi, tab_vendor, tab_riwayat = st.tabs(["Rekomendasi", "Data Vendor", "Riwayat"])

with tab_rekomendasi:
    c1, c2, c3, c6 = st.columns([2, 1, 1, 1])
    material_id = c1.selectbox(
        "Material / jasa", materials.index,
        format_func=lambda m: f"{materials.loc[m, 'nama_material']} ({materials.loc[m, 'satuan']})")
    m = materials.loc[material_id]
    qty = c2.number_input(f"Jumlah ({m['satuan']})", min_value=1, value=int(m["qty_median"]), step=10)
    hari = c3.number_input("Hari tersedia sampai deadline", min_value=1, value=int(m["lead_time_max"]),
                           help=f"Lead time wajar untuk material ini {m['lead_time_min']}–{m['lead_time_max']} hari.")
    hari_ini = min(date.today(), CALENDAR_END.date())
    tanggal = c6.date_input("Tanggal pesan", value=hari_ini, min_value=hari_ini, max_value=CALENDAR_END.date(),
                            format="DD/MM/YYYY",
                            help="Menentukan musim ramai (Ramadan–Lebaran, musim nikah Syawal/Dzulhijjah), "
                                 "saat vendor lebih sering telat.")
    musim_input = peak_period(tanggal)
    if musim_input:
        c6.caption(f"🌙 {musim_input['nama']}")

    c4, c5 = st.columns([1, 2])
    pakai_budget = c4.checkbox("Batasi total budget")
    budget = c4.number_input("Budget total (Rp)", min_value=1_000, value=10_000_000, step=500_000,
                             disabled=not pakai_budget)
    preset = c5.radio("Prioritas", [*PRESETS, "Kustom"], horizontal=True,
                      help="Bobot skor: P(berhasil), harga murah, dan lead time cepat.")
    weights = None
    if preset == "Kustom":
        s1, s2, s3 = c5.columns(3)
        weights = {"w_berhasil": s1.slider("Bobot berhasil", 0.0, 1.0, 0.5, 0.05),
                   "w_harga": s2.slider("Bobot harga", 0.0, 1.0, 0.3, 0.05),
                   "w_cepat": s3.slider("Bobot cepat", 0.0, 1.0, 0.2, 0.05)}
    else:
        w = PRESETS[preset]
        c5.caption(f"Bobot: berhasil {w['w_berhasil']:.2f} · harga {w['w_harga']:.2f} · cepat {w['w_cepat']:.2f}")

    if st.button("Cari rekomendasi vendor", type="primary"):
        need = {"material_id": material_id, "qty": int(qty), "hari_tersedia": int(hari),
                "budget": float(budget) if pakai_budget else None, "tanggal": tanggal.isoformat()}
        try:
            hasil_baru = recommend(need, preset=preset if weights is None else "Seimbang",
                                   weights=weights, data=data, model=model)
        except ValueError as e:
            for k in ("hasil", "penjelasan", "riwayat_id"):
                st.session_state.pop(k, None)
            st.error(str(e))
        else:
            with st.spinner("Menyusun penjelasan..."):
                penjelasan_baru = get_explanation(hasil_baru) if hasil_baru["top"] else None
            st.session_state.update(hasil=hasil_baru, penjelasan=penjelasan_baru,
                                    riwayat_id=storage.safe_save(hasil_baru, penjelasan_baru, "streamlit"))

    hasil = st.session_state.get("hasil")
    if hasil:
        st.divider()
        if not hasil["ranking"]:
            st.warning(hasil["pesan"] + " Coba tambah hari tersedia, naikkan jumlah sampai memenuhi MOQ vendor, atau naikkan budget.")
        else:
            ranking = pd.DataFrame(hasil["ranking"])
            k = hasil["kebutuhan"]
            st.subheader(f"Hasil untuk {k['qty']} {materials.loc[k['material_id'], 'satuan']} "
                         f"{materials.loc[k['material_id'], 'nama_material']}, {k['hari_tersedia']} hari")
            st.caption(f"{hasil['jumlah_kandidat']} vendor lolos saringan · preset {hasil['preset']} · "
                       f"tanggal pesan {tgl_id(hasil['tanggal_pesan'])} · "
                       f"model {hasil['model']} · rekam jejak vendor dihitung s.d. 30 Sep 2026")
            if hasil.get("musim_ramai"):
                mr = hasil["musim_ramai"]
                st.info(f"🌙 **Musim ramai: {mr['nama']}** ({tgl_id(mr['mulai'])} – {tgl_id(mr['selesai'])}). "
                        "Pada periode ini vendor lebih sering telat, sehingga peluang berhasil umumnya lebih rendah "
                        "dan ranking bisa berubah. Kolom *Berhasil saat musim ramai* menunjukkan rekam jejak "
                        "vendor di periode serupa.")

            cols = st.columns(min(3, len(ranking)))
            for col, (_, r) in zip(cols, ranking.head(3).iterrows()):
                with col.container(border=True):
                    st.markdown(f"**#{r['peringkat']} {r['nama_vendor']}** · {r['kota']}")
                    st.metric("Peluang berhasil", persen(r["p_berhasil"]))
                    st.markdown(f"{rupiah(r['harga_satuan'])}/satuan · total {rupiah(r['total_harga'])}  \n"
                                f"Lead time {r['lead_time_janji']} hari · skor {r['skor']:.3f}")
                    if r["data_terbatas"]:
                        st.caption(f"⚠️ Data terbatas ({r['jumlah_transaksi']} transaksi)")

            st.markdown("##### Penjelasan")
            penjelasan = st.session_state.get("penjelasan") or get_explanation(hasil)
            with st.container(border=True):
                st.markdown(to_markdown(penjelasan))
            caption = explanation_caption(penjelasan)
            if st.session_state.get("riwayat_id"):
                caption += f" Tersimpan di riwayat (#{st.session_state['riwayat_id']})."
            st.caption(caption)

            st.markdown("##### Trade-off harga vs peluang berhasil")
            st.plotly_chart(tradeoff_chart(ranking, top_k=3), width="stretch", theme="streamlit")

            st.markdown("##### Semua kandidat")
            st.dataframe(
                ranking[["peringkat", "nama_vendor", "kota", "harga_satuan", "total_harga", "lead_time_janji", "moq",
                         "p_berhasil", "skor", "jumlah_transaksi", "tingkat_berhasil_10", "berhasil_musim_ramai",
                         "rata_hari_telat", "tingkat_cacat", "data_terbatas"]],
                hide_index=True, width="stretch",
                column_config={
                    "peringkat": st.column_config.NumberColumn("#"),
                    "nama_vendor": "Vendor",
                    "kota": "Kota",
                    "harga_satuan": st.column_config.NumberColumn("Harga/satuan (Rp)", format="localized"),
                    "total_harga": st.column_config.NumberColumn("Total (Rp)", format="localized"),
                    "lead_time_janji": st.column_config.NumberColumn("Lead time (hari)"),
                    "moq": "MOQ",
                    "p_berhasil": st.column_config.ProgressColumn("P(berhasil)", min_value=0, max_value=1,
                                                                  format="percent"),
                    "skor": st.column_config.NumberColumn("Skor", format="%.3f"),
                    "jumlah_transaksi": "Transaksi",
                    "tingkat_berhasil_10": st.column_config.NumberColumn("Berhasil (10 terakhir)", format="percent"),
                    "berhasil_musim_ramai": st.column_config.NumberColumn("Berhasil saat musim ramai",
                                                                          format="percent"),
                    "rata_hari_telat": st.column_config.NumberColumn("Rata telat (hari)", format="%.1f"),
                    "tingkat_cacat": st.column_config.NumberColumn("Tingkat cacat", format="percent"),
                    "data_terbatas": "Data terbatas",
                },
            )

with tab_vendor:
    st.caption("Ringkasan rekam jejak seluruh vendor dari data transaksi (lihat saja).")
    stats = vendor_stats(data)
    kategori = st.multiselect("Kategori", sorted(stats["kategori"].unique()), default=sorted(stats["kategori"].unique()))
    st.dataframe(
        stats[stats["kategori"].isin(kategori)][
            ["vendor_id", "nama_vendor", "kategori", "kota", "tanggal_bergabung", "transaksi", "tingkat_berhasil",
             "rata_hari_telat", "rata_cacat", "pesanan_terakhir", "material"]],
        hide_index=True, width="stretch",
        column_config={
            "vendor_id": "ID", "nama_vendor": "Vendor", "kategori": "Kategori", "kota": "Kota",
            "tanggal_bergabung": st.column_config.DateColumn("Bergabung"),
            "transaksi": "Transaksi",
            "tingkat_berhasil": st.column_config.ProgressColumn("Tingkat berhasil", min_value=0, max_value=1,
                                                                format="percent"),
            "rata_hari_telat": st.column_config.NumberColumn("Rata telat (hari)", format="%.1f"),
            "rata_cacat": st.column_config.NumberColumn("Rata cacat", format="percent"),
            "pesanan_terakhir": st.column_config.DateColumn("Pesanan terakhir"),
            "material": "Material yang ditawarkan",
        },
    )

    st.markdown("##### Daftar penawaran per material")
    mat = st.selectbox("Material", materials.index, key="mat_vendor",
                       format_func=lambda x: materials.loc[x, "nama_material"])
    offers = data["offers"][data["offers"]["material_id"] == mat].merge(
        data["vendors"][["vendor_id", "nama_vendor", "kota"]], on="vendor_id")
    st.dataframe(
        offers[["offer_id", "nama_vendor", "kota", "harga_satuan", "lead_time_janji", "moq"]].sort_values("harga_satuan"),
        hide_index=True, width="stretch",
        column_config={"offer_id": "ID", "nama_vendor": "Vendor", "kota": "Kota",
                       "harga_satuan": st.column_config.NumberColumn("Harga/satuan (Rp)", format="localized"),
                       "lead_time_janji": "Lead time (hari)", "moq": "MOQ"},
    )

    st.markdown("##### Histori transaksi per vendor")
    tx_all = prepared_transactions(data)
    vendors = data["vendors"].set_index("vendor_id")
    h1, h2 = st.columns(2)
    vid = h1.selectbox("Vendor", vendors.index, key="vendor_histori",
                       format_func=lambda v: f"{vendors.loc[v, 'nama_vendor']} ({vendors.loc[v, 'kota']})")
    vtx = tx_all[tx_all["vendor_id"] == vid]
    mat_opsi = ["Semua", *sorted(vtx["material_id"].unique())]
    mat_h = h2.selectbox("Material", mat_opsi, key="mat_histori",
                         format_func=lambda x: x if x == "Semua" else materials.loc[x, "nama_material"])
    if mat_h != "Semua":
        vtx = vtx[vtx["material_id"] == mat_h]

    if vtx.empty:
        st.info("Vendor ini belum punya transaksi.")
    else:
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Transaksi", len(vtx))
        g2.metric("Berhasil", persen(vtx["berhasil"].mean()))
        g3.metric("Telat dari deadline", persen((vtx["telat_deadline"] > 0).mean()))
        g4.metric("Rata cacat", f"{vtx['persen_cacat'].mean() * 100:.1f}%")
        tabel = vtx.sort_values("tanggal_pesan", ascending=False).assign(
            material=lambda d: d["material_id"].map(materials["nama_material"]),
            musim_ramai=lambda d: d["musim_ramai"].astype(bool),
            berhasil=lambda d: d["berhasil"].astype(bool),
        )
        st.dataframe(
            tabel[["transaksi_id", "tanggal_pesan", "material", "qty", "harga_satuan", "total_harga",
                   "lead_time_janji", "deadline", "tanggal_datang", "telat_deadline", "qty_cacat", "persen_cacat",
                   "musim_ramai", "berhasil"]],
            hide_index=True, width="stretch", height=320,
            column_config={
                "transaksi_id": "ID",
                "tanggal_pesan": st.column_config.DateColumn("Dipesan"),
                "material": "Material", "qty": "Jumlah",
                "harga_satuan": st.column_config.NumberColumn("Harga/satuan (Rp)", format="localized"),
                "total_harga": st.column_config.NumberColumn("Total (Rp)", format="localized"),
                "lead_time_janji": "Janji (hari)",
                "deadline": st.column_config.DateColumn("Deadline"),
                "tanggal_datang": st.column_config.DateColumn("Datang"),
                "telat_deadline": st.column_config.NumberColumn("Telat (hari)"),
                "qty_cacat": "Cacat (pcs)",
                "persen_cacat": st.column_config.NumberColumn("Cacat", format="percent"),
                "musim_ramai": "Musim ramai",
                "berhasil": "Berhasil",
            },
        )
        st.caption("Berhasil = datang paling lambat di deadline dan cacat ≤ 3%. Histori inilah yang dipakai model "
                   "untuk menghitung rekam jejak vendor (hanya transaksi yang sudah datang sebelum tanggal pesanan).")

with tab_riwayat:
    st.caption("Riwayat permintaan rekomendasi dari aplikasi ini dan dari API, disimpan di database (Supabase Postgres).")
    if not storage.is_configured():
        st.info("Penyimpanan riwayat belum dikonfigurasi (DATABASE_URL kosong).")
    else:
        st.button("Muat ulang", key="muat_riwayat")
        try:
            riwayat = pd.DataFrame(storage.list_history(limit=50))
        except Exception as e:
            st.error(f"Gagal membaca riwayat: {type(e).__name__}")
        else:
            if riwayat.empty:
                st.write("Belum ada riwayat.")
            else:
                riwayat["material"] = riwayat["material_id"].map(materials["nama_material"])
                st.dataframe(
                    riwayat[["id", "waktu", "kanal", "material", "qty", "hari_tersedia", "budget", "preset",
                             "jumlah_kandidat", "vendor_teratas", "sumber_penjelasan", "versi_model"]],
                    hide_index=True, width="stretch",
                    column_config={
                        "id": "#", "waktu": st.column_config.DatetimeColumn("Waktu", format="D MMM YYYY, HH:mm"),
                        "kanal": "Kanal", "material": "Material", "qty": "Jumlah", "hari_tersedia": "Hari",
                        "budget": st.column_config.NumberColumn("Budget (Rp)", format="localized"),
                        "preset": "Prioritas", "jumlah_kandidat": "Kandidat", "vendor_teratas": "Vendor #1",
                        "sumber_penjelasan": "Penjelasan", "versi_model": "Model",
                    },
                )
