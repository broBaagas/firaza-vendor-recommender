"""Antarmuka Streamlit: halaman Rekomendasi + tab Data Vendor (lihat saja).

Memanggil modul core langsung, bukan lewat API.
Jalankan: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.data import load_data  # noqa: E402
from core.features import LIMITED_DATA_MIN_TX, prepare_transactions  # noqa: E402
from core.ranking import PRESETS, load_model, recommend  # noqa: E402

TOP_COLOR, OTHER_COLOR = "#2a78d6", "#a3a29c"

st.set_page_config(page_title="Firaza Vendor Recommender", page_icon="📦", layout="wide")


@st.cache_data
def get_data() -> dict[str, pd.DataFrame]:
    return load_data()


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def vendor_stats(_data: dict) -> pd.DataFrame:
    tx = prepare_transactions(_data["transactions"])
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


def summary_text(ranking: pd.DataFrame) -> str:
    """Ringkasan deterministik sementara (sebelum penjelasan Gemini di Minggu 3)."""
    top = ranking.iloc[0]
    cheapest = ranking.loc[ranking["harga_satuan"].idxmin()]
    text = (f"**{top['nama_vendor']}** berada di peringkat 1 dengan peluang berhasil {persen(top['p_berhasil'])}, "
            f"harga {rupiah(top['harga_satuan'])}/satuan, dan lead time {top['lead_time_janji']} hari.")
    if cheapest["vendor_id"] != top["vendor_id"]:
        selisih = f"{(top['harga_satuan'] / cheapest['harga_satuan'] - 1) * 100:.1f}".replace(".", ",")
        text += (f" Vendor termurah, **{cheapest['nama_vendor']}** ({rupiah(cheapest['harga_satuan'])}), "
                 f"peluang berhasilnya {persen(cheapest['p_berhasil'])}; pilihan #1 lebih mahal "
                 f"{selisih}% tetapi peluang berhasilnya "
                 f"{(top['p_berhasil'] - cheapest['p_berhasil']) * 100:+.0f} poin.")
    else:
        text += " Vendor ini sekaligus yang termurah di antara kandidat."
    if top["data_terbatas"]:
        text += f" ⚠️ Rekam jejak vendor ini masih sedikit (< {LIMITED_DATA_MIN_TX} transaksi)."
    return text


data = get_data()
model = get_model()
materials = data["materials"].set_index("material_id")

st.title("📦 Firaza Vendor Recommender")
st.caption("Decision-support pemilihan vendor: prediksi peluang berhasil (tepat deadline dan cacat ≤ 3%) per vendor, "
           "lalu ranking berdasarkan peluang berhasil, harga, dan kecepatan. Keputusan akhir tetap di tangan Anda. "
           "**Data vendor dan transaksi sintetis.**")

tab_rekomendasi, tab_vendor = st.tabs(["Rekomendasi", "Data Vendor"])

with tab_rekomendasi:
    c1, c2, c3 = st.columns([2, 1, 1])
    material_id = c1.selectbox(
        "Material / jasa", materials.index,
        format_func=lambda m: f"{materials.loc[m, 'nama_material']} ({materials.loc[m, 'satuan']})")
    m = materials.loc[material_id]
    qty = c2.number_input(f"Jumlah ({m['satuan']})", min_value=1, value=int(m["qty_median"]), step=10)
    hari = c3.number_input("Hari tersedia sampai deadline", min_value=1, value=int(m["lead_time_max"]),
                           help=f"Lead time wajar untuk material ini {m['lead_time_min']}–{m['lead_time_max']} hari.")

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
                "budget": float(budget) if pakai_budget else None}
        try:
            st.session_state["hasil"] = recommend(need, preset=preset if weights is None else "Seimbang",
                                                  weights=weights, data=data, model=model)
        except ValueError as e:
            st.session_state.pop("hasil", None)
            st.error(str(e))

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
                       f"model {hasil['model']} · rekam jejak vendor dihitung s.d. 30 Sep 2026")

            cols = st.columns(min(3, len(ranking)))
            for col, (_, r) in zip(cols, ranking.head(3).iterrows()):
                with col.container(border=True):
                    st.markdown(f"**#{r['peringkat']} {r['nama_vendor']}** · {r['kota']}")
                    st.metric("Peluang berhasil", persen(r["p_berhasil"]))
                    st.markdown(f"{rupiah(r['harga_satuan'])}/satuan · total {rupiah(r['total_harga'])}  \n"
                                f"Lead time {r['lead_time_janji']} hari · skor {r['skor']:.3f}")
                    if r["data_terbatas"]:
                        st.caption(f"⚠️ Data terbatas ({r['jumlah_transaksi']} transaksi)")

            st.markdown("##### Ringkasan")
            st.markdown(summary_text(ranking))
            st.caption("Penjelasan AI (Gemini) beserta trade-off dan saran akan menggantikan ringkasan ini di Minggu 3.")

            st.markdown("##### Trade-off harga vs peluang berhasil")
            st.plotly_chart(tradeoff_chart(ranking, top_k=3), width="stretch", theme="streamlit")

            st.markdown("##### Semua kandidat")
            st.dataframe(
                ranking[["peringkat", "nama_vendor", "kota", "harga_satuan", "total_harga", "lead_time_janji", "moq",
                         "p_berhasil", "skor", "jumlah_transaksi", "tingkat_berhasil_10", "rata_hari_telat",
                         "tingkat_cacat", "data_terbatas"]],
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
