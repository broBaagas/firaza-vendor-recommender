"""Antarmuka Streamlit: tab Rekomendasi, Data Vendor (rekam jejak, penawaran, histori transaksi), dan Riwayat.

Memanggil modul core langsung, bukan lewat API.
Jalankan: streamlit run app/streamlit_app.py
"""

import os
import sys
from datetime import date, timedelta
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
from core.seasons import CALENDAR_END, holiday_on, peak_period  # noqa: E402

# Palet monokrom per mode. "ink" = warna teks utama, "paper" = latar. Kartu vendor peringkat 1 memakai
# kontras terbalik (latar ink, teks paper): satu-satunya elemen yang menonjol.
PALETTE = {
    "light": {"ink": "#111111", "paper": "#FFFFFF", "muted": "#6B6B6B", "soft": "#F6F6F6",
              "other": "#B5B5B5"},
    "dark": {"ink": "#F2F2F2", "paper": "#000000", "muted": "#9A9A9A", "soft": "#0F0F0F",
             "other": "#5C5C5C"},
}


def theme_mode() -> str:
    """'light' atau 'dark' sesuai tema yang aktif di browser (default terang)."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # noqa: BLE001 - st.context.theme belum tersedia (mis. saat tes)
        return "light"


def app_css(mode: str) -> str:
    p = PALETTE[mode]
    return APP_CSS.replace("__INK__", p["ink"]).replace("__PAPER__", p["paper"]) \
        .replace("__MUTED__", p["muted"]).replace("__SOFT__", p["soft"])


APP_CSS = """
<style>
[data-testid="stMainBlockContainer"] { max-width: 1120px; padding-top: 3rem; }
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] p { color: __PAPER__ !important; }
.brand {
    color: __MUTED__;
    font-weight: 700;
    font-size: 0.95rem;
    margin: 0 0 0.35rem 0;
}
h1.hero-title {
    font-size: clamp(1.9rem, 3.6vw, 2.75rem);
    line-height: 1.12;
    font-weight: 800;
    letter-spacing: -0.02em;
    max-width: 22ch;
    padding: 0;
    margin: 0 0 0.75rem 0;
}
.hero-sub {
    font-size: 1.05rem;
    line-height: 1.6;
    max-width: 68ch;
    margin: 0 0 0.25rem 0;
}
.st-key-form_kebutuhan { background: __SOFT__; }
.st-key-kartu_1 {
    background: __INK__;
    border-color: __INK__ !important;
}
.st-key-kartu_1 p, .st-key-kartu_1 span, .st-key-kartu_1 div, .st-key-kartu_1 label,
.st-key-kartu_1 [data-testid="stMetricValue"], .st-key-kartu_1 [data-testid="stMetricLabel"] {
    color: __PAPER__ !important;
}
.st-key-kartu_1 [data-testid="stCaptionContainer"] p { color: __PAPER__ !important; opacity: 0.7; }
.label-rekomendasi {
    display: inline-block;
    background: __PAPER__;
    color: __INK__ !important;
    font-weight: 700;
    font-size: 0.78rem;
    padding: 0.1rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.25rem;
}
.st-key-kartu_1 .label-rekomendasi { color: __INK__ !important; }
.peringkat-lain {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 600;
    opacity: 0.7;
    margin-bottom: 0.25rem;
}
</style>
"""

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


DATE_FMT = "D MMM YYYY"


def desimal(x: float, n: int) -> str:
    """Angka desimal gaya Indonesia (koma), misalnya 0,643."""
    return f"{x:.{n}f}".replace(".", ",")


def as_percent(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Salinan untuk tampilan: kolom rasio (0–1) dikali 100 agar bisa diformat "%.0f%%" (tanpa 2 desimal)."""
    out = df.copy()
    for c in cols:
        out[c] = out[c] * 100
    return out


BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def tgl_id(x) -> str:
    d = pd.Timestamp(x)
    return f"{d.day} {BULAN[d.month - 1]} {d.year}"


def tradeoff_chart(ranking: pd.DataFrame, top_k: int) -> go.Figure:
    p = PALETTE[MODE]
    fig = go.Figure()
    for is_top, name, color in [(False, "Kandidat lain", p["other"]), (True, f"{top_k} teratas", p["ink"])]:
        d = ranking[(ranking["peringkat"] <= top_k) == is_top]
        fig.add_trace(go.Scatter(
            x=d["harga_satuan"], y=d["p_berhasil"], mode="markers+text", name=name,
            text=d["nama_vendor"], textposition="top center",
            marker={"size": 12, "color": color, "line": {"width": 2, "color": p["paper"]}},
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


def calendar_section(hasil: dict) -> None:
    """Bagian 📅 Kalender pesanan: status hari (hari besar / biasa), hari besar s.d. deadline,
    musim ramai, dan saran vendor paling aman saat waktu rawan."""
    k = hasil.get("kalender")
    if not k:
        return
    st.markdown("##### 📅 Kalender pesanan")
    pesan = tgl_id(k["tanggal_pesan"])
    lines = [f"**{pesan}** adalah **hari besar: {k['hari_besar_tanggal_pesan']}**." if k["hari_besar_tanggal_pesan"]
             else f"**{pesan}** adalah **hari biasa** (bukan hari besar nasional)."]
    lain = [h for h in k["hari_besar_periode"] if h["tanggal"] != k["tanggal_pesan"]]
    if lain:
        daftar = "; ".join(f"{tgl_id(h['tanggal'])} {h['nama']}" for h in lain)
        lines.append(f"Sampai deadline {tgl_id(k['deadline'])} ada hari besar: {daftar}. Vendor bisa libur atau "
                     "pengiriman melambat di sekitar tanggal itu.")
    elif not k["hari_besar_tanggal_pesan"]:
        lines.append(f"Tidak ada hari besar sampai deadline {tgl_id(k['deadline'])}.")
    mr = k.get("musim_ramai")
    if mr:
        lines.append(f"🌙 Masuk **musim ramai {mr['nama']}** ({tgl_id(mr['mulai'])} – {tgl_id(mr['selesai'])}): vendor "
                     "lebih sering telat. Peluang berhasil di bawah sudah memperhitungkan musim ini (lihat kolom "
                     "*Berhasil saat musim ramai*).")
    elif k["hari_besar_periode"]:
        lines.append("Catatan: hari besar belum menjadi fitur model (data historis tidak mencatatnya), jadi peluang "
                     "berhasil di bawah belum memperhitungkannya. Karena itu disarankan memilih vendor paling aman.")

    aman = hasil.get("paling_aman")
    with st.container(border=True):
        st.markdown("  \n".join(lines))
        if k["waktu_rawan"] and aman:
            v1 = hasil["ranking"][0]
            if aman["peringkat"] == 1:
                st.success(f"**Saran waktu rawan:** tetap pilih **{aman['nama_vendor']}**. Vendor peringkat 1 ini juga "
                           f"yang peluang berhasilnya paling tinggi ({persen(aman['p_berhasil'])}).")
            else:
                selisih = aman["harga_satuan"] / v1["harga_satuan"] - 1
                harga = (f"{selisih * 100:.1f}% lebih mahal".replace(".", ",") if selisih > 0 else
                         f"{-selisih * 100:.1f}% lebih murah".replace(".", ",") if selisih < 0 else "harga sama")
                st.warning(
                    f"🛡️ **Saran waktu rawan:** pilih **{aman['nama_vendor']}** (peringkat {aman['peringkat']}), "
                    f"vendor dengan **peluang berhasil tertinggi: {persen(aman['p_berhasil'])}** vs "
                    f"{persen(v1['p_berhasil'])} milik {v1['nama_vendor']}. Harga {rupiah(aman['harga_satuan'])}/satuan "
                    f"({harga} dari peringkat 1), lead time {aman['lead_time_janji']} hari. Peringkat 1 tetap "
                    "pilihan terbaik menurut bobot prioritas; pilih vendor ini jika ketepatan waktu paling penting.")


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

MODE = theme_mode()
st.markdown(app_css(MODE), unsafe_allow_html=True)
st.markdown(
    '<p class="brand">Firaza Vendor Recommender</p>'
    '<h1 class="hero-title">Vendor mana yang paling aman untuk pesanan ini?</h1>'
    '<p class="hero-sub">Isi kebutuhan Anda. Model memperkirakan peluang tiap vendor mengirim '
    '<b>tepat deadline</b> dengan <b>cacat paling banyak 3%</b>, lalu mengurutkannya bersama harga dan '
    'kecepatan. Keputusan akhir tetap di tangan Anda.</p>',
    unsafe_allow_html=True)
st.caption("Data vendor dan transaksi di aplikasi ini sintetis, dibuat untuk demonstrasi.")

REPO_URL = "https://github.com/broBaagas/firaza-vendor-recommender"

with st.expander("ℹ️ Cara pakai"):
    st.markdown(
        "1. **Rekomendasi**: pilih material atau jasa, isi jumlah, sisa hari sampai deadline, dan tanggal pesan "
        "(opsional: batasi total budget), pilih prioritas, lalu klik **Cari rekomendasi vendor**. "
        "Bagian **📅 Kalender pesanan** memberi tahu apakah tanggal itu hari besar (🎉, misalnya 17 Agustus) "
        "atau hari biasa, hari besar sampai deadline, dan musim ramai (🌙 Ramadan–Lebaran, musim nikah). "
        "Di waktu rawan ini muncul saran vendor dengan peluang berhasil paling tinggi.\n"
        "2. Baca kartu **3 vendor teratas**, **penjelasan** (rekomendasi, alasan, trade-off, saran), grafik "
        "harga vs peluang berhasil, dan tabel semua kandidat.\n"
        "3. Ganti **Prioritas** untuk melihat ranking berubah: *Seimbang*, *Termurah*, *Tercepat*, "
        "atau *Kustom* (atur bobot sendiri).\n"
        "4. **Data Vendor**: rekam jejak semua vendor, penawaran per material, dan histori transaksi per vendor.\n"
        "5. **Riwayat**: permintaan rekomendasi yang pernah dibuat (tersimpan di database).\n\n"
        "**Apa itu peluang berhasil?**\n\n"
        "Pesanan dianggap **berhasil** jika memenuhi **dua syarat sekaligus**: barang **datang paling lambat "
        "di deadline** dan barang **cacat tidak lebih dari 3%**. Telat sehari saja, atau cacat 4%, dihitung gagal.\n\n"
        "**Peluang berhasil** (P(berhasil)) adalah perkiraan seberapa besar kemungkinan pesanan Anda berhasil "
        "jika dipesan ke vendor itu. Contoh: peluang **83%** berarti dari 100 pesanan serupa ke vendor tersebut, "
        "kira-kira **83 berhasil** dan 17 telat atau cacatnya terlalu banyak.\n\n"
        "Angka ini dihitung oleh model *machine learning* (Random Forest) yang belajar dari ±6.000 transaksi "
        "sebelumnya. Yang diperhitungkan:\n"
        "- **Rekam jejak vendor**: seberapa sering berhasil di 10 transaksi terakhir, rata-rata telat, tingkat "
        "cacat, kinerja saat musim ramai, dan jumlah transaksinya.\n"
        "- **Kondisi pesanan**: sisa waktu dibanding lead time yang dijanjikan vendor (makin longgar, makin aman), "
        "jumlah barang, harga, dan apakah tanggal pesan jatuh di **musim ramai**.\n\n"
        "Cara membaca:\n"
        "- **80% ke atas**: umumnya aman.\n"
        "- **Di bawah 80%**: lebih berisiko; tambah waktu pemesanan, siapkan vendor cadangan, atau pilih vendor "
        "dengan peluang lebih tinggi.\n"
        "- Ini **perkiraan, bukan jaminan**. ⚠️ *Data terbatas* (vendor dengan kurang dari 5 transaksi) berarti "
        "perkiraannya kurang pasti.\n\n"
        "**Istilah lain.** *Skor* = gabungan peluang berhasil, harga murah, dan lead time cepat sesuai bobot "
        "prioritas; vendor diurutkan dari skor tertinggi. Ranking ditentukan model, bukan Gemini; Gemini hanya "
        "menulis penjelasannya.\n\n"
        f"Kode sumber, dokumentasi, dan evaluasi model: [GitHub]({REPO_URL}) · "
        f"[Laporan evaluasi]({REPO_URL}/blob/main/docs/evaluasi_model.md) · "
        f"[Asumsi data]({REPO_URL}/blob/main/docs/asumsi_data.md)"
    )

tab_rekomendasi, tab_vendor, tab_riwayat = st.tabs(["Rekomendasi", "Data Vendor", "Riwayat"])

with tab_rekomendasi:
    box = st.container(border=True, key="form_kebutuhan")
    box.markdown("##### Kebutuhan pengadaan")
    c1, c2, c3, c6 = box.columns([2, 1, 1, 1])
    material_id = c1.selectbox(
        "Material atau jasa", materials.index,
        format_func=lambda m: f"{materials.loc[m, 'nama_material']} ({materials.loc[m, 'satuan']})")
    m = materials.loc[material_id]
    qty = c2.number_input(f"Jumlah ({m['satuan']})", min_value=1, value=int(m["qty_median"]), step=10)
    hari = c3.number_input("Sisa waktu (hari)", min_value=1, value=int(m["lead_time_max"]),
                           help="Jumlah hari dari tanggal pesan sampai barang harus sudah diterima (deadline). "
                                f"Lead time wajar untuk material ini {m['lead_time_min']}–{m['lead_time_max']} hari.")
    hari_ini = min(date.today(), CALENDAR_END.date())
    tanggal = c6.date_input("Tanggal pesan", value=hari_ini, min_value=hari_ini, max_value=CALENDAR_END.date(),
                            format="DD/MM/YYYY",
                            help="Menentukan musim ramai (Ramadan–Lebaran, musim nikah Syawal/Dzulhijjah), "
                                 "saat vendor lebih sering telat.")
    hari_besar_input = holiday_on(tanggal)
    musim_input = peak_period(tanggal)
    status = [f"Deadline {tgl_id(tanggal + timedelta(days=int(hari)))}."]
    if hari_besar_input:
        status.append(f"🎉 Tanggal pesan jatuh di hari besar: {hari_besar_input}.")
    if musim_input:
        status.append(f"🌙 Masuk musim ramai {musim_input['nama']}.")
    if not hari_besar_input and not musim_input:
        status.append("Tanggal pesan adalah hari biasa, di luar musim ramai.")
    box.caption(" ".join(status))

    c5, c4 = box.columns([3, 2], gap="large")
    preset = c5.segmented_control(
        "Prioritas", [*PRESETS, "Kustom"], default="Seimbang", key="prioritas",
        help="Menentukan bobot skor: peluang berhasil, harga murah, dan lead time cepat.") or "Seimbang"
    pakai_budget = c4.toggle("Batasi total budget")
    budget = (c4.number_input("Budget total (Rp)", min_value=1_000, value=10_000_000, step=500_000)
              if pakai_budget else None)
    weights = None
    if preset == "Kustom":
        s1, s2, s3 = c5.columns(3)
        weights = {"w_berhasil": s1.slider("Bobot berhasil", 0.0, 1.0, 0.5, 0.05),
                   "w_harga": s2.slider("Bobot harga", 0.0, 1.0, 0.3, 0.05),
                   "w_cepat": s3.slider("Bobot cepat", 0.0, 1.0, 0.2, 0.05)}
    else:
        w = PRESETS[preset]
        c5.caption(f"Bobot: peluang berhasil {w['w_berhasil']:.0%}, harga {w['w_harga']:.0%}, "
                   f"kecepatan {w['w_cepat']:.0%}.")

    if box.button("Cari rekomendasi vendor", type="primary"):
        need = {"material_id": material_id, "qty": int(qty), "hari_tersedia": int(hari),
                "budget": float(budget) if budget else None, "tanggal": tanggal.isoformat()}
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
            st.caption(f"{hasil['jumlah_kandidat']} vendor lolos saringan dengan prioritas {hasil['preset']}, "
                       f"tanggal pesan {tgl_id(hasil['tanggal_pesan'])}. Model {hasil['model']}; rekam jejak vendor "
                       "dihitung sampai 30 Sep 2026.")
            cols = st.columns(min(3, len(ranking)))
            for col, (_, r) in zip(cols, ranking.head(3).iterrows()):
                with col.container(border=True, key=f"kartu_{r['peringkat']}"):
                    if r["peringkat"] == 1:
                        st.markdown('<span class="label-rekomendasi">Rekomendasi</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="peringkat-lain">Peringkat {r["peringkat"]}</span>',
                                    unsafe_allow_html=True)
                    st.markdown(f"#### {r['nama_vendor']}")
                    st.caption(r["kota"])
                    st.metric("Peluang berhasil", persen(r["p_berhasil"]))
                    st.markdown(f"{rupiah(r['harga_satuan'])} per satuan  \n"
                                f"Total {rupiah(r['total_harga'])}  \n"
                                f"Dikirim dalam {r['lead_time_janji']} hari  \n"
                                f"Skor {desimal(r['skor'], 3)}")
                    if r["data_terbatas"]:
                        st.caption(f"⚠️ Data terbatas ({r['jumlah_transaksi']} transaksi)")

            calendar_section(hasil)

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
                as_percent(ranking, ["p_berhasil", "tingkat_berhasil_10", "berhasil_musim_ramai",
                                     "tingkat_cacat"])[["peringkat", "nama_vendor", "kota", "harga_satuan", "total_harga", "lead_time_janji", "moq",
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
                    "p_berhasil": st.column_config.ProgressColumn("Peluang berhasil", min_value=0, max_value=100,
                                                                  format="%.0f%%"),
                    "skor": st.column_config.NumberColumn("Skor", format="%.3f"),
                    "jumlah_transaksi": "Transaksi",
                    "tingkat_berhasil_10": st.column_config.NumberColumn("Berhasil (10 terakhir)", format="%.0f%%"),
                    "berhasil_musim_ramai": st.column_config.NumberColumn("Berhasil saat musim ramai",
                                                                          format="%.0f%%"),
                    "rata_hari_telat": st.column_config.NumberColumn("Rata telat (hari)", format="%.1f"),
                    "tingkat_cacat": st.column_config.NumberColumn("Tingkat cacat", format="%.1f%%"),
                    "data_terbatas": "Data terbatas",
                },
            )

with tab_vendor:
    st.markdown("##### Rekam jejak vendor")
    st.caption("Ringkasan kinerja setiap vendor dari seluruh transaksi.")
    stats = vendor_stats(data)
    kategori = st.segmented_control("Kategori", ["Semua", *sorted(stats["kategori"].unique())], default="Semua",
                                    key="kategori_vendor")
    tampil = stats if kategori in (None, "Semua") else stats[stats["kategori"] == kategori]
    st.dataframe(
        as_percent(tampil, ["rata_cacat", "tingkat_berhasil"])[
            ["vendor_id", "nama_vendor", "kategori", "kota", "tanggal_bergabung", "transaksi", "tingkat_berhasil",
             "rata_hari_telat", "rata_cacat", "pesanan_terakhir", "material"]],
        hide_index=True, width="stretch",
        column_config={
            "vendor_id": "ID", "nama_vendor": "Vendor", "kategori": "Kategori", "kota": "Kota",
            "tanggal_bergabung": st.column_config.DateColumn("Bergabung", format=DATE_FMT),
            "transaksi": "Transaksi",
            "tingkat_berhasil": st.column_config.ProgressColumn("Tingkat berhasil", min_value=0, max_value=100,
                                                                format="%.0f%%"),
            "rata_hari_telat": st.column_config.NumberColumn("Rata telat (hari)", format="%.1f"),
            "rata_cacat": st.column_config.NumberColumn("Rata cacat", format="%.1f%%"),
            "pesanan_terakhir": st.column_config.DateColumn("Pesanan terakhir", format=DATE_FMT),
            "material": "Material yang ditawarkan",
        },
    )

    st.markdown("##### Penawaran per material")
    st.caption("Harga, lead time yang dijanjikan, dan jumlah minimum pesanan (MOQ) tiap vendor, dari yang termurah.")
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
    st.caption("Setiap pesanan ke satu vendor: kapan dipesan, kapan datang, dan berapa yang cacat.")
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
            as_percent(tabel, ["persen_cacat"])[["transaksi_id", "tanggal_pesan", "material", "qty", "harga_satuan", "total_harga",
                   "lead_time_janji", "deadline", "tanggal_datang", "telat_deadline", "qty_cacat", "persen_cacat",
                   "musim_ramai", "berhasil"]],
            hide_index=True, width="stretch", height=320,
            column_config={
                "transaksi_id": "ID",
                "tanggal_pesan": st.column_config.DateColumn("Dipesan", format=DATE_FMT),
                "material": "Material", "qty": "Jumlah",
                "harga_satuan": st.column_config.NumberColumn("Harga/satuan (Rp)", format="localized"),
                "total_harga": st.column_config.NumberColumn("Total (Rp)", format="localized"),
                "lead_time_janji": "Janji (hari)",
                "deadline": st.column_config.DateColumn("Deadline", format=DATE_FMT),
                "tanggal_datang": st.column_config.DateColumn("Datang", format=DATE_FMT),
                "telat_deadline": st.column_config.NumberColumn("Telat (hari)"),
                "qty_cacat": "Cacat (pcs)",
                "persen_cacat": st.column_config.NumberColumn("Cacat", format="%.1f%%"),
                "musim_ramai": "Musim ramai",
                "berhasil": "Berhasil",
            },
        )
        st.caption("Berhasil = datang paling lambat di deadline dan cacat ≤ 3%. Histori inilah yang dipakai model "
                   "untuk menghitung rekam jejak vendor (hanya transaksi yang sudah datang sebelum tanggal pesanan).")

with tab_riwayat:
    st.markdown("##### Riwayat rekomendasi")
    st.caption("Setiap permintaan rekomendasi dari aplikasi ini dan dari API tersimpan di database (Supabase Postgres).")
    if not storage.is_configured():
        st.info("Penyimpanan riwayat belum dikonfigurasi (DATABASE_URL kosong).")
    else:
        st.button("Muat ulang", key="muat_riwayat")
        try:
            riwayat = pd.DataFrame(storage.list_history(limit=50))
        except Exception as e:
            st.error(f"Gagal membaca riwayat: {storage.describe_error(e)}")
        else:
            if riwayat.empty:
                st.info("Belum ada riwayat. Riwayat muncul setelah Anda menekan **Cari rekomendasi vendor**.")
            else:
                riwayat["material"] = riwayat["material_id"].map(materials["nama_material"])
                riwayat["budget"] = pd.to_numeric(riwayat["budget"], errors="coerce")  # None → sel kosong
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
