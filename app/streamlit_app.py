"""Antarmuka Streamlit: halaman Rekomendasi + tab Data Vendor (lihat saja).

Memanggil modul core langsung, bukan lewat API.
Jalankan: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="Firaza Vendor Recommender", layout="wide")
st.title("Firaza Vendor Recommender")

tab_rekomendasi, tab_vendor = st.tabs(["Rekomendasi", "Data Vendor"])

with tab_rekomendasi:
    st.info("Form kebutuhan, tabel ranking, grafik, dan penjelasan akan ditambahkan di Minggu 4.")

with tab_vendor:
    st.info("Tabel data vendor (lihat saja) akan ditambahkan di Minggu 4.")
