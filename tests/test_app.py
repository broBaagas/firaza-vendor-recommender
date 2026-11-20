from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")  # jangan memanggil Gemini sungguhan saat tes
    monkeypatch.setenv("DATABASE_URL", "")  # jangan menulis ke database sungguhan
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


def test_app_loads(app):
    assert not app.exception
    assert len(app.tabs) == 3
    assert any("belum dikonfigurasi" in i.value for i in app.info)  # tab Riwayat tanpa DATABASE_URL


def test_recommendation_flow(app):
    app.button[0].click().run()
    assert not app.exception
    assert sum(m.label == "Peluang berhasil" for m in app.metric) == 3  # kartu 3 teratas
    assert any(m.value.startswith("Pilih ") and "**Trade-off**" in m.value for m in app.markdown)
    assert any("Penjelasan template" in c.value for c in app.caption)


def test_custom_weights(app):
    app.radio[0].set_value("Kustom").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("prioritas Kustom" in c.value for c in app.caption)


def test_peak_season_banner(app):
    from datetime import date
    app.date_input[0].set_value(date(2027, 2, 15)).run()
    assert any("Ramadan-Lebaran 2027" in c.value for c in app.caption)  # penanda di bawah input
    app.button[0].click().run()
    assert not app.exception
    assert any("musim ramai Ramadan-Lebaran 2027" in m.value for m in app.markdown)
    assert any("tanggal pesan 15 Feb 2027" in c.value for c in app.caption)
    assert any("Saran waktu rawan" in s.value for s in [*app.success, *app.warning])


def test_holiday_section(app):
    from datetime import date
    app.date_input[0].set_value(date(2027, 8, 17)).run()
    assert any("Hari Kemerdekaan Republik Indonesia" in c.value for c in app.caption)
    app.button[0].click().run()
    assert not app.exception
    assert any("adalah **hari besar: Hari Kemerdekaan Republik Indonesia**" in m.value for m in app.markdown)
    assert any("Saran waktu rawan" in s.value for s in [*app.success, *app.warning])


def test_normal_day_section(app):
    app.button[0].click().run()
    assert any("adalah **hari biasa**" in m.value for m in app.markdown)


def test_usage_guide_present(app):
    assert any(e.label.endswith("Cara pakai") for e in app.expander)
    assert any("Apa itu peluang berhasil?" in m.value for m in app.markdown)


def test_vendor_transaction_history(app):
    vendor_box = app.selectbox(key="vendor_histori")
    assert not app.exception
    assert any(m.label == "Transaksi" for m in app.metric)
    vendor_box.set_value(vendor_box.options[1]).run()
    app.selectbox(key="mat_histori").set_value(app.selectbox(key="mat_histori").options[1]).run()
    assert not app.exception
    assert any(c.value.startswith("Berhasil = datang") for c in app.caption)


def test_no_candidates_shows_warning(app):
    app.selectbox[0].set_value("M05").run()
    app.number_input[1].set_value(3).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.warning and "Tidak ada vendor" in app.warning[0].value
