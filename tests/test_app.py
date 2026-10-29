from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


@pytest.fixture
def app():
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


def test_app_loads(app):
    assert not app.exception
    assert len(app.tabs) == 2


def test_recommendation_flow(app):
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 3  # kartu 3 teratas
    assert any("peringkat 1" in m.value for m in app.markdown)


def test_custom_weights(app):
    app.radio[0].set_value("Kustom").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("preset Kustom" in c.value for c in app.caption)


def test_no_candidates_shows_warning(app):
    app.selectbox[0].set_value("M05").run()
    app.number_input[1].set_value(3).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.warning and "Tidak ada vendor" in app.warning[0].value
