import pytest

from core.ranking import PRESETS


@pytest.mark.parametrize("name", list(PRESETS))
def test_preset_weights_sum_to_one(name):
    assert sum(PRESETS[name].values()) == pytest.approx(1.0)
