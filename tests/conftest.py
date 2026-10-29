import pytest

from core.data import load_data
from core.features import build_training_frame
from generator.generate import generate


@pytest.fixture(scope="session")
def generated():
    return generate()


@pytest.fixture(scope="session")
def training_frame():
    d = load_data()
    return build_training_frame(d["transactions"], d["offers"], d["materials"])
