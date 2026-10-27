import pytest

from gridcast.data.synthetic import make_synthetic


@pytest.fixture(scope="session")
def synthetic():
    return make_synthetic(start="2023-01-01", days=500, seed=1)
