import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded generator: every property test is reproducible."""
    return np.random.default_rng(20261018)


def close(a, b, tol=1e-12):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol, rtol=0)
