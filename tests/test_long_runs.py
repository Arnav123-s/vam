"""Long-run machine audits (DESIGN 6.2): constant storage and no drift in positivity or normalization."""

import numpy as np
import pytest

from vam.bitkernel import BitMachine
from vam.nested import NUMBA_AVAILABLE, NestedConfig, NestedMachine
from vam.realizations.vam1 import Config, DenseMaps

MILLION = 1_000_000


def test_bit_kernel_storage_is_constant_over_a_million_events(rng):
    bm = BitMachine(Config(cells=16, context=16, consolidating=True, dt=1.0))
    size = bm.nbytes()
    for _ in range(4):
        bm.run(rng.integers(0, 2, MILLION // 4).astype(np.uint8))
        assert bm.nbytes() == size


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="the compiled nested machine needs numba")
def test_nested_storage_is_constant_over_a_million_events(rng):
    nm = NestedMachine(NestedConfig(Config(cells=1, context=1, consolidating=True, dt=1.0),
                                    depths=(8, 16, 32, 64), table_bits=16))
    size = nm.nbytes()
    for _ in range(4):
        nm.run(rng.integers(0, 2, MILLION // 4).astype(np.uint8))
        assert nm.nbytes() == size
    assert nm.evictions > 0                  # the tables did fill, and the storage still did not grow


@pytest.mark.slow
def test_coherent_machine_stays_a_valid_density_over_twenty_thousand_events(rng):
    """Every event of the full loop (observe, inject, evolve, emit, re-enter, learn) on a massive,
    partly coherent chain: the machine validates positivity and normalization at every event; at the
    end the accumulated drift is checked directly, and the storage has not grown."""
    cfg = Config(cells=2, context=2, theta=0.5, dephasing=0.4, dt=0.1)
    m = DenseMaps(cfg).reader()
    size = m.state.nbytes()
    for b in rng.integers(0, 2, 20_000):
        m.step(int(b), rng)
    R = m.state.R
    assert m.state.nbytes() == size
    assert abs(np.trace(R) - 1.0) < 1e-12
    assert np.max(np.abs(R - R.conj().T)) < 1e-14
    assert np.linalg.eigvalsh(R).min() > -1e-12
