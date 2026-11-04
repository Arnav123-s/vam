"""F-Part 7 with T3.4: density tells vectors how to flow; vectors tell the density where to curve."""

import itertools

import numpy as np

from vam.dirac import variable_speed_dirac_hamiltonian
from vam.geometry import site_speeds_from_density
from vam.linalg import is_hermitian, unitary_from_hamiltonian

N = 48
SITES = np.arange(N)


def packet(width=2.0):
    psi = np.zeros((N, 2), dtype=complex)
    psi[:, 0] = np.exp(-(SITES - N / 2) ** 2 / (2 * width ** 2))
    return (psi / np.linalg.norm(psi)).reshape(-1)


def populations(psi):
    return (np.abs(psi.reshape(N, 2)) ** 2).sum(axis=1)


def loop(coupling, T=10.0, dt=0.25, record=None):
    psi = packet()
    for _ in range(round(T / dt)):
        v = site_speeds_from_density(populations(psi), g0=1.0, coupling=coupling)
        H = variable_speed_dirac_hamiltonian(v)
        assert is_hermitian(H)
        psi = unitary_from_hamiltonian(H, dt) @ psi
        if record is not None:
            record.append(v)
    return psi


def spread(psi):
    p = populations(psi)
    m = (p * SITES).sum()
    return float(((SITES - m) ** 2 * p).sum())


def test_metric_sets_slower_transport_where_density_is_higher():
    p = np.array([0.0, 0.1, 0.4, 0.9])
    v = site_speeds_from_density(p, g0=1.0, coupling=2.0)
    assert np.all(v > 0) and np.all(np.diff(v) < 0)
    assert np.allclose(site_speeds_from_density(p, 1.0, 0.0), 1.0)


def test_loop_is_norm_preserving():
    assert np.isclose(np.linalg.norm(loop(20.0)), 1.0, atol=1e-12)


def test_vector_concentration_bends_its_own_flow():
    """A stronger density-to-geometry coupling holds the packet together (monotone in the coupling)."""
    s = [spread(loop(c)) for c in (0.0, 5.0, 20.0, 50.0)]
    assert all(a > b for a, b in itertools.pairwise(s))


def test_the_field_is_dynamic_and_driven_by_the_vectors():
    rec = []
    loop(20.0, record=rec)
    assert not np.allclose(rec[0], rec[-1])
    centre = N // 2
    assert rec[0][centre] < rec[-1][centre]          # the packet spread, so the centre sped up again
