"""T7 and T13.4: coherent path sums, environmental influence, dark-port interference, precision."""

import numpy as np
import pytest

from vam.linalg import random_density, random_unitary
from vam.paths import (
    PathDrivenEnvironment,
    coherent_sum,
    dark_port_probability,
    dark_port_probability_from_interferometer,
    influence_functional,
    path_amplitudes,
    path_sum,
    phase_error_bound,
)


def test_finite_path_sum_is_exact(rng):
    Us = [random_unitary(3, rng) for _ in range(4)]
    prod = Us[3] @ Us[2] @ Us[1] @ Us[0]
    for i in range(3):
        for f in range(3):
            assert np.isclose(path_sum(Us, i, f), prod[f, i], atol=1e-13)
    assert len(path_amplitudes(Us, 0, 0)) == 3 ** 3


@pytest.mark.parametrize("mixed", [False, True])
def test_two_path_influence_functional_equals_partial_trace(rng, mixed):
    d_sys, d_env, N = 2, 3, 3
    env = {(k, s): random_unitary(d_env, rng) for k in range(N) for s in range(d_sys)}
    model = PathDrivenEnvironment(tuple(random_unitary(d_sys, rng) for _ in range(N)),
                                  lambda k, s: env[(k, s)], d_sys, d_env)
    psi0 = random_unitary(d_sys, rng)[:, 0]
    sigma = random_density(d_env, rng) if mixed else np.diag([1, 0, 0]).astype(complex)
    assert np.allclose(model.reduced_density_by_paths(psi0, sigma), model.reduced_density_by_evolution(psi0, sigma),
                       atol=1e-12)


def test_pure_environment_influence_is_an_overlap(rng):
    U1, U2 = random_unitary(3, rng), random_unitary(3, rng)
    e0 = np.array([1, 0, 0], dtype=complex)
    F = influence_functional(U1, U2, np.outer(e0, e0))
    assert np.isclose(F, np.vdot(U2 @ e0, U1 @ e0))


@pytest.mark.parametrize("d,expected", [(1.0, 0.0), (0.5, 0.25), (0.0, 0.5)])
def test_dark_port_table(d, expected):
    assert np.isclose(dark_port_probability(d), expected)
    assert np.isclose(dark_port_probability_from_interferometer(d), expected, atol=1e-14)


def test_dark_port_complex_overlap(rng):
    for _ in range(20):
        d = rng.uniform(0, 1) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        assert np.isclose(dark_port_probability_from_interferometer(d), dark_port_probability(d), atol=1e-14)


def test_phase_error_bound(rng):
    """T13.4: |A~ - A| <= (delta/hbar) sum |a_j|; tight-ish when many phases shift together."""
    for hbar in (1.0, 0.05):
        a = rng.standard_normal(50) + 1j * rng.standard_normal(50)
        S = rng.uniform(0, 10, 50)
        delta = 1e-3
        Spert = S + rng.uniform(-delta, delta, 50)
        err = abs(coherent_sum(a, Spert, hbar) - coherent_sum(a, S, hbar))
        assert err <= phase_error_bound(a, delta, hbar) + 1e-15
    # strong cancellation: the relative error of a tiny sum can be large
    a = np.ones(2)
    S = np.array([0.0, np.pi])
    A = coherent_sum(a, S)
    At = coherent_sum(a, S + np.array([0.0, 1e-3]))
    assert abs(A) < 1e-15 and abs(At - A) > 1e-4
