"""T12: forward structural learning (correlation-to-weight law, consolidation, generalization bound)."""

from fractions import Fraction

import numpy as np
import sympy as sp

from vam.canalization import consolidation_superop, relation_preserving_map
from vam.channels import KrausMap, apply_superop, random_channel, random_instrument, replacement_channel
from vam.linalg import random_density, random_hermitian, trace_distance
from vam.plasticity import CorrelationWeightLaw, GatedCorrelationLaw, consolidated_state, consolidation_multiplier
from vam.space import Space

LAW = CorrelationWeightLaw(g=1.0, kappa=1.0, gamma=1.0, kT=0.02)


# -- T12.1 ----------------------------------------------------------------------------------

def test_exact_table():
    assert np.isclose(LAW.mean(0.0, 0.0, 0.8), 0.0) and LAW.variance(0.0) == 0.0
    assert np.isclose(LAW.mean(np.log(2), 0.0, 0.8), 0.4, atol=1e-15)
    assert np.isclose(LAW.variance(np.log(2)), 0.015, atol=1e-15)
    assert np.isclose(LAW.mean(60.0, 0.0, 0.8), 0.8) and np.isclose(LAW.variance(60.0), 0.02)


def test_closed_forms_solve_the_moment_equations_symbolically():
    t, g, k, gam, kT, c0, w0 = sp.symbols("t g kappa gamma kT c0 w0", positive=True)
    mean = g * c0 / k + (w0 - g * c0 / k) * sp.exp(-k * t / gam)
    var = kT / k * (1 - sp.exp(-2 * k * t / gam))
    # dE[w]/dt = (g c0 - kappa E[w]) / gamma ;  dVar/dt = -2 kappa Var / gamma + 2 kT / gamma
    assert sp.simplify(sp.diff(mean, t) - (g * c0 - k * mean) / gam) == 0
    assert sp.simplify(sp.diff(var, t) - (-2 * k * var / gam + 2 * kT / gam)) == 0
    assert mean.subs(t, 0) == w0 and var.subs(t, 0) == 0
    vals = {g: 1, k: 1, gam: 1, kT: sp.Rational(2, 100), c0: sp.Rational(8, 10), w0: 0, t: sp.log(2)}
    assert sp.nsimplify(mean.subs(vals)) == sp.Rational(2, 5)
    assert sp.nsimplify(var.subs(vals)) == sp.Rational(3, 200)


def test_reciprocity_from_one_energy():
    w, C = sp.symbols("w C")
    g, k = sp.Rational(3, 2), sp.Rational(1, 2)
    law = CorrelationWeightLaw(g=float(g), kappa=float(k))
    E = k / 2 * w ** 2 - g * w * C
    for wv, Cv in [(0.3, -0.2), (-1.1, 0.7)]:
        assert np.isclose(law.force_on_weight(wv, Cv), float(-sp.diff(E, w).subs({w: wv, C: Cv})))
        assert np.isclose(law.coupling_seen_by_vectors(wv), float(-sp.diff(E, C).subs({w: wv, C: Cv})))
        assert np.isclose(law.energy(wv, Cv), float(E.subs({w: wv, C: Cv})))


def test_exact_ou_transitions_reproduce_the_table_statistically(rng):
    n, steps = 200_000, 7
    dt = np.log(2) / steps
    w = np.zeros(n)
    for _ in range(steps):
        w = LAW.mean(dt, w, 0.8) + np.sqrt(LAW.variance(dt)) * rng.standard_normal(n)
    se_mean = np.sqrt(0.015 / n)
    assert abs(w.mean() - 0.4) < 5 * se_mean
    se_var = 0.015 * np.sqrt(2 / (n - 1))
    assert abs(w.var(ddof=1) - 0.015) < 5 * se_var


def test_zero_noise_idealization_is_deterministic():
    law = CorrelationWeightLaw(g=1.0, kappa=1.0, gamma=1.0, kT=0.0)
    w = 0.0
    for _ in range(10):
        w = law.step(w, 0.8, np.log(2) / 10)
    assert np.isclose(w, 0.4, atol=1e-14)


def test_gated_law_reaches_a_conditional_expectation_forward(rng):
    """Sector weights converge to E[z | sector] from a stream, with nothing stored but w.

    Many independent streams run side by side; each keeps only its two weights.
    The update is affine in ``z``, so its stationary mean is exactly ``E[z | s]``
    and its stationary variance is ``Var(z | s) (1 - e^-dt) / (1 + e^-dt)``.
    """
    law = GatedCorrelationLaw(g=1.0, kappa=1.0, gamma=1.0, dt=0.002)
    streams, steps = 2000, 20_000
    p1, ez = 0.3, np.array([-0.2, 0.6])
    w = np.zeros((streams, 2))
    for _ in range(steps):
        s = (rng.random(streams) < p1).astype(int)
        z = np.where(rng.random(streams) < (1 + ez[s]) / 2, 1.0, -1.0)
        a = np.eye(2)[s]
        w = law.update(w, a, a * z[:, None])
    e = np.exp(-law.dt)
    se = np.sqrt((1 - ez ** 2) * (1 - e) / (1 + e) / streams)
    assert np.all(np.abs(w.mean(axis=0) - ez) < 6 * se)
    assert np.allclose(w.std(axis=0), np.sqrt((1 - ez ** 2) * (1 - e) / (1 + e)), rtol=0.1)


def test_gated_law_leaves_inactive_sectors_unchanged():
    law = GatedCorrelationLaw(dt=0.1)
    w = np.array([0.3, -0.5, 0.9])
    out = law.update(w, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert out[1] == -0.5 and out[2] == 0.9 and out[0] > 0.3


# -- T12.2 ----------------------------------------------------------------------------------

def test_consolidation_multipliers_exact():
    r = Fraction(1, 2)
    assert [consolidation_multiplier(r, n) for n in (1, 2, 5, 10)] == [Fraction(1, 2), Fraction(1, 4),
                                                                      Fraction(1, 32), Fraction(1, 1024)]


def test_consolidation_power_formula(rng):
    sp_ = Space.of(R=2, N=2)
    Pi = relation_preserving_map(sp_, ["N"], random_density(2, rng))
    P = Pi.superop()
    for r in (0.0, 0.5, 0.9):
        F = consolidation_superop(r, P)
        rho = random_density(4, rng)
        for n in (1, 2, 5, 10):
            Fn = np.linalg.matrix_power(F, n)
            assert np.allclose(Fn, P + r ** n * (np.eye(16) - P), atol=1e-13)
            assert np.allclose(apply_superop(Fn, rho), consolidated_state(rho, Pi(rho), r, n), atol=1e-13)


def test_repetition_also_consolidates_a_mistaken_organization(rng):
    wrong = random_density(2, rng)
    Pi_wrong = replacement_channel(wrong, 2)
    F = consolidation_superop(0.5, Pi_wrong)
    rho = random_density(2, rng)
    out = apply_superop(np.linalg.matrix_power(F, 40), rho)
    assert np.allclose(out, wrong, atol=1e-11)


# -- T12.3 ----------------------------------------------------------------------------------

def random_effect(d, rng):
    H = random_hermitian(d, rng)
    _w, V = np.linalg.eigh(H)
    return (V * rng.random(d)) @ V.conj().T          # 0 <= M <= I


def test_generalization_bound_under_a_common_process(rng):
    d = 3
    for _ in range(50):
        R, Rp = random_density(d, rng), random_density(d, rng)
        Phi = random_channel(d, rng)
        M = random_effect(d, rng)
        D = trace_distance(R, Rp)
        assert trace_distance(Phi(R), Phi(Rp)) <= D + 1e-12
        diff = abs(np.trace(M @ Phi(R)) - np.trace(M @ Phi(Rp)))
        assert diff <= D + 1e-12


def test_bound_includes_adaptive_output_feedback(rng):
    """Instrument + outcome-dependent feedback = one channel with a classical output factor."""
    d = 3
    J = random_instrument(d, (0, 1), rng)
    E = {t: random_channel(d, rng) for t in (0, 1)}
    kraus = []
    for t in (0, 1):
        flag = np.zeros((2, 1))
        flag[t, 0] = 1.0
        for Ke in E[t].kraus:
            for Kj in J[t].kraus:
                kraus.append(np.kron(flag, Ke @ Kj))
    Lam = KrausMap(kraus)
    assert Lam.is_trace_preserving()
    for _ in range(30):
        R, Rp = random_density(d, rng), random_density(d, rng)
        M = random_effect(2 * d, rng)
        assert abs(np.trace(M @ Lam(R)) - np.trace(M @ Lam(Rp))) <= trace_distance(R, Rp) + 1e-12


def test_postselection_can_amplify_differences():
    R, Rp = np.diag([0.99, 0.01]).astype(complex), np.diag([0.999, 0.001]).astype(complex)
    K = np.diag([np.sqrt(0.001), 1.0])
    post = lambda S: (K @ S @ K) / np.trace(K @ S @ K)
    assert trace_distance(R, Rp) < 0.01
    assert trace_distance(post(R), post(Rp)) > 0.4
