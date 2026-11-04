"""T8: Newtonian limit of the Dirac band, stationary phase, 25-configuration interference, threading."""

import itertools

import numpy as np
import pytest
import sympy as sp

from vam.readout import coarse_then_fine, nested_instrument
from vam.realizations import fourier25 as f25
from vam.semiclassical import (
    bump,
    characteristics,
    dirac_packet_evolution,
    free_action,
    gaussian_packet_moments,
    h_pos,
    newton_trajectory,
    oscillatory_integral,
    plane_wave_energies,
    stationary_phase_leading,
    thread_intermediate,
)


# ---------------------------------------------------------------- T8.1
@pytest.mark.parametrize("p", [[0.7], [0.3, -1.1, 0.4]])
def test_dirac_plane_wave_dispersion(p):
    m, c, V = 1.3, 2.1, 0.25
    E = np.sqrt(c ** 2 * np.dot(p, p) + m ** 2 * c ** 4)
    ev = plane_wave_energies(p, m, c, V)
    assert np.allclose(np.unique(np.round(ev, 10)), [V - E, V + E])


def test_wkb_dispersion_relation_symbolically():
    """Insert Psi = a exp(iS/hbar) at leading order: (-S_t - V)^2 = c^2 S_x^2 + m^2 c^4 for the 1-D Dirac operator."""
    St, Sx, V, m, c = sp.symbols("S_t S_x V m c", real=True)
    alpha = sp.Matrix([[0, 1], [1, 0]])
    beta = sp.Matrix([[1, 0], [0, -1]])
    # leading order: (-S_t) a = (c alpha S_x + beta m c^2 + V) a has a nonzero solution a
    M = (-St - V) * sp.eye(2) - (c * alpha * Sx + beta * m * c ** 2)
    assert sp.expand(M.det()) == sp.expand((-St - V) ** 2 - c ** 2 * Sx ** 2 - m ** 2 * c ** 4)
    p = sp.Symbol("p", real=True)
    mp = sp.Symbol("m", positive=True)
    eps = sp.Symbol("eps", positive=True)             # eps = 1/c
    Hpos = V + sp.sqrt(mp ** 2 * eps ** -4 + eps ** -2 * p ** 2)
    ser = sp.series(Hpos, eps, 0, 3).removeO()
    # m c^2 + p^2/2m + V, with the next term -p^4/(8 m^3 c^2) = O(p^4/(m^3 c^2))
    assert sp.simplify(ser - (mp / eps ** 2 + p ** 2 / (2 * mp) + V - eps ** 2 * p ** 4 / (8 * mp ** 3))) == 0


def test_characteristics_approach_newton_as_c_grows():
    """Error of Newton's trajectory relative to the Dirac-band characteristics shrinks as 1/c^2."""
    k = 1.0
    grad_V = lambda q: k * q
    m, q0, p0, T = 1.0, [1.0], [0.5], 3.0
    errs = []
    for c in (10.0, 20.0, 40.0):
        _, qr, _ = characteristics(q0, p0, grad_V, m, c, T)
        _, qn = newton_trajectory(q0, [p0[0] / m], grad_V, m, T)
        errs.append(np.max(np.abs(qr - qn)))
    assert 3.6 < errs[0] / errs[1] < 4.4 and 3.6 < errs[1] / errs[2] < 4.4
    assert np.isclose(h_pos(np.array([0.0]), m, 10.0), m * 100.0)


# ---------------------------------------------------------------- T8.2
def test_stationary_phase_leading_term_and_error_order():
    a = lambda x: bump(x, 0.0, 1.0)
    x_s = 0.2
    S = lambda x: 0.5 * (x - x_s) ** 2 + 0.3 * (x - x_s) ** 3
    errs = []
    for eps in (4e-3, 2e-3, 1e-3):
        num = oscillatory_integral(a, S, eps, -1, 1)
        lead = stationary_phase_leading(float(bump(np.array([x_s]))[0]), 0.0, 1.0, eps)
        errs.append(abs(num - lead))
        assert abs(num - lead) < 0.05 * abs(lead)
    # error is O(eps^{d/2 + 1}) = O(eps^{3/2}): halving eps divides it by about 2^{1.5}
    for e1, e2 in itertools.pairwise(errs):
        assert 2.0 < e1 / e2 < 3.6


def test_nonstationary_channel_decays_faster_than_any_power():
    a = lambda x: bump(x, 0.0, 1.0)
    S = lambda x: x + 0.2 * x ** 2      # S' = 1 + 0.4 x > 0 on the support
    vals = [abs(oscillatory_integral(a, S, eps, -1, 1)) for eps in (0.1, 0.05, 0.025, 0.0125)]
    exps = [np.log(v) / np.log(e) for v, e in zip(vals, (0.1, 0.05, 0.025, 0.0125), strict=True)]
    assert all(e2 > e1 for e1, e2 in itertools.pairwise(exps))   # effective power keeps growing
    # relative to a stationary channel's eps^{1/2} scale, the nonstationary channel vanishes
    rel = [v / np.sqrt(e) for v, e in zip(vals, (0.1, 0.05, 0.025, 0.0125), strict=True)]
    assert all(r2 < r1 / 3 for r1, r2 in itertools.pairwise(rel))


def test_needle_separation():
    """One stationary channel among nonstationary ones: its normalized intensity tends to one."""
    a = lambda x: bump(x, 0.0, 1.0)
    channels = [lambda x: 0.5 * x ** 2, lambda x: x, lambda x: -0.8 * x + 0.1 * x ** 2, lambda x: 1.5 * x]
    shares = []
    for eps in (0.05, 0.01):
        I = np.array([abs(oscillatory_integral(a, S, eps, -1, 1)) ** 2 for S in channels])
        shares.append(I[0] / I.sum())
    assert shares[1] > shares[0] and shares[1] > 0.999


# ---------------------------------------------------------------- T8.3
def test_25_configuration_interference_for_every_r():
    for r in range(25):
        amp = f25.amplitudes(r)
        assert np.isclose(amp[r], 1, atol=1e-14)
        assert np.max(np.abs(np.delete(amp, r))) < 5e-15
    # T15.2 reports residual wrong-channel amplitudes of about 1.3e-15 for r = 7
    assert np.max(np.abs(np.delete(f25.amplitudes(7), 7))) < 1.5e-15


def test_geometric_series_vanishes_exactly():
    """For z = w^d, w a primitive 25th root of unity and 25 not dividing d, sum_j z^j = 0.

    Exact proof by polynomial arithmetic: sum_j x^{(d j) mod 25} is divisible by
    the cyclotomic polynomial Phi_25, which annihilates w.
    """
    x = sp.Symbol("x")
    phi = sp.cyclotomic_poly(25, x)
    for d in range(1, 25):
        P = sum(x ** ((d * j) % 25) for j in range(25))
        assert sp.rem(sp.Poly(P, x), sp.Poly(phi, x)).is_zero
    assert not sp.rem(sp.Poly(sum(x ** 0 for _ in range(25)), x), sp.Poly(phi, x)).is_zero  # d = 0: sum is 25


def test_nested_readout_identifies_seven():
    coarse, fine = f25.nested_sectors()
    amp = f25.amplitudes(7)
    v = f25.F @ amp                      # state in the phase basis
    R = np.outer(v, v.conj())
    two = coarse_then_fine(R, coarse, fine)
    assert np.isclose(two[(1, 2)][0], 1) and np.isclose(two[(1, 2)][1], 1)
    inst = nested_instrument(coarse, fine)
    probs = inst.probabilities(R)
    for key, p in probs.items():
        pj, pa = two[key]
        joint = pj * pa if pj > 1e-12 else 0.0
        assert np.isclose(p, joint, atol=1e-13)


def test_streaming_residue_machine_reads_needle_with_certainty(rng):
    m = f25.machine()
    value = 0
    for b in rng.integers(0, 2, 200):
        rec = m.step(int(b), rng)
        value = (2 * value + int(b)) % 25
        assert rec.outcome == value and np.isclose(rec.p_outcome, 1.0, atol=1e-10)
    assert m.state.nbytes() == f25.initial_state().nbytes()


# ---------------------------------------------------------------- T8.4
def test_macro_micro_threading_free_particle():
    t, T = 0.25, 1.0
    th = thread_intermediate(lambda z: free_action(0.0, z, t), lambda z: free_action(z, 2.0, T - t), (-5, 5))
    assert np.isclose(th.z, 0.5, atol=1e-9)
    assert np.isclose(th.first_action, 0.5) and np.isclose(th.second_action, 1.5) and np.isclose(th.total_action, 2.0)
    assert np.isclose(th.total_action, free_action(0.0, 2.0, T))
    assert np.isclose(th.hessian, 16 / 3, rtol=1e-5)
    assert abs(th.momentum_mismatch) < 1e-6


def test_threading_symbolically():
    z, t = sp.symbols("z t", positive=True)
    S = z ** 2 / (2 * t) + (2 - z) ** 2 / (2 * (1 - t))
    sol = sp.solve(sp.diff(S, z), z)[0]
    assert sp.simplify(sol - 2 * t) == 0
    assert sp.nsimplify(sol.subs(t, sp.Rational(1, 4))) == sp.Rational(1, 2)
    assert sp.simplify(sp.diff(S, z, 2).subs(t, sp.Rational(1, 4))) == sp.Rational(16, 3)


def test_wave_packet_endpoint_and_spread():
    mean, var = gaussian_packet_moments(1.0, 0.0, 2.0, 0.1, 1.0, 0.01)
    assert np.isclose(mean, 2.0) and np.isclose(var, 0.0125)
    for t in (0.0, 0.5, 1.0):
        assert np.isclose(gaussian_packet_moments(t, 0, 2, 0.1, 1, 0.01)[1], 0.01 + 0.0025 * t ** 2)


def test_dirac_packet_follows_newton_for_large_c():
    """The same Dirac generator propagates the packet; its centre and spread match the Newtonian prediction."""
    for c in (200.0, 400.0):
        res = dirac_packet_evolution(1.0, 0.0, 2.0, 0.1, 1.0, c, 0.01, L=6.0, N=8192)
        v_rel = 2.0 / np.sqrt(1 + 4 / c ** 2)
        assert np.isclose(res.mean, v_rel, atol=2e-6)
        assert np.isclose(res.variance, 0.0125, rtol=2e-3)
    res0 = dirac_packet_evolution(0.0, 0.0, 2.0, 0.1, 1.0, 400.0, 0.01, L=6.0, N=8192)
    assert np.isclose(res0.variance, 0.01, rtol=1e-3)
