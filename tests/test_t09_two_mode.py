"""T9: exact vector-weight-output-feedback calculation (symbolic and through the machine)."""

import numpy as np
import pytest
import sympy as sp

from vam.linalg import Y, Z, kron
from vam.realizations import two_mode as tm

sX = sp.Matrix([[0, 1], [1, 0]])
sY = sp.Matrix([[0, -sp.I], [sp.I, 0]])
sZ = sp.Matrix([[1, 0], [0, -1]])
sI = sp.eye(2)
kp = sp.kronecker_product


def ptrace(R, keep):
    R = sp.Matrix(R)
    out = sp.zeros(2, 2)
    for a in range(2):
        for b in range(2):
            if keep == 0:
                out[a, b] = sum(R[2 * a + k, 2 * b + k] for k in range(2))
            else:
                out[a, b] = sum(R[2 * k + a, 2 * k + b] for k in range(2))
    return out


@pytest.fixture(scope="module")
def exact():
    H = kp(sX, sZ) + kp(sI, sX)
    assert H * H == 2 * sp.eye(4)
    U = sp.eye(4) / sp.sqrt(2) - sp.I / 2 * H           # cos(pi/4) I - i sin(pi/4)/sqrt2 H
    psi = U * sp.Matrix([1, 0, 0, 0])
    return H, U, psi


def test_state_after_ts(exact):
    _, U, psi = exact
    assert sp.simplify(psi - sp.Matrix([1 / sp.sqrt(2), -sp.I / 2, -sp.I / 2, 0])) == sp.zeros(4, 1)
    R = psi * psi.H
    RQ, RW = ptrace(R, 0), ptrace(R, 1)
    target = sp.Matrix([[sp.Rational(3, 4), sp.I / (2 * sp.sqrt(2))], [-sp.I / (2 * sp.sqrt(2)), sp.Rational(1, 4)]])
    assert sp.simplify(RQ - target) == sp.zeros(2) and sp.simplify(RW - target) == sp.zeros(2)
    assert sp.simplify((RQ * RQ).trace()) == sp.Rational(7, 8)
    assert np.allclose(U.evalf().tolist(), tm.U_S, atol=1e-15)


def test_correlations_change_the_differential_equation(exact):
    _, _, psi = exact
    R = psi * psi.H
    exact_rate = sp.simplify(2 * (R * kp(sY, sZ)).trace())
    assert exact_rate == -sp.sqrt(2)
    RQ, RW = ptrace(R, 0), ptrace(R, 1)
    product_rate = sp.simplify(2 * (RQ * sY).trace() * (RW * sZ).trace())
    assert product_rate == -1 / sp.sqrt(2)
    # identity d<Z_Q>/dt = 2 <Y_Q Z_W> from i[H, Z_Q]
    H = kp(sX, sZ) + kp(sI, sX)
    assert sp.simplify(sp.I * (H * kp(sZ, sI) - kp(sZ, sI) * H) - 2 * kp(sY, sZ)) == sp.zeros(4)


def test_readout_feedback_and_next_output(exact):
    _H, U, psi = exact
    P = {0: kp(sp.Matrix([[1, 0], [0, 0]]), sI), 1: kp(sp.Matrix([[0, 0], [0, 1]]), sI)}
    Vb = {b: sp.cos(sp.pi / 4) * sI - sp.I * (2 * b - 1) * sp.sin(sp.pi / 4) * sX for b in (0, 1)}
    probs, post = {}, {}
    for t in (0, 1):
        phi = P[t] * psi
        probs[t] = sp.simplify((phi.H * phi)[0])
        post[t] = phi / sp.sqrt(probs[t])
    assert probs == {0: sp.Rational(3, 4), 1: sp.Rational(1, 4)}
    w0 = sp.Matrix([sp.sqrt(sp.Rational(2, 3)), -sp.I * sp.sqrt(sp.Rational(1, 3))])
    same_ray = lambda u, v: sp.simplify(u * u.H - v * v.H) == sp.zeros(4)    # equal up to global phase
    assert same_ray(post[0], kp(sp.Matrix([1, 0]), w0))
    assert same_ray(post[1], kp(sp.Matrix([0, 1]), sp.Matrix([1, 0])))
    zw = {t: sp.simplify((post[t].H * kp(sI, sZ) * post[t])[0]) for t in (0, 1)}
    assert zw == {0: sp.Rational(1, 3), 1: 1}
    fed = {t: kp(Vb[t], Vb[t]) * post[t] for t in (0, 1)}
    bloch = lambda v, op: sp.simplify((v.H * op * v)[0])
    for t in (0, 1):
        assert [bloch(fed[t], kp(o, sI)) for o in (sX, sY, sZ)] == [0, 1, 0]
    assert [bloch(fed[0], kp(sI, o)) for o in (sX, sY, sZ)] == [0, sp.Rational(1, 3), 2 * sp.sqrt(2) / 3]
    assert [bloch(fed[1], kp(sI, o)) for o in (sX, sY, sZ)] == [0, -1, 0]
    rate = {t: bloch(fed[t], 2 * kp(sY, sZ)) for t in (0, 1)}
    assert rate == {0: 4 * sp.sqrt(2) / 3, 1: 0}
    assert np.isclose(float(rate[0]), 1.885618083)
    # Heisenberg identity for the next readout
    ident = sp.simplify(U.H * kp(sZ, sI) * U - (kp(sZ, sI) / 2 + kp(sY, sY) / 2 + kp(sY, sZ) / sp.sqrt(2)))
    assert ident == sp.zeros(4)
    nxt = {t: sp.simplify(bloch(U * fed[t], kp(sZ, sI))) for t in (0, 1)}
    assert nxt == {0: sp.Rational(5, 6), 1: -sp.Rational(1, 2)}
    p_next0 = {t: (1 + nxt[t]) / 2 for t in (0, 1)}
    assert p_next0 == {0: sp.Rational(11, 12), 1: sp.Rational(1, 4)}
    pairs = {(0, 0): probs[0] * p_next0[0], (0, 1): probs[0] * (1 - p_next0[0]),
             (1, 0): probs[1] * p_next0[1], (1, 1): probs[1] * (1 - p_next0[1])}
    assert pairs == {(0, 0): sp.Rational(11, 16), (0, 1): sp.Rational(1, 16),
                     (1, 0): sp.Rational(1, 16), (1, 1): sp.Rational(3, 16)}


def test_machine_reproduces_output_pairs():
    m = tm.machine()
    dist = m.output_distribution([None, None])
    expected = {(0, 0): 11 / 16, (0, 1): 1 / 16, (1, 0): 1 / 16, (1, 1): 3 / 16}
    assert set(dist) == set(expected)
    for k, v in expected.items():
        assert np.isclose(dist[k], v, atol=1e-14)


def test_five_steps_all_32_branches():
    m = tm.machine()
    frontier = [((), 1.0, m.state)]
    for _ in range(5):
        nxt = []
        for outs, p, st in frontier:
            for tau, _q, pt, st2 in m.branches(st, None):
                assert np.linalg.eigvalsh(st2.R)[0] > -1e-12 and np.isclose(np.trace(st2.R), 1)
                nxt.append(((*outs, tau), p * pt, st2))
        frontier = nxt
    assert len(frontier) == 32
    assert np.isclose(sum(p for _, p, _ in frontier), 1.0, atol=1e-13)


def test_same_vector_marginal_different_weights_different_response():
    m = tm.machine()
    br = {tau: st for tau, _q, _p, st in m.branches(m.state, None)}
    RQ = {t: tm.SPACE.reduce(br[t].R, "Q") for t in (0, 1)}
    assert np.allclose(RQ[0], RQ[1])
    rate = {t: 2 * np.real(np.trace(br[t].R @ kron(Y, Z))) for t in (0, 1)}
    assert np.isclose(rate[0], 4 * np.sqrt(2) / 3) and np.isclose(rate[1], 0)
