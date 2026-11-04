"""T2: bits, incomplete configurations, order-sensitive composition."""

from fractions import Fraction

import numpy as np
import sympy as sp
from scipy.linalg import logm

from vam.bits import bits_to_bytes, bytes_to_bits, iter_bits
from vam.linalg import bloch_vector, commutator, projector
from vam.realizations import incomplete_codes as ic
from vam.realizations import noncommuting_pulses as ncp


# ---------------------------------------------------------------- T2.1 / F-Part 10
def test_bits_round_trip_for_any_bytes(rng):
    data = bytes(rng.integers(0, 256, 257, dtype=np.uint8))
    bits = bytes_to_bits(data)
    assert bits.size == 8 * len(data) and set(np.unique(bits)) <= {0, 1}
    assert bits_to_bytes(bits) == data
    assert list(iter_bits(b"\x80\x01")) == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


# ---------------------------------------------------------------- T2.2
def test_kraus_completeness_exact():
    total = sum(K.conj().T @ K for b in (0, 1) for K in ic.KRAUS[b].values())
    assert np.allclose(total, np.eye(3), atol=1e-15)


def test_population_maps_match_kraus_and_columns_sum_to_one():
    assert np.allclose((ic.M0 + ic.M1).sum(axis=0), 1)
    for b, M in ((0, ic.M0), (1, ic.M1)):
        for col in range(3):
            R = np.zeros((3, 3), dtype=complex)
            R[col, col] = 1
            out = ic.input_map(b)(R)
            assert np.allclose(np.real(np.diag(out)), M[:, col])
            assert np.allclose(out, np.diag(np.diag(out)))  # diagonal stays diagonal


def test_exact_table_with_rationals():
    """Exact rational arithmetic reproduces the T2.2 table."""
    M = {0: sp.Matrix([[Fraction(1, 3), 0, 1], [Fraction(1, 3), 0, 0], [0, 0, 0]]),
         1: sp.Matrix([[0, 1, 0], [0, 0, 0], [Fraction(1, 3), 0, 0]])}
    z = sp.Matrix([1, 0, 0])
    expected = [((sp.Rational(1, 2), sp.Rational(1, 2), 0), sp.Rational(2, 3), sp.Rational(1, 2)),
                ((sp.Rational(3, 4), 0, sp.Rational(1, 4)), sp.Rational(2, 3), sp.Rational(3, 4)),
                ((sp.Rational(2, 3), sp.Rational(1, 3), 0), sp.Rational(3, 4), sp.Rational(2, 3))]
    total = sp.Integer(1)
    for u, (pops, p_bit, p_end) in zip((0, 1, 0), expected, strict=True):
        nz = M[u] * z
        p = sum(nz)
        z = nz / p
        total *= p
        assert tuple(z) == pops and p == p_bit and z[0] == p_end
    assert total == sp.Rational(1, 3)


def test_machine_reproduces_table_and_string_probability():
    m = ic.machine()
    rng = np.random.default_rng(0)
    table = [((1 / 2, 1 / 2, 0), 2 / 3, 1 / 2), ((3 / 4, 0, 1 / 4), 2 / 3, 3 / 4), ((2 / 3, 1 / 3, 0), 3 / 4, 2 / 3)]
    for b, (pops, p_bit, p_end) in zip((0, 1, 0), table, strict=True):
        assert np.isclose(m.predict(b), p_bit, atol=1e-14)
        m.step(b, rng)
        assert np.allclose(np.real(np.diag(m.state.R)), pops, atol=1e-14)
        assert np.isclose(np.real(np.trace(ic.P_COMPLETE @ m.state.R)), p_end, atol=1e-14)
    assert np.isclose(ic.machine().sequence_probability([0, 1, 0]), 1 / 3, atol=1e-14)


def test_three_parses_each_one_ninth_and_state_keeps_the_ambiguity():
    parses = ic.brute_force_parses("010")
    assert set(parses) == {"AC", "BA", "B+B"}
    assert all(np.isclose(p, 1 / 9) for p in parses.values())
    # the density state's unnormalized populations equal the parse sums
    complete = parses["AC"] + parses["BA"]
    incomplete_b = parses["B+B"]
    m = ic.machine()
    R = m.state.R
    for b in (0, 1, 0):
        R = ic.input_map(b)(R)
    assert np.isclose(np.real(R[0, 0]), complete) and np.isclose(np.real(R[1, 1]), incomplete_b)


def test_brute_force_agrees_with_density_on_all_short_strings():
    import itertools
    for n in range(1, 8):
        for s in itertools.product("01", repeat=n):
            s = "".join(s)
            brute = sum(ic.brute_force_parses(s).values())
            dens = ic.machine().sequence_probability([int(c) for c in s]) if brute > 0 else 0.0
            assert np.isclose(brute, dens, atol=1e-13), s


# ---------------------------------------------------------------- T2.3
def test_order_survives_in_present_state():
    psi01 = ncp.final_state("01")
    psi10 = ncp.final_state("10")
    assert np.allclose(bloch_vector(projector(psi01)), [0, -1, 0], atol=1e-14)
    assert np.allclose(bloch_vector(projector(psi10)), [1, 0, 0], atol=1e-14)
    plus = np.array([1, 1]) / np.sqrt(2)
    assert np.isclose(abs(np.vdot(plus, psi01)) ** 2, 0.5, atol=1e-14)
    assert np.isclose(abs(np.vdot(plus, psi10)) ** 2, 1.0, atol=1e-14)
    # up to a global phase
    ref01 = np.array([1, -1j]) / np.sqrt(2)
    ref10 = np.array([1, 1]) / np.sqrt(2)
    assert np.isclose(abs(np.vdot(ref01, psi01)), 1) and np.isclose(abs(np.vdot(ref10, psi10)), 1)


def test_bch_second_order_term_is_the_commutator():
    G0, G1 = ncp.GENERATORS[0], ncp.GENERATORS[1]
    errs = []
    for delta in (1e-2, 5e-3):
        exact = logm(__import__("scipy").linalg.expm(delta * G1) @ __import__("scipy").linalg.expm(delta * G0))
        approx = delta * (G1 + G0) + delta ** 2 / 2 * commutator(G1, G0)
        errs.append(np.max(np.abs(exact - approx)))
        # without the commutator the error is second order
        assert np.max(np.abs(exact - delta * (G1 + G0))) > 0.4 * delta ** 2 * np.max(np.abs(commutator(G1, G0)))
    assert 7.0 < errs[0] / errs[1] < 9.0  # third order
