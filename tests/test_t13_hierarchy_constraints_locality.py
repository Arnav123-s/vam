"""T13: hierarchy, hard constraints, locality and the cost of bit-level execution."""

import itertools

import numpy as np

from vam.canalization import coarse_channel_with_remainder
from vam.channels import is_completely_positive_superop
from vam.constraints import constraint_report, forbidden_population, leakage_rate
from vam.density import from_factor
from vam.linalg import (
    X,
    dag,
    expectation,
    kron,
    random_density,
    random_hermitian,
    random_pure_state,
    random_unitary,
    unitary_from_hamiltonian,
    von_neumann_entropy,
)
from vam.lindblad import evolve, liouvillian
from vam.paths import coherent_sum, phase_error_bound
from vam.space import Space

# -- T13.1 hierarchy -------------------------------------------------------------------------

def mutual_information(R, space, a, b):
    S = lambda names: von_neumann_entropy(space.reduce(R, names))
    return S(a) + S(b) - S(a + b)


def test_factor_form_with_hierarchical_isometries_keeps_correlations(rng):
    sp = Space.of(A=2, B=2)
    Xf = rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))
    R = from_factor(Xf)                              # R = X X^dag / Tr: positive by construction
    assert np.linalg.eigvalsh(R)[0] > -1e-14
    big = Space.of(A=3, B=4)
    VA, VB = random_unitary(3, rng)[:, :2], random_unitary(4, rng)[:, :2]
    V = kron(VA, VB)
    R2 = V @ R @ dag(V)
    assert np.isclose(mutual_information(R2, big, ["A"], ["B"]), mutual_information(R, sp, ["A"], ["B"]), atol=1e-10)


def test_coarse_isometry_alone_loses_trace_and_the_remainder_flag_restores_it(rng):
    V = random_unitary(5, rng)[:, :2]
    P = V @ dag(V)
    C = coarse_channel_with_remainder(V)
    assert C.is_trace_preserving() and is_completely_positive_superop(C.superop(), 5, 3)
    for _ in range(5):
        R = random_density(5, rng)
        bare = dag(V) @ R @ V
        assert np.isclose(np.trace(bare), np.trace(P @ R))
        out = C(R)
        assert np.isclose(np.trace(out), 1.0)
        assert np.isclose(out[2, 2], np.trace((np.eye(5) - P) @ R))
        assert np.allclose(out[:2, :2], bare)


def test_coarse_channel_removes_cross_sector_coherence(rng):
    V = np.eye(3, dtype=complex)[:, :2]
    a, b = random_pure_state(2, rng), 1.0
    psi_plus = np.concatenate([a / np.sqrt(2), [b / np.sqrt(2)]])
    psi_minus = np.concatenate([a / np.sqrt(2), [-b / np.sqrt(2)]])
    C = coarse_channel_with_remainder(V)
    out = [C(np.outer(p, p.conj())) for p in (psi_plus, psi_minus)]
    assert np.allclose(out[0], out[1])               # coherence between sectors is not retained


def test_a_mixture_is_not_evidence_of_cross_layer_coherence(rng):
    """Two layer purifications with the same marginal mixture differ in layer interference."""
    p = 0.3
    psi0, psi1 = random_pure_state(2, rng), random_pure_state(2, rng)
    mixture = p * np.outer(psi0, psi0.conj()) + (1 - p) * np.outer(psi1, psi1.conj())
    sp = Space.of(S=2, L=2)
    xs = []
    for phase in (0.0, np.pi / 2, np.pi):
        Psi = np.sqrt(p) * kron(psi0, [1, 0]) + np.sqrt(1 - p) * np.exp(1j * phase) * kron(psi1, [0, 1])
        R = np.outer(Psi, Psi.conj())
        assert np.allclose(sp.reduce(R, "S"), mixture)
        xs.append(np.real(expectation(R, sp.embed(X, "L"))))
    assert not np.isclose(xs[0], xs[2])             # the coherent construction carries a phase the mixture lacks


# -- T13.2 hard constraints ------------------------------------------------------------------

def supported(P, A):
    return P @ A @ P


def test_support_invariant_dynamics_never_populates_the_forbidden_complement(rng):
    d = 5
    P = np.diag([1, 1, 1, 0, 0]).astype(complex)
    R = supported(P, random_density(d, rng))
    R /= np.trace(R)
    H = supported(P, random_hermitian(d, rng))
    jumps = [supported(P, rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) * 0.3 for _ in range(2)]
    assert constraint_report(P, R, H, jumps).persistent
    L = liouvillian(H, jumps)
    for t in (0.1, 1.0, 10.0):
        assert abs(forbidden_population(P, evolve(R, L, t))) < 1e-12


def test_a_zero_eigenvalue_alone_is_not_a_constraint(rng):
    d = 4
    P = np.diag([1, 1, 1, 0]).astype(complex)
    R = supported(P, random_density(d, rng))
    R /= np.trace(R)
    H = random_hermitian(d, rng)                      # not supported on P
    rep = constraint_report(P, R, H)
    assert rep.state_supported < 1e-12 and not rep.persistent
    pops = [forbidden_population(P, evolve(R, liouvillian(H), t)) for t in (0.2, 0.5, 1.0)]
    assert max(pops) > 1e-3
    assert abs(leakage_rate(P, R, H)) < 1e-12        # first derivative vanishes at a zero population ...
    assert pops[0] > 0                                # ... yet the population grows at second order


# -- T13.3 locality --------------------------------------------------------------------------

def brick(n, gates, parity):
    U = np.eye(2 ** n, dtype=complex)
    for j in range(parity, n - 1, 2):
        U = kron(np.eye(2 ** j), gates[j], np.eye(2 ** (n - j - 2))) @ U
    return U


def test_disjoint_updates_commute_and_overlapping_ones_do_not(rng):
    n = 6
    gates = [random_unitary(4, rng) for _ in range(n - 1)]
    ops = {j: kron(np.eye(2 ** j), gates[j], np.eye(2 ** (n - j - 2))) for j in range(n - 1)}
    psi = random_pure_state(2 ** n, rng)
    even = [0, 2, 4]
    results = []
    for order in itertools.permutations(even):
        v = psi.copy()
        for j in order:
            v = ops[j] @ v
        results.append(v)
    assert all(np.allclose(r, results[0], atol=1e-13) for r in results)
    assert not np.allclose(ops[0] @ ops[1] @ psi, ops[1] @ ops[0] @ psi)


def test_an_input_changes_distant_observables_only_inside_the_cone(rng):
    n, layers = 8, 2
    gates = [random_unitary(4, rng) for _ in range(n - 1)]
    sp = Space.qubits(*[f"q{j}" for j in range(n)])
    psi = random_pure_state(2 ** n, rng)
    kick = sp.embed(unitary_from_hamiltonian(X, 0.9), "q0")       # the input touches site 0 only
    def run(v):
        for _ in range(layers):
            v = brick(n, gates, 1) @ (brick(n, gates, 0) @ v)
        return np.outer(v, v.conj())
    R0, R1 = run(psi), run(kick @ psi)
    reach = 2 * layers                                # each brickwork layer moves the cone by at most one site
    for j in range(n):
        same = np.allclose(sp.reduce(R0, f"q{j}"), sp.reduce(R1, f"q{j}"), atol=1e-13)
        if j > reach:
            assert same
    assert not np.allclose(sp.reduce(R0, "q0"), sp.reduce(R1, "q0"))
    assert not np.allclose(R0, R1)                    # the global state changed immediately


# -- T13.4 cost of execution -----------------------------------------------------------------

def test_trotter_local_error_is_second_order_in_the_commutator(rng):
    H1, H2 = random_hermitian(4, rng), random_hermitian(4, rng)
    comm = np.linalg.norm(H1 @ H2 - H2 @ H1, 2)
    ratios = []
    for h in (1e-1, 5e-2, 2.5e-2, 1.25e-2):
        exact = unitary_from_hamiltonian(H1 + H2, h)
        split = unitary_from_hamiltonian(H1, h) @ unitary_from_hamiltonian(H2, h)
        err = np.linalg.norm(exact - split, 2)
        assert err <= 0.5 * h * h * comm * (1 + 1e-9) + 1e-15
        ratios.append(err / h ** 2)
    assert np.isclose(ratios[-1], ratios[-2], rtol=0.05)          # err ~ C h^2
    D = np.diag(rng.standard_normal(4)).astype(complex)
    D2 = np.diag(rng.standard_normal(4)).astype(complex)
    assert np.allclose(unitary_from_hamiltonian(D + D2, 0.3),
                       unitary_from_hamiltonian(D, 0.3) @ unitary_from_hamiltonian(D2, 0.3))


def test_strong_cancellation_needs_high_precision(rng):
    n = 200
    a = np.full(n, 1.0 / n)
    S = 2 * np.pi * np.arange(n) / n                   # perfect cancellation: A = 0
    delta = 1e-6
    A = coherent_sum(a, S)
    A_tilde = coherent_sum(a, S + delta * rng.uniform(-1, 1, n))
    assert abs(A) < 1e-13
    assert abs(A_tilde - A) <= phase_error_bound(a, delta)
    assert abs(A_tilde - A) / max(abs(A), 1e-300) > 1e3              # relative error is enormous


def test_exact_recovery_needs_as_many_bits_as_inputs():
    """Every map from 3 bits to 2 bits merges two inputs (exhaustive over all 4^8 maps)."""
    codes = np.array(list(itertools.product(range(4), repeat=8)), dtype=np.int8)   # row = images of the 8 inputs
    distinct = (np.sort(codes, axis=1)[:, 1:] != np.sort(codes, axis=1)[:, :-1]).sum(axis=1) + 1
    n_inputs, n_codes = 2 ** 3, 2 ** 2
    assert distinct.max() == n_codes < n_inputs       # at most 2^B recoverable inputs out of 2^n


def test_dense_storage_counts():
    for D in (2, 16, 1024):
        assert np.zeros((D, D), dtype=complex).size == D * D
