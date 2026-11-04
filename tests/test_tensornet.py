"""Acceptance tests for vam.tensornet (T13.1, T13.4, F-Part 9).

Conventions under test
----------------------
* Site 0 is the leftmost (most significant) tensor factor:
  ``MPS.product([v0, v1, ...]).to_dense() == kron(v0, v1, ...)``.
* MPS site tensors have shape ``(D_left, d, D_right)``; LPDO site tensors
  have shape ``(D_left, d, k, D_right)`` and represent ``rho = Tr_K |X><X|``,
  so every LPDO is positive by construction.
* ``apply_two`` truncates in the canonical gauge and returns the discarded
  weight of that truncation (fraction of the squared norm removed); the
  state is renormalized. ``.discarded`` accumulates these fractions.
"""

import numpy as np
import pytest

from vam.channels import KrausMap, random_channel
from vam.dirac import staggered_chain_hamiltonian, staggered_chain_terms
from vam.linalg import (
    P0,
    P1,
    X,
    Y,
    Z,
    kron,
    random_density,
    random_pure_state,
    random_unitary,
    unitary_from_hamiltonian,
    von_neumann_entropy,
)
from vam.space import Space
from vam.tensornet import LPDO, MPS, diagonal_mpo


def embed(op, site, dims):
    """Dense operator acting on consecutive sites starting at ``site``."""
    d_op = round(np.sqrt(op.size))
    left = int(np.prod(dims[:site]))
    width, acc = 0, 1
    while acc < d_op:
        acc *= dims[site + width]
        width += 1
    right = int(np.prod(dims[site + width:]))
    return kron(np.eye(left), op, np.eye(right))


def dense_space(dims):
    return Space(tuple(f"s{j}" for j in range(len(dims))), tuple(dims))


# -- MPS -------------------------------------------------------------------------------------

@pytest.mark.parametrize("dims", [(2, 2, 2, 2, 2), (2, 3, 2, 4), (3,), (2, 2)])
def test_round_trip_is_exact(rng, dims):
    psi = random_pure_state(int(np.prod(dims)), rng)
    m = MPS.from_dense(psi, dims)
    assert m.dims == tuple(dims)
    assert np.allclose(m.to_dense(), psi, atol=1e-13)
    for b, D in enumerate(m.bond_dims):
        assert D <= min(np.prod(dims[:b + 1]), np.prod(dims[b + 1:]))


def test_product_ordering(rng):
    vs = [random_pure_state(d, rng) for d in (2, 3, 2)]
    m = MPS.product(vs)
    assert np.allclose(m.to_dense(), kron(*vs))
    assert m.bond_dims == (1, 1)


def test_local_gates_match_dense(rng):
    dims = (2, 3, 2, 2, 2)
    psi = random_pure_state(int(np.prod(dims)), rng)
    m = MPS.from_dense(psi, dims)
    for _ in range(12):
        s = int(rng.integers(len(dims)))
        U = random_unitary(dims[s], rng)
        m.apply_one(U, s)
        psi = embed(U, s, dims) @ psi
        s = int(rng.integers(len(dims) - 1))
        G = random_unitary(dims[s] * dims[s + 1], rng)
        disc = m.apply_two(G, s)
        psi = embed(G, s, dims) @ psi
        assert disc < 1e-14
    assert np.allclose(m.to_dense(), psi, atol=1e-11)
    assert np.isclose(m.norm(), 1.0)


def test_non_unitary_local_operator_and_norm(rng):
    dims = (2, 2, 2)
    psi = random_pure_state(8, rng)
    m = MPS.from_dense(psi, dims)
    m.apply_one(P1, 1)
    target = embed(P1, 1, dims) @ psi
    assert np.isclose(m.norm(), np.linalg.norm(target))
    assert np.allclose(m.to_dense(), target, atol=1e-13)


def test_truncation_is_optimal_and_accounted(rng):
    n = 8
    dims = (2,) * n
    psi = random_pure_state(2 ** n, rng)
    m = MPS.from_dense(psi, dims)
    G = random_unitary(4, rng)
    exact = embed(G, 3, dims) @ psi
    disc = m.apply_two(G, 3, max_bond=4)
    assert 0 < disc < 1
    assert m.bond_dims[3] == 4
    overlap = abs(np.vdot(exact, m.to_dense())) ** 2
    assert np.isclose(overlap, 1 - disc, atol=1e-12)
    assert np.isclose(m.norm(), 1.0) and np.isclose(m.discarded, disc)


def test_cutoff_drops_only_negligible_weight(rng):
    vs = [random_pure_state(2, rng) for _ in range(6)]
    m = MPS.product(vs)
    m.apply_two(np.eye(4), 2, cutoff=1e-12)
    assert m.bond_dims[2] == 1


def test_expectations_and_reduced_densities(rng):
    dims = (2, 2, 3, 2)
    psi = random_pure_state(24, rng)
    m = MPS.from_dense(psi, dims)
    sp = dense_space(dims)
    R = np.outer(psi, psi.conj())
    ops = {0: Z, 2: np.diag([1.0, -1.0, 0.5]), 3: X}
    full = kron(Z, np.eye(2), np.diag([1.0, -1.0, 0.5]), X)
    assert np.isclose(m.expectation(ops), np.vdot(psi, full @ psi))
    for sites in ([1], [0, 1], [1, 3], [0, 2, 3]):
        assert np.allclose(m.reduced_density(sites), sp.reduce(R, [f"s{j}" for j in sites]), atol=1e-12)


def test_entanglement_entropy_matches_schmidt(rng):
    dims = (2, 2, 2, 2, 2, 2)
    psi = random_pure_state(64, rng)
    m = MPS.from_dense(psi, dims)
    sp = dense_space(dims)
    R = np.outer(psi, psi.conj())
    for b in range(5):
        left = sp.reduce(R, [f"s{j}" for j in range(b + 1)])
        assert np.isclose(m.entanglement_entropy(b), von_neumann_entropy(left), atol=1e-10)


def test_canonical_gauge_does_not_change_the_state(rng):
    dims = (2, 2, 2, 2)
    psi = random_pure_state(16, rng)
    m = MPS.from_dense(psi, dims)
    for c in (0, 3, 1):
        m.canonicalize(c)
        assert np.allclose(m.to_dense(), psi, atol=1e-13)
        for j in range(c):                                 # left-orthonormal to the left of the centre
            A = m.tensors[j].reshape(-1, m.tensors[j].shape[2])
            assert np.allclose(A.conj().T @ A, np.eye(A.shape[1]), atol=1e-12)
        for j in range(c + 1, len(dims)):                  # right-orthonormal to the right
            B = m.tensors[j].reshape(m.tensors[j].shape[0], -1)
            assert np.allclose(B @ B.conj().T, np.eye(B.shape[0]), atol=1e-12)


def test_storage_count_is_the_t13_4_formula(rng):
    n, d = 10, 2
    m = MPS.from_dense(random_pure_state(2 ** n, rng), (d,) * n, max_bond=8)
    chi = max(m.bond_dims)
    assert chi == 8
    assert m.num_entries() == sum(t.size for t in m.tensors)
    assert m.num_entries() <= n * d * chi ** 2


def brickwork_gates(n, dt, hop, mass):
    """First-order brickwork of the staggered Dirac chain: even bonds, odd bonds, then the mass layer."""
    hopping = 0.5 * hop * (np.kron(X, X) + np.kron(Y, Y))
    G = unitary_from_hamiltonian(hopping, dt)
    masses = [unitary_from_hamiltonian(0.5 * mass * (-1) ** j * Z, dt) for j in range(n)]
    return G, masses


def brickwork_step_dense(psi, n, G, masses):
    dims = (2,) * n
    for parity in (0, 1):
        for j in range(parity, n - 1, 2):
            psi = embed(G, j, dims) @ psi
    for j in range(n):
        psi = embed(masses[j], j, dims) @ psi
    return psi


def brickwork_step_mps(m, n, G, masses, max_bond=None):
    for parity in (0, 1):
        for j in range(parity, n - 1, 2):
            m.apply_two(G, j, max_bond=max_bond)
    for j in range(n):
        m.apply_one(masses[j], j)


def test_staggered_dirac_chain_evolution_matches_dense(rng):
    n = 10
    G, masses = brickwork_gates(n, 0.1, 1.0, 0.5)
    bits = rng.integers(0, 2, n)
    vs = [np.eye(2)[b] for b in bits]
    psi = kron(*vs).astype(complex)
    m = MPS.product([v.astype(complex) for v in vs])
    for _ in range(15):
        psi = brickwork_step_dense(psi, n, G, masses)
        brickwork_step_mps(m, n, G, masses)
    assert np.allclose(m.to_dense(), psi, atol=1e-10)
    # the brickwork is generated by the chain's own local terms
    H = staggered_chain_hamiltonian(n, 1.0, 0.5)
    assert len(staggered_chain_terms(n, 1.0, 0.5)) == 2 * n - 1 and H.shape == (2 ** n, 2 ** n)


def test_exact_event_cone_at_scale():
    """60 qubits: an input at site 0 leaves every site beyond the brickwork cone exactly unchanged.

    Exact when nothing inside the cone is truncated: three brickwork steps from a product state put at most
    three gates across any cut, so a bond of 4^3 = 64 is lossless.
    """
    n, layers = 60, 3
    G, masses = brickwork_gates(n, 0.3, 1.0, 0.5)
    base = [np.array([1, 0], dtype=complex) if j % 2 else np.array([0, 1], dtype=complex) for j in range(n)]
    a, b = MPS.product(base), MPS.product(base)
    b.apply_one(unitary_from_hamiltonian(X, 0.7), 0)
    for _ in range(layers):
        brickwork_step_mps(a, n, G, masses, max_bond=64)
        brickwork_step_mps(b, n, G, masses, max_bond=64)
    assert a.discarded < 1e-20 and b.discarded < 1e-20
    reach = 2 * layers
    for j in range(reach + 1, n):
        assert np.allclose(a.reduced_density([j]), b.reduced_density([j]), atol=1e-13)
    assert not np.allclose(a.reduced_density([0]), b.reduced_density([0]), atol=1e-6)
    assert np.isclose(a.norm(), 1.0) and np.isclose(b.norm(), 1.0)


def test_truncation_breaks_the_cone_only_by_the_discarded_weight():
    """With a bond cap the far marginals can move, but by no more than the truncation error allows.

    Each truncation with discarded fraction e_i moves the state by trace distance sqrt(e_i); by the triangle
    inequality and Cauchy-Schwarz the total is at most sqrt(N sum e_i) for N truncations.
    """
    n, layers, cap = 40, 6, 8
    G, masses = brickwork_gates(n, 0.3, 1.0, 0.5)
    base = [np.array([1, 0], dtype=complex) if j % 2 else np.array([0, 1], dtype=complex) for j in range(n)]
    a, b = MPS.product(base), MPS.product(base)
    b.apply_one(unitary_from_hamiltonian(X, 0.7), 0)
    for _ in range(layers):
        brickwork_step_mps(a, n, G, masses, max_bond=cap)
        brickwork_step_mps(b, n, G, masses, max_bond=cap)
    assert a.discarded > 0
    N = layers * (n - 1)
    bound = np.sqrt(N * a.discarded) + np.sqrt(N * b.discarded)
    for j in range(2 * layers + 1, n):
        diff = 0.5 * np.abs(np.linalg.eigvalsh(a.reduced_density([j]) - b.reduced_density([j]))).sum()
        assert diff <= bound


# -- LPDO ------------------------------------------------------------------------------------

def test_lpdo_product_and_from_mps(rng):
    rhos = [random_density(d, rng) for d in (2, 3, 2)]
    lp = LPDO.product(rhos)
    assert np.allclose(lp.to_dense(), kron(*rhos), atol=1e-13)
    assert np.isclose(lp.trace(), 1.0)
    psi = random_pure_state(12, rng)
    lp2 = LPDO.from_mps(MPS.from_dense(psi, (2, 3, 2)))
    assert np.allclose(lp2.to_dense(), np.outer(psi, psi.conj()), atol=1e-13)


def test_lpdo_channels_and_gates_match_dense(rng):
    dims = (2, 2, 2, 2, 2)
    rhos = [random_density(2, rng) for _ in dims]
    lp = LPDO.product(rhos)
    R = kron(*rhos)
    for step in range(10):
        s = int(rng.integers(len(dims)))
        ch = random_channel(2, rng, n_kraus=2)
        lp.apply_channel(ch, s)
        R = KrausMap([embed(K, s, dims) for K in ch.kraus])(R)
        s = int(rng.integers(len(dims) - 1))
        G = random_unitary(4, rng)
        lp.apply_two(G, s)
        R = embed(G, s, dims) @ R @ embed(G, s, dims).conj().T
        if step % 3 == 0:
            ch2 = random_channel(4, rng, n_kraus=3)
            lp.apply_channel_two(ch2, s)
            R = KrausMap([embed(K, s, dims) for K in ch2.kraus])(R)
    out = lp.to_dense()
    assert np.allclose(out, R, atol=1e-10)
    assert np.isclose(lp.trace(), 1.0)
    assert np.linalg.eigvalsh(out)[0] > -1e-12


def test_lpdo_accepts_kraus_lists(rng):
    rhos = [random_density(2, rng) for _ in range(3)]
    lp = LPDO.product(rhos)
    gamma = 0.3
    amp_damp = [np.array([[1, 0], [0, np.sqrt(1 - gamma)]]), np.array([[0, np.sqrt(gamma)], [0, 0]])]
    lp.apply_channel(amp_damp, 1)
    target = kron(rhos[0], KrausMap(amp_damp)(rhos[1]), rhos[2])
    assert np.allclose(lp.to_dense(), target, atol=1e-13)


def test_lpdo_conditioning_on_an_outcome(rng):
    dims = (2, 2, 2)
    psi = random_pure_state(8, rng)
    lp = LPDO.from_mps(MPS.from_dense(psi, dims))
    lp.apply_channel(random_channel(2, rng), 0)
    R = lp.to_dense()
    p = lp.condition([P1], 2)
    M = embed(P1, 2, dims)
    assert np.isclose(p, np.real(np.trace(M @ R)))
    assert np.allclose(lp.to_dense(), M @ R @ M / p, atol=1e-12)
    assert np.isclose(lp.trace(), 1.0)


def test_lpdo_kraus_dimension_is_compressed_exactly(rng):
    """Repeated channels on one qubit never need more than d * D_left * D_right Kraus vectors."""
    lp = LPDO.product([random_density(2, rng) for _ in range(3)])
    R = lp.to_dense()
    for _ in range(6):
        ch = random_channel(2, rng, n_kraus=4)
        lp.apply_channel(ch, 1)
        R = KrausMap([embed(K, 1, (2, 2, 2)) for K in ch.kraus])(R)
    assert np.allclose(lp.to_dense(), R, atol=1e-12)
    t = lp.tensors[1]
    assert t.shape[2] <= t.shape[0] * t.shape[1] * t.shape[3]


def test_lpdo_truncation_keeps_positivity_and_trace(rng):
    n = 8
    lp = LPDO.product([random_density(2, rng) for _ in range(n)])
    for _ in range(4):
        for j in range(n - 1):
            lp.apply_two(random_unitary(4, rng), j, max_bond=3)
            lp.apply_channel(random_channel(2, rng), j, max_kraus=2)
    out = lp.to_dense()
    assert np.isclose(np.trace(out), 1.0)
    assert np.linalg.eigvalsh(out)[0] > -1e-12
    assert max(lp.bond_dims) <= 3 and max(t.shape[2] for t in lp.tensors) <= 2
    assert lp.discarded > 0


def test_lpdo_expectation_and_reduced_density(rng):
    dims = (2, 3, 2)
    lp = LPDO.from_mps(MPS.from_dense(random_pure_state(12, rng), dims))
    lp.apply_channel(random_channel(3, rng), 1)
    rho = lp.to_dense()
    sp = dense_space(dims)
    assert np.isclose(lp.expectation({0: Z, 2: X}), np.trace(rho @ kron(Z, np.eye(3), X)))
    for sites in ([0], [1, 2], [0, 2]):
        assert np.allclose(lp.reduced_density(sites), sp.reduce(rho, [f"s{j}" for j in sites]), atol=1e-12)


def test_lpdo_dephasing_keeps_sector_coherence_like_dense(rng):
    """The T7.3 sector projection applied by the tensor network equals the dense map."""
    dims = (2, 2, 2)
    lp = LPDO.from_mps(MPS.from_dense(random_pure_state(8, rng), dims))
    R = lp.to_dense()
    lp.apply_channel([P0, P1], 0)
    target = sum(embed(P, 0, dims) @ R @ embed(P, 0, dims) for P in (P0, P1))
    assert np.allclose(lp.to_dense(), target, atol=1e-13)


def test_lpdo_storage_count(rng):
    lp = LPDO.product([random_density(2, rng) for _ in range(5)])
    assert lp.num_entries() == sum(t.size for t in lp.tensors)


# -- additions for the VAM-1 tensor engine: MPO application, basis probabilities, swaps, purification ---

def mixed_lpdo(rng, dims, rounds=2):
    """A generic mixed LPDO: random local densities, random two-site gates and local channels."""
    lp = LPDO.product([random_density(d, rng) for d in dims])
    for _ in range(rounds):
        for j in range(len(dims) - 1):
            lp.apply_two(random_unitary(dims[j] * dims[j + 1], rng), j)
            lp.apply_channel(random_channel(dims[j], rng, n_kraus=2), j)
    return lp


def mpo_to_dense(mpo):
    M = np.ones((1, 1, 1))
    for W in mpo:
        M = np.einsum("ija,astb->isjtb", M, W).reshape(M.shape[0] * W.shape[1], M.shape[1] * W.shape[2], W.shape[3])
    return M[:, :, 0]


@pytest.mark.parametrize("dims", [(2, 2), (2, 2, 2), (2, 3, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2)])
def test_diagonal_mpo_is_exact_with_the_tensor_train_bond(rng, dims):
    values = rng.normal(size=dims) + 1j * rng.normal(size=dims)
    mpo = diagonal_mpo(values)
    assert np.allclose(mpo_to_dense(mpo), np.diag(values.reshape(-1)), atol=1e-12)
    for j, W in enumerate(mpo[:-1]):
        assert W.shape[3] <= min(np.prod(dims[:j + 1]), np.prod(dims[j + 1:]))
    if dims == (2,) * 8:
        assert max(W.shape[3] for W in mpo) == 2 ** 4                 # 2^floor(k/2) for a generic table
    assert [W.shape[3] for W in diagonal_mpo(np.ones(dims))][:-1] == [1] * (len(dims) - 1)   # constant: bond 1


@pytest.mark.parametrize("first,width", [(0, 3), (1, 4), (3, 3), (2, 1)])
def test_lpdo_mpo_application_matches_dense(rng, first, width):
    dims = (2, 2, 3, 2, 2, 2)
    lp = mixed_lpdo(rng, dims)
    R = lp.to_dense()
    mpo = []
    for i in range(width):
        d = dims[first + i]
        Dl = 1 if i == 0 else 3
        Dr = 1 if i == width - 1 else 3
        mpo.append(rng.normal(size=(Dl, d, d, Dr)) + 1j * rng.normal(size=(Dl, d, d, Dr)))
    W = embed(mpo_to_dense(mpo), first, dims)
    target = W @ R @ W.conj().T
    disc = lp.apply_mpo(mpo, first)
    assert disc < 1e-20 and lp.center == first
    assert np.allclose(lp.to_dense(), target, atol=1e-10 * np.abs(target).max())
    assert np.isclose(lp.trace(), np.trace(target).real)


@pytest.mark.parametrize("first,width", [(0, 3), (1, 4), (3, 3), (2, 1)])
def test_mps_mpo_application_matches_dense(rng, first, width):
    dims = (2, 2, 3, 2, 2, 2)
    m = MPS.from_dense(random_pure_state(int(np.prod(dims)), rng), dims)
    psi = m.to_dense()
    mpo = []
    for i in range(width):
        d = dims[first + i]
        Dl = 1 if i == 0 else 3
        Dr = 1 if i == width - 1 else 3
        mpo.append(rng.normal(size=(Dl, d, d, Dr)) + 1j * rng.normal(size=(Dl, d, d, Dr)))
    target = embed(mpo_to_dense(mpo), first, dims) @ psi
    disc = m.apply_mpo(mpo, first)
    assert disc < 1e-20 and m.center == first
    assert np.allclose(m.to_dense(), target, atol=1e-10 * np.abs(target).max())
    assert np.isclose(m.norm(), np.linalg.norm(target))


def test_mps_mpo_truncation_is_canonical_and_accounted(rng):
    """Truncated at the orthogonality centre: the discarded fraction is the removed weight, and the error of the
    normalized state is at most the square root of the summed fractions (one per bond, each orthogonal)."""
    dims = (2,) * 8
    m = MPS.from_dense(random_pure_state(2 ** 8, rng), dims)
    mpo = diagonal_mpo(rng.uniform(0.2, 1.0, size=dims))
    exact = m.copy()
    exact.apply_mpo(mpo, 0)
    disc = m.apply_mpo(mpo, 0, max_bond=4)
    assert 0 < disc < 1 and max(m.bond_dims) <= 4 and m.center == 0
    assert np.isclose(m.norm(), exact.norm())                         # the squared norm is kept
    a, b = exact.to_dense() / exact.norm(), m.to_dense() / m.norm()
    assert np.linalg.norm(a - b * np.vdot(b, a) / abs(np.vdot(b, a))) <= 2 * np.sqrt(disc)


def test_lpdo_mpo_truncation_is_canonical_positive_and_accounted(rng):
    dims = (2,) * 7
    lp = mixed_lpdo(rng, dims, rounds=2)
    mpo = diagonal_mpo(rng.uniform(0.2, 1.0, size=dims))
    exact = lp.copy()
    exact.apply_mpo(mpo, 0)
    disc = lp.apply_mpo(mpo, 0, max_bond=4)
    assert 0 < disc < 1 and max(lp.bond_dims) <= 4
    assert np.isclose(lp.trace(), exact.trace())                      # the squared norm is kept
    every = list(range(len(dims)))                                    # transfer contraction: X is never expanded
    a, b = exact.reduced_density(every), lp.reduced_density(every)
    assert np.linalg.eigvalsh(b)[0] > -1e-12                          # positive by construction
    # each truncation moves the normalized purification by at most sqrt(e_i) (see the docstring of
    # test_truncation_breaks_the_cone_only_by_the_discarded_weight); the reduced state follows
    bound = 2 * np.sqrt(len(dims) * disc)
    assert 0.5 * np.abs(np.linalg.eigvalsh(a - b)).sum() <= bound


@pytest.mark.parametrize("sites", [[0], [1, 2], [0, 3, 4], [2, 5], [0, 1, 2, 3, 4, 5]])
def test_lpdo_basis_probabilities_are_the_diagonal_of_the_reduced_density(rng, sites):
    dims = (2, 3, 2, 2, 2, 2)
    lp = mixed_lpdo(rng, dims)
    rho = lp.to_dense()
    sp = dense_space(dims)
    expected = np.real(np.diag(sp.reduce(rho, [f"s{j}" for j in sites])))
    assert np.allclose(lp.probabilities(sites), expected / expected.sum(), atol=1e-13)


def test_lpdo_swap_exchanges_physical_and_purification_legs(rng):
    dims = (2, 3, 2, 2)
    lp = mixed_lpdo(rng, dims)
    lp.apply_channel(random_channel(2, rng, n_kraus=3), 3)
    R, kraus = lp.to_dense(), lp.kraus_dims
    disc = lp.swap(2)
    S = np.zeros((4, 4))
    for a in range(2):
        for b in range(2):
            S[2 * b + a, 2 * a + b] = 1.0
    G = embed(S, 2, dims)
    assert disc < 1e-20
    assert np.allclose(lp.to_dense(), G @ R @ G.T, atol=1e-12)
    assert lp.kraus_dims[2:] == kraus[2:][::-1]                      # each Kraus leg moved with its qubit
    lp.swap(0)                                                       # unequal local dimensions travel too
    assert lp.dims == (3, 2, 2, 2)


def test_concentrated_purification_is_minimal_and_keeps_rho(rng):
    dims = (2, 2, 2, 2, 2)
    lp = mixed_lpdo(rng, dims, rounds=1)
    lp.apply_channel([P0, P1], 4)
    R = lp.to_dense()
    disc = lp.concentrate_purification()
    assert disc < 1e-20
    assert np.allclose(lp.to_dense(), R, atol=1e-12)
    assert lp.kraus_dims[:-1] == (1,) * (len(dims) - 1)
    rank = lambda M: int(np.sum(np.linalg.eigvalsh(M) > 1e-12 * np.abs(M).max()))
    assert lp.kraus_dims[-1] == rank(R)
    sp = dense_space(dims)
    for j, D in enumerate(lp.bond_dims):
        assert D == rank(sp.reduce(R, [f"s{i}" for i in range(j + 1)]))
