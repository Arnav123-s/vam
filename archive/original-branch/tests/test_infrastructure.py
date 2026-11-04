"""Linear algebra, spaces and channel representations (used by every section)."""

import numpy as np
import pytest

from vam.channels import (
    ChannelError,
    Instrument,
    KrausMap,
    choi_to_kraus,
    collision_channel,
    map_from_function,
    superop_to_choi,
)
from vam.density import DensityError, ZeroProbabilityError, check_density, condition, pure
from vam.linalg import (
    I2,
    dag,
    is_unitary,
    kron,
    partial_trace,
    permute_operator,
    purity,
    random_density,
    random_hermitian,
    random_unitary,
    trace_distance,
    unitary_from_hamiltonian,
    von_neumann_entropy,
)
from vam.space import Space


def test_partial_trace_of_product(rng):
    A, B, C = random_density(2, rng), random_density(3, rng), random_density(2, rng)
    R = kron(A, B, C)
    assert np.allclose(partial_trace(R, [2, 3, 2], [0]), A)
    assert np.allclose(partial_trace(R, [2, 3, 2], [1]), B)
    assert np.allclose(partial_trace(R, [2, 3, 2], [0, 2]), kron(A, C))
    assert np.isclose(np.trace(partial_trace(R, [2, 3, 2], [])), 1)


def test_space_embed_matches_kron_and_permutation(rng):
    sp = Space.of(V=2, W=3, B=2)
    Aw = random_hermitian(3, rng)
    assert np.allclose(sp.embed(Aw, "W"), kron(I2, Aw, I2))
    Avb = random_hermitian(4, rng)  # acts on (V, B) in that order
    full = sp.embed(Avb, ["V", "B"])
    direct = permute_operator(kron(Avb, np.eye(3)), [2, 2, 3], [0, 2, 1])
    assert np.allclose(full, direct)
    # embedding on (B, V) is the swapped operator
    SWAP = np.eye(4)[[0, 2, 1, 3]]
    assert np.allclose(sp.embed(SWAP @ Avb @ SWAP, ["B", "V"]), full)


def test_space_reduce_round_trip(rng):
    sp = Space.of(V=2, W=2, E=3)
    loc = {"V": random_density(2, rng), "W": random_density(2, rng), "E": random_density(3, rng)}
    R = sp.product_density(loc)
    assert np.allclose(sp.reduce(R, "W"), loc["W"])
    assert np.allclose(sp.reduce(R, ["E", "V"]), kron(loc["V"], loc["E"]))


def test_density_checks():
    with pytest.raises(DensityError):
        check_density(np.diag([1.5, -0.5]))
    with pytest.raises(DensityError):
        check_density(np.array([[0.5, 1], [0, 0.5]]))
    assert np.isclose(purity(np.eye(4) / 4), 0.25)
    assert np.isclose(purity(pure(np.array([1, 1j]) / np.sqrt(2))), 1.0)


def test_purity_bounds(rng):
    for d in (2, 3, 5):
        R = random_density(d, rng)
        assert 1 / d - 1e-12 <= purity(R) <= 1 + 1e-12


def test_condition_refuses_zero_probability():
    with pytest.raises(ZeroProbabilityError):
        condition(np.zeros((2, 2)))


def test_kraus_representations_agree(rng):
    d_in, d_out = 3, 2
    ks = [rng.standard_normal((d_out, d_in)) + 1j * rng.standard_normal((d_out, d_in)) for _ in range(3)]
    Phi = KrausMap(ks)
    R = random_density(d_in, rng)
    S = Phi.superop()
    out = (S @ R.reshape(-1, order="F")).reshape(d_out, d_out, order="F")
    assert np.allclose(out, Phi(R))
    assert np.allclose(superop_to_choi(S, d_in, d_out), Phi.choi())
    Psi = choi_to_kraus(Phi.choi(), d_in, d_out)
    assert np.allclose(Psi(R), Phi(R))
    assert np.allclose(map_from_function(Phi, d_in, d_out), S)
    # dual is the adjoint for the Hilbert-Schmidt inner product
    A = random_hermitian(d_out, rng)
    assert np.isclose(np.trace(A @ Phi(R)), np.trace(Phi.dual(A) @ R))


def test_compose_and_compress(rng):
    U = random_unitary(4, rng)
    Phi = collision_channel(U, random_density(2, rng), 2)
    Psi = collision_channel(random_unitary(4, rng), random_density(2, rng), 2)
    R = random_density(2, rng)
    C = Psi.compose(Phi)
    assert np.allclose(C(R), Psi(Phi(R)))
    assert len(C.kraus) <= 4
    assert C.is_trace_preserving()


def test_collision_channel_is_cptp_and_matches_partial_trace(rng):
    for d_sys, d_env in ((2, 2), (3, 2), (2, 3)):
        U = random_unitary(d_sys * d_env, rng)
        sig = random_density(d_env, rng)
        Phi = collision_channel(U, sig, d_sys)
        assert Phi.is_trace_preserving()
        R = random_density(d_sys, rng)
        full = U @ kron(R, sig) @ dag(U)
        assert np.allclose(Phi(R), partial_trace(full, [d_sys, d_env], [0]))


def test_instrument_requires_completeness(rng):
    P0 = np.diag([1, 0]).astype(complex)
    with pytest.raises(ChannelError):
        Instrument({0: KrausMap([P0])})
    I = Instrument({0: KrausMap([P0]), 1: KrausMap([np.diag([0, 1]).astype(complex)])})
    R = random_density(2, rng)
    p = I.probabilities(R)
    assert np.isclose(sum(p.values()), 1)


def test_trace_distance_and_entropy():
    assert np.isclose(trace_distance(np.diag([1, 0]), np.diag([0, 1])), 1)
    assert np.isclose(von_neumann_entropy(np.eye(4) / 4), 2)
    assert np.isclose(von_neumann_entropy(np.diag([1, 0])), 0)


def test_unitary_from_hamiltonian_is_unitary_for_long_times(rng):
    H = random_hermitian(6, rng)
    assert is_unitary(unitary_from_hamiltonian(H, 1e4), atol=1e-9)
