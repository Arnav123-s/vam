"""T6 and T13.3: Dirac generators inside the open-system setting; einselection; locks; light cones."""

import numpy as np
import pytest

from vam.channels import collision_channel, superop_to_choi
from vam.dirac import (
    chain_operator,
    checkerboard_propagator,
    clifford_defect,
    dirac_matrices,
    hamiltonian_light_cone_leakage,
    light_cone_leakage,
    one_excitation_block,
    split_step_walk,
    staggered_chain_hamiltonian,
    walk_dispersion,
    wilson_dirac_hamiltonian,
    wilson_dispersion,
)
from vam.linalg import X, Z, is_psd, is_unitary, random_density, random_hermitian, unitary_from_hamiltonian, unvec, vec
from vam.lindblad import (
    dephasing_closed_form,
    dephasing_jump,
    evolve,
    liouvillian,
    sector_dissipator,
    sector_dissipator_closed_form,
    sector_projection_superop,
    semigroup,
    spectral_lock,
)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_clifford_relations(d):
    a, b = dirac_matrices(d)
    assert clifford_defect(a, b) == 0.0


def test_wilson_spectrum_matches_dispersion():
    N, a, c, m, r = 24, 0.5, 1.3, 0.4, 1.0
    H = wilson_dirac_hamiltonian(N, a, c, m, r=r)
    assert np.allclose(H, H.conj().T)
    k = 2 * np.pi * np.arange(N) / (N * a)
    E = wilson_dispersion(k, a, c, m, r)
    expected = np.sort(np.concatenate([E, -E]))
    assert np.allclose(np.sort(np.linalg.eigvalsh(H)), expected, atol=1e-12)
    # the doubler at k = pi/a is lifted to m c^2 + 2 r hbar c / a
    assert np.isclose(wilson_dispersion(np.pi / a, a, c, m, r), m * c ** 2 + 2 * r * c / a)
    # without the Wilson term the doubler is massless-like (energy m c^2 at k = pi/a)
    assert np.isclose(wilson_dispersion(np.pi / a, a, c, m, r=0.0), m * c ** 2)


def test_staggered_chain_is_number_conserving_and_maps_to_hopping():
    n, hop, mass = 6, 0.7, 0.3
    H = staggered_chain_hamiltonian(n, hop, mass)
    Ntot = chain_operator(n, [((j,), (np.eye(2) - Z) / 2) for j in range(n)])
    assert np.allclose(H @ Ntot, Ntot @ H)
    H1 = one_excitation_block(H, n)
    c0 = 0.5 * mass * sum((-1) ** j for j in range(n))
    expected = np.diag([c0 - mass * (-1) ** j for j in range(n)]).astype(complex)
    for j in range(n - 1):
        expected[j, j + 1] = expected[j + 1, j] = hop
    assert np.allclose(H1, expected)


def test_staggered_ring_dispersion_is_lattice_dirac():
    """Single-particle ring with the same hopping and staggered mass: E = +-sqrt(m^2 + 4 hop^2 cos^2 q)."""
    n, hop, mass = 16, 0.7, 0.3
    H1 = np.diag([-mass * (-1) ** j for j in range(n)]).astype(complex)
    for j in range(n):
        H1[j, (j + 1) % n] = H1[(j + 1) % n, j] = hop
    q = 2 * np.pi * np.arange(n // 2) / n
    E = np.sqrt(mass ** 2 + 4 * hop ** 2 * np.cos(q) ** 2)
    assert np.allclose(np.sort(np.linalg.eigvalsh(H1)), np.sort(np.concatenate([E, -E])))


def test_split_step_walk_unitary_dispersion_and_exact_cone():
    N, theta = 40, 0.3
    W = split_step_walk(N, theta)
    assert is_unitary(W)
    k = 2 * np.pi * np.arange(N) / N
    omega = walk_dispersion(k, theta)
    phases = np.sort(np.abs(np.angle(np.linalg.eigvals(W))))
    assert np.allclose(phases, np.sort(np.concatenate([omega, omega])), atol=1e-10)
    for steps in (1, 5, 12):
        assert light_cone_leakage(W, N, 2, 20, steps) == 0.0
    # small-k, small-theta limit: omega^2 ~ k^2 + theta^2 (Dirac, c = 1, rest energy theta)
    kk, th = 0.01, 0.02
    assert np.isclose(walk_dispersion(kk, th) ** 2, kk ** 2 + th ** 2, rtol=1e-3)


def test_continuous_time_lattice_hamiltonian_has_tails_beyond_the_cone():
    N = 40
    H = wilson_dirac_hamiltonian(N, a=1.0, c=1.0, m=0.3)
    assert hamiltonian_light_cone_leakage(H, N, 2, 20, t=5.0, cone=5) > 1e-6


@pytest.mark.parametrize("steps", [1, 4, 9])
def test_checkerboard_path_sum_equals_walk(steps):
    """Feynman's checkerboard: the sum over 2^steps chirality bit strings is the walk propagator."""
    N, theta, x0 = 2 * steps + 5, 0.37, steps + 2
    W = split_step_walk(N, theta, periodic=False)
    for chi0 in (0, 1):
        psi = np.zeros(2 * N, dtype=complex)
        psi[2 * x0 + chi0] = 1
        for _ in range(steps):
            psi = W @ psi
        paths = checkerboard_propagator(steps, theta, x0, chi0)
        dense = {(x, c): psi[2 * x + c] for x in range(N) for c in (0, 1) if abs(psi[2 * x + c]) > 1e-15}
        assert set(dense) <= set(paths)
        for key, amp in paths.items():
            assert np.isclose(amp, dense.get(key, 0.0), atol=1e-13)


def test_dirac_collision_channel_is_cptp(rng):
    """T6.2: a Dirac system plus a boundary-coupled environment unit gives a CPTP current-state channel."""
    Hs = wilson_dirac_hamiltonian(3, m=0.5, periodic=False)          # 6-dim system
    He = random_hermitian(2, rng)
    boundary = np.diag([0.0, 0.0, 1.0])                               # site 2 is the boundary
    Hint = 0.4 * np.kron(np.kron(boundary, Z), X)
    Htot = np.kron(Hs, np.eye(2)) + np.kron(np.eye(6), He) + Hint
    U = unitary_from_hamiltonian(Htot, 0.8)
    Phi = collision_channel(U, random_density(2, rng), 6)
    assert Phi.is_trace_preserving()
    assert is_psd(Phi.choi(), 1e-10)


def test_lindblad_semigroup_is_cptp(rng):
    d = 3
    H = random_hermitian(d, rng)
    Ls = [rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)) for _ in range(2)]
    S = semigroup(liouvillian(H, Ls), 0.7)
    assert is_psd(superop_to_choi(S, d, d), 1e-9)
    R = random_density(d, rng)
    assert np.isclose(np.trace(unvec(S @ vec(R), d)), 1)


def test_dephasing_einselection_closed_form(rng):
    """T6.4: R_aa'(t) = exp(-gamma (a-a')^2 t / 2) R_aa'(0); halving at gamma t = (log 2)/2 for a - a' = 2."""
    a = np.array([-1.0, 1.0, 0.5])
    A = np.diag(a).astype(complex)
    gamma, t = 0.9, 1.3
    R = random_density(3, rng)
    L = liouvillian(None, [dephasing_jump(A, gamma)], dim=3)
    assert np.allclose(evolve(R, L, t), dephasing_closed_form(R, a, gamma, t))
    t_half = np.log(2) / 2 / gamma
    out = evolve(R, L, t_half)
    assert np.isclose(out[0, 1], R[0, 1] / 2)
    assert np.allclose(np.diag(out), np.diag(R))
    # with a commuting Hamiltonian the same formula holds in the interaction picture
    H = np.diag(rng.standard_normal(3)).astype(complex)
    U = unitary_from_hamiltonian(H, t)
    L2 = liouvillian(H, [dephasing_jump(A, gamma)])
    assert np.allclose(U.conj().T @ evolve(R, L2, t) @ U, dephasing_closed_form(R, a, gamma, t))


def test_sector_dissipator_keeps_interior_coherence(rng):
    """T7.3: e^{tD_B} R = Pi_B(R) + e^{-gamma t}(R - Pi_B(R)); within-sector coherence survives."""
    P = [np.diag([1, 1, 0, 0]).astype(complex), np.diag([0, 0, 1, 1]).astype(complex)]
    gamma, t = 1.7, 0.6
    R = random_density(4, rng)
    out = evolve(R, sector_dissipator(P, gamma), t)
    assert np.allclose(out, sector_dissipator_closed_form(R, P, gamma, t))
    assert np.isclose(out[0, 1], R[0, 1]) and np.isclose(out[2, 3], R[2, 3])
    assert np.isclose(out[0, 2], np.exp(-gamma * t) * R[0, 2])
    # it is a Lindblad generator with jumps sqrt(gamma) P_r
    assert np.allclose(sector_dissipator(P, gamma), liouvillian(None, [np.sqrt(gamma) * p for p in P], dim=4))


def test_locks_from_dynamics():
    """T11.1: Pi = lim exp(tL) exists under the spectral condition; oscillating modes are not a lock."""
    P = [np.diag([1, 1, 0, 0]).astype(complex), np.diag([0, 0, 1, 1]).astype(complex)]
    Pi, rep = spectral_lock(sector_dissipator(P, 2.0), 4)
    assert rep.settled and np.isclose(rep.gap, 2.0)
    assert np.allclose(Pi, sector_projection_superop(P))
    assert np.allclose(Pi, semigroup(sector_dissipator(P, 2.0), 40.0), atol=1e-12)
    H = np.diag([0.0, 1.0]).astype(complex)
    Pi2, rep2 = spectral_lock(liouvillian(H), 2)
    assert Pi2 is None and not rep2.settled and rep2.oscillating_modes == 2
