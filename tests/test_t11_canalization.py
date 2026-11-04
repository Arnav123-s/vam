"""T11: canalization, idempotent locks and retention (plus the algebra construction of decision D7)."""

import itertools

import numpy as np

from vam.canalization import (
    append_factor_isometry,
    idempotence_defect,
    invariant_algebra,
    isometric_nesting,
    relation_preserving_map,
    retention_defect,
)
from vam.channels import (
    KrausMap,
    is_completely_positive_superop,
    is_trace_preserving_superop,
    projective_dephasing,
    random_channel,
    random_instrument,
)
from vam.linalg import I2, P0, P1, Z, expectation, kron, random_density, random_hermitian, random_unitary
from vam.lindblad import dephasing_jump, liouvillian, semigroup, spectral_lock
from vam.machine import EventProtocol, Machine
from vam.realizations import incomplete_codes as ic
from vam.realizations import two_mode as tm
from vam.space import Space

SP = Space.of(R=2, N=3)          # relational qubit, nuisance qutrit


def assert_cptp_idempotent(S, d):
    assert is_completely_positive_superop(S, d)
    assert is_trace_preserving_superop(S, d)
    assert idempotence_defect(S) < 1e-12


# -- T11.2 relation-preserving many-to-one map -----------------------------------------------

def test_relation_preserving_map(rng):
    sigma = random_density(3, rng)
    Pi = relation_preserving_map(SP, ["N"], sigma)
    assert_cptp_idempotent(Pi.superop(), 6)
    for _ in range(5):
        rho = random_density(6, rng)
        out = Pi(rho)
        assert np.allclose(out, kron(SP.reduce(rho, "R"), sigma), atol=1e-13)
        A = SP.embed(random_hermitian(2, rng), "R")
        assert np.isclose(expectation(out, A), expectation(rho, A), atol=1e-13)


def test_many_fine_configurations_become_one(rng):
    """Different fine states with the same relational marginal give one retained state."""
    sigma = random_density(3, rng)
    Pi = relation_preserving_map(SP, ["N"], sigma)
    rho_R = random_density(2, rng)
    fines = [kron(rho_R, random_density(3, rng)) for _ in range(4)]
    # a correlated fine state with the same R marginal
    U = SP.embed(random_unitary(3, rng), "N")
    fines.append(U @ fines[0] @ U.conj().T)
    outs = [Pi(f) for f in fines]
    assert all(np.allclose(o, outs[0], atol=1e-13) for o in outs)


# -- T11.3 retention identity ---------------------------------------------------------------

def relational_events(rng):
    """Instrument on R combined with arbitrary trace-preserving dynamics of N."""
    J = random_instrument(2, (0, 1, 2), rng)
    PhiN = random_channel(3, rng)
    return {t: J[t].tensor(PhiN) for t in J.outcomes}


def test_retention_identity_holds_for_relational_events(rng):
    Pi = relation_preserving_map(SP, ["N"], random_density(3, rng))
    for K in relational_events(rng).values():
        assert retention_defect(Pi, K) < 1e-13


def test_retention_gives_identical_future_probabilities_and_states(rng):
    Pi = relation_preserving_map(SP, ["N"], random_density(3, rng))
    Ks = relational_events(rng)
    rho = random_density(6, rng)
    for seq in itertools.product(Ks, repeat=4):
        plain, compiled = rho.copy(), Pi(rho)
        for t in seq:
            plain = Ks[t](plain)
            compiled = Pi(Ks[t](compiled))
        p_plain = np.real(np.trace(plain))
        assert np.isclose(np.real(np.trace(compiled)), p_plain, atol=1e-13)
        if p_plain > 1e-10:
            assert np.allclose(Pi(plain) / p_plain, compiled / np.trace(compiled), atol=1e-10)


def test_retention_fails_when_the_nuisance_feeds_back(rng):
    """If N later influences R, discarding N changes future probabilities; the defect detects it."""
    Pi = relation_preserving_map(SP, ["N"], np.eye(3) / 3)
    swapish = SP.embed(random_unitary(6, rng), ["R", "N"])
    K = KrausMap([SP.embed(P0, "R") @ swapish])
    assert retention_defect(Pi, K) > 1e-3
    rho = kron(np.diag([0.5, 0.5]), np.diag([1.0, 0, 0])).astype(complex)
    assert abs(np.trace(K(rho)) - np.trace(K(Pi(rho)))) > 1e-3


# -- decision D7: the retained algebra -------------------------------------------------------

def test_algebra_canalization_satisfies_the_identity_and_is_cptp(rng):
    Ks = list(relational_events(rng).values())
    A = invariant_algebra(Ks, 6)
    assert A.dimension == 4                      # M_2 (x) I_3: only relational distinctions matter
    S = A.superop()
    assert_cptp_idempotent(S, 6)
    for K in Ks:
        assert retention_defect(S, K) < 1e-12
    # E_A is the relation-preserving map with the uniform nuisance state
    assert np.allclose(S, relation_preserving_map(SP, ["N"], np.eye(3) / 3).superop(), atol=1e-12)


def test_generic_dynamics_retains_everything(rng):
    J = random_instrument(4, (0, 1), rng)
    A = invariant_algebra(list(J.maps.values()), 4)
    assert A.dimension == 16
    assert np.allclose(A.superop(), np.eye(16), atol=1e-10)


def test_classical_dynamics_retains_only_populations(rng):
    d = 4
    T = rng.random((d, d))
    T /= T.sum(axis=0)                           # column-stochastic transitions
    ks = {0: [], 1: []}
    for i in range(d):
        for j in range(d):
            out = 0 if i < 2 else 1              # outcome reports the target block
            E = np.zeros((d, d), dtype=complex)
            E[i, j] = np.sqrt(T[i, j])
            ks[out].append(E)
    maps = [KrausMap(v) for v in ks.values()]
    A = invariant_algebra(maps, d)
    assert A.dimension == d
    deph = projective_dephasing([np.diag(np.eye(d)[k]).astype(complex) for k in range(d)])
    assert np.allclose(A.superop(), deph.superop(), atol=1e-12)


def test_incomplete_codes_need_only_populations():
    """For T2.2 the future of every bit string depends only on the parse-state populations."""
    maps = [ic.input_map(b) for b in (0, 1)]
    A = invariant_algebra(maps, ic.DIM)
    assert A.dimension == 3
    Pi = A.channel()
    base = ic.protocol()
    compiled = EventProtocol(input_map=base.input_map, output_instrument=base.output_instrument,
                             feedback=base.feedback, canalization=lambda s: Pi, label="T2.2 compiled")
    m_plain, m_comp = ic.machine(), Machine(compiled, ic.initial_state())
    for n in range(1, 7):
        for bits in itertools.product((0, 1), repeat=n):
            assert np.isclose(m_plain.sequence_probability(bits), m_comp.sequence_probability(bits), atol=1e-13)


def test_two_mode_machine_keeps_every_distinction():
    J = tm.output_instrument()
    maps = [tm.feedback(t).compose(J[t]) for t in (0, 1)]
    A = invariant_algebra(maps, 4)
    for K in maps:
        assert retention_defect(A.superop(), K) < 1e-12
    assert A.dimension == 16


def test_algebra_respects_declared_seeds(rng):
    """Seeds add observables that must stay readable (e.g. a declared output observable)."""
    Ks = list(relational_events(rng).values())
    A = invariant_algebra(Ks, 6, seeds=[SP.embed(np.diag([1.0, 0, 0]).astype(complex), "N")])
    assert A.dimension > 4
    assert A.contains(SP.embed(np.diag([1.0, 0, 0]).astype(complex), "N"))
    for K in Ks:
        assert retention_defect(A.superop(), K) < 1e-12


# -- T11.1 continuous retention and the lock ------------------------------------------------

def test_continuous_retention_identity(rng):
    """Pi L = Pi L Pi implies Pi e^{tL} = Pi e^{tL} Pi."""
    H = SP.embed(random_hermitian(2, rng), "R")
    jumps = [SP.embed(dephasing_jump(np.diag([1.0, -1.0, 0.0]).astype(complex), 0.7), "N"),
             SP.embed(dephasing_jump(np.diag([0.0, 1.0, -1.0]).astype(complex), 0.4), "N")]
    L = liouvillian(H, jumps)
    Pi = relation_preserving_map(SP, ["N"], np.eye(3) / 3).superop()
    assert np.max(np.abs(Pi @ L - Pi @ L @ Pi)) < 1e-12
    for t in (0.3, 1.7, 5.0):
        E = semigroup(L, t)
        assert np.max(np.abs(Pi @ E - Pi @ E @ Pi)) < 1e-11


def test_lock_converges_exponentially_with_the_gap():
    L = liouvillian(None, [dephasing_jump(Z, 0.5)], dim=2)
    Pi, rep = spectral_lock(L, 2)
    assert rep.settled and np.isclose(rep.gap, 1.0)
    assert is_completely_positive_superop(Pi, 2)
    for t in (2.0, 4.0, 8.0):
        err = np.max(np.abs(semigroup(L, t) - Pi))
        assert np.isclose(err, np.exp(-rep.gap * t), rtol=1e-9)


# -- T11.4 incompatible locks vs nested retention --------------------------------------------

def test_z_then_x_locks_erase_z():
    Pz = projective_dephasing([P0, P1])
    plus = np.array([1, 1]) / np.sqrt(2)
    Px = projective_dephasing([np.outer(plus, plus), I2 - np.outer(plus, plus)])
    for S in (Pz, Px):
        assert idempotence_defect(S) < 1e-15
    rho = np.diag([0.9, 0.1]).astype(complex)
    assert np.isclose(expectation(Pz(rho), Z), 0.8)
    assert np.isclose(expectation(Px(Pz(rho)), Z), 0.0)


def test_isometric_nesting_preserves_expectations(rng):
    V = random_unitary(5, rng)[:, :3]
    for _ in range(5):
        rho, A = random_density(3, rng), random_hermitian(3, rng)
        r2, A2 = isometric_nesting(V, rho, A)
        assert np.isclose(expectation(r2, A2), expectation(rho, A), atol=1e-13)
    W = append_factor_isometry(2, 3)
    assert np.allclose(W.conj().T @ W, np.eye(2))


def test_nesting_the_old_distinction_survives_a_new_lock():
    """Copy the Z distinction into a new factor, then lock the old factor in X: Z survives in the nest."""
    V = np.zeros((4, 2), dtype=complex)
    V[0, 0] = V[3, 1] = 1.0                     # |i> -> |i>|i>
    rho = np.diag([0.9, 0.1]).astype(complex)
    nested, _Z_new = isometric_nesting(V, rho, Z)
    plus = np.array([1, 1]) / np.sqrt(2)
    Px_old = projective_dephasing([kron(np.outer(plus, plus), I2), kron(I2 - np.outer(plus, plus), I2)])
    after = Px_old(nested)
    assert np.isclose(expectation(after, kron(I2, Z)), 0.8)
    assert np.isclose(expectation(after, kron(Z, I2)), 0.0)


# -- T11.5 current-state storage versus chronological storage --------------------------------

def test_fixed_dimension_state_keeps_the_function_not_the_history():
    from vam.realizations import fourier25 as f25
    states = {}
    for n in range(1, 9):
        for bits in itertools.product((0, 1), repeat=n):
            m = f25.machine()
            v = 0
            for b in bits:
                v = (2 * v + b) % 25
                m.force(b, v)
            states.setdefault(v, []).append(m.state.R)
    assert len(states) == 25                          # 510 histories, 25 present states
    for Rs in states.values():
        assert all(np.allclose(R, Rs[0], atol=1e-12) for R in Rs)
    reps = [Rs[0] for Rs in states.values()]
    overlaps = [abs(np.trace(a @ b)) for a, b in itertools.combinations(reps, 2)]
    assert max(overlaps) < 1e-12                      # residues perfectly distinguishable
