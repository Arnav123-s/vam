"""T4: present-state sufficiency (see also T10 tests), boundary mediation, statistical blankets."""

import numpy as np

from vam.boundary import (
    accessible_activity_configurations,
    classical_permutation,
    conditional_mutual_information,
    interaction_boundary,
    interior_derivative_via_boundary,
    pointer_blanket_state,
    screening_violations,
)
from vam.hamiltonian import JointHamiltonian, heisenberg_derivative
from vam.linalg import P1, X, Y, Z, kron, random_density, random_hermitian, random_pure_state
from vam.lindblad import lindblad_rhs
from vam.space import Space


def _ibe_hamiltonian(rng, with_direct=False):
    sp = Space.of(I=2, B=3, E=2)
    H = JointHamiltonian(sp)
    H.add("I", random_hermitian(2, rng), "H_I")
    H.add(["I", "B"], random_hermitian(6, rng), "H_IB")
    H.add(["B", "E"], random_hermitian(6, rng), "H_BE")
    H.add("E", random_hermitian(2, rng), "H_E")
    if with_direct:
        H.add(["I", "E"], random_hermitian(4, rng), "H_IE")
    return sp, H


def test_generator_level_boundary_hamiltonian(rng):
    """T4.2: d<O_I>/dt depends only on R_IB and H_IB."""
    sp, H = _ibe_hamiltonian(rng)
    assert screening_violations(H.terms, ["I"], ["E"]) == []
    R = random_density(sp.dim, rng)
    O = random_hermitian(2, rng)
    full = heisenberg_derivative(H.matrix(), sp.embed(O, "I"), R)
    local = interior_derivative_via_boundary(H, R, O, ["I"], ["B"])
    assert np.isclose(full, local)
    # two global states with the same R_IB but different exterior give the same derivative
    R2 = kron(sp.reduce(R, ["I", "B"]), random_density(2, rng))
    assert np.isclose(heisenberg_derivative(H.matrix(), sp.embed(O, "I"), R2), local)


def test_direct_interior_exterior_term_is_detected(rng):
    sp, H = _ibe_hamiltonian(rng, with_direct=True)
    assert screening_violations(H.terms, ["I"], ["E"]) == ["H_IE"]
    R = random_density(sp.dim, rng)
    O = random_hermitian(2, rng)
    full = heisenberg_derivative(H.matrix(), sp.embed(O, "I"), R)
    local = interior_derivative_via_boundary(H, R, O, ["I"], ["B"])
    assert not np.isclose(full, local)


def test_generator_level_boundary_with_local_dissipators(rng):
    """T4.2 for Lindblad generators L_IB + L_BE with trace-preserving local dissipators."""
    sp, H = _ibe_hamiltonian(rng)
    L_be = [sp.embed(rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6)), ["B", "E"]) for _ in range(2)]
    R = random_density(sp.dim, rng)
    O = sp.embed(random_hermitian(2, rng), "I")
    rhs = lindblad_rhs(R, H.matrix(), L_be)
    with_diss = np.trace(O @ rhs)
    without = np.trace(O @ lindblad_rhs(R, H.matrix(), []))
    assert np.isclose(with_diss, without)


def test_pointer_blanket_has_zero_cmi(rng):
    """T4.3."""
    probs = [0.2, 0.5, 0.3]
    RI = [random_density(2, rng) for _ in probs]
    RE = [random_density(2, rng) for _ in probs]
    R = pointer_blanket_state(probs, RI, RE)
    sp = Space.of(I=2, B=3, E=2)
    assert abs(conditional_mutual_information(R, sp, ["I"], ["B"], ["E"])) < 1e-10


def test_quantum_markov_chain_form_has_zero_cmi(rng):
    """T4.3 general form: B = B_L kron B_R with R = R_{I B_L} kron R_{B_R E} (entanglement allowed)."""
    psi_l = random_pure_state(4, rng)
    psi_r = random_pure_state(4, rng)
    R = kron(np.outer(psi_l, psi_l.conj()), np.outer(psi_r, psi_r.conj()))
    sp = Space.of(I=2, BL=2, BR=2, E=2)
    assert abs(conditional_mutual_information(R, sp, ["I"], ["BL", "BR"], ["E"])) < 1e-9
    # generic state: nonzero
    assert conditional_mutual_information(random_density(16, rng), sp, ["I"], ["BL", "BR"], ["E"]) > 1e-3


def test_relay_counterexample_cmi_one_bit():
    """T4.4: every interaction goes through B, yet I(I:E|B) = 1 bit at the end."""
    # factor order (I, B, E); E uniform, B = I = 0
    R = np.zeros((8, 8), dtype=complex)
    R[0, 0] = R[1, 1] = 0.5
    steps = [lambda s: [s[0], s[1] ^ s[2], s[2]],   # B <- B xor E   (acts on B, E)
             lambda s: [s[0] ^ s[1], s[1], s[2]],   # I <- I xor B   (acts on I, B)
             lambda s: [s[0], s[1] ^ s[2], s[2]]]   # B <- B xor E
    for f in steps:
        P = classical_permutation(3, f)
        R = P @ R @ P.conj().T
    sp = Space.of(I=2, B=2, E=2)
    assert np.allclose(sp.reduce(R, "B"), np.diag([1, 0]))
    RIE = sp.reduce(R, ["I", "E"])
    assert np.allclose(RIE, np.diag([0.5, 0, 0, 0.5]))  # I = E
    assert np.isclose(conditional_mutual_information(R, sp, ["I"], ["B"], ["E"]), 1.0)


def test_activity_registers_switch_boundaries(rng):
    """T4.5: d<Z1>/dt = 2 <N12 Y1 X2>; the interaction is inactive on N12 = 0."""
    sp = Space.of(N12=2, N23=2, V1=2, V2=2, V3=2)
    H = JointHamiltonian(sp)
    H.add(["V1", "V2"], kron(X, X), "h12", activity="N12")
    H.add(["V2", "V3"], kron(X, X), "h23", activity="N23")
    Hm = H.matrix()
    R = random_density(sp.dim, rng)
    lhs = heisenberg_derivative(Hm, sp.embed(Z, "V1"), R)
    rhs = 2 * np.trace(R @ sp.embed(P1, "N12") @ sp.embed(kron(Y, X), ["V1", "V2"]))
    assert np.isclose(lhs, rhs)
    # weight density supported on N12 = 0: no influence on V1
    Rv = random_density(8, rng)
    R0 = kron(np.diag([1, 0]), random_density(2, rng), Rv)
    assert np.isclose(heisenberg_derivative(Hm, sp.embed(Z, "V1"), R0), 0)
    cfg0 = accessible_activity_configurations(sp, R0, ["N12", "N23"])
    assert all(c["N12"] == 0 for c in cfg0)
    assert interaction_boundary(H.terms, ["V1"], cfg0) == frozenset({"N12"})
    # re-entry flips N12 to 1: the interaction becomes active and V2 joins the boundary
    R1 = kron(np.diag([0, 1]), random_density(2, rng), Rv)
    cfg1 = accessible_activity_configurations(sp, R1, ["N12", "N23"])
    assert interaction_boundary(H.terms, ["V1"], cfg1) == frozenset({"N12", "V2"})
    # coherent superposition of N12 sectors: union of accessible supports
    plus = np.full((2, 2), 0.5, dtype=complex)
    Rs = kron(plus, random_density(2, rng), Rv)
    cfgs = accessible_activity_configurations(sp, Rs, ["N12", "N23"])
    assert interaction_boundary(H.terms, ["V1"], cfgs) == frozenset({"N12", "V2"})


def test_zero_mean_weight_is_not_absence(rng):
    """T4.2 remark: a zero-mean interaction weight can still contribute through correlations."""
    sp = Space.of(W=2, V1=2, V2=2)
    H = JointHamiltonian(sp)
    H.add(["W", "V1", "V2"], kron(Z, X, X), "signed coupling")   # weight observable Z_W, mean 0 below
    Hm = H.matrix()
    plus = np.array([1, 1]) / np.sqrt(2)
    a = np.kron(np.array([1, 1j]) / np.sqrt(2), plus)    # <Y1 X2> = +1
    b = np.kron(np.array([1, -1j]) / np.sqrt(2), plus)   # <Y1 X2> = -1
    psi = (np.kron([1, 0], a) + np.kron([0, 1], b)) / np.sqrt(2)
    R = np.outer(psi, psi.conj())
    assert abs(np.trace(R @ sp.embed(Z, "W"))) < 1e-14  # zero mean weight
    # d<Z1>/dt = 2 <Z_W Y1 X2> = 2: correlations carry the influence
    assert np.isclose(heisenberg_derivative(Hm, sp.embed(Z, "V1"), R), 2.0)
