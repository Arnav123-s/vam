"""VAM-1 reference machine (DESIGN section 5): every map, every engine and the learning law."""

import itertools
import math

import numpy as np
import pytest

from vam.canalization import invariant_algebra, retention_defect
from vam.density import is_density
from vam.dirac import coin, split_step_walk
from vam.linalg import P0, P1, Y, kron, random_density, unitary_from_hamiltonian
from vam.machine import EventProtocol, Machine, MachineState, macro_micro_step
from vam.realizations import vam1
from vam.realizations.vam1 import Config, DenseMaps, PointerEngine
from vam.space import Space

CONFIGS = [
    Config(cells=2, context=1),
    Config(cells=2, context=2, theta=0.4, dephasing=0.7),
    Config(cells=2, context=1, theta=0.9, einselect="all"),
    Config(cells=3, context=2, theta=0.3, dephasing=0.0),
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_function_maps_equal_their_kraus_forms(rng, cfg):
    M = DenseMaps(cfg)
    w = rng.uniform(-1, 1, cfg.sectors)
    pairs = [(M.observe(w, 0), M.kraus_observe(w, 0)), (M.observe(w, 1), M.kraus_observe(w, 1)),
             (M.inject(0), M.kraus_inject(0)), (M.inject(1), M.kraus_inject(1)),
             (M.reenter(0), M.kraus_reenter(0)), (M.reenter(1), M.kraus_reenter(1)),
             (M.evolve(), M.kraus_evolve())]
    for _ in range(3):
        R = random_density(cfg.dim, rng)
        A = random_density(cfg.dim, rng)
        for F, K in pairs:
            assert np.allclose(F(R), K(R), atol=1e-13)
            assert np.allclose(F.dual(A), K.dual(A), atol=1e-13)
    for K in pairs[2:]:
        assert K[1].is_trace_preserving()
    assert np.allclose(M.kraus_observe(w, 0).effect() + M.kraus_observe(w, 1).effect(), np.eye(cfg.dim))


def test_observation_is_the_reduced_port_measurement(rng):
    """``O_b`` equals: port |0>, controlled rotation sum_s P_s kron exp(-i theta_s Y/2), read the port."""
    cfg = Config(cells=2, context=2)
    M = DenseMaps(cfg)
    w = rng.uniform(-1, 1, cfg.sectors)
    p0 = vam1.predictions(w, cfg.eps)
    theta = 2 * np.arccos(np.sqrt(p0))
    U = sum(kron(np.diag((M.lay.sector == s).astype(complex)), unitary_from_hamiltonian(Y, theta[s] / 2))
            for s in range(cfg.sectors))
    R = random_density(cfg.dim, rng)
    for b, Pb in ((0, P0), (1, P1)):
        joint = U @ kron(R, P0) @ U.conj().T
        proj = kron(np.eye(cfg.dim), Pb)
        reduced = (proj @ joint @ proj).reshape(cfg.dim, 2, cfg.dim, 2).trace(axis1=1, axis2=3)
        assert np.allclose(reduced, M.observe(w, b)(R), atol=1e-13)


def single_excitation(cfg, cell, chirality):
    q = 2 * cell + chirality
    v = np.zeros(cfg.dim, dtype=complex)
    v[1 << (cfg.qubits - 1 - q)] = 1.0
    return v


def test_one_excitation_sector_is_the_split_step_dirac_walk():
    """One event acts on one excitation as ``E = C S``; the walk is ``W = S C = C^-1 E C``."""
    c, theta, steps = 5, 0.35, 2                 # cells 0..4; two steps from cell 2 never reach an exit
    cfg = Config(cells=c, context=1, theta=theta, dephasing=0.0)
    lay = vam1.Layout.build(cfg)
    U = np.ones((1, 1), dtype=complex)
    for _ in range(c):
        U = np.kron(U, vam1.cell_unitary(theta))
    U = np.kron(U, np.eye(2))                    # the active port (empty here) is not part of the walk
    # chain amplitude of an excitation at (x, chi) <-> walk amplitude at x * 2 + chi
    basis = np.array([[single_excitation(cfg, x, ch) for ch in (0, 1)] for x in range(c)]).reshape(2 * c, -1)

    def event(psi):                      # transport with vacuum input, then the coin
        out = np.zeros_like(psi)
        np.add.at(out, lay.target[0], psi)   # exact permutation on configurations whose exits are empty
        return U @ out

    C = np.kron(np.eye(c), coin(theta))
    W = split_step_walk(c, theta, periodic=False)
    x0 = c // 2
    phi = np.zeros(2 * c, dtype=complex)
    phi[2 * x0] = 1.0
    psi = basis.T @ (C @ phi)             # chain state of C phi
    for _ in range(steps):
        psi = event(psi)
    lhs = basis @ psi                     # back to walk coordinates: E^n C phi
    rhs = C @ np.linalg.matrix_power(W, steps) @ phi
    assert np.allclose(lhs, rhs, atol=1e-13)


def test_exact_event_cone():
    """Two machines that differ only in the first bit agree on every cell the bit cannot have reached."""
    cfg = Config(cells=4, context=1, theta=0.6, dephasing=0.0)
    M = DenseMaps(cfg)
    ev = M.evolve()
    states = []
    for first in (0, 1):
        R = M.initial_state().R
        for b in (first, 0, 0):
            R = ev(M.inject(b)(R))
        states.append(R)
    sp = Space.qubits(*[f"q{j}" for j in range(cfg.qubits)])
    for cell in range(cfg.cells):
        names = [f"q{2 * cell}", f"q{2 * cell + 1}"]
        same = np.allclose(sp.reduce(states[0], names), sp.reduce(states[1], names), atol=1e-13)
        if cell >= 3:
            assert same                     # three events move a disturbance at most three cells
        if cell in (0, 2):
            assert not same                 # the bit itself (cell 2) and its reflected echo (cell 0)


@pytest.mark.parametrize("dephasing", [0.0, 0.4])
def test_each_cell_evolves_from_its_neighbours_alone(dephasing, rng):
    """Dirac dynamics inside Zurek's open-system setting, localized (T6, T13.3, T4.2).

    After one event (transport, then the mass coin and the dissipator in every cell), cell 1 holds what
    cells 0 and 2 sent it and the coin made of it: its new reduced density is fixed by
    the reduced density of cells 0 and 2 alone, whatever the rest of the chain holds or how it is
    correlated. Influence runs both ways: what cell 1 holds now decides part of its neighbours' next state.
    """
    cfg = Config(cells=4, context=1, theta=0.6, dephasing=dephasing)
    M = DenseMaps(cfg)
    evolve, transport = M.evolve(), M.inject(1)
    ev = lambda R: evolve(transport(R))
    names = [f"q{j}" for j in range(cfg.qubits)]
    sp = Space.qubits(*names)
    cell = {c: [f"q{2 * c}", f"q{2 * c + 1}"] for c in range(cfg.cells)}
    near = cell[0] + cell[2]
    far = [n for n in names if n not in near]

    def with_marginal(R, keep, other):
        """A state whose reduced density on ``keep`` is R's, with ``other`` (uncorrelated) everywhere else."""
        order = keep + [n for n in names if n not in keep]
        return Space.qubits(*order).reorder_operator(kron(sp.reduce(R, keep), other), names)

    R1 = random_density(sp.dim, rng)
    R2 = with_marginal(R1, near, random_density(2 ** len(far), rng))    # same neighbours, all else different
    assert not np.allclose(sp.reduce(R1, cell[1]), sp.reduce(R2, cell[1]))
    assert np.allclose(sp.reduce(ev(R1), cell[1]), sp.reduce(ev(R2), cell[1]), atol=1e-12)

    R3 = with_marginal(R1, cell[1] + cell[2] + cell[3] + names[-1:], random_density(4, rng))  # a new cell 0
    assert not np.allclose(sp.reduce(ev(R1), cell[1]), sp.reduce(ev(R3), cell[1]), atol=1e-6)
    R4 = with_marginal(R1, cell[0] + cell[2] + cell[3] + names[-1:], random_density(4, rng))  # a new cell 1
    for c in (0, 2):                                                    # and the other way round
        assert not np.allclose(sp.reduce(ev(R1), cell[c]), sp.reduce(ev(R4), cell[c]), atol=1e-6)


def test_massless_chain_is_an_exact_sliding_context(rng):
    cfg = Config(cells=3, context=3)
    m = DenseMaps(cfg).reader()
    bits = [int(b) for b in rng.integers(0, 2, 40)]
    for t, b in enumerate(bits):
        m.step(b, rng)
        diag = np.real(np.diag(m.state.R))
        i = int(np.argmax(diag))
        assert diag[i] == pytest.approx(1.0)
        recent = bits[max(0, t - 2):t + 1][::-1]
        R_modes = [int(x) for x in DenseMaps(cfg).lay.bits[i, 0::2]]
        assert R_modes[:len(recent)] == recent


def run_dense(cfg, bits, rng):
    m = DenseMaps(cfg).reader()
    qs, outs = [], []
    for b in bits:
        qs.append(m.predict(b))
        outs.append(m.step(b, rng).outcome)
    return qs, m, outs


@pytest.mark.parametrize("cfg", [Config(cells=3, context=2),
                                 Config(cells=3, context=2, theta=0.5, einselect="all"),
                                 Config(cells=2, context=2, theta=1.1, einselect="all", consolidating=True)])
def test_pointer_engine_equals_the_dense_engine(rng, cfg):
    """Emission for emission: given equal generators, both engines emit the same bits and agree exactly."""
    bits = [int(b) for b in rng.integers(0, 2, 60)]
    qs, m, outs = run_dense(cfg, bits, np.random.default_rng(11))
    pe = PointerEngine(cfg, rng=np.random.default_rng(11))
    qp, outp = [], []
    for b in bits:
        qp.append(pe.step(b))
        outp.append(pe.last_output)
    assert outs == outp
    assert np.allclose(qs, qp, atol=1e-12)
    assert np.allclose(m.state.structural["w"], pe.w, atol=1e-12)
    assert np.allclose(np.real(np.diag(m.state.R)), pe.p, atol=1e-12)


def test_every_external_bit_is_followed_by_an_emission_with_the_machines_own_prediction(rng):
    """T14 steps 11-12: after each external bit the machine emits at the active port with exactly its
    prediction of the next bit, conditions on that output once, and the output sits in the active port."""
    cfg = Config(cells=3, context=2, theta=0.4, dephasing=0.5)
    m = DenseMaps(cfg).reader()
    for b in rng.integers(0, 2, 25):
        R_b, _ = m.input_stage(m.state, int(b))
        probs = m.outcome_probabilities(m.state, R_b)
        after_input = Machine(m.protocol, MachineState(R_b, m.state.structural), validate=False)
        assert np.isclose(probs[0], after_input.predict(0)) and np.isclose(probs[1], after_input.predict(1))
        rec = m.step(int(b), rng)
        port = DenseMaps(cfg).lay.port
        assert np.isclose(np.real(np.diag(m.state.R))[port == rec.outcome].sum(), 1.0)


def test_massless_emission_and_reentry_leave_every_prediction_unchanged(rng):
    """theta = 0: whatever the machine emits, its predictions and weights are those of the bit kernel
    (tests/test_bitkernel.py checks the bit kernel against the dense engine)."""
    cfg = Config(cells=3, context=3, dephasing=0.0, dt=0.2)
    bits = [int(b) for b in rng.integers(0, 2, 60)]
    runs = [run_dense(cfg, bits, np.random.default_rng(seed)) for seed in (1, 2)]
    assert runs[0][2] != runs[1][2]                         # different emissions...
    assert np.allclose(runs[0][0], runs[1][0], atol=1e-12)   # ...same predictions
    assert np.allclose(runs[0][1].state.structural["w"], runs[1][1].state.structural["w"], atol=1e-12)


def test_with_mass_the_reentered_output_reaches_the_context(rng):
    """theta > 0: an emitted bit enters the chain as a left-mover at the active port, and the coin mixes it
    into the right-movers, so what the machine said changes what it predicts next."""
    cfg = Config(cells=2, context=2, theta=0.7, einselect="all", dt=0.2)
    bits = [int(b) for b in rng.integers(0, 2, 30)]
    m = DenseMaps(cfg).reader()
    for b in bits:
        m.step(b, rng)
    preds = []
    for tau in (0, 1):
        branch = Machine(m.protocol, m.state.copy())
        branch.force(0, tau)
        branch.force(0, 0)            # one more transport carries the port's content into the chain
        preds.append(branch.predict(0))
    assert abs(preds[0] - preds[1]) > 1e-3


def test_reentered_outputs_are_not_evidence(rng):
    """T10.3: the weights after an event depend on the external bit, never on the emitted one."""
    cfg = Config(cells=2, context=2, theta=0.5, einselect="all", dt=0.2)
    m = DenseMaps(cfg).reader()
    for b in rng.integers(0, 2, 20):
        m.step(int(b), rng)
    weights = []
    for tau in (0, 1):
        branch = Machine(m.protocol, m.state.copy())
        branch.force(1, tau)
        weights.append(branch.state.structural["w"])
    assert np.array_equal(weights[0], weights[1])


def test_coherent_engine_keeps_every_event_a_valid_density(rng):
    cfg = Config(cells=3, context=2, theta=0.5, dephasing=0.3)
    m = DenseMaps(cfg).reader()
    for b in rng.integers(0, 2, 30):
        assert np.isclose(m.predict(0) + m.predict(1), 1.0)
        m.step(int(b), rng)
        assert is_density(m.state.R, atol=1e-9)


def test_floor_keeps_every_bit_possible():
    cfg = Config(cells=1, context=1, eps=2.0 ** -10)
    for w in (-5.0, -1.0, 0.0, 1.0, 7.0):
        p = vam1.bit_probabilities(np.array([w]), cfg.eps, 0)[0]
        assert cfg.eps - 1e-15 <= p <= 1 - cfg.eps + 1e-15


def markov_source(rng, n, table):
    """Order-2 bit source: P(next = 0 | last two bits) from ``table`` keyed by s = b_{t-1} + 2 b_{t-2}."""
    bits = [0, 0]
    for _ in range(n):
        s = bits[-1] + 2 * bits[-2]
        bits.append(0 if rng.random() < table[s] else 1)
    return bits[2:]


@pytest.mark.parametrize("consolidating", [False, True])
def test_weights_learn_the_conditional_statistics_forward(rng, consolidating):
    table = np.array([0.9, 0.2, 0.6, 0.35])
    cfg = Config(cells=2, context=2, dt=0.01, consolidating=consolidating)
    pe = PointerEngine(cfg)
    for b in markov_source(rng, 60_000, table):
        pe.step(b)
    assert np.allclose(vam1.predictions(pe.w, cfg.eps), table, atol=0.06 if not consolidating else 0.02)


def test_the_literal_law_consolidates_geometrically_as_in_t12_2():
    """T12.2: under one repeated lesson the literal T12.1 law shrinks the transient by a fixed factor r per
    repetition. With kappa dt / gamma = ln 2 and full activity, r = 1/2: multipliers 1/2, 1/4, 1/32, 1/1024."""
    cfg = Config(cells=1, context=1, dt=math.log(2))
    w, n = np.array([0.0]), np.array([0.0])
    target, multipliers = 1.0, {}
    for rep in range(1, 11):
        w, n = vam1.learn(cfg, w, n, np.array([1.0]), 0)
        multipliers[rep] = abs(w[0] - target) / abs(0.0 - target)
    for rep, expected in ((1, 1 / 2), (2, 1 / 4), (5, 1 / 32), (10, 1 / 1024)):
        assert multipliers[rep] == pytest.approx(expected, rel=1e-12)


def test_the_count_dependent_friction_is_frequency_counting_not_t12_2():
    """Decision D10: with gamma_s = gamma + kappa dt n_s and g = kappa = gamma = dt = 1, the transient after
    n repeated lessons is exp(-H_n) (H_n the harmonic number), not r^n: the weight is a running average."""
    cfg = Config(cells=1, context=1, dt=1.0, consolidating=True)
    w, n = np.array([0.0]), np.array([0.0])
    harmonic = 0.0
    for rep in range(1, 101):
        w, n = vam1.learn(cfg, w, n, np.array([1.0]), 0)
        harmonic += 1.0 / rep
        assert abs(w[0] - 1.0) == pytest.approx(math.exp(-harmonic), rel=1e-12)


def test_consolidation_beats_constant_friction_on_a_stationary_source(rng):
    table = np.array([0.9, 0.2, 0.6, 0.35])
    bits = markov_source(rng, 20_000, table)
    loss = {}
    for cons in (False, True):
        pe = PointerEngine(Config(cells=2, context=2, dt=0.05, consolidating=cons))
        loss[cons] = -np.mean([np.log2(pe.step(b)) for b in bits])
    assert loss[True] < loss[False]


def test_generation_emits_the_learned_continuation(rng):
    """Trained on a periodic stream, macro/micro generation continues the period."""
    pattern = [0, 1, 1, 0, 1, 0, 0, 0]
    cfg = Config(cells=4, context=4, dt=0.2)
    reader = DenseMaps(cfg).reader()
    stream = pattern * 40
    for b in stream:
        reader.step(b, rng)
    gen = DenseMaps(cfg).generator(reader.state)
    emitted = []
    for _ in range(len(pattern)):
        plan = macro_micro_step(gen, horizon=3)
        emitted.append(plan.first)
    assert emitted == pattern
    assert np.allclose(gen.state.structural["w"], reader.state.structural["w"])   # outputs are not evidence


def test_generation_outputs_are_the_machines_own_predictions(rng):
    cfg = Config(cells=3, context=2, theta=0.4, dephasing=0.5)
    reader = DenseMaps(cfg).reader()
    for b in rng.integers(0, 2, 25):
        reader.step(int(b), rng)
    gen = DenseMaps(cfg).generator(reader.state)
    R_b, _ = gen.input_stage(gen.state, None)
    probs = gen.outcome_probabilities(gen.state, R_b)
    assert np.isclose(probs[0], reader.predict(0)) and np.isclose(probs[1], reader.predict(1))


def event_maps(M, w):
    """Every unnormalized event map of the reading loop at weights ``w``: external bit b, emission tau."""
    return [M.kraus_reenter(t).compose(M.kraus_observe(w, t)).compose(M.kraus_evolve())
            .compose(M.kraus_inject(b)).compose(M.kraus_observe(w, b)) for b in (0, 1) for t in (0, 1)]


@pytest.mark.parametrize("cells,context,theta,expected", [(2, 1, 0.0, 2), (2, 2, 0.0, 4), (2, 1, 0.7, None)])
def test_canalization_keeps_exactly_what_the_future_can_reveal_at_every_weight(rng, cells, context, theta,
                                                                                 expected):
    """D7 for the learning machine: one algebra serves every weight, including w = 0 and equal weights,
    where the maps alone cannot tell the sectors apart."""
    cfg = Config(cells=cells, context=context, theta=theta)
    M = DenseMaps(cfg)
    A = M.canalization()
    if expected is not None:
        assert A.dimension == expected          # massless: only the context populations matter (2^k)
    else:
        assert A.dimension > 2 ** context       # mass brings left-movers back into the context
    Pi = A.channel()
    for w in (np.zeros(cfg.sectors), np.full(cfg.sectors, 0.4), rng.uniform(-1, 1, cfg.sectors)):
        for K in event_maps(M, w):
            assert retention_defect(Pi, K) < 1e-10


def test_a_canalization_built_from_one_weight_alone_would_erase_what_learning_needs():
    """The audited failure: closing the algebra over the maps at w = 0 only (no sector seeds) keeps the
    scalars alone, so it cannot serve a machine whose weights move."""
    cfg = Config(cells=2, context=1)
    M = DenseMaps(cfg)
    narrow = invariant_algebra(event_maps(M, np.zeros(cfg.sectors)), cfg.dim)
    assert narrow.dimension == 1
    assert max(retention_defect(narrow.channel(), K) for K in event_maps(M, np.array([-0.9, 0.9]))) > 1e-3
    assert M.canalization().dimension == 2


@pytest.mark.parametrize("theta", [0.0, 0.6])
def test_canalized_learning_machine_predicts_exactly_as_the_plain_one(rng, theta):
    """With Pi applied after every event, the learning machine's predictions, emissions and weights are
    those of the uncanalized machine (alternating stream, prescribed emissions)."""
    cfg = Config(cells=2, context=1, theta=theta, dephasing=0.3, dt=0.3)
    M = DenseMaps(cfg)
    Pi = M.canalization().channel()
    plain, compiled = M.reader(), M.reader(canalization=lambda s: Pi)
    for b in [0, 1] * 30:
        assert plain.predict(b) == pytest.approx(compiled.predict(b), abs=1e-12)
        tau = int(rng.integers(0, 2))
        plain.force(b, tau)
        compiled.force(b, tau)
        assert np.allclose(plain.state.structural["w"], compiled.state.structural["w"], atol=1e-12)
    assert abs(plain.state.structural["w"]).max() > 0.5     # the machine did learn the alternation


def test_canalized_machine_keeps_every_output_string_probability(rng):
    cfg = Config(cells=2, context=1, theta=0.7)
    M = DenseMaps(cfg)
    Pi = M.canalization().channel()
    w = np.array([0.3, -0.6])
    proto = M.reading_protocol(learning=False)
    canal = EventProtocol(input_map=proto.input_map, output_instrument=proto.output_instrument,
                          feedback=proto.feedback, canalization=lambda s: Pi, input_mode="condition")
    start = MachineState(random_density(cfg.dim, rng), {"w": w, "n": np.zeros(2)})
    plain, compiled = Machine(proto, start), Machine(canal, start)
    for bits in itertools.product((0, 1), repeat=4):
        assert math.isclose(plain.sequence_probability(bits), compiled.sequence_probability(bits), abs_tol=1e-12)
