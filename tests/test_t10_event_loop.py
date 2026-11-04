"""T10: the complete present-state event loop (event maps, theorem, feedback is not evidence)."""

import itertools

import numpy as np
import pytest

from vam.channels import (
    ChannelError,
    Instrument,
    KrausMap,
    identity_channel,
    random_channel,
    random_instrument,
    replacement_channel,
)
from vam.density import ZeroProbabilityError, is_density
from vam.linalg import P0, P1, random_density
from vam.machine import EventProtocol, Machine, MachineState
from vam.realizations import incomplete_codes as ic

D = 4
OUTS = ("a", "b", "c")


def random_protocol(rng, with_plasticity=False):
    """Random CP event maps satisfying exactly the assumptions of T10.2."""
    B = random_instrument(D, (0, 1), rng, label="B")
    J = random_instrument(D, OUTS, rng, label="J")
    E = {t: random_channel(D, rng, label=f"E{t}") for t in OUTS}
    Pi = random_channel(D, rng, n_kraus=2, label="Pi")
    return EventProtocol(input_map=lambda b, s: B[b], output_instrument=lambda s: J,
                         feedback=lambda t, s: E[t], canalization=lambda s: Pi,
                         plasticity=slow_coordinate if with_plasticity else None, label="random"), (B, J, E, Pi)


def slow_coordinate(ctx, s):
    """A structural coordinate driven by the present state only."""
    return {"w": 0.9 * s["w"] + 0.1 * np.real(np.diag(ctx.R_new))}


def test_event_equation_equals_composed_maps(rng):
    proto, (B, J, E, Pi) = random_protocol(rng)
    for _ in range(5):
        R = random_density(D, rng)
        m = Machine(proto, MachineState(R))
        for b in (0, 1):
            R_b, q = m.input_stage(m.state, b)
            assert np.isclose(q, np.real(np.trace(B[b](R))))
            for t in OUTS:
                new, p = m.complete_event(m.state, b, q, R_b, t)
                num = Pi(E[t](J[t](B[b](R))))
                den = np.real(np.trace(J[t](B[b](R))))
                assert np.allclose(new.R, num / den, atol=1e-12)
                assert np.isclose(q * p, den, atol=1e-13)


def test_theorem_positivity_normalization_and_sum(rng):
    proto, _ = random_protocol(rng)
    m = Machine(proto, MachineState(random_density(D, rng)))
    for _ in range(300):
        b = int(rng.integers(2))
        R_b, _q = m.input_stage(m.state, b)
        probs = m.outcome_probabilities(m.state, R_b)
        assert all(p >= -1e-14 for p in probs.values())
        assert np.isclose(sum(probs.values()), 1.0, atol=1e-12)
        m.step(b, rng)
        assert is_density(m.state.R, atol=1e-10)


def test_predicted_stream_probabilities_sum_to_one(rng):
    proto, _ = random_protocol(rng)
    m = Machine(proto, MachineState(random_density(D, rng)))
    for n in (1, 2, 3, 4):
        total = sum(m.sequence_probability(bits) for bits in itertools.product((0, 1), repeat=n))
        assert np.isclose(total, 1.0, atol=1e-12)


def test_induction_sequential_normalized_events_equal_one_linear_composition(rng):
    """Product of per-event probabilities and the final state equal the composed unnormalized maps."""
    proto, (B, J, E, Pi) = random_protocol(rng)
    R0 = random_density(D, rng)
    for history in [((0, "a"), (1, "c"), (1, "b")), ((1, "b"), (0, "b"), (0, "a"), (1, "c"))]:
        m = Machine(proto, MachineState(R0.copy()))
        weight, linear = 1.0, R0.copy()
        for b, t in history:
            rec = m.force(b, t)
            weight *= rec.q_bit * rec.p_outcome
            linear = Pi(E[t](J[t](B[b](linear))))
        assert np.isclose(weight, np.real(np.trace(linear)), atol=1e-13)
        assert np.allclose(m.state.R, linear / np.trace(linear), atol=1e-12)


def test_markov_property_against_the_generative_process():
    """Histories ``0``, ``00`` and ``000`` of T2.2 give one present state, and the
    generative process itself (brute-force unit enumeration, no densities) gives
    the same conditional future for each of them."""
    def prefix_prob(s):
        return sum(ic.brute_force_parses(s).values()) if s else 1.0

    states = {}
    for h in ("0", "00", "000"):
        m = ic.machine()
        for ch in h:
            m.force(int(ch), None)
        states[h] = m.state
    assert states["0"].distance(states["00"]) < 1e-15 and states["0"].distance(states["000"]) < 1e-15
    m = ic.machine()
    for n in range(1, 5):
        for f in itertools.product("01", repeat=n):
            f = "".join(f)
            cond = {h: prefix_prob(h + f) / prefix_prob(h) for h in states}
            assert np.isclose(cond["0"], cond["00"], atol=1e-14) and np.isclose(cond["0"], cond["000"], atol=1e-14)
            assert np.isclose(m.sequence_probability([int(c) for c in f], state=states["0"]), cond["0"], atol=1e-14)


def test_machine_keeps_only_the_present_state(rng):
    proto, _ = random_protocol(rng, with_plasticity=True)
    m = Machine(proto, MachineState(random_density(D, rng), {"w": np.zeros(D)}))
    assert set(Machine.__slots__) == {"protocol", "state", "validate"}
    nbytes, shapes = m.state.nbytes(), m.state.shapes()
    for b in rng.integers(0, 2, 2000):
        m.step(int(b), rng)
        assert m.state.nbytes() == nbytes and m.state.shapes() == shapes


def test_equal_present_states_have_identical_futures(rng):
    """Two machines in the same complete configuration produce identical futures for identical inputs."""
    proto, _ = random_protocol(rng, with_plasticity=True)
    R = random_density(D, rng)
    m1 = Machine(proto, MachineState(R.copy(), {"w": np.zeros(D)}))
    # reach a state by some history, then clone it into a machine with no history at all
    for b in (0, 1, 1, 0, 1):
        m1.step(b, rng)
    m2 = Machine(proto, m1.state.copy())
    bits = rng.integers(0, 2, 50)
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    out1 = [m1.step(int(b), r1).outcome for b in bits]
    out2 = [m2.step(int(b), r2).outcome for b in bits]
    assert out1 == out2 and m1.state.distance(m2.state) == 0.0


# -- T10.3 feedback is not evidence ---------------------------------------------------------

def noisy_readout(eps):
    """Z readout with flip probability eps (outcome tau reports the hidden bit)."""
    K = {0: [np.sqrt(1 - eps) * P0, np.sqrt(eps) * P1], 1: [np.sqrt(eps) * P0, np.sqrt(1 - eps) * P1]}
    return Instrument({t: KrausMap(K[t], label=f"J{t}") for t in (0, 1)}, label="noisy Z")


def test_output_is_conditioned_on_once():
    eps, prior = 0.2, 0.5
    R = np.diag([prior, 1 - prior]).astype(complex)
    proto = EventProtocol(input_map=lambda b, s: identity_channel(2), output_instrument=lambda s: noisy_readout(eps),
                          feedback=lambda t, s: identity_channel(2), settle=lambda s: identity_channel(2),
                          input_mode="inject")
    m = Machine(proto, MachineState(R))
    m.force(None, 0)
    once = prior * (1 - eps) / (prior * (1 - eps) + (1 - prior) * eps)            # 0.8
    twice = prior * (1 - eps) ** 2 / (prior * (1 - eps) ** 2 + (1 - prior) * eps ** 2)  # 16/17
    assert np.isclose(np.real(m.state.R[0, 0]), once) and not np.isclose(once, twice)


def test_feedback_must_be_trace_preserving():
    """Re-entry is an interaction, never a second conditioning: a non-TP feedback map is rejected."""
    second_conditioning = KrausMap([P0], label="condition again")
    proto = EventProtocol(input_map=lambda b, s: identity_channel(2), output_instrument=lambda s: noisy_readout(0.2),
                          feedback=lambda t, s: second_conditioning, settle=lambda s: identity_channel(2),
                          input_mode="inject")
    m = Machine(proto, MachineState(np.eye(2, dtype=complex) / 2))
    with pytest.raises(ChannelError):
        m.force(None, 0)


def test_feedback_changes_state_without_changing_total_probability(rng):
    """Averaged over outcomes, feedback only transforms; it never re-weights outcomes."""
    proto, (_B, J, E, _Pi) = random_protocol(rng)
    m = Machine(proto, MachineState(random_density(D, rng)))
    R_b, _ = m.input_stage(m.state, 0)
    probs = m.outcome_probabilities(m.state, R_b)
    for t in OUTS:
        assert np.isclose(np.real(np.trace(E[t](J[t](R_b)))), probs[t], atol=1e-13)


# -- zero-probability observations (T10.2 last paragraph) -----------------------------------

def test_zero_probability_observation_is_not_normalized():
    m = ic.machine()
    m.force(1, None)                           # C has started: the next bit must be 0
    assert np.isclose(m.predict(1), 0.0)
    with pytest.raises(ZeroProbabilityError):
        m.force(1, None)
    assert m.sequence_probability([1, 1]) == 0.0


def test_zero_probability_inject_policy_keeps_interpretation_unresolved():
    fallback = lambda b, s: replacement_channel(np.eye(ic.DIM, dtype=complex) / ic.DIM, ic.DIM, label="unresolved")
    base = ic.protocol()
    proto = EventProtocol(input_map=base.input_map, output_instrument=base.output_instrument, feedback=base.feedback,
                          zero_policy="inject", fallback_injection=fallback, label="inject policy")
    m = Machine(proto, ic.initial_state())
    m.force(1, None)
    rec = m.force(1, None)
    assert rec.q_bit == 0.0
    assert np.allclose(m.state.R, np.eye(ic.DIM) / ic.DIM)


def test_conditioning_requires_a_complete_prediction(rng):
    B = random_instrument(D, (0, 1), rng)
    half = KrausMap([K * np.sqrt(0.5) for K in B[1].kraus])
    J = random_instrument(D, OUTS, rng)
    proto = EventProtocol(input_map=lambda b, s: B[0] if b == 0 else half, output_instrument=lambda s: J,
                          feedback=lambda t, s: identity_channel(D))
    m = Machine(proto, MachineState(random_density(D, rng)))
    with pytest.raises(ChannelError):
        m.force(0, "a")
