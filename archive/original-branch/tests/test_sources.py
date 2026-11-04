"""Known test sources and the reference learner of experiment E1 (experiments/PROTOCOL.md)."""

import itertools

import numpy as np
import pytest

from vam.baselines import contexts, kt_context_q
from vam.realizations import incomplete_codes as ic
from vam.sources import BitHMM, ByteMarkov, DelayedEcho, IncompleteCodes, MarkovBits, conditional_entropy, h2


def test_binary_entropy():
    assert np.allclose(h2(np.array([0.0, 0.5, 1.0])), [0.0, 1.0, 0.0])
    assert np.isclose(h2(np.array([0.11]))[0], -(0.11 * np.log2(0.11) + 0.89 * np.log2(0.89)))


def test_hmm_rejects_an_invalid_model():
    with pytest.raises(ValueError):
        BitHMM(np.full((2, 2, 2), 0.3))


def test_markov_bits_floor_is_the_entropy_rate_once_the_window_covers_the_order(rng):
    src = MarkovBits()
    rate = src.entropy_rate()
    for k in (2, 3, 6):
        assert np.isclose(src.window_floor(k), rate, atol=1e-12)
    assert src.window_floor(1) > rate + 1e-3 and src.window_floor(0) >= src.window_floor(1)
    sample = src.sample(400_000, rng)
    se = np.std(-np.log2(sample.oracle_q)) / np.sqrt(len(sample.bits))
    assert abs(sample.oracle_loss - rate) < 5 * se


def test_markov_bits_emit_the_table(rng):
    src = MarkovBits()
    sample = src.sample(200_000, rng)
    s = contexts(sample.bits, 2)
    for ctx in range(4):
        sel = s == ctx
        sel[:2] = False
        assert abs(np.mean(sample.bits[sel] == 0) - src.table[ctx]) < 0.01


def test_incomplete_codes_oracle_is_the_t22_machine():
    """Sequence probabilities of the forward filter equal the T2.2 density machine's, for every string."""
    src = IncompleteCodes()
    machine = ic.machine()
    prior = np.array([1.0, 0.0, 0.0])
    for n in range(1, 7):
        total = 0.0
        for bits in itertools.product((0, 1), repeat=n):
            p_filter = float(np.prod(src.hmm().filter_q(np.array(bits), prior)))
            assert np.isclose(p_filter, machine.sequence_probability(list(bits)), atol=1e-14)
            total += p_filter
        assert np.isclose(total, 1.0)


def test_incomplete_codes_floors_fall_toward_the_entropy_rate(rng):
    src = IncompleteCodes()
    floors = [src.window_floor(k) for k in range(0, 13)]
    assert all(a >= b - 1e-12 for a, b in itertools.pairwise(floors))
    sample = src.sample(300_000, rng)
    assert sample.oracle_loss < floors[12] + 0.01        # the filter is at least as good as any window


def byte_markov_hmm(src: ByteMarkov) -> BitHMM:
    """Independent bit-level HMM of a byte chain: states (phase f, previous letter, current letter)."""
    m = len(src.letters)
    lb = src._letter_bits()
    idx = lambda f, a, b: (f * m + a) * m + b
    T = np.zeros((2, 8 * m * m, 8 * m * m))
    for f, a, b in itertools.product(range(8), range(m), range(m)):
        bit = lb[b, f]
        if f < 7:
            T[bit, idx(f, a, b), idx(f + 1, a, b)] = 1.0
        else:
            for c in range(m):
                T[bit, idx(f, a, b), idx(0, b, c)] = src.P[b, c]
    return BitHMM(T)


@pytest.mark.parametrize("m,ks", [(16, range(1, 6)), (4, (8, 11, 12))])
def test_byte_markov_floor_matches_an_independent_bit_hmm(m, ks):
    src = ByteMarkov.random(seed=7, letters=bytes(range(97, 97 + m)))
    hmm = byte_markov_hmm(src)
    for k in ks:
        assert np.isclose(src.window_floor(k), hmm.window_floor(k), atol=1e-10)


def test_byte_markov_oracle_multiplies_to_the_letter_probabilities(rng):
    src = ByteMarkov.random()
    sample = src.sample(8 * 20_000, rng)
    per_letter = sample.oracle_q.reshape(-1, 8).prod(axis=1)
    letters = np.packbits(sample.bits)
    idx = letters - src.letters[0]
    assert np.allclose(per_letter[1:], src.P[idx[:-1], idx[1:]], atol=1e-12)
    assert np.isclose(per_letter[0], src.stationary[idx[0]])
    se = np.std(-np.log2(sample.oracle_q).reshape(-1, 8).sum(axis=1)) / np.sqrt(len(idx)) / 8
    assert abs(sample.oracle_loss - src.entropy_rate()) < 5 * se


def test_byte_markov_floors_are_monotone_and_above_the_rate():
    src = ByteMarkov.random()
    floors = [src.window_floor(k) for k in range(1, 17)]
    assert all(a >= b - 1e-12 for a, b in itertools.pairwise(floors))
    assert floors[-1] >= src.entropy_rate() - 1e-12


def test_delayed_echo_floor_matches_its_hmm(rng):
    src = DelayedEcho(d=3, p=0.1)
    hmm = src.hmm()
    for k in range(0, 6):
        assert np.isclose(src.window_floor(k), hmm.window_floor(k), atol=1e-12)
    sample = src.sample(100_001, rng)
    assert np.mean(sample.bits[3:] != sample.bits[:-3]) == pytest.approx(0.1, abs=0.005)
    assert np.array_equal(sample.oracle_q[3:] == 0.1, sample.bits[3:] != sample.bits[:-3])


def test_conditional_entropy_of_a_table():
    joint = np.array([[0.25, 0.25], [0.5, 0.0]])
    assert np.isclose(conditional_entropy(joint), 0.5)


def test_contexts_hold_the_previous_bits(rng):
    bits = rng.integers(0, 2, 200).astype(np.uint8)
    s = contexts(bits, 5)
    for t in range(len(bits)):
        expected = sum(int(bits[t - j]) << (j - 1) for j in range(1, 6) if t - j >= 0)
        assert s[t] == expected


def test_kt_equals_a_naive_loop(rng):
    bits = rng.integers(0, 2, 3000).astype(np.uint8)
    for k in (0, 3, 7):
        counts = np.zeros((1 << k, 2))
        q_ref = []
        for t, s in enumerate(contexts(bits, k)):
            c = counts[s]
            p1 = (c[1] + 0.5) / (c.sum() + 1.0)
            q_ref.append(p1 if bits[t] else 1 - p1)
            counts[s, bits[t]] += 1
        assert np.allclose(kt_context_q(bits, k), q_ref, atol=1e-14)
