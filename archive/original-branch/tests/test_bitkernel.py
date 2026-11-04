"""Acceptance tests for vam.bitkernel: the massless, einselected VAM-1 on packed bits (F-Part 8).

In that limit the chain is an exact sliding window, every context sector is
either certain or empty, and the canalization algebra says only the last
``k`` bits matter (tests/test_vam1.py). The bit kernel keeps exactly that:
one integer register of the last ``k`` bits (newest bit in bit 0) and the
weights. It must reproduce the dense and pointer engines exactly.
"""

import time

import numpy as np
import pytest

from vam.baselines import contexts
from vam.bitkernel import NUMBA_AVAILABLE, BitMachine
from vam.bits import bits_to_bytes, bytes_to_bits
from vam.machine import macro_micro_step
from vam.realizations.vam1 import Config, DenseMaps, PointerEngine


@pytest.mark.parametrize("k,c,cons", [(1, 1, False), (2, 3, False), (3, 3, True), (2, 2, True)])
def test_equals_the_pointer_engine(rng, k, c, cons):
    cfg = Config(cells=c, context=k, consolidating=cons, dt=0.1)
    bits = rng.integers(0, 2, 300).astype(np.uint8)
    pe = PointerEngine(cfg)
    q_ref = np.array([pe.step(int(b)) for b in bits])
    bm = BitMachine(cfg)
    q = bm.run(bits)
    assert np.allclose(q, q_ref, atol=1e-12)
    assert np.allclose(bm.w, pe.w, atol=1e-12) and np.allclose(bm.n, pe.n, atol=1e-12)


def test_equals_the_dense_coherent_engine(rng):
    cfg = Config(cells=3, context=2, dephasing=0.0)          # no einselection at all: still exact when massless
    bits = [int(b) for b in rng.integers(0, 2, 80)]
    m = DenseMaps(cfg).reader()
    q_ref = []
    for b in bits:
        q_ref.append(m.predict(b))
        m.step(b, rng)
    bm = BitMachine(cfg)
    assert np.allclose([bm.step(b) for b in bits], q_ref, atol=1e-12)
    assert np.allclose(bm.w, m.state.structural["w"], atol=1e-12)


def test_rejects_a_massive_field():
    with pytest.raises(ValueError):
        BitMachine(Config(cells=2, context=2, theta=0.3))


def test_step_and_run_agree(rng):
    cfg = Config(cells=8, context=8, consolidating=True)
    bits = rng.integers(0, 2, 5000).astype(np.uint8)
    a, b = BitMachine(cfg), BitMachine(cfg)
    qa = np.array([a.step(int(x)) for x in bits])
    qb = b.run(bits)
    assert np.allclose(qa, qb, atol=1e-13)
    assert np.allclose(a.w, b.w, atol=1e-13) and a.context == b.context


def test_compiled_and_reference_paths_agree(rng):
    cfg = Config(cells=10, context=10, consolidating=True)
    bits = rng.integers(0, 2, 20_000).astype(np.uint8)
    fast, slow = BitMachine(cfg), BitMachine(cfg, use_numba=False)
    assert np.allclose(fast.run(bits), slow.run(bits), atol=1e-13)
    assert np.allclose(fast.w, slow.w, atol=1e-13)


def test_register_holds_the_last_k_bits(rng):
    cfg = Config(cells=6, context=6)
    bm = BitMachine(cfg)
    bits = [int(x) for x in rng.integers(0, 2, 40)]
    bm.run(np.array(bits, dtype=np.uint8))
    expected = sum(b << j for j, b in enumerate(reversed(bits[-6:])))
    assert bm.context == expected


def test_bits_per_byte(rng):
    cfg = Config(cells=12, context=12, consolidating=True)
    data = bytes(rng.integers(0, 256, 2000, dtype=np.uint8))
    bm = BitMachine(cfg)
    bpb = bm.bits_per_byte(data)
    q = BitMachine(cfg).run(bytes_to_bits(data))
    assert np.isclose(bpb, -np.log2(q).sum() / len(data))
    assert 7.5 < bpb < 8.6                          # random bytes cannot be compressed


def test_learns_a_deterministic_stream_down_to_its_window_floor():
    """With consolidating friction and dt = gamma/kappa the prior weighs one observation, so every
    context with a single continuation becomes certain (error falling like 1/n). Six positions per
    sentence share their 16-bit context with a different continuation ("e q" against "e l", and two
    more), and alternate, so any 16-bit window pays about one bit at each of them."""
    cfg = Config(cells=16, context=16, consolidating=True, dt=1.0)
    text = b"the quick brown fox jumps over the lazy dog. " * 200
    bits = bytes_to_bits(text)
    period = 8 * 45
    ctx = contexts(bits, 16)[period:2 * period]
    nxt = bits[period:2 * period]
    ambiguous = sum(int((ctx == c).sum()) for c in np.unique(ctx) if len(set(nxt[ctx == c])) > 1)
    assert ambiguous == 6
    bm = BitMachine(cfg)
    q = bm.run(bits)
    per_sentence = -np.log2(q).reshape(200, -1).sum(axis=1)
    assert per_sentence[-20:].mean() < ambiguous + 1.0
    assert per_sentence[-20:].mean() < per_sentence[0] / 40


def test_state_size_is_constant(rng):
    cfg = Config(cells=12, context=12)
    bm = BitMachine(cfg)
    size = bm.nbytes()
    for _ in range(5):
        bm.run(rng.integers(0, 2, 10_000).astype(np.uint8))
        assert bm.nbytes() == size


def test_continuation_distribution_equals_the_machine(rng):
    cfg = Config(cells=3, context=3, dt=0.3)
    reader = DenseMaps(cfg).reader()
    bm = BitMachine(cfg)
    for b in rng.integers(0, 2, 60):
        reader.step(int(b), rng)
        bm.step(int(b))
    gen = DenseMaps(cfg).generator(reader.state)
    ref = gen.output_distribution([None] * 4)
    dist = bm.continuations(4)
    assert set(dist) == set(ref)
    for key, p in ref.items():
        assert np.isclose(dist[key], p, atol=1e-12)


def test_generation_matches_macro_micro_and_does_not_learn(rng):
    pattern = [0, 1, 1, 0, 1, 0, 0, 0]
    cfg = Config(cells=4, context=4, dt=0.2)
    reader = DenseMaps(cfg).reader()
    bm = BitMachine(cfg)
    for b in pattern * 40:
        reader.step(b, rng)
        bm.step(b)
    gen = DenseMaps(cfg).generator(reader.state)
    ref = [macro_micro_step(gen, horizon=3).first for _ in range(16)]
    w_before = bm.w.copy()
    out = bm.generate(16, horizon=3)
    assert out == ref == pattern * 2
    assert np.array_equal(bm.w, w_before)


def test_sampled_generation_is_reproducible(rng):
    cfg = Config(cells=8, context=8)
    bm = BitMachine(cfg)
    bm.run(rng.integers(0, 2, 3000).astype(np.uint8))
    a, b = bm.copy(), bm.copy()
    assert a.generate(64, horizon=4, select="sample", rng=np.random.default_rng(3)) == \
        b.generate(64, horizon=4, select="sample", rng=np.random.default_rng(3))


def test_generated_bits_pack_into_bytes():
    cfg = Config(cells=24, context=24, consolidating=True)
    text = b"abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc" * 30
    bm = BitMachine(cfg)
    bm.run(bytes_to_bits(text))
    out = bits_to_bytes(np.array(bm.generate(8 * 12, horizon=8), dtype=np.uint8))
    assert out == b"abcabcabcabc"


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="throughput target applies to the compiled kernel")
def test_measured_throughput(rng):
    cfg = Config(cells=22, context=22, consolidating=True)
    bits = rng.integers(0, 2, 8_000_000).astype(np.uint8)
    bm = BitMachine(cfg)
    bm.run(bits[:1000])                             # compile
    t0 = time.perf_counter()
    bm.run(bits)
    rate = len(bits) / (time.perf_counter() - t0)
    # events (observe + learn + inject) per second, end to end in a fresh process on a shared machine;
    # warm runs reach about 4e7, and the first touch of the 64 MB output costs the rest
    assert rate > 1e7
