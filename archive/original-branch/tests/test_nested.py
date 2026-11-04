"""Nested context levels (T14 step 14, decision D9): isometric nesting, coarse-to-fine readout, storage."""

import math
import time

import numpy as np
import pytest

from vam.baselines import contexts
from vam.bitkernel import BitMachine
from vam.bits import bytes_to_bits
from vam.nested import (
    EV,
    NUMBA_AVAILABLE,
    NestedConfig,
    NestedMachine,
    NestedReference,
    _learn,
    _w_of,
    coherent_mix,
    context_hash,
)
from vam.realizations.vam1 import Config
from vam.sources import MarkovBits

pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="the compiled nested machine needs numba")

CONS = Config(cells=1, context=1, consolidating=True, dt=1.0)
CONST = Config(cells=1, context=1, consolidating=False, dt=0.05)


def vam_learn(w, n, b):
    return _learn(CONS, w, n, b)


def ref_run(cfg, bits, learning=True):
    m = NestedReference(cfg)
    return np.array([m.step(int(b), learning) for b in bits]), m


@pytest.mark.parametrize("law", [CONS, CONST])
def test_one_level_is_the_bit_kernel(rng, law):
    bits = rng.integers(0, 2, 20_000).astype(np.uint8)
    k = 9
    nm = NestedMachine(NestedConfig(law, depths=(k,), table_bits=12, mature=0))
    bm = BitMachine(Config(cells=k, context=k, consolidating=law.consolidating, dt=law.dt))
    assert np.allclose(nm.run(bits), bm.run(bits), atol=1e-13)
    assert np.allclose(nm.level_table(0)[:, 0], bm.w, atol=1e-13)


def test_without_inheritance_every_level_is_its_own_bit_kernel(rng):
    bits = rng.integers(0, 2, 8000).astype(np.uint8)
    depths = (3, 6, 10)
    nm = NestedMachine(NestedConfig(CONS, depths=depths, table_bits=12, readout="mature", inherit=False))
    q = nm.run(bits)
    for lv, d in enumerate(depths):
        bm = BitMachine(Config(cells=d, context=d, consolidating=True, dt=1.0))
        q_d = bm.run(bits)
        assert np.allclose(nm.level_table(lv)[:, 0], bm.w, atol=1e-13)
    # the "mature" readout with mature = 0 uses the finest level whenever its sector was seen before
    ctx = contexts(bits, depths[-1])
    seen = np.zeros(len(bits), dtype=bool)
    visited = set()
    for t, c in enumerate(ctx):
        seen[t] = c in visited
        visited.add(c)
    assert np.allclose(q[seen], q_d[seen], atol=1e-13) and seen.mean() > 0.8


CASES = [(CONS, "mixture", 0, True, 0.0), (CONST, "mature", 1, True, 0.0), (CONS, "mature", 0, False, 0.0),
         (CONS, "mixture", 3, True, 4.0)]


@pytest.mark.parametrize("law,readout,mature,inherit,prior", CASES)
def test_compiled_machine_equals_the_exact_reference_until_the_first_eviction(rng, law, readout, mature, inherit,
                                                                              prior):
    cfg = NestedConfig(law, depths=(2, 5, 9, 17, 33, 64), table_bits=22, readout=readout, mature=mature,
                       inherit=inherit, prior=prior)
    bits = MarkovBits().sample(6000, rng).bits
    q_ref, _ = ref_run(cfg, bits)
    nm = NestedMachine(cfg)
    q = nm.run(bits)
    upto = len(bits) if nm.first_eviction < 0 else nm.first_eviction
    assert upto > 100
    assert np.allclose(q[:upto], q_ref[:upto], atol=1e-12)


@pytest.mark.parametrize("law,readout,mature,inherit,prior", CASES)
def test_compiled_machine_equals_the_table_reference_with_forgetting(rng, law, readout, mature, inherit, prior):
    cfg = NestedConfig(law, depths=(2, 5, 9, 17, 33, 64), table_bits=9, readout=readout, mature=mature,
                       inherit=inherit, prior=prior)
    bits = MarkovBits().sample(6000, rng).bits
    ref = NestedReference(cfg, hashed=True)
    q_ref = np.array([ref.step(int(b)) for b in bits])
    nm = NestedMachine(cfg)
    q = nm.run(bits[:2500])
    q = np.concatenate([q, nm.run(bits[2500:])])           # state carries across calls
    assert nm.evictions == ref.evictions > 1000
    assert np.allclose(q, q_ref, atol=1e-12)
    assert int(nm.reg) == ref.reg
    for b in (0, 1):
        assert np.isclose(nm.predict(b), ref.predict(b), atol=1e-12)


def test_python_hash_matches_the_compiled_tables(rng):
    cfg = NestedConfig(CONS, depths=(4, 40), table_bits=10, mature=0)
    nm = NestedMachine(cfg)
    bits = rng.integers(0, 2, 300).astype(np.uint8)
    nm.run(bits)
    reg = 0
    for b in bits:
        reg = ((reg << 1) | int(b)) & ((1 << 64) - 1)
    # the context just completed has not been visited yet; the one before it has
    prev = (reg >> 1) & ((1 << 40) - 1)
    h = context_hash(prev, 1)
    assert int(nm._tags[nm._hidx[1], h >> (64 - 10)]) == ((h & 0xFFFFFFFF) | 1)


@pytest.mark.parametrize("law", [CONS, CONST])
def test_a_new_distinction_starts_as_the_coarser_readout(rng, law):
    """Isometric nesting (T11.4): a new fine sector's own prediction is the coarser readout at that moment
    (the mixture of the levels above it), its evidence is zero, and it then learns the current bit."""
    bits = rng.integers(0, 2, 4000).astype(np.uint8)
    m = NestedReference(NestedConfig(law, depths=(2, 5, 11), table_bits=22))
    checked = 0
    for b in bits:
        keys, entries = m._lookup()
        chain = m._chain(entries)
        new = entries[2] is None and entries[1] is not None
        n_parent = entries[1][1] if new else None
        m.step(int(b))
        if new:
            fine = m.tables[2][keys[2][0]]
            w, n = _learn(law, _w_of(law, chain[1]), min(n_parent, 0.0), int(b))
            assert fine[0] == pytest.approx(w, abs=1e-15) and fine[1] == n and fine[2] == 0.0
            checked += 1
    assert checked > 500


def test_creating_a_distinction_leaves_the_prediction_unchanged(rng):
    """Adding the new sector, before any evidence reaches it, gives the machine exactly the prediction it
    had without it, whatever the coarser levels' evidence (this is what failed when a new sector copied
    its parent's raw relation instead of the mixed readout)."""
    bits = rng.integers(0, 2, 6000).astype(np.uint8)
    m = NestedReference(NestedConfig(CONS, depths=(3, 8, 20), table_bits=22))
    checked, moved = 0, 0.0
    for t, b in enumerate(bits):
        keys, entries = m._lookup()
        if t > 2000 and entries[1] is not None and entries[2] is None:
            chain = m._chain(entries)
            with_new = m._chain([*entries[:2], [_w_of(CONS, chain[1]), 0.0, 0.0, keys[2][1]]])
            assert with_new[-1] == pytest.approx(chain[-1], abs=1e-15)
            raw_copy = m._chain([*entries[:2], [entries[1][0], 0.0, 0.0, keys[2][1]]])
            moved = max(moved, abs(raw_copy[-1] - chain[-1]))
            checked += 1
        m.step(int(b))
    assert checked > 100
    assert moved > 1e-3                    # negative control: copying the parent's raw relation moves it


def test_storage_is_fixed_and_forgetting_is_counted(rng):
    cfg = NestedConfig(CONS, depths=(4, 16, 32), table_bits=8, mature=1)
    nm = NestedMachine(cfg)
    size = nm.nbytes()
    for _ in range(4):
        q = nm.run(rng.integers(0, 2, 20_000).astype(np.uint8))
        assert nm.nbytes() == size
        assert np.all((q >= cfg.law.eps - 1e-15) & (q <= 1 - cfg.law.eps + 1e-15))
    assert nm.evictions > 0


def test_frozen_reading_changes_no_table(rng):
    nm = NestedMachine(NestedConfig(CONS, depths=(4, 12, 40), table_bits=10))
    nm.run(rng.integers(0, 2, 5000).astype(np.uint8))
    wn, tags = nm._wn.copy(), nm._tags.copy()
    nm.run(rng.integers(0, 2, 5000).astype(np.uint8), learning=False)
    assert np.array_equal(nm._wn, wn) and np.array_equal(nm._tags, tags)


def test_deep_levels_resolve_what_a_short_window_cannot():
    """A 16-bit window pays about 6.8 bits per sentence on this stream (tests/test_bitkernel.py): six
    positions share their window with a different continuation. Deeper levels tell "e quick" from
    "e lazy d", so only the slowly vanishing learning error remains."""
    text = b"the quick brown fox jumps over the lazy dog. " * 200
    bits = bytes_to_bits(text)
    nm = NestedMachine(NestedConfig(CONS, depths=(8, 16, 32, 64), table_bits=16))
    per_sentence = -np.log2(nm.run(bits)).reshape(200, -1).sum(axis=1)
    window = -np.log2(BitMachine(Config(cells=16, context=16, consolidating=True, dt=1.0)).run(bits))
    assert per_sentence[-20:].mean() < 1.5
    assert per_sentence[-20:].mean() < window.reshape(200, -1).sum(axis=1)[-20:].mean() - 4.0


def test_deep_levels_cost_little_on_a_short_memory_source(rng):
    """Levels deeper than the source's memory carry no information; the mixture learns to give them
    little weight, so the loss stays near the entropy rate (the 16-bit window alone is worse)."""
    src = MarkovBits()
    bits = src.sample(1 << 20, rng).bits
    last = lambda q: -np.log2(q[3 * len(q) // 4:]).mean()
    nested = last(NestedMachine(NestedConfig(CONS, depths=(2, 4, 8, 16), table_bits=16)).run(bits))
    window = last(BitMachine(Config(cells=16, context=16, consolidating=True, dt=1.0)).run(bits))
    assert nested < src.entropy_rate() + 0.01
    assert nested < window


def test_a_prior_slows_a_new_sector_without_moving_it(rng):
    """The prior caps the inherited activity count at ``prior``; the inherited prediction is unchanged."""
    bits = rng.integers(0, 2, 4000).astype(np.uint8)
    m = NestedReference(NestedConfig(CONS, depths=(2, 6), table_bits=8, prior=3.0))
    checked = 0
    for b in bits:
        fine_ctx, coarse_ctx = m.reg & 0x3F, m.reg & 0x3
        new = fine_ctx not in m.tables[1] and coarse_ctx in m.tables[0]
        chain = m._chain(m._entries())
        coarse_before = list(m.tables[0].get(coarse_ctx, [0.0, 0.0]))
        m.step(int(b))
        if new:
            w, n = vam_learn(_w_of(CONS, chain[0]), min(coarse_before[1], 3.0), int(b))
            assert np.isclose(m.tables[1][fine_ctx][0], w) and m.tables[1][fine_ctx][1] == n
            checked += 1
    assert checked > 20


def test_continuations_are_the_machines_own_predictions(rng):
    nm = NestedMachine(NestedConfig(CONST, depths=(3, 7, 20), table_bits=10))
    nm.run(rng.integers(0, 2, 4000).astype(np.uint8))
    dist = nm.continuations(3)
    assert np.isclose(sum(dist.values()), 1.0)
    for key, p in dist.items():
        m = nm.copy()
        prob = 1.0
        for t in key:
            prob *= m.predict(t)
            m.run(np.array([t], dtype=np.uint8), learning=False)
        assert np.isclose(p, prob, atol=1e-14)


def test_generation_continues_a_learned_text_and_does_not_learn():
    text = b"one two three four. " * 60
    nm = NestedMachine(NestedConfig(CONS, depths=(8, 16, 32, 64), table_bits=14))
    nm.run(bytes_to_bits(text))
    before = nm._wn.copy()
    out = np.packbits(np.array(nm.generate(8 * 20, horizon=8), dtype=np.uint8)).tobytes()
    assert out == b"one two three four. "
    assert np.array_equal(nm._wn, before)


def test_rejects_bad_configurations():
    with pytest.raises(ValueError):
        NestedConfig(CONS, depths=(8, 8))
    with pytest.raises(ValueError):
        NestedConfig(CONS, depths=(24, 32), table_bits=20)
    with pytest.raises(ValueError):
        NestedConfig(Config(cells=2, context=2, theta=0.2), depths=(2,))
    with pytest.raises(ValueError):
        NestedConfig(CONS, readout="softmax")


def test_measured_throughput(rng):
    nm = NestedMachine(NestedConfig(CONS, depths=(8, 16, 24, 32, 48, 64), table_bits=22))
    bits = rng.integers(0, 2, 4_000_000).astype(np.uint8)
    nm.run(bits[:1000])
    t0 = time.perf_counter()
    nm.run(bits)
    rate = len(bits) / (time.perf_counter() - t0)
    # six levels per bit, each with a table access, a hash and the law's exponential: about 2e6 measured
    assert rate > 1.5e6


# ---------------------------------------------------------------------------
# Layers 3, 5 and 6 of the nested fractal architecture (spec N)
# ---------------------------------------------------------------------------

LAYERS = [dict(coherence=0.5), dict(coherence=1.0), dict(verify=2), dict(verify=3, door_bits=6),
          dict(governor=True), dict(governor=True, door_bits=4),
          dict(governor=True, verify=2, coherence=0.7, protect=0.5, sink_rate=0.5)]


@pytest.mark.parametrize("kw", LAYERS, ids=lambda kw: "-".join(f"{k}{v}" for k, v in kw.items()))
def test_layers_compiled_equals_the_table_reference(rng, kw):
    cfg = NestedConfig(CONS, depths=(2, 5, 9, 17, 33, 64), table_bits=9, prior=4.0, **kw)
    bits = MarkovBits().sample(6000, rng).bits
    ref = NestedReference(cfg, hashed=True)
    q_ref = np.array([ref.step(int(b)) for b in bits])
    nm = NestedMachine(cfg)
    q = np.concatenate([nm.run(bits[:2500]), nm.run(bits[2500:])])
    assert np.allclose(q, q_ref, atol=1e-12)
    assert (nm.evictions, nm.refused, nm.protected) == (ref.evictions, ref.refused, ref.protected)
    for b in (0, 1):
        assert np.isclose(nm.predict(b), ref.predict(b), atol=1e-12)
    if kw.get("verify", 1) > 1 or kw.get("governor"):
        assert nm.refused > 0


# The machine before N (commit bfc1e1b, the one E1 and E2 ran), on 60,000 bits of MarkovBits drawn with
# seed 20261022, depths (2, 9, 33), prior 4: bits per bit while learning, bits per bit of a frozen copy on
# the first 20,000 bits, the last prediction, evictions and the first eviction. Computed by running that
# commit's own package; the current code gives the same values bit for bit.
PRE_N = [
    ("CONS", 8, 0.6537889551267188, 0.6543979652037425, 0.8970552176831675, 68893, 37),
    ("CONS", 22, 0.6529990375107724, 0.5532545209992804, 0.8970552176831675, 268, 2535),
    ("CONST", 8, 0.6651323446822267, 0.6578787363384002, 0.845582984951714, 68893, 37),
    ("CONST", 22, 0.6630978299525624, 0.6294195541611673, 0.845582984951714, 268, 2535),
]


@pytest.mark.parametrize("law, table_bits, learn_bpb, frozen_bpb, q_last, evictions, first", PRE_N,
                         ids=lambda v: str(v) if isinstance(v, (str, int)) else None)
def test_defaults_leave_the_machine_unchanged(law, table_bits, learn_bpb, frozen_bpb, q_last, evictions, first):
    """With every N option off the machine is the one E1 and E2 ran (values from the pre-N commit)."""
    bits = MarkovBits().sample(60000, np.random.default_rng(20261022)).bits
    cfg = NestedConfig({"CONS": CONS, "CONST": CONST}[law], depths=(2, 9, 33), table_bits=table_bits, prior=4.0)
    assert (cfg.coherence, cfg.verify, cfg.governor) == (0.0, 1, False)
    m = NestedMachine(cfg)
    q = m.run(bits)
    frozen = m.copy().run(bits[:20000], learning=False)
    assert float(-np.log2(q).mean()) == pytest.approx(learn_bpb, rel=1e-12)
    assert float(-np.log2(frozen).mean()) == pytest.approx(frozen_bpb, rel=1e-12)
    assert float(q[-1]) == pytest.approx(q_last, rel=1e-12)
    assert (m.evictions, m.first_eviction) == (evictions, first)


def test_coherent_mix_is_the_two_hypothesis_density_readout():
    """T_b = Tr(P_b rho) with rho = (1-c) diag(1-s, s) + c |a><a| and P_b = |phi_b><phi_b|, normalized."""
    for p, o, s, c in [(0.5, 0.99, 0.5, 1.0), (0.2, 0.7, 0.3, 0.4), (0.9, 0.1, 0.8, 0.75)]:
        a = np.sqrt([1 - s, s])
        rho = (1 - c) * np.diag([1 - s, s]) + c * np.outer(a, a)
        assert np.all(np.linalg.eigvalsh(rho) > -1e-15) and np.isclose(np.trace(rho), 1.0)
        T = [np.sqrt([p, o]) @ rho @ np.sqrt([p, o]), np.sqrt([1 - p, 1 - o]) @ rho @ np.sqrt([1 - p, 1 - o])]
        assert coherent_mix(p, o, s, c) == pytest.approx(T[0] / sum(T), abs=1e-15)
    assert coherent_mix(0.3, 0.8, 0.4, 0.0) == pytest.approx(0.3 + 0.4 * 0.5, abs=1e-15)     # c = 0: the mixture
    assert coherent_mix(0.5, 0.99, 0.5, 1.0) > coherent_mix(0.5, 0.99, 0.5, 0.0)            # agreement reinforces
    assert coherent_mix(0.3, 0.8, 0.4, 0.6) == pytest.approx(1 - coherent_mix(0.7, 0.2, 0.4, 0.6), abs=1e-15)


@pytest.mark.parametrize("c", [0.5, 1.0])
def test_coherent_nesting_stays_isometric(rng, c):
    """A new sector whose prediction is the coarser readout leaves the coherent readout unchanged."""
    cfg = NestedConfig(CONS, depths=(3, 8, 20), table_bits=22, coherence=c)
    bits = rng.integers(0, 2, 6000).astype(np.uint8)
    m = NestedReference(cfg)
    checked = 0
    for t, b in enumerate(bits):
        keys, entries = m._lookup()
        if t > 2000 and entries[1] is not None and entries[2] is None:
            chain = m._chain(entries)
            with_new = m._chain([*entries[:2], [_w_of(CONS, chain[1]), 0.0, 0.0, keys[2][1]]])
            assert with_new[-1] == pytest.approx(chain[-1], abs=1e-15)
            checked += 1
        m.step(int(b))
    assert checked > 100


def test_a_one_off_context_takes_no_slot(rng):
    """Verified growth (N-L5): a 40-bit context of random bits never repeats, so with verify=2 the hashed
    level stays empty in the exact reference; repeated text is admitted at its second sighting."""
    cfg = NestedConfig(CONS, depths=(4, 40), table_bits=10, verify=2)
    m = NestedReference(cfg)
    stream = np.concatenate([rng.integers(0, 2, 3000), bytes_to_bits(b"verified patterns grow the memory. " * 2)])
    visits: dict[int, int] = {}
    for t, b in enumerate(stream):
        visits[m.reg & ((1 << 40) - 1)] = visits.get(m.reg & ((1 << 40) - 1), 0) + 1
        m.step(int(b))
        if t == 2999:
            assert m.occupancy(1) == 0 and m.refused > 2900
    assert m.occupancy(1) == sum(1 for v in visits.values() if v >= 2) > 200


def test_verified_patterns_survive_noise(rng):
    """Learn a sentence in a small table, then read random bits. Without the layers the noise evicts every
    deep sector of the sentence; with verified growth or the governor almost all stay."""
    text = b"the quick brown fox jumps over the lazy dog. "
    noise = rng.integers(0, 2, 60_000).astype(np.uint8)
    ctxs, reg = set(), 0
    for b in bytes_to_bits(text * 3):
        reg = ((reg << 1) | int(b)) & ((1 << 64) - 1)
        ctxs.add(reg)
    kept = {}
    for name, kw in {"plain": {}, "verify": {"verify": 2}, "governor": {"governor": True}}.items():
        nm = NestedMachine(NestedConfig(CONS, depths=(8, 40, 64), table_bits=9, **kw))
        nm.run(bytes_to_bits(text * 100))
        before = np.mean([nm.stored(c, 2) for c in ctxs])
        nm.run(noise)
        kept[name] = (before, np.mean([nm.stored(c, 2) for c in ctxs]))
    assert kept["plain"][0] > 0.5 and kept["plain"][1] < 0.02
    for name in ("verify", "governor"):
        assert kept[name][0] > 0.5 and kept[name][1] > 0.97 * kept[name][0]


def test_governor_halts_growth_at_half_and_drains_protected_sectors(rng):
    """Below half of its slots a hashed level admits at first sight; from half on a newcomer needs a second
    sighting, and a newcomer knocking on a protected sector (ev > protect) multiplies its evidence by
    exp(-2 rate): the sink's eigenvalue is Re lambda = -2 rate."""
    cfg = NestedConfig(CONS, depths=(4, 33), table_bits=8, governor=True, protect=0.5, sink_rate=0.75)
    m = NestedReference(cfg, hashed=True)
    bits = MarkovBits().sample(40_000, rng).bits
    knocks = 0
    for b in bits:
        before = {k: v[EV] for k, v in m.tables[1].items()}
        governed = m.governed(1)
        refused, protected = m.refused, m.protected
        m.step(int(b))
        if not governed:
            assert m.refused == refused
        if m.protected > protected:
            changed = [k for k in before if k in m.tables[1] and m.tables[1][k][EV] != before[k]]
            ratios = [m.tables[1][k][EV] / before[k] for k in changed if before[k] > cfg.protect]
            assert any(r == pytest.approx(math.exp(-2 * 0.75), rel=1e-12) for r in ratios)
            knocks += 1
    assert m.governed(1) and m.refused > 0 and knocks > 0
    assert 0.5 <= m.occupancy(1) / 256 <= 1.0
