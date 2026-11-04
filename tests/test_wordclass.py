"""E6b (PROTOCOL E6b.5): the class store, the class map, the K levels and the gates, on small machines."""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from vam.bits import bytes_to_bits
from vam.nested import NestedConfig
from vam.nfa import hier, wcprepass
from vam.nfa import wordclass as WC
from vam.realizations.vam1 import Config

SPEC = Path(__file__).resolve().parents[1] / "spec"
LAW = Config(cells=1, context=1, consolidating=True, dt=1.0)


@pytest.fixture(scope="module")
def text() -> bytes:
    return (b" ".join(p.read_bytes() for p in sorted(SPEC.glob("*.md"))) * 3)[:420000]


def small() -> hier.HierConfig:
    cfg = NestedConfig(LAW, depths=(8, 16, 24, 32, 48, 64), table_bits=13, prior=4.0, coherence=1.0, verify=3,
                       governor=True, protect=0.5, sink_rate=0.0)
    return hier.HierConfig(cfg, upper_bits=11, upper_verify=4, segment="text")


def kmachine(mode: str, **kw) -> WC.KMachine:
    base = small()
    upper = base.upper if mode == "off" else WC.K_ORDER
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(hier.HierConfig)}
    fields["upper"] = upper
    wc = WC.WordClassConfig(mode=mode, row_bits=12, feats=64, V=2048, C=16, d=16, iters=6, min_count=2,
                            compile_bytes=1 << 17, **kw)
    return WC.KMachine(WC.KHierConfig(**fields), wc)


# -- the pre-pass is E4's plus the E6b lines ------------------------------------------------------------------------

def strip_e6b(src: str) -> str:
    out, skip = [], False
    for line in src.splitlines():
        if "# E6b begin" in line:
            skip = True
        if not skip:
            out.append(line.replace("  # E6b signature", ""))
        if "# E6b end" in line:
            skip = False
    return "\n".join(out)


def test_the_k_prepass_is_the_e4_prepass_plus_the_e6b_lines():
    a = inspect.getsource(hier._make_prepass)
    b = inspect.getsource(wcprepass.make_prepass_k)
    a = re.sub(r'""".*?"""', '""""""', a, flags=re.S)
    b = strip_e6b(re.sub(r'""".*?"""', '""""""', b, flags=re.S))
    a = a.replace("def _make_prepass(jit):", "def make_prepass_k(jit):")
    b = b.replace("text, comm,\n                  rbits, ctag, ccls, codes, ncodes, cunk):", "text, comm):")
    assert a.split() == b.split()


# -- gate (a): K off is E4 bit for bit ---------------------------------------------------------------------------

def test_k_off_equals_e4_bit_for_bit_learning_then_frozen(text):
    bits = bytes_to_bits(text[:150000])
    ref, off = hier.HierMachine(small()), kmachine("off")
    assert np.array_equal(ref.run(bits, learning=True), off.run(bits, learning=True))
    more = bytes_to_bits(text[150000:200000])
    assert np.array_equal(ref.copy().run(more, learning=False), off.copy().run(more, learning=False))
    for name in ("_wn", "_tags", "_door", "_occ"):
        assert np.array_equal(getattr(ref, name), getattr(off, name))


def test_the_e4_keys_of_the_k_prepass_equal_hiers(text):
    bits = bytes_to_bits(text[:40000])
    kinds, dups = np.arange(5, dtype=np.int64), np.zeros(5, dtype=np.int64)
    e1 = np.empty((bits.shape[0], 5), dtype=np.uint64)
    e2 = np.empty_like(e1)
    st1 = np.array(hier.initial_state(), dtype=np.uint64)
    st2 = np.array(hier.initial_state() + [WC.UNK] * 4, dtype=np.uint64)
    hier._fill_keys(bits, st1, e1, kinds, dups, True, False)
    codes, nc = np.empty(len(bits), dtype=np.uint64), np.zeros(1, dtype=np.int64)
    WC._fill_keys_k(bits, st2, e2, kinds, dups, True, False, 4, np.zeros(16, np.uint64), np.full(16, WC.NOMAP,
                    np.uint16), codes, nc, WC.UNK)
    assert np.array_equal(e1, e2) and np.array_equal(st1, st2[:11])


# -- gate (b): the K keys equal the pure Python writing ------------------------------------------------------------

def synthetic_map(data: bytes, row_bits: int, C: int = 16) -> WC.ClassMap:
    codes = WC.unit_codes(np.frombuffer(data, dtype=np.uint8))
    uniq = np.unique(codes)
    rng = np.random.default_rng(0)
    pick = rng.choice(uniq, size=min(len(uniq), 300), replace=False)
    rows = 1 << row_bits
    tag, cls = np.zeros(rows, dtype=np.uint64), np.full(rows, WC.NOMAP, dtype=np.uint16)
    for c in pick:
        r = int(c) >> (64 - row_bits)
        tag[r] = c
        cls[r] = int(rng.integers(C))
    return WC.ClassMap(tag, cls, np.flatnonzero(cls != WC.NOMAP))


def test_k_keys_equal_the_python_writing_and_unit_codes_are_the_units(text):
    data = text[:3000]
    cmap = synthetic_map(data, 8)
    bits = bytes_to_bits(data)
    kinds = np.array([WC.K1, WC.K2, WC.K4], dtype=np.int64)
    ext = np.empty((bits.shape[0], 3), dtype=np.uint64)
    st = np.array(hier.initial_state() + [WC.UNK] * 4, dtype=np.uint64)
    codes, nc = np.empty(len(bits), dtype=np.uint64), np.zeros(1, dtype=np.int64)
    WC._fill_keys_k(bits, st, ext, kinds, np.zeros(3, dtype=np.int64), True, False, 8, cmap.tag, cmap.cls, codes, nc,
                    WC.UNK)
    ref = hier.initial_state() + [WC.UNK] * 4
    for i in range(len(bits)):
        for j, lv in enumerate((WC.K1, WC.K2, WC.K4)):
            assert int(ext[i, j]) == WC.level_key_k(ref, lv)
        ref = WC.advance_k(ref, int(bits[i]), cmap, 8)
    assert list(st) == ref
    assert np.array_equal(codes[:nc[0]], WC.unit_codes(np.frombuffer(data, dtype=np.uint8)))


# -- the count store ----------------------------------------------------------------------------------------------

def test_the_store_counts_are_sums_whatever_the_chunking(text):
    codes = WC.unit_codes(np.frombuffer(text[:60000], dtype=np.uint8))
    cfg = WC.WordClassConfig(row_bits=10, feats=32)
    one, many = WC.ClassStore(cfg), WC.ClassStore(cfg)
    one.update(codes)
    for a in range(0, len(codes), 997):
        many.update(codes[a:a + 997])
    assert one.digest() == many.digest()
    assert one.counts.sum() == 2 * (len(codes) - 2)           # every interior unit adds a left and a right neighbor


# -- gate (c): the compile, the control ----------------------------------------------------------------------------

def class_stream(n_units: int = 60000, C: int = 8, words: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    vocab = [[f"w{c}x{k}" for k in range(words)] for c in range(C)]
    out, truth, cur = [], {}, 0
    for _ in range(n_units):
        cur = (cur + 1) % C if rng.random() < 0.8 else int(rng.integers(C))
        w = vocab[cur][int(rng.integers(words))]
        out.append(w + " ")
        truth[w + " "] = cur
    return "".join(out).encode(), truth


def cfg_small(**kw) -> WC.WordClassConfig:
    return WC.WordClassConfig(row_bits=12, feats=64, V=1024, C=8, d=8, iters=10, min_count=2, **kw)


def test_the_compile_reads_the_store_and_the_previous_map_and_is_deterministic():
    data, _ = class_stream()
    store = WC.ClassStore(cfg_small())
    store.update(WC.unit_codes(np.frombuffer(data, dtype=np.uint8)))
    a, b = WC.compile_classes(store, None, cfg_small(), 0), WC.compile_classes(store, None, cfg_small(), 0)
    assert a.digest() == b.digest()
    assert list(inspect.signature(WC.compile_classes).parameters) == ["store", "prev", "cfg", "index"]
    other = WC.ClassStore(cfg_small())
    other.update(WC.unit_codes(np.frombuffer(data, dtype=np.uint8)))
    assert WC.compile_classes(other, None, cfg_small(), 0).digest() == a.digest()
    c = WC.compile_classes(store, a, cfg_small(), 1)
    d = WC.compile_classes(store, a, cfg_small(), 1)
    assert c.digest() == d.digest() and 0.0 <= c.info["churn"] <= 1.0


def purity(C: int, seed: int) -> tuple[float, int]:
    """Purity of the compiled classes against 8 known classes (splitting a class does not lower purity)."""
    data, truth = class_stream()
    cfg = WC.WordClassConfig(row_bits=12, feats=64, V=1024, C=C, d=8, iters=10, min_count=2, seed=seed)
    store = WC.ClassStore(cfg)
    store.update(WC.unit_codes(np.frombuffer(data, dtype=np.uint8)))
    m = WC.compile_classes(store, None, cfg, 0)
    assert m is not None
    by_code = {int(WC.unit_codes(np.frombuffer(w.encode(), dtype=np.uint8))[0]): c for w, c in truth.items()}
    clusters: dict[int, list[int]] = {}
    for r in m.sel:
        t = by_code.get(int(m.tag[r]))
        if t is not None:
            clusters.setdefault(int(m.cls[r]), []).append(t)
    return sum(max(np.bincount(v)) for v in clusters.values()) / sum(len(v) for v in clusters.values()), len(clusters)


def test_the_compile_recovers_known_classes():
    """Gate (e) as built: 8 known classes, 16 clusters (twice the true number, as C = 256 exceeds the true number on
    text), the run's seed 0: purity at least 0.9. It depends on the k-means start (other seeds: 0.87 to 0.996)."""
    p, n = purity(16, 0)
    assert n >= 8 and p >= 0.9


def test_the_control_has_the_real_row_counts_and_persists():
    data, _ = class_stream()
    store = WC.ClassStore(cfg_small())
    store.update(WC.unit_codes(np.frombuffer(data, dtype=np.uint8)))
    real = WC.compile_classes(store, None, cfg_small(), 0)
    ctrl = WC.control_map(real, None, cfg_small(), 0)
    assert np.array_equal(np.sort(real.cls[real.sel]), np.sort(ctrl.cls[ctrl.sel]))
    assert (real.cls[real.sel] != ctrl.cls[ctrl.sel]).mean() > 0.5               # not the real partition
    real2 = WC.compile_classes(store, real, cfg_small(), 1)
    ctrl2 = WC.control_map(real2, ctrl, cfg_small(), 1)
    kept = ctrl.cls[real2.sel] != WC.NOMAP
    assert kept.any() and np.array_equal(ctrl2.cls[real2.sel][kept], ctrl.cls[real2.sel][kept])


# -- gate (d): causality, frozen reads ------------------------------------------------------------------------------

def test_the_map_in_force_for_a_block_does_not_depend_on_that_block(text):
    block = hier.BLOCK_BITS // 8
    data = text[:3 * block]
    pos = block + 5000
    bad = data[:pos] + bytes([data[pos] ^ 0x55]) + data[pos + 1:]
    outs = []
    for d in (data, bad):
        m = kmachine("real")
        q = []
        digests = []
        for j in range(3):
            digests.append(m.table_hashes()["map"])
            q.append(m.run(bytes_to_bits(d[j * block:(j + 1) * block]), learning=True))
        outs.append((digests, q))
    assert outs[0][0][1] == outs[1][0][1]                     # the map for block 1 comes from block 0 alone
    assert np.array_equal(outs[0][1][0], outs[1][1][0])
    assert np.array_equal(outs[0][1][1][:8 * 5000], outs[1][1][1][:8 * 5000])


def test_frozen_reads_write_no_tables_counts_or_map(text):
    m = kmachine("real")
    m.run(bytes_to_bits(text[:300000]), learning=True)
    assert m._real is not None and len(m.compiles) >= 1
    before, snap = m.table_hashes(), m.snapshot()
    m.run(bytes_to_bits(text[300000:340000]), learning=False)
    assert m.table_hashes() == before
    m.restore(snap)
    assert m.snapshot()[0] == snap[0]


def test_class_levels_are_in_the_declared_order_and_the_control_runs(text):
    assert WC.K_ORDER == (5, 0, 6, 1, 7, 2, 3, 4)
    for mode in ("real", "random"):
        m = kmachine(mode)
        assert list(m._kinds) == list(WC.K_ORDER)
        q = m.run(bytes_to_bits(text[:300000]), learning=True)
        assert np.isfinite(q).all() and (q > 0).all()
        assert len(m.compiles) >= 1 and m.compiles[-1]["rows"] > 0


def test_context_keys_identify_the_last_two_units_and_the_partial_unit():
    a = WC.context_keys(b"the cat sat on the mat. the cat sat", 1)
    assert a[0:0].size == 0 and len(a) == 35
    # "the cat sat": the key at the first byte after "the cat " (position 8) equals the one after "the cat " later
    b = b"the cat sat on the mat. the cat sat"
    assert a[8] == a[b.index(b"the cat sat", 1) + 8]
    assert a[8] != a[9]


# -- the development check's flags (E6b Amendment 1): off by default, unchanged when off ---------------------------

def test_the_new_flags_are_off_by_default_and_the_declared_order_is_kept(text):
    d = WC.WordClassConfig()
    assert d.order == "interleaved" and d.fix_map is False
    assert list(kmachine("real")._kinds) == list(WC.K_ORDER)         # flags off: the declared order of E6b.3
    assert small().upper == hier.ALL_UPPER                           # and E4's own levels are untouched


def test_order_after_puts_the_class_levels_after_s1():
    fields = {f.name: getattr(small(), f.name) for f in dataclasses.fields(hier.HierConfig)}
    fields["upper"] = WC.K_AFTER
    m = WC.KMachine(WC.KHierConfig(**fields), WC.WordClassConfig(mode="real", order="after", row_bits=12, feats=64,
                                                                  V=2048, C=16, d=16, iters=6, min_count=2))
    assert list(m._kinds) == [0, 1, 2, 3, 4, 5, 6, 7]
    with pytest.raises(ValueError):
        WC.WordClassConfig(order="middle")


def test_a_fixed_map_is_compiled_once(text):
    m = kmachine("real", fix_map=True)
    m.run(bytes_to_bits(text[:400000]), learning=True)
    assert len(m.compiles) == 1
    free = kmachine("real")
    free.run(bytes_to_bits(text[:400000]), learning=True)
    assert len(free.compiles) >= 2


def test_the_control_reports_its_own_churn():
    data, _ = class_stream()
    store = WC.ClassStore(cfg_small())
    store.update(WC.unit_codes(np.frombuffer(data, dtype=np.uint8)))
    real = WC.compile_classes(store, None, cfg_small(), 0)
    ctrl = WC.control_map(real, None, cfg_small(), 0)
    real2 = WC.compile_classes(store, real, cfg_small(), 1)
    ctrl2 = WC.control_map(real2, ctrl, cfg_small(), 1)
    assert "churn" not in ctrl.info and ctrl2.info["real_churn"] == real2.info["churn"]
    assert ctrl2.info["churn"] == 0.0                     # assignments persist, so no row it already held changed
