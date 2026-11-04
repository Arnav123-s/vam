"""E4 (DESIGN section 11): the upper levels, layer by layer. The criteria are PROTOCOL E4, Part 1."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from vam.bits import bytes_to_bits
from vam.nested import GOLDEN, MASK64, NestedConfig, NestedMachine, _learn, _p0, _q, _share, coherent_mix
from vam.nfa import bounds, hier
from vam.realizations.vam1 import Config

LAW = Config(cells=1, context=1, consolidating=True, dt=1.0)
SPEC = Path(__file__).resolve().parents[1] / "spec"


@pytest.fixture(scope="module")
def text() -> bytes:
    return b" ".join(p.read_bytes() for p in sorted(SPEC.glob("*.md")))[:120000]


def base(**kw):
    opts = dict(depths=(8, 16, 24, 32, 48, 64), table_bits=12, prior=4.0, coherence=1.0, verify=3, governor=True,
                protect=0.5, sink_rate=0.1)
    return NestedConfig(LAW, **{**opts, **kw})


# -- Part 0: identity with E3's machine -------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [dict(), dict(verify=1, governor=False, coherence=0.0), dict(verify=2, governor=False)],
                         ids=["e3_row", "plain", "verify2"])
def test_no_upper_level_is_the_nested_machine_bit_for_bit(text, kw):
    cfg = base(**kw)
    bits = bytes_to_bits(text)
    a, b = NestedMachine(cfg), hier.HierMachine(hier.HierConfig(cfg, upper=()))
    assert np.array_equal(a.run(bits), b.run(bits))
    assert np.array_equal(a.run(bits[:40000], learning=False), b.run(bits[:40000], learning=False))
    assert (a.evictions, a.refused, a.protected) == (b.evictions, b.refused, b.protected)
    assert np.array_equal(a._wn, b._wn)


# -- N-L1: registers, state, the pre-pass equals the Python state machine ---------------------------------------

def test_registers_respect_the_bounds_of_n_l1():
    assert hier.ARITY <= bounds.Hardware().line_words and bounds.register_cap() == 64
    with pytest.raises(ValueError):
        hier.HierConfig(base(), upper=(hier.S0, hier.W1))        # the declared order is kept
    with pytest.raises(ValueError):
        hier.HierConfig(base(), segment="words")


def test_state_size_does_not_grow_with_the_stream(text):
    m = hier.HierMachine(hier.HierConfig(base(), upper_bits=10))
    before, size = m.state_bytes(), m.nbytes()
    m.run(bytes_to_bits(text))
    assert m.state_bytes() == before < 128 and m.nbytes() == size


@pytest.mark.parametrize("segment", ["content", "text"])
@pytest.mark.parametrize("control", ["none", "comm", "dup"])
def test_prepass_keys_equal_the_python_state_machine(text, segment, control):
    bits = bytes_to_bits(text[:6000])
    m = hier.HierMachine(hier.HierConfig(base(), upper_bits=10, segment=segment, control=control))
    ext = np.empty((len(bits), 5), dtype=np.uint64)
    st = np.array(hier.initial_state(), dtype=np.uint64)
    hier._fill_keys(bits, st, ext, m._kinds, m._dups, segment == "text", control == "comm")
    py = hier.initial_state()
    for i, b in enumerate(bits):
        keys = [hier.level_key(py, k, control == "comm", hier.DUP_DEPTHS[k] if control == "dup" else 0)
                for k in range(5)]
        assert [int(x) for x in ext[i]] == keys
        py = hier.advance(py, int(b), segment == "text", control == "comm")
    assert [int(x) for x in st] == py


def test_units_and_sentences_have_the_declared_sizes(text):
    st, units, closed = hier.initial_state(), 0, 0
    for by in text:
        for k in range(7, -1, -1):
            old = st[hier.ST_P1]
            st = hier.advance(st, (by >> k) & 1)
        units += st[hier.ST_C] == hier.SEED_C
        closed += st[hier.ST_P1] != old
    assert 4 < len(text) / units < 9 and closed > 0 and len(text) / closed > 40


def test_unit_codes_collide_rarely(text):
    seen: dict[int, bytes] = {}
    collisions, st, start = 0, hier.initial_state(), 0
    for i, by in enumerate(text):
        for k in range(7, -1, -1):
            st = hier.advance(st, (by >> k) & 1)
        if st[hier.ST_C] == hier.SEED_C:
            unit, code = text[start:i + 1], st[hier.ST_R0]
            start = i + 1
            collisions += seen.setdefault(code, unit) != unit
    assert collisions == 0


# -- the compiled loop equals a plain Python writing (per-level sizes, verify, global governor, raised bar) ------

class Reference:
    """Plain dictionaries for the full machine, the semantics of ``hier._make_loop`` written independently."""

    def __init__(self, m: hier.HierMachine) -> None:
        self.m, self.cfg = m, m.cfg
        self.st = hier.initial_state()
        self.tables = [dict() for _ in range(m.levels)]       # slot -> [w, n, ev, tag]
        self.doors = [np.zeros(1 << (64 - int(m._door_shifts[int(m._hidx[lv])])), dtype=np.int64)
                      if m._hidx[lv] >= 0 else None for lv in range(m.levels)]
        self.ev = self.ref = self.prot = 0
        self.refused = self.protected = 0

    def keys(self):
        m, st, cfg = self.m, self.st, self.cfg
        out = []
        for lv in range(m.levels):
            h = int(m._hidx[lv])
            if lv < m._Lb:
                ctx = st[0] & ((1 << cfg.depths[lv]) - 1)
                x = ctx if h < 0 else hier.mix64(ctx ^ (((lv + 1) * GOLDEN) & MASK64))
            else:
                x = m._keys(st)[lv - m._Lb]
            if h < 0:
                out.append((ctx, 0))
            else:
                out.append((x >> int(m._shifts[h]), (x & 0xFFFFFFFF) | 1, x))
        return out

    def sighting(self, lv, x, need):
        """One sighting at the two-way doorkeeper bucket pair of context key ``x`` (as ``NestedReference``)."""
        door = self.doors[lv]
        y = hier.mix64(x ^ hier.DOOR_SALT)
        d = y >> int(self.m._door_shifts[int(self.m._hidx[lv])])
        tag = (y & 0xFFFF) | 1
        d0, d1 = d & ~1, d | 1
        e0, e1 = int(door[d0]), int(door[d1])
        if e0 >> 16 == tag:
            idx, count = d0, min(255, (e0 & 0xFFFF) + 1)
        elif e1 >> 16 == tag:
            idx, count = d1, min(255, (e1 & 0xFFFF) + 1)
        else:
            idx, count = (d0 if (e0 & 0xFFFF) <= (e1 & 0xFFFF) else d1), 1
        door[idx] = 0 if count >= need else (tag << 16) | count
        return count

    def occ(self, lv):
        return len(self.tables[lv])

    def governed(self, lv):
        m = self.m
        if not self.cfg.governor:
            return False
        hashed = [v for v in range(m.levels) if m._hidx[v] >= 0]
        if m.hcfg.scope == "global":
            return 2 * sum(self.occ(v) for v in hashed) >= m._tcap
        return 2 * self.occ(lv) >= int(m._tsizes[int(m._hidx[lv])])

    def step(self, b):
        m, cfg = self.m, self.cfg
        keys = self.keys()
        entries = []
        for lv in range(m.levels):
            k = keys[lv]
            e = self.tables[lv].get(k[0])
            if e is not None and int(m._hidx[lv]) >= 0 and e[3] != k[1]:
                e = None
            entries.append(e)
        chain = []
        p0 = 0.0
        for lv, e in enumerate(entries):
            if lv == 0:
                p0 = _p0(cfg.law, e[0] if e else 0.0)
            elif e is not None and e[1] >= cfg.mature:
                p0 = coherent_mix(p0, _p0(cfg.law, e[0]), _share(e[2]), cfg.coherence)
            chain.append(p0)
        q = _q(chain[-1], b)
        parent_n = 0.0
        for lv in range(m.levels):
            h, k, e = int(m._hidx[lv]), keys[lv], entries[lv]
            if e is None:
                if h >= 0:
                    table, key = self.tables[lv], k[0]
                    gov = self.governed(lv)
                    need = int(m._verify[lv])
                    if gov:
                        need = need + 1 if m.hcfg.raise_bar else max(need, 2)
                    if m._door_on and need > 1:
                        if self.sighting(lv, k[2], need) < need:
                            self.refused += 1
                            continue
                    occupant = table.get(key)
                    if m._door_on and gov and occupant is not None and occupant[2] > cfg.protect:
                        occupant[2] *= cfg.sink_factor
                        self.protected += 1
                        continue
                    if occupant is not None:
                        self.ev += 1
                    inherited = (_w(cfg, chain[lv - 1]) if lv > 0 else 0.0, min(parent_n, cfg.prior) if lv > 0 else 0.0)
                    e = table[key] = [*inherited, 0.0, k[1]]
                else:
                    inherited = (_w(cfg, chain[lv - 1]) if lv > 0 else 0.0, min(parent_n, cfg.prior) if lv > 0 else 0.0)
                    e = self.tables[lv][k[0]] = [*inherited, 0.0, 0]
            elif lv > 0:
                own, prev = _q(_p0(cfg.law, e[0]), b), _q(chain[lv - 1], b)
                e[2] += math.log2(own / prev)
            parent_n = e[1]
            e[0], e[1] = _learn(cfg.law, e[0], e[1], b)
        self.st = hier.advance(self.st, b, self.m.hcfg.segment == "text", self.m.hcfg.control == "comm")
        return q


def _w(cfg, p0):
    return min(1.0, max(-1.0, (2.0 * p0 - 1.0) / (1.0 - 2.0 * cfg.law.eps)))


OPTIONS = [dict(), dict(scope="global"), dict(scope="global", raise_bar=True), dict(raise_bar=True, upper_verify=2),
           dict(segment="text", scope="global"), dict(control="dup"), dict(control="comm", upper_verify=2),
           dict(upper=(hier.W2, hier.S1), upper_verify=4, scope="global", raise_bar=True)]


@pytest.mark.parametrize("opts", OPTIONS, ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()) or "defaults")
def test_compiled_equals_the_plain_reference(text, opts):
    cfg = base(depths=(8, 16, 24, 32, 48, 64), table_bits=9)
    m = hier.HierMachine(hier.HierConfig(cfg, upper_bits=7, **opts))
    ref = Reference(m)
    bits = bytes_to_bits(text[:5000])
    q = np.concatenate([m.run(bits[:7000]), m.run(bits[7000:])])
    qr = np.array([ref.step(int(b)) for b in bits])
    assert np.allclose(q, qr, atol=1e-12)
    assert (m.refused, m.protected, m.evictions) == (ref.refused, ref.protected, ref.ev)
    assert m.refused > 0
    assert np.isclose(m.predict(0), 1.0 - m.predict(1))


def test_the_raised_bar_changes_the_refusals_at_every_verify_setting(text):
    """N-L6 (c): the raise is kept only if it changes behaviour at verify 2, 3 and 4."""
    bits = bytes_to_bits(text[:30000])
    for v in (2, 3, 4):
        counts = []
        for raise_bar in (False, True):
            cfg = base(verify=v, table_bits=9)
            m = hier.HierMachine(hier.HierConfig(cfg, upper_bits=7, scope="global", raise_bar=raise_bar))
            m.run(bits)
            counts.append(m.refused)
        assert counts[0] != counts[1], v


# -- N-L3, N-L4, N-L5, N-L6, N-L7 ---------------------------------------------------------------------------------

def test_the_readout_lies_between_the_component_readouts(text):
    m = hier.HierMachine(hier.HierConfig(base(), upper_bits=10))
    m.run(bytes_to_bits(text))
    rng = np.random.default_rng(0)
    st = m.state
    for _ in range(300):
        st = m.advance(st, int(rng.integers(0, 2)))
        p0 = m.p0_state(st)
        assert 0.0 < p0 < 1.0 and np.isfinite(p0)
    assert np.isfinite(m._wn).all()


def test_a_new_upper_sector_equal_to_the_coarser_readout_changes_nothing():
    for c in (0.0, 0.5, 1.0):
        for p0 in (0.1, 0.5, 0.93):
            for s in (0.0, 0.3, 1.0):
                assert abs(coherent_mix(p0, p0, s, c) - p0) < 1e-12


def test_containment_a_perturbed_byte_has_no_effect_beyond_the_span(text):
    """N-L4: on a frozen machine, flipping one byte changes no prediction beyond the registers' span."""
    m = hier.HierMachine(hier.HierConfig(base(table_bits=14), upper_bits=12))
    m.run(bytes_to_bits(text[:100000]))
    seg = text[100000:112000]
    rng = np.random.default_rng(3)
    clean = m.copy().run(bytes_to_bits(seg), learning=False).reshape(len(seg), 8)
    spans = []
    for _ in range(5):
        pos = int(rng.integers(100, 1000))
        bad = bytearray(seg)
        bad[pos] ^= 0x20
        out = m.copy().run(bytes_to_bits(bytes(bad)), learning=False).reshape(len(seg), 8)
        differs = np.flatnonzero(np.any(out != clean, axis=1))
        assert differs[0] >= pos
        spans.append(int(differs[-1]) - pos)
        assert not np.any(out[differs[-1] + 1:] != clean[differs[-1] + 1:])     # equal again, and it stays equal
    assert max(spans) < 3000                 # a sentence of the content rule is about 150 bytes: bounded, finite


def test_verified_growth_keeps_noise_out_of_the_upper_levels():
    """N-L5 (a). In i.i.d. bytes the short units (one byte: 256 codes) repeat by construction, so W1 and S0 hold a
    few real contexts; the long levels W2, W4 and S1 hold none under verify 3 and fill under verify 1. (PROTOCOL's
    criterion, every level below 1% at 2**21 slots on 8 MiB, is evaluated at full size by the run.)"""
    rng = np.random.default_rng(1)
    noise = bytes_to_bits(bytes(rng.integers(0, 256, 300000, dtype=np.uint8)))
    occ = {}
    for v in (1, 3):
        m = hier.HierMachine(hier.HierConfig(base(verify=v, governor=False, table_bits=14), upper_bits=16))
        size = m.nbytes()
        m.run(noise)
        assert m.nbytes() == size
        occ[v] = {hier.LEVEL_NAMES[lv - m._Lb]: m.occupancy()[lv] for lv in range(m._Lb, m.levels)}
    for name in ("W2", "W4", "S1"):
        assert occ[3][name] < 0.01 and occ[1][name] > 0.5, (name, occ)
    assert all(occ[3][k] < 0.5 * occ[1][k] for k in occ[1])


def stored(m, lv, x):
    h = int(m._hidx[lv])
    slot = x >> int(m._shifts[h])
    return int(m._tags[m._tagoff[h] + slot]) == ((x & 0xFFFFFFFF) | 1)


def test_proven_upper_sectors_survive_noise_and_plain_ones_do_not():
    """N-L6 (a) on the W4 level, tables of 512 slots: the contexts of a learned sentence, then 8000 random bytes."""
    sentence = b"the quick brown fox jumps over the lazy dog. "
    noise = bytes_to_bits(bytes(np.random.default_rng(2).integers(0, 256, 8000, dtype=np.uint8)))
    kept = {}
    for name, kw in {"plain": dict(verify=1, governor=False), "verify2": dict(verify=2, governor=False),
                     "governor": dict(verify=1, governor=True, protect=0.5, sink_rate=0.1)}.items():
        m = hier.HierMachine(hier.HierConfig(base(table_bits=9, **kw), upper_bits=9, scope="global"))
        m.run(bytes_to_bits(sentence * 100))
        lv = m._Lb + hier.ALL_UPPER.index(hier.W4)
        st, keys = m.state, []
        for bit in bytes_to_bits(sentence * 3):
            keys.append(hier.level_key(st, hier.W4))
            st = m.advance(st, int(bit))
        held = [k for k in set(keys) if stored(m, lv, k)]
        m.run(noise)
        kept[name] = (len(held), sum(stored(m, lv, k) for k in held))
        assert m.global_occupancy() <= 1.0
    assert kept["plain"][1] < 0.02 * max(1, kept["plain"][0]) and kept["plain"][0] > 0
    for name in ("verify2", "governor"):
        assert kept[name][0] > 0 and kept[name][1] >= 0.97 * kept[name][0], kept


def test_label_entropy_per_cut_is_at_most_one_bit():
    for s in np.linspace(0.0, 1.0, 11):
        h = 0.0 if s in (0.0, 1.0) else -(s * math.log2(s) + (1 - s) * math.log2(1 - s))
        assert h <= 1.0 + 1e-12                                 # bond 2: the area law is vacuous (DESIGN 11.3)


def test_the_instrument_equals_the_prediction_on_the_full_state(text):
    from vam.nfa import collapse
    m = hier.HierMachine(hier.HierConfig(base(table_bits=14), upper_bits=12))
    m.run(bytes_to_bits(text[:60000]))
    m.run(bytes_to_bits(b"the quick brown fox"), learning=False)
    ro = collapse.Readout(m)
    st = ro.state0
    worst = 0.0
    for _ in range(24):
        z = collapse.first_step(ro, st, 8, 1.0)["Z"][0]
        worst = max(worst, abs(z - ro.p0(st)))
        st = ro.advance(st, int(z < 0.5))
    assert worst < 1e-10


def test_the_planner_hbar0_equals_the_machines_max_generation(text):
    from vam.nfa import collapse
    m = hier.HierMachine(hier.HierConfig(base(table_bits=14), upper_bits=12))
    m.run(bytes_to_bits(text[:60000]))
    a, b = m.copy(), m.copy()
    got = collapse.generate(a, 16, horizon=8, hbar=0.0)
    assert got == b.generate(16, horizon=8, select="max")
