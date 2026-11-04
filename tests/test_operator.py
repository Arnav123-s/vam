"""E6 (PROTOCOL E6.5): the shared operator level O, the machine with the O stage, and the gates, on small machines."""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from vam.bits import bytes_to_bits
from vam.nested import NestedConfig
from vam.nfa import hier, oploop
from vam.nfa import operator as O
from vam.realizations.vam1 import Config

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
SPEC = Path(__file__).resolve().parents[1] / "spec"
LAW = Config(cells=1, context=1, consolidating=True, dt=1.0)


@pytest.fixture(scope="module")
def text() -> bytes:
    return (b" ".join(p.read_bytes() for p in sorted(SPEC.glob("*.md"))) * 2)[:260000]


def small(table_bits: int = 13):
    cfg = NestedConfig(LAW, depths=(8, 16, 24, 32, 48, 64), table_bits=table_bits, prior=4.0, coherence=1.0, verify=3,
                       governor=True, protect=0.5, sink_rate=0.0)
    return hier.HierConfig(cfg, upper_bits=11, upper_verify=4, segment="text")


def ocfg(**kw):
    return O.OperatorConfig(**{"compile_bytes": 1 << 15, "past_bits": 10, **kw})


def machine(mode="spectral", **kw):
    return O.OperatorMachine(small(), ocfg(mode=mode, **kw))


# -- the copied loop is E4's loop plus the E6 lines -----------------------------------------------------------

def strip_e6(src: str) -> str:
    out, skip = [], False
    for line in src.splitlines():
        if "# E6 begin" in line:
            skip = True
        if not skip:
            out.append(line.replace("  # E6 signature", ""))
        if "# E6 end" in line:
            skip = False
    return "\n".join(out)


def test_the_o_loop_is_the_e4_loop_plus_the_e6_lines():
    a = inspect.getsource(hier._make_loop)
    b = inspect.getsource(oploop.make_loop_o)
    a = re.sub(r'""".*?"""', '""""""', a, flags=re.S)
    b = strip_e6(re.sub(r'""".*?"""', '""""""', b, flags=re.S))
    a = a.replace("def _make_loop(jit):", "def make_loop_o(jit):")
    b = b.replace("door_on,\n             extp, extu, ev_o, acc, o_on, o_top):", "door_on):")
    assert a.split() == b.split()
    assert oploop.CHUNK == hier.CHUNK


# -- gate (a): O off is E4 bit for bit ----------------------------------------------------------------------

def test_o_off_equals_e4_bit_for_bit_learning_then_frozen(text):
    bits = bytes_to_bits(text[:120000])
    ref, off = hier.HierMachine(small()), machine("off")
    assert np.array_equal(ref.run(bits, learning=True), off.run(bits, learning=True))
    more = bytes_to_bits(text[120000:160000])
    assert np.array_equal(ref.copy().run(more, learning=False), off.copy().run(more, learning=False))
    for name in ("_wn", "_tags", "_door", "_occ"):
        assert np.array_equal(getattr(ref, name), getattr(off, name))


def test_the_hier_config_is_the_one_hier_build_makes():
    spec = {"machine": "hier", "law": "counting", "prior": 4.0, "coherence": 1.0, "governor": True, "protect": 0.5,
            "sink_rate": 0.0, "verify": 3, "hier_levels": "WS", "hier_segment": "text", "hier_verify": 4}
    assert O.hier_config(spec, LAW) == hier.build(spec, LAW).hcfg


# -- the moment store ---------------------------------------------------------------------------------------

def test_the_store_counts_are_sums_whatever_the_chunking(text):
    data = text[:50000]
    one, many = O.MomentStore(ocfg()), O.MomentStore(ocfg())
    one.update(data)
    for a in range(0, len(data), 7001):
        many.update(data[a:a + 7001])
    assert one.digest() == many.digest()
    assert one.n2 == len(data) - 4 and one.n3 == len(data) - 5


# -- gate (b): compile reads counts only ------------------------------------------------------------------

def test_compile_reads_the_counts_and_nothing_else(text):
    store = O.MomentStore(ocfg())
    store.update(text[:80000])
    a, b = O.compile_operator(store, ocfg()), O.compile_operator(store, ocfg())
    assert np.array_equal(a.Bs, b.Bs) and np.array_equal(a.W, b.W) and a.k == b.k
    assert list(inspect.signature(O.compile_operator).parameters) == ["store", "cfg"]
    other = O.MomentStore(ocfg())
    other.update(text[:80000])                       # equal counts built separately: identical operators
    c = O.compile_operator(other, ocfg())
    assert np.array_equal(a.Bs, c.Bs)


def test_the_page_rank_rule_is_e4s_floor_rank(text):
    from scipy import sparse
    sys.path.insert(0, str(EXPERIMENTS))
    import e4_hier
    store = O.MomentStore(ocfg(floor=20))
    store.update(text[:200000])
    full = sparse.csr_matrix(store.c2.astype(float))
    halves = [sparse.csr_matrix(h.astype(float)) for h in store.c2h]
    assert e4_hier.floor_rank(full, halves, 20)["rank"] == O.page_rank_of(store, ocfg(floor=20))


# -- gate (c): convergence to a known HMM -------------------------------------------------------------------

def hmm_stream(n: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    T = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.25, 0.25, 0.5]])
    E = np.array([[0.6, 0.3, 0.1, 0, 0, 0, 0, 0], [0.05, 0.1, 0.05, 0.5, 0.3, 0, 0, 0],
                  [0, 0, 0.05, 0, 0.1, 0.4, 0.3, 0.15]])
    s, out = 0, np.empty(n, dtype=np.uint8)
    u = rng.random((n, 2))
    cT, cE = T.cumsum(1), E.cumsum(1)
    for t in range(n):
        out[t] = int(np.searchsorted(cE[s], u[t, 0]))
        s = int(np.searchsorted(cT[s], u[t, 1]))
    return out, T, E


def true_predictive(obs, T, E):
    alpha = np.array([1 / 3] * 3)
    preds = np.empty((len(obs), 8))
    for t, x in enumerate(obs):
        preds[t] = (alpha @ T) @ E if t else (alpha @ E)
        alpha = (alpha @ T if t else alpha) * E[:, x]
        alpha = alpha / alpha.sum()
    return preds


def test_the_compile_recovers_a_three_state_hmm_with_the_bad_update_rule():
    n = 1_000_000
    obs, T, E = hmm_stream(n)
    cfg = O.OperatorConfig(L=1, past_bits=8, compile_bytes=1 << 20, floor=20, k_force=3)
    store = O.MomentStore(cfg)
    store.update(obs.tobytes())
    model = O.compile_operator(store, cfg)
    assert O.page_rank_of(store, cfg) <= 3
    test_obs, _, _ = hmm_stream(20000, seed=2)
    truth = true_predictive(test_obs, T, E)
    b, errs, resets = model.b1.copy(), [], 0
    for t, x in enumerate(test_obs):
        raw = model.W @ b
        p = np.maximum(raw / raw.sum(), cfg.eps_o)
        p = (p / p.sum())[:8]
        errs.append(np.abs(p - truth[t][:8]).mean() * 8 / 8)
        if not raw[x] > cfg.eps_o * np.abs(raw).sum():
            b, resets = model.b1.copy(), resets + 1
        else:
            b = (model.Bs[x] @ b) / raw[x]
    assert float(np.mean(errs[100:])) < 0.02
    assert resets < 0.05 * len(test_obs)


# -- gate (d): the rank-1 arm is the unigram -----------------------------------------------------------------

def test_the_rank_1_arm_is_the_unigram_and_does_not_depend_on_the_state(text):
    store = O.MomentStore(ocfg(mode="unigram"))
    store.update(text[:60000])
    m = O.compile_operator(store, ocfg(mode="unigram"))
    pi = store.c2.sum(1) / store.c2.sum()
    assert m.k == 1 and np.allclose(m.W[:, 0] * m.b1[0], pi)
    for b in (np.array([1.0]), np.array([3.7])):
        assert np.allclose((m.W @ b) / (m.W @ b).sum(), pi)
    assert np.allclose(m.Bs[:, 0, 0], pi)


def test_a_rank_1_spectral_compile_returns_the_unigram_on_iid_data():
    rng = np.random.default_rng(5)
    pi = np.array([0.4, 0.3, 0.2, 0.1])
    obs = rng.choice(4, size=300000, p=pi).astype(np.uint8)
    cfg = O.OperatorConfig(L=1, past_bits=8, k_force=1)
    store = O.MomentStore(cfg)
    store.update(obs.tobytes())
    m = O.compile_operator(store, cfg)
    p = (m.W @ m.b1)[:4]
    assert np.allclose(p / p.sum(), pi, atol=0.01)


# -- gate (e): frozen reads change nothing, the state is restored --------------------------------------------

def test_frozen_reads_leave_store_tables_and_operators_unchanged_and_the_snapshot_restores(text):
    m = machine()
    m.run(bytes_to_bits(text[:100000]), learning=True)
    assert m._model is not None and m.compiles
    before, snap = m.table_hashes(), m.snapshot()
    st_before = m.state
    m.run(bytes_to_bits(text[100000:140000]), learning=False)
    assert m.table_hashes() == before
    assert m.state != st_before
    m.restore(snap)
    assert m.state == st_before and np.array_equal(m._b, snap[2])
    again = m.run(bytes_to_bits(text[100000:140000]), learning=False)
    m.restore(snap)
    assert np.array_equal(again, m.run(bytes_to_bits(text[100000:140000]), learning=False))


def test_a_copy_shares_the_store_and_the_operators_but_not_the_state(text):
    m = machine()
    m.run(bytes_to_bits(text[:70000]), learning=True)
    c = m.copy()
    assert c._store is m._store and c._model is m._model and c._b is not m._b
    c.run(bytes_to_bits(text[70000:80000]), learning=False)
    assert not np.array_equal(c._b, m._b)


# -- gate (f): the stage regret ------------------------------------------------------------------------------

def test_the_stage_costs_at_most_two_bits_per_node_against_its_input(text):
    m = machine()
    m.run(bytes_to_bits(text[:200000]), learning=True)
    r = m.stage_regret()
    assert r["nodes_used"] > 100 and r["max_excess_bits"] <= 2.0 + 1e-6
    assert r["total_excess_bits"] <= 2.0 * 255


# -- gate (g): re-entry and causality -------------------------------------------------------------------------

def test_reentry_changes_the_state_unless_an_eigenvector_and_never_the_counts(text):
    m = machine()
    m.run(bytes_to_bits(text[:100000]), learning=True)
    before = m.table_hashes()
    changed = [m.reenter(x) for x in text[100000:100200]]
    assert any(changed)
    assert m.table_hashes() == before
    u = machine("unigram")
    u.run(bytes_to_bits(text[:100000]), learning=True)
    assert u._model is not None and u.reenter(104) is False        # k = 1: every state is an eigenvector
    assert machine().reenter(65) is False                           # no operators yet: nothing to change


def test_reading_does_not_reenter_external_bytes(text):
    m = machine()
    calls = []
    m.reenter = lambda byte: calls.append(byte)
    m.run(bytes_to_bits(text[:60000]), learning=True)
    assert calls == []


def test_the_operators_used_for_a_block_do_not_depend_on_its_bytes(text):
    period = 1 << 15
    a, b = machine(), machine()
    head = text[:3 * period]
    a.run(bytes_to_bits(head), learning=True)
    b.run(bytes_to_bits(head), learning=True)
    assert a._model is not b._model
    assert np.array_equal(a._model.Bs, b._model.Bs)
    block = bytearray(text[3 * period:4 * period])
    flipped = bytearray(block)
    flipped[period // 2] ^= 0x55
    ma, mb = a._model, b._model
    qa = a.run(bytes_to_bits(bytes(block)), learning=True)
    qb = b.run(bytes_to_bits(bytes(flipped)), learning=True)
    assert np.array_equal(ma.Bs, mb.Bs)                              # the operators that read this block
    cut = (period // 2) * 8
    assert np.array_equal(qa[:cut], qb[:cut])                        # nothing before the changed byte differs
    assert not np.array_equal(qa[cut:], qb[cut:])


# -- the bad-update rule and the negative-mass share -------------------------------------------------------------

def test_a_nonpositive_mass_resets_the_state_and_is_counted():
    k = 2
    W = np.zeros((256, k))
    W[:, 0] = 1.0 / 256
    W[7, 0] = -0.5                                    # byte 7 has negative raw mass
    Bs = np.zeros((256, k, k)) + np.eye(k)
    b1 = np.array([1.0, 0.0])
    b = np.array([1.0, 0.3])
    extp, extu, ostat = np.zeros(16), np.zeros(16, dtype=np.int64), np.zeros(4)
    O._o_pass(np.array([7, 9], dtype=np.uint8), b, Bs, W, b1, np.full(256, 1 / 256), 2.0 ** -12, 2.0 ** -12, extp, extu,
              ostat)
    assert ostat[3] >= 1 and ostat[2] == 2 and ostat[1] >= 1
    assert np.all(extp > 0) and np.all(extp < 1)
    assert set(extu[:8]) == {(1 << j) | (7 >> (8 - j)) for j in range(8)}



def test_c3_over_observed_rows_equals_dense():
    """E6 Amendment 1 (R-a): C3 over the observed bytes compiles to the same operators as the dense C3."""
    rng = np.random.default_rng(3)
    data = bytes(rng.choice(np.array([10, 32, 97, 101, 110, 116, 200], dtype=np.uint8), size=60000,
                            p=[.05, .2, .15, .2, .15, .2, .05]))
    base = dict(L=2, past_bits=6, compile_bytes=1 << 15)
    dense = O.MomentStore(O.OperatorConfig(**base))
    rows = O.MomentStore(O.OperatorConfig(**base, c3_bytes=tuple(sorted(set(data)))))
    for st in (dense, rows):
        st.update(data[:30000])
        st.update(data[30000:])
    assert rows.c3.shape == (7, 7, 64) and rows.n3 == dense.n3 and rows.max_cell == dense.max_cell
    a = O.compile_operator(dense, O.OperatorConfig(**base, k_force=5))
    b = O.compile_operator(rows, O.OperatorConfig(**base, k_force=5, c3_bytes=tuple(sorted(set(data)))))
    np.testing.assert_allclose(a.Bs, b.Bs, atol=1e-12)
    np.testing.assert_allclose(a.b1, b.b1, atol=1e-12)


def test_e6_diagnostic_flags_off_change_nothing():
    """With --past-bits and --k-force off the run script names no variant and the config is E6's own."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import e6_operator as run

    assert run.variant(None, None) == ("", {})
    assert run.variant(14, None) == ("e6_ra", {"past_bits": 14})
    assert run.variant(None, 24) == ("e6_rb", {"k_force": 24})
    with pytest.raises(SystemExit):
        run.variant(14, 24)
    for arm in run.ARMS:
        assert run.ocfg_for(arm, False) == O.OperatorConfig(mode={"E4": "off", "E4+O": "spectral",
                                                                    "E4+O rank 1": "unigram"}[arm])
    assert O.OperatorConfig().c3_bytes == ()
    assert O.MomentStore(O.OperatorConfig(past_bits=4)).c3.shape == (256, 256, 16)
