"""E7 (PROTOCOL E7.5): the family stages F4 and F5, the machine that mixes them in, and the gates, on small machines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from vam.bits import bytes_to_bits
from vam.nested import NestedConfig
from vam.nfa import hier
from vam.nfa.families import (
    CPStage,
    FamilyConfig,
    FamilyMachine,
    Lattice,
    LatticeStage,
    Spelling,
    bit_probs,
    cp_fit,
    fourier_readout,
)
from vam.realizations.vam1 import Config

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
LAW = Config(cells=1, context=1, consolidating=True, dt=1.0)


def small(table_bits: int = 13):
    cfg = NestedConfig(LAW, depths=(8, 16, 24, 32, 48, 64), table_bits=table_bits, prior=4.0, coherence=1.0, verify=3,
                       governor=True, protect=0.5, sink_rate=0.0)
    return hier.HierConfig(cfg, upper_bits=11, upper_verify=4, segment="text")


def machine(route: str, slot: str = "after0", **kw) -> FamilyMachine:
    return FamilyMachine(small(), FamilyConfig(route=route, slot=slot, **kw))


@pytest.fixture(scope="module")
def e7():
    sys.path.insert(0, str(EXPERIMENTS))
    import e7_families
    return e7_families


def stream(seed: int = 0, h: float = 0.2, epochs: int = 6):
    import e7_families as X
    X.EPOCHS = epochs
    held, seen = X.split(10, h, seed)
    eqs = [(a, b, a + b) for a, b in X.ordered(seen)]
    return held, seen, eqs, X.epoch_bytes(eqs, seed)


# -- gate (a): the stages off is E4 bit for bit ---------------------------------------------------------------

@pytest.mark.parametrize("slot", ["after0", "top"])
def test_family_off_equals_e4_bit_for_bit(slot):
    rng = np.random.default_rng(1)
    data = bytes(rng.integers(0, 256, size=30000, dtype=np.uint8))
    ref, off = hier.HierMachine(small()), machine("off", slot)
    assert np.array_equal(ref.run(bytes_to_bits(data), learning=True), off.run(bytes_to_bits(data), learning=True))
    more = bytes_to_bits(data[:10000])
    assert np.array_equal(ref.copy().run(more, learning=False), off.copy().run(more, learning=False))


# -- gate (b): the lattice code, and V5's closed case -------------------------------------------------------------

def test_lattice_equals_hermite_membership_and_is_exact_when_complete(e7):
    out = e7.gate_lattice(60)
    assert out["agree"] == 60 and out["complete_stage1_unique_and_right"]


def test_lattice_canonical_form_is_a_span_test():
    lat = Lattice(3)
    lat.add([1, 1, -2])
    lat.add([0, 2, -2])
    assert lat.canon([1, 3, -4]) == lat.canon([0, 0, 0])           # sum of the two
    assert lat.canon([0, 1, -1]) != lat.canon([0, 0, 0])           # half of the second: torsion, not in the span


def test_f5_closed_case_and_stage_one_complete(e7):
    out = e7.gate_closed_case()
    assert out["mod10_top1"] == 1.0 and out["stage1_complete_top1"] >= 0.99


def test_parser_agrees_with_the_text_rule(e7):
    assert e7.gate_parser()["checked"]


# -- gate (d): soundness of F4 ---------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_f4_determined_candidates_are_the_true_sums(e7, seed):
    held, seen = e7.split(10, 0.2, seed)
    lat = LatticeStage.build(19, [(a, b, a + b) for a, b in e7.ordered(seen)])
    det = 0
    for a, b in held:
        cs = lat.candidates(a, b)
        if a + b in cs:
            det += 1
            assert cs == [a + b]
        assert not cs or cs == [a + b] or a + b not in cs
    assert det > 0


def test_f4_does_not_fire_on_a_random_table(e7):
    for seed in range(3):
        table = e7.random_table(seed, 10, 19)
        held, seen = e7.split(10, 0.2, seed)
        lat = LatticeStage.build(19, [(a, b, int(table[a, b])) for a, b in e7.ordered(seen)])
        assert all(lat.dist(a, b) is None for a, b in held)


# -- the distributions the stages hand the machine -----------------------------------------------------------------

def test_spelling_marginal_sums_to_one_and_matches_the_spelled_result():
    sp = Spelling(19)
    p = np.zeros(19)
    p[13], p[5] = 0.75, 0.25
    q0 = sp.byte_dist(p, b"")
    assert q0.sum() == pytest.approx(1.0)
    assert q0[ord("1")] == pytest.approx(0.75) and q0[ord("5")] == pytest.approx(0.25)
    q1 = sp.byte_dist(p, b"1")
    assert q1[ord("3")] == pytest.approx(0.75) and q1.sum() == pytest.approx(0.75)
    assert sp.byte_dist(p, b"7") is None or sp.byte_dist(p, b"7").sum() == 0.0


def test_bit_probabilities_reproduce_the_floored_byte_distribution():
    rng = np.random.default_rng(2)
    q = rng.random(256) ** 4
    for x in (0, 17, 255, 200):
        p0, u = bit_probs(q, x, eps_o=1e-9, eps=1e-12)
        prob = 1.0
        for j in range(8):
            bit = (x >> (7 - j)) & 1
            prob *= p0[j] if bit == 0 else 1.0 - p0[j]
        assert prob == pytest.approx(q[x] / q.sum(), rel=1e-6)
        assert list(u) == [(1 << j) | (x >> (8 - j)) for j in range(8)]


def test_fourier_readout_is_a_distribution():
    held, seen = [], [(a, b, a + b) for a in range(10) for b in range(a, 10)]
    A, B, G, resid = cp_fit(10, 19, seen, 19, 0, sweeps=60)
    p = fourier_readout(A, B, G, 2, 3)
    assert p.sum() == pytest.approx(1.0) and (p > 0).all() and held == []
    assert resid.shape == (10,)


# -- the machine ------------------------------------------------------------------------------------------------

def read(m: FamilyMachine, epochs: list[bytes]) -> None:
    for chunk in epochs:
        m.run(bytes_to_bits(chunk), learning=True)
        m.end_epoch()


def test_stage_is_built_once_after_the_first_epoch_and_never_again():
    _, _, eqs, epochs = stream()
    m = machine("f4")
    m.run(bytes_to_bits(epochs[0]), learning=True)
    assert m._fam is None
    assert m.end_epoch() is True and m._fam is not None
    for chunk in epochs[1:]:
        m.run(bytes_to_bits(chunk), learning=True)
        assert m.end_epoch() is False
    assert len(m.fits) == 1 and m._eqs == set(eqs)


def test_frozen_reads_leave_tables_equations_and_evidence_unchanged_and_restore_the_state():
    held, _, _, epochs = stream()
    m = machine("f4")
    read(m, epochs)
    before, snap = m.table_hashes(), m.snapshot()
    a, b = held[0]
    m.run(bytes_to_bits(f"{a}+{b}=".encode()), learning=False)
    m.run(bytes_to_bits(b"5\n99+1=2"), learning=False)
    assert m.table_hashes() == before
    m.restore(snap)
    assert m.snapshot()[1] == snap[1]


@pytest.mark.parametrize("route", ["f4", "f5"])
def test_causality_the_stage_of_epoch_e_does_not_depend_on_lines_of_epoch_e(route):
    held, _, _, epochs = stream()
    other = list(epochs)
    # change one line of epoch 2 (a held-out equation appears there in the second stream)
    a, b = held[0]
    other[1] = epochs[1] + f"{a}+{b}={a + b}\n".encode()
    m1, m2 = machine(route, sweeps=40), machine(route, sweeps=40)
    for m, ep in ((m1, epochs), (m2, other)):
        m.run(bytes_to_bits(ep[0]), learning=True)
        m.end_epoch()
        m.run(bytes_to_bits(ep[1]), learning=True)
    d1, d2 = m1.stage_dist(2, 3), m2.stage_dist(2, 3)
    assert (d1 is None and d2 is None) or np.array_equal(d1, d2)
    assert m1._fit_eqs == m2._fit_eqs


def test_stage_reads_only_answer_bytes_and_is_inactive_elsewhere():
    _, seen, _, epochs = stream()
    m = machine("f4")
    read(m, epochs[:2])
    ep, eu = np.empty(8 * 20), np.zeros(8 * 20, dtype=np.int64)
    a, b = seen[0]
    line = f"{a}+{b}={a + b}\n"
    _, _, on = m._stage(np.frombuffer(line.encode(), dtype=np.uint8), ep[:8 * len(line)], eu[:8 * len(line)],
                        (None, None), False)
    n = len(f"{a}+{b}=")
    assert on and np.isnan(ep[:8 * n]).all() and not np.isnan(ep[8 * n:8 * (n + len(str(a + b)) + 1)]).any()


# -- gate (f), and the point of the experiment: the family stage works at the top and is diluted at O's slot ----------

def test_stage_regret_is_within_two_bits_per_node_and_top_slot_generalizes():
    held, _, _, epochs = stream(epochs=8)
    sp = Spelling(19)
    scores = {}
    for slot in ("after0", "top"):
        m = machine("f4", slot)
        read(m, epochs)
        assert m.stage_regret()["max_excess_bits"] <= 2.0 + 1e-6
        hit = []
        for a, b in held:
            cs = m.stage_dist(a, b)
            if cs is None:
                continue
            start = m.snapshot()
            m.run(bytes_to_bits(f"{a}+{b}=".encode()), learning=False)
            mid = m.snapshot()
            costs = []
            for o in sp.spell:
                m.restore(mid)
                costs.append(float(-np.log2(m.run(bytes_to_bits(o), learning=False)).sum()))
            m.restore(start)
            hit.append(int(np.argmin(costs) == a + b))
        scores[slot] = float(np.mean(hit))
    assert scores["top"] == 1.0


def test_cp_stage_drops_tokens_it_cannot_fit_and_abstains_on_them():
    rng = np.random.default_rng(0)
    eqs = [(a, b, int(rng.integers(0, 19))) for a in range(10) for b in range(a, 10)]
    both = eqs + [(b, a, c) for a, b, c in eqs if a != b]
    st = CPStage.build(10, 19, both, seed=0, sweeps=60)
    assert len(st.dropped) > 0
    some = next(iter(st.dropped))
    assert st.dist(some, 1) is None


def test_the_default_slot_is_the_top_after_amendment_1(e7):
    assert e7.DEFAULT_SLOT == "top" and FamilyConfig().slot == "top"
    assert FamilyConfig(route="f4", slot="after0").slot == "after0"
    with pytest.raises(ValueError):
        FamilyConfig(slot="middle")
