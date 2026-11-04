"""N-L4, N-L5, N-L6 on the quantum chain: Markov blankets (conditional mutual information, containment of a
disturbance), the Renyi-2 Page measure, unit-rate sinks with Re lambda = -2, and the governed MPO engine."""

import math

import numpy as np
import pytest
from scipy.linalg import expm

from vam.bits import bytes_to_bits
from vam.nfa import blankets as bl
from vam.nfa import bounds
from vam.nfa.governor import (
    EntropyBond,
    GovernedEngine,
    SinkMaps,
    interior_sites,
    operator_entropies,
    sink_liouvillian,
    sink_superoperator,
)
from vam.realizations.vam1 import Config, DenseMaps
from vam.realizations.vam1_mpo import OperatorEngine

TEXT = bytes_to_bits(b"Blankets screen the cells. A sink drains what is not verified. ")


def test_cmi_of_known_states():
    ghz = np.zeros(8)
    ghz[[0, 7]] = 1 / np.sqrt(2)
    assert bl.cmi(bl.dense_reduced(np.outer(ghz, ghz), 3), [0], [1], [2]) == pytest.approx(1.0, abs=1e-12)
    p1, T = np.array([0.3, 0.7]), np.array([[0.9, 0.1], [0.2, 0.8]])
    P = np.array([p1[a] * T[a, b] * T[b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    assert bl.cmi(bl.dense_reduced(np.diag(P), 3), [0], [1], [2]) == pytest.approx(0.0, abs=1e-12)  # Markov chain
    assert bl.cmi(bl.dense_reduced(np.eye(8) / 8, 3), [0], [1], [2]) == pytest.approx(0.0, abs=1e-12)


def test_engine_reduced_densities_match_dense():
    cfg = Config(3, 2, theta=0.4, dephasing=0.7)
    oe, dense = OperatorEngine(cfg), DenseMaps(cfg).reader()
    rng = np.random.default_rng(1)
    for b in TEXT[:12]:
        rec = dense.step(int(b), rng)
        oe.force(int(b), int(rec.outcome))
    red = bl.dense_reduced(dense.state.R, 7)
    for sites in ([0, 1], [2, 3, 4], [1, 5, 6]):
        assert np.abs(red(sites) - oe.reduced_density(sites)).max() < 1e-10
    assert bl.cmi(red, [0, 1], [2, 3], [4, 5]) == pytest.approx(bl.cmi_profile(oe)[0], abs=1e-9)


def test_sink_eigenvalue_is_minus_two():
    L = sink_liouvillian(1.0)
    assert sorted(np.linalg.eigvals(L).real) == pytest.approx([-2.0, -2.0, 0.0, 0.0])
    assert np.abs(expm(L) - sink_superoperator(1.0)).max() < 1e-14
    assert np.abs(expm(0.3 * sink_liouvillian(0.5)) - sink_superoperator(0.5, 0.3)).max() < 1e-14


def test_purity_is_the_norm_of_the_stored_vector():
    cfg = Config(3, 2, theta=0.4)
    e = GovernedEngine(cfg, always=True)
    for b in TEXT[:30]:
        e.force(int(b), 0)
    R = e.to_dense()
    assert e.purity() == pytest.approx(np.real(np.trace(R @ R)), abs=1e-12)
    assert e.renyi2() == pytest.approx(-math.log2(np.real(np.trace(R @ R))), abs=1e-10)


@pytest.mark.parametrize("cfg", [Config(3, 2, theta=0.4, dephasing=0.7), Config(3, 1, theta=1.0)])
def test_governed_engine_is_the_dense_machine_with_sinks(cfg):
    """Untruncated, the governed engine equals the dense machine whose evolution is followed by the sinks."""
    e = GovernedEngine(cfg, max_bond=10_000, cutoff=0.0, always=True)
    machine = SinkMaps(cfg, e.sink_sites).reader()
    rng = np.random.default_rng(3)
    for b in TEXT[:20]:
        tau = int(rng.integers(0, 2))
        rec = machine.force(int(b), tau)
        q = e.force(int(b), tau)
        assert q == pytest.approx(rec.q_bit, abs=1e-10)
    assert np.abs(e.to_dense() - machine.state.R).max() < 1e-10
    assert np.allclose(e.w, machine.state.structural["w"], atol=1e-12)
    assert set(e.sink_sites).isdisjoint(e.context_sites)                  # verified sectors are never drained
    assert e.sink_sites == interior_sites(cfg)


def test_page_threshold_starts_the_governor_and_halts_growth():
    cfg = Config(6, 2, theta=0.5)
    e = GovernedEngine(cfg, threshold=0.05, max_bond=48)
    for b in TEXT[:60]:
        e.force(int(b), 0)
        if e.governed:
            break
    assert e.governed and e.page_fraction() >= 0.05
    cap = e.max_bond
    assert cap <= 48
    for b in TEXT[60:90]:
        e.force(int(b), 0)
    assert max(e.bond_dims) <= cap                                         # the volume stopped expanding


def test_cache_bond_is_the_default():
    e = GovernedEngine(Config(4, 2, theta=0.3))
    assert e.max_bond == bounds.max_mpo_bond()


@pytest.mark.slow
def test_governor_keeps_the_mpo_faithful_and_the_blankets_tight():
    """8 cells, context einselection, theta = 0.5: the plain engine at bond 64 discards about 1.4 of the
    Hilbert-Schmidt weight over 40 events and leaks a disturbance outside its cone by about 6e-3; governed
    (unit-rate interior sinks, cache bond) it discards < 1e-5 and leaks < 1e-5, and neighbouring cells are
    nearly conditionally independent given the cell between them."""
    cfg = Config(8, 2, theta=0.5)
    plain = OperatorEngine(cfg, rng=np.random.default_rng(0), max_bond=64)
    gov = GovernedEngine(cfg, rng=np.random.default_rng(0), always=True)
    for b in TEXT[:40]:
        plain.step(int(b))
        gov.step(int(b))
    assert plain.discarded > 0.5 and gov.discarded < 1e-5
    rng = np.random.default_rng(2)
    nb, nt = [int(x) for x in rng.integers(0, 2, 5)], [int(x) for x in rng.integers(0, 2, 5)]
    leak_plain = bl.cone_violation(bl.containment(plain, 6, nb, nt), 6, 2)
    leak_gov = bl.cone_violation(bl.containment(gov, 6, nb, nt), 6, 2)
    assert leak_plain > 1e-3 and leak_gov < 1e-5
    assert max(bl.cmi_profile(gov)) < 0.05 < max(bl.cmi_profile(plain))
    assert max(operator_entropies(gov)) <= math.log2(bounds.max_mpo_bond()) + 1e-9


def test_containment_is_exact_without_truncation():
    """With learned weights and no truncation a disturbance outside the context region stays in its cone."""
    cfg = Config(6, 2, theta=0.6, einselect="all")
    e = OperatorEngine(cfg, rng=np.random.default_rng(5))
    for b in TEXT[:30]:
        e.step(int(b))
    assert e.discarded < 1e-20
    rng = np.random.default_rng(6)
    nb, nt = [int(x) for x in rng.integers(0, 2, 3)], [int(x) for x in rng.integers(0, 2, 3)]
    ch = bl.containment(e, 4, nb, nt)
    assert bl.cone_violation(ch, 4, 2) < 1e-12
    assert ch[0, 5] > 1e-2 and ch[1].max() > 1e-2                          # the disturbance is really there
    assert ch[0, :4].max() < 1e-12 and ch[0, -1] < 1e-12                   # and one cell per event is its speed


@pytest.mark.parametrize("cfg,free", [(Config(3, 2, theta=0.5), True), (Config(3, 1, theta=1.1), True),
                                      (Config(4, 2, theta=0.3), True), (Config(4, 1, theta=0.7), False),
                                      (Config(3, 2, theta=0.5, dephasing=5.0), False)])
def test_sinks_are_free_when_no_coherence_can_return(cfg, free):
    """With the context einselected exactly (g = inf), a coherence the coin makes in a cell splits into a
    right-moving and a left-moving part. With at most two cells outside the context the two parts can never
    meet again in one cell, so no interior coherence ever reaches a population: the sinks change no
    prediction at all. A longer interior, or a finite g, lets coherence return, and then they do. (Proof:
    reaching a population needs a coherence to annihilate by the coin in a cell m whose R part came from
    R_{m-1} and L part from L_{m+1}, so k + 1 <= m <= c - 2; AUDIT.md, Audit 2.)"""
    plain = OperatorEngine(cfg, max_bond=4096, cutoff=0.0)
    sinks = GovernedEngine(cfg, max_bond=4096, cutoff=0.0, always=True, sink_rate=40.0)
    taus = np.random.default_rng(7).integers(0, 2, 200)
    change = max(abs(plain.force(int(b), int(t)) - sinks.force(int(b), int(t)))
                 for b, t in zip(np.tile(TEXT, 2)[:200], taus, strict=True))
    assert (change < 1e-12) if free else (change > 1e-8)


def test_entropy_bond_is_the_effective_number_of_directions():
    rule = EntropyBond(cap=90, margin=2.0)
    assert rule(np.array([1.0, 0.0, 0.0])) == 2                            # a product cut: 2^0 = 1, times the margin
    assert rule(np.full(8, 1 / 8)) == 16                                   # 3 bits: 8 directions, times 2
    assert rule(np.full(4096, 1 / 4096)) == 90                             # never more than the cache allows
    assert EntropyBond(90, 1.0)(np.array([0.5, 0.5])) == 2


def test_entropy_governed_bonds_follow_the_area_law():
    """The entropy rule holds each cut at the size its entanglement needs: the predictions stay those of the
    fixed cache bond while the bonds are smaller."""
    cfg = Config(6, 2, theta=0.5)
    fixed = GovernedEngine(cfg, always=True)
    area = GovernedEngine(cfg, always=True, bond="entropy", margin=4.0)
    taus = np.random.default_rng(4).integers(0, 2, 48)
    change = max(abs(fixed.force(int(b), int(t)) - area.force(int(b), int(t))) for b, t in zip(TEXT[:48], taus,
                                                                                               strict=True))
    assert change < 1e-3
    assert sum(area.bond_dims) < sum(fixed.bond_dims)
    other = area.copy()
    assert other.max_bond is not area.max_bond and other.cap == area.cap
