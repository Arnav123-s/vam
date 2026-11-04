"""VAM-1 on an MPO density (vam1_mpo): exactness against the dense, pointer and bit engines (finite dephasing
included), the event cone, a bond set by the register, and truncated runs measured against exact engines."""

import math
import time

import numpy as np
import pytest

from vam.bitkernel import BitMachine
from vam.channels import random_channel
from vam.linalg import random_density, random_unitary
from vam.realizations.vam1 import Config, DenseMaps, PointerEngine
from vam.realizations.vam1_mpo import OperatorEngine, superoperator, superoperator_two
from vam.tensornet import MPS

GRID = [Config(c, k, theta=th, dephasing=g, einselect=e)
        for c in (2, 3) for k in range(1, c + 1) for th in (0.0, 0.4, 1.0)
        for g in (0.0, 0.7, math.inf) for e in ("context", "all")]
#: a representative slice of GRID that runs by default; the whole grid is marked slow
FAST = [Config(2, 1, theta=0.4), Config(2, 2, theta=1.0, dephasing=0.0, einselect="all"),
        Config(3, 2, theta=0.4, dephasing=0.7), Config(3, 3, theta=1.0, dephasing=0.7, einselect="all"),
        Config(3, 3, theta=1.0, einselect="all"), Config(3, 1, theta=0.0, dephasing=0.0)]
#: English text for the truncated runs (the experiments read real books; a test must not depend on them)
TEXT = (b"A textbook states a rule, shows it at work on an example, and then asks the reader to use it "
        b"alone. The machine reads every byte as eight bits, oldest bit first along the chain. ")


def ids(cfg):
    return f"c{cfg.cells}k{cfg.context}t{cfg.theta}g{cfg.dephasing}{cfg.einselect}"


def text_bits(n):
    return [int(b) for b in np.unpackbits(np.frombuffer(TEXT * (1 + n // (8 * len(TEXT))), np.uint8))[:n]]


def vec_state(R, n):
    """``vec(R)`` of an ``n``-qubit density as an exact MPS with local index ``2 s + s'``."""
    v = np.asarray(R).reshape((2,) * (2 * n)).transpose([q for j in range(n) for q in (j, n + j)])
    return MPS.from_dense(v.reshape(-1), [4] * n)


def assert_same_as_dense(oe, m, tol=1e-11):
    assert np.allclose(oe.to_dense(), m.state.R, atol=tol, rtol=0)
    assert np.allclose(oe.w, m.state.structural["w"], atol=1e-12, rtol=0)
    assert np.allclose(oe.n, m.state.structural["n"], atol=1e-12, rtol=0)
    assert abs(oe.trace() - 1.0) < 1e-12


def compare_with_dense(cfg, seed, drawn=8, prescribed=4):
    """Equal generators (drawn emissions), then prescribed emissions; every event is compared."""
    rng = np.random.default_rng(seed)
    bits = [int(b) for b in rng.integers(0, 2, drawn + prescribed)]
    taus = [int(t) for t in rng.integers(0, 2, prescribed)]
    m = DenseMaps(cfg).reader()
    oe = OperatorEngine(cfg, rng=np.random.default_rng(seed + 1))
    g = np.random.default_rng(seed + 1)
    for b in bits[:drawn]:
        for x in (0, 1):
            assert oe.predict(x) == pytest.approx(m.predict(x), abs=1e-12)
        rec = m.step(b, g)
        assert oe.step(b) == pytest.approx(rec.q_bit, abs=1e-12)
        assert oe.last_output == rec.outcome
        assert oe.last_output_probability == pytest.approx(rec.p_outcome, abs=1e-12)
        assert_same_as_dense(oe, m)
    for b, tau in zip(bits[drawn:], taus, strict=True):
        rec = m.force(b, tau)
        assert oe.force(b, tau) == pytest.approx(rec.q_bit, abs=1e-12)
        assert oe.last_output_probability == pytest.approx(rec.p_outcome, abs=1e-12)
        assert_same_as_dense(oe, m)
    assert oe.discarded < 1e-20                       # only numerically zero singular values were dropped


@pytest.mark.parametrize("cfg", FAST, ids=ids)
def test_untruncated_engine_equals_the_dense_reader(cfg):
    compare_with_dense(cfg, seed=11)


@pytest.mark.slow
@pytest.mark.parametrize("cfg", GRID, ids=ids)
def test_untruncated_engine_equals_the_dense_reader_on_the_whole_grid(cfg):
    """cells 2 and 3, every context, theta in {0, 0.4, 1}, dephasing in {0, 0.7, inf}, both einselections."""
    compare_with_dense(cfg, seed=12)


# -- the maps on vec(rho) ----------------------------------------------------------------------------------------

def test_superoperators_act_as_the_maps(rng):
    """``sum_K K (x) conj(K)`` on one site and ``U (x) conj(U)`` on two sites act on ``vec(rho)`` as the channel
    and the unitary act on ``rho``; the swap moves a site's whole content (ket and bra index)."""
    R1 = random_density(2, rng)
    ch = random_channel(2, rng, n_kraus=3)
    x = vec_state(R1, 1)
    x.apply_one(superoperator(ch.kraus), 0)
    assert np.allclose(x.to_dense().reshape(2, 2), ch(R1), atol=1e-13)
    R2 = random_density(4, rng)
    U = random_unitary(4, rng)
    oe = OperatorEngine(Config(1, 1))
    oe.x = vec_state(np.kron(R2, np.diag([1.0, 0.0])), 3)
    oe.x.apply_two(superoperator_two(U), 0)
    assert np.allclose(oe.to_dense(), np.kron(U @ R2 @ U.conj().T, np.diag([1.0, 0.0])), atol=1e-13)
    oe.x.apply_two(oe._swap, 1)                        # qubit 1 <-> qubit 2: |0><0| moves to the middle
    expected = np.einsum("abcABC->acbACB", np.kron(U @ R2 @ U.conj().T, np.diag([1.0, 0.0])).reshape((2,) * 6))
    assert np.allclose(oe.to_dense(), expected.reshape(8, 8), atol=1e-13)


@pytest.mark.parametrize("einselect", ["context", "all"])
@pytest.mark.parametrize("cells,context", [(2, 1), (3, 2), (3, 3)])
def test_finite_dephasing_mpo_is_the_sector_dissipator(rng, cells, context, einselect):
    """The bond-2 MPO ``e^-g Id + (1 - e^-g) prod_i Delta_i`` is ``R -> e^-g R + (1 - e^-g) R * mask`` of DenseMaps,
    on a random full-rank density of the whole register."""
    cfg = Config(cells, context, dephasing=0.7, einselect=einselect)
    M = DenseMaps(cfg)
    R = random_density(cfg.dim, rng)
    oe = OperatorEngine(cfg)
    oe.x = vec_state(R, cfg.qubits)
    oe._evolve()                                       # theta = 0: the dissipator alone
    assert np.allclose(oe.to_dense(), M.keep * R + (1 - M.keep) * R * M.mask, atol=1e-13)
    assert max(W.shape[3] for W in oe._dissipator) == (2 if len(oe._dissipator) > 1 else 1)


def test_reentered_outputs_are_not_evidence(rng):
    cfg = Config(2, 2, theta=0.5, dephasing=0.7, einselect="all", dt=0.2)
    oe = OperatorEngine(cfg)
    for b in rng.integers(0, 2, 12):
        oe.step(int(b))
    branches = [oe.copy() for _ in (0, 1)]
    for tau, branch in enumerate(branches):
        branch.force(1, tau)
    assert np.array_equal(branches[0].w, branches[1].w)
    assert not np.allclose(branches[0].to_dense(), branches[1].to_dense())   # the emission did change the state


# -- the pointer engine, the bit kernel, and a bond set by the register -----------------------------------------

def test_equals_the_pointer_engine_for_300_bits_with_a_bond_set_by_the_register():
    """Fully einselected, c = k = 4, theta = 0.5, equal generators, untruncated: every prediction, emission and
    weight equal the pointer engine, and the whole distribution over the 512 configurations every 50 events.

    The exact bond stays small: at most 12 (measured), where the register allows 256 and the local
    purification of vam1_tn outgrows memory within a few events at six cells. The density holds no history."""
    cfg = Config(4, 4, theta=0.5, einselect="all")
    oe = OperatorEngine(cfg, rng=np.random.default_rng(7))
    pe = PointerEngine(cfg, rng=np.random.default_rng(7))
    for t, b in enumerate(np.random.default_rng(1).integers(0, 2, 300)):
        assert oe.predict(int(b)) == pytest.approx(pe.predict(int(b)), abs=1e-12)
        assert oe.step(int(b)) == pytest.approx(pe.step(int(b)), abs=1e-12)
        assert oe.last_output == pe.last_output
        assert np.allclose(oe.w, pe.w, atol=1e-12, rtol=0)
        if (t + 1) % 50 == 0:
            assert np.allclose(np.real(np.diag(oe.to_dense())), pe.p, atol=1e-12, rtol=0)
    assert max(oe.bond_dims) <= 16
    assert oe.discarded < 1e-20


def run_against_bit_kernel(nbits):
    """theta = 0, c = 16, k = 8: predictions and weights equal the bit kernel; the state stays a product."""
    cfg = Config(16, 8)
    oe, bm = OperatorEngine(cfg, rng=np.random.default_rng(3)), BitMachine(cfg, use_numba=False)
    for b in np.random.default_rng(2).integers(0, 2, nbits):
        assert oe.step(int(b)) == pytest.approx(bm.step(int(b)), abs=1e-12)
        assert np.allclose(oe.w, bm.w, atol=1e-12, rtol=0) and np.allclose(oe.n, bm.n, atol=1e-12, rtol=0)
    assert max(oe.bond_dims) == 1
    assert oe.discarded < 1e-20


def test_massless_engine_equals_the_bit_kernel_short():
    run_against_bit_kernel(300)


@pytest.mark.slow
def test_massless_engine_equals_the_bit_kernel_for_2000_bits():
    run_against_bit_kernel(2000)


# -- generation ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dephasing", [math.inf, 0.7])
def test_generation_equals_the_dense_generator(rng, dephasing):
    """c = 3: read with prescribed emissions, then generate with prescribed outputs and with equal generators."""
    cfg = Config(3, 2, theta=0.6, dephasing=dephasing)
    reader = DenseMaps(cfg).reader()
    oe = OperatorEngine(cfg, rng=np.random.default_rng(5))
    for b, tau in zip(rng.integers(0, 2, 8), rng.integers(0, 2, 8), strict=True):
        reader.force(int(b), int(tau))
        oe.force(int(b), int(tau))
    w = oe.w.copy()
    gen = DenseMaps(cfg).generator(reader.state)
    for tau in rng.integers(0, 2, 8):
        rec = gen.force(None, int(tau))
        assert oe.generate(int(tau)) == int(tau)
        assert oe.last_output_probability == pytest.approx(rec.p_outcome, abs=1e-12)
        assert_same_as_dense(oe, gen)
    g = np.random.default_rng(5)                       # oe.rng is unused so far: equal generators
    for _ in range(8):
        rec = gen.step(None, g)
        assert oe.generate() == rec.outcome
        assert_same_as_dense(oe, gen)
    assert np.array_equal(oe.w, w)                     # plasticity is off during generation


# -- the event cone ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dephasing", [0.0, 0.7, math.inf])
def test_exact_event_cone_at_twenty_cells(dephasing):
    """Two engines that differ only in the first bit agree, after T events, on every cell >= T and on the port.

    Learning is off and the emissions are prescribed and equal, so the weights stay 0 and both observations
    are proportional to the identity: what separates the engines is the field alone."""
    cfg, T = Config(20, 3, theta=0.6, dephasing=dephasing), 8
    rng = np.random.default_rng(4)
    common, taus = [int(b) for b in rng.integers(0, 2, T)], [int(t) for t in rng.integers(0, 2, T)]
    engines = []
    for first in (0, 1):
        oe = OperatorEngine(cfg)
        for t in range(T):
            oe.force(first if t == 0 else common[t], taus[t], learning=False)
        assert oe.discarded < 1e-20
        engines.append(oe)
    a, b = engines
    for j in range(cfg.cells):
        same = np.allclose(a.reduced_density([2 * j, 2 * j + 1]), b.reduced_density([2 * j, 2 * j + 1]),
                           atol=1e-12, rtol=0)
        if j >= T:
            assert same                                # beyond the cone: exactly unchanged
        if j == T - 1:
            assert not same                            # the first bit itself, T - 1 cells in
    assert np.allclose(a.reduced_density([2 * cfg.cells]), b.reduced_density([2 * cfg.cells]), atol=1e-12)


# -- truncation, measured against exact engines, and scale -------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("theta", [0.3, 0.6])
def test_truncated_engine_follows_the_pointer_engine_at_eight_cells(theta, capsys):
    """c = 8, k = 4, fully einselected, max_bond = 64, 600 bits of English text, equal generators: every emission
    equal and every prediction within 1e-5 of the exact pointer engine (measured: 1.6e-6 at theta 0.3 and
    2.4e-7 at theta 0.6; the truncated purification of vam1_tn misses by 1.9e-2 at six cells)."""
    cfg = Config(8, 4, theta=theta, einselect="all")
    oe = OperatorEngine(cfg, rng=np.random.default_rng(7), max_bond=64)
    pe = PointerEngine(cfg, rng=np.random.default_rng(7))
    err = 0.0
    for b in text_bits(600):
        err = max(err, abs(oe.step(b) - pe.step(b)))
        assert oe.last_output == pe.last_output
    with capsys.disabled():
        print(f"\n[vam1_mpo c=8 theta={theta} D=64] max |q_MPO - q_pointer| = {err:.2e}, "
              f"discarded = {oe.discarded:.2e}")
    assert err < 1e-5
    assert abs(oe.trace() - 1.0) < 1e-10


@pytest.mark.slow
def test_truncated_engine_follows_the_exact_one_with_context_einselection_at_six_cells(capsys):
    """c = 6, k = 4, theta = 0.3, context einselection (coherent L movers): max_bond = 64 against the untruncated
    engine (whose bond reaches 172) over 60 bits of text: every emission equal, every prediction within 1e-5."""
    cfg = Config(6, 4, theta=0.3)
    exact = OperatorEngine(cfg, rng=np.random.default_rng(7))
    trunc = OperatorEngine(cfg, rng=np.random.default_rng(7), max_bond=64)
    err = 0.0
    for b in text_bits(60):
        err = max(err, abs(exact.step(b) - trunc.step(b)))
        assert exact.last_output == trunc.last_output
    with capsys.disabled():
        print(f"\n[vam1_mpo c=6 context D=64] max |q_D64 - q_exact| = {err:.2e}, exact bond {max(exact.bond_dims)}")
    assert exact.discarded < 1e-20 and err < 1e-5


@pytest.mark.slow
@pytest.mark.parametrize("dephasing", [math.inf, 0.5])
def test_scale_run_stays_normalized_bounded_and_nearly_positive(dephasing, capsys):
    """c = 24, k = 4, theta = 0.3, context einselection, max_bond = 64, 100 bits.

    Asserted: trace 1 to 1e-10 at every event, bonds and storage within the cap, and every reduced density
    tested positive to within 1e-8 (positivity is not structural for an MPO density, so it is measured).
    Printed: events per second, discarded Hilbert-Schmidt weight, the smallest reduced eigenvalue."""
    cfg = Config(24, 4, theta=0.3, dephasing=dephasing)
    max_bond = 64
    oe = OperatorEngine(cfg, rng=np.random.default_rng(7), max_bond=max_bond)
    cap = cfg.qubits * max_bond * 4 * max_bond * 16 + oe.w.nbytes + oe.n.nbytes
    worst_trace, worst_bytes = 0.0, 0
    t0 = time.perf_counter()
    for b in np.random.default_rng(1).integers(0, 2, 100):
        oe.step(int(b))
        worst_trace = max(worst_trace, abs(oe.trace() - 1.0))
        worst_bytes = max(worst_bytes, oe.nbytes())
    rate = 100 / (time.perf_counter() - t0)
    smallest = min(float(np.linalg.eigvalsh(oe.reduced_density(s))[0])
                   for s in ([0, 1], [10, 11], [46, 47], [0, 2, 4, 6], [20, 21, 22, 23], [48]))
    with capsys.disabled():
        print(f"\n[vam1_mpo c=24 g={dephasing}] {rate:.2f} events/s, discarded = {oe.discarded:.2e}, "
              f"max state {worst_bytes / 1e6:.2f} MB, smallest reduced eigenvalue {smallest:.2e}")
    assert worst_trace < 1e-10
    assert max(oe.bond_dims) <= max_bond and worst_bytes <= cap
    assert smallest > -1e-8
