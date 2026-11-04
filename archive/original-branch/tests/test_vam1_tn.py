"""VAM-1 on an LPDO (vam1_tn): exactness against the dense, pointer and bit engines, the event cone, the two
purification modes, and a measured run at scale."""

import math
import time

import numpy as np
import pytest

from vam.bitkernel import BitMachine
from vam.channels import random_channel
from vam.linalg import random_density, random_unitary
from vam.realizations.vam1 import Config, DenseMaps, PointerEngine, bit_probabilities
from vam.realizations.vam1_tn import TensorEngine, lpdo_density
from vam.tensornet import LPDO

GRID = [Config(c, k, theta=th, dephasing=g, einselect=e)
        for c in (2, 3) for k in range(1, c + 1) for th in (0.0, 0.4, 1.0)
        for g in (0.0, math.inf) for e in ("context", "all")]
#: a representative slice of GRID that runs by default; the whole grid is marked slow
FAST = [Config(2, 1, theta=0.4), Config(2, 2, theta=1.0, dephasing=0.0, einselect="all"),
        Config(3, 2, theta=0.4), Config(3, 3, theta=1.0, einselect="all"), Config(3, 1, theta=0.0, dephasing=0.0)]
#: (drawn, prescribed) events per run: the global mode is bounded by the register, the local one grows with the
#: number of events (see test_local_purification_keeps_the_history_and_global_purification_is_minimal)
EVENTS = {"global": (6, 6), "local": (3, 2)}


def ids(cfg):
    return f"c{cfg.cells}k{cfg.context}t{cfg.theta}g{cfg.dephasing}{cfg.einselect}"


def assert_same_as_dense(te, m, tol=1e-11):
    assert np.allclose(te.to_dense(), m.state.R, atol=tol, rtol=0)
    assert np.allclose(te.w, m.state.structural["w"], atol=1e-12, rtol=0)
    assert np.allclose(te.n, m.state.structural["n"], atol=1e-12, rtol=0)
    assert abs(te.trace() - 1.0) < 1e-12


def compare_with_dense(cfg, purification, seed):
    """Equal generators (drawn emissions), then prescribed emissions; every event is compared."""
    drawn, prescribed = EVENTS[purification]
    rng = np.random.default_rng(seed)
    bits = [int(b) for b in rng.integers(0, 2, drawn + prescribed)]
    taus = [int(t) for t in rng.integers(0, 2, prescribed)]
    m = DenseMaps(cfg).reader()
    te = TensorEngine(cfg, rng=np.random.default_rng(seed + 1), purification=purification)
    g = np.random.default_rng(seed + 1)
    for b in bits[:drawn]:
        for x in (0, 1):
            assert te.predict(x) == pytest.approx(m.predict(x), abs=1e-12)
        rec = m.step(b, g)
        assert te.step(b) == pytest.approx(rec.q_bit, abs=1e-12)
        assert te.last_output == rec.outcome
        assert te.last_output_probability == pytest.approx(rec.p_outcome, abs=1e-12)
        assert_same_as_dense(te, m)
    for b, tau in zip(bits[drawn:], taus, strict=True):
        rec = m.force(b, tau)
        assert te.force(b, tau) == pytest.approx(rec.q_bit, abs=1e-12)
        assert te.last_output_probability == pytest.approx(rec.p_outcome, abs=1e-12)
        assert_same_as_dense(te, m)
    assert te.discarded < 1e-20                       # only numerically zero singular values were dropped


@pytest.mark.parametrize("purification", ["global", "local"])
@pytest.mark.parametrize("cfg", FAST, ids=ids)
def test_untruncated_engine_equals_the_dense_reader(cfg, purification):
    compare_with_dense(cfg, purification, seed=11)


@pytest.mark.slow
@pytest.mark.parametrize("purification", ["global", "local"])
@pytest.mark.parametrize("cfg", GRID, ids=ids)
def test_untruncated_engine_equals_the_dense_reader_on_the_whole_grid(cfg, purification):
    """cells 2 and 3, every context, theta in {0, 0.4, 1}, dephasing in {0, inf}, both einselections."""
    compare_with_dense(cfg, purification, seed=12)


def test_only_dephasing_zero_and_infinity_are_supported():
    for g in (0.3, 1.0, 50.0):
        with pytest.raises(ValueError, match="dephasing"):
            TensorEngine(Config(2, 1, dephasing=g))
    with pytest.raises(ValueError, match="purification"):
        TensorEngine(Config(2, 1), purification="other")
    TensorEngine(Config(2, 1, dephasing=0.0))
    TensorEngine(Config(2, 1, dephasing=math.inf))


@pytest.mark.parametrize("cells,context", [(2, 2), (3, 2), (3, 3)])
def test_observation_mpo_is_the_dense_observation_operator(rng, cells, context):
    """The MPO on sites 0..2k-2 (identity elsewhere) is exactly O_b of DenseMaps: the sector convention
    s = sum_j R_j 2^j holds, and its bond is at most 2^floor(k/2)."""
    cfg = Config(cells, context)
    te, M = TensorEngine(cfg), DenseMaps(cfg)
    w = rng.uniform(-1, 1, cfg.sectors)
    for b in (0, 1):
        mpo = te.observation_mpo(bit_probabilities(w, cfg.eps, b))
        D = np.ones((1, 1, 1))
        for W in mpo:
            D = np.einsum("ija,astb->isjtb", D, W).reshape(D.shape[0] * 2, D.shape[1] * 2, W.shape[3])
        rest = cfg.qubits - len(mpo)
        dense = np.kron(D[:, :, 0], np.eye(2 ** rest))
        assert np.allclose(dense, M.kraus_observe(w, b).kraus[0], atol=1e-13)
        assert max(W.shape[3] for W in mpo) <= 2 ** (context // 2)


def test_reentered_outputs_are_not_evidence(rng):
    cfg = Config(2, 2, theta=0.5, einselect="all", dt=0.2)
    te = TensorEngine(cfg, purification="global")
    for b in rng.integers(0, 2, 12):
        te.step(int(b))
    branches = [te.copy() for _ in (0, 1)]
    for tau, branch in enumerate(branches):
        branch.force(1, tau)
    assert np.array_equal(branches[0].w, branches[1].w)
    assert not np.allclose(branches[0].to_dense(), branches[1].to_dense())   # the emission did change the state


# -- the pointer engine and the bit kernel ------------------------------------------------------------------

def run_against_pointer(cfg, purification, nbits, full_every):
    bits = [int(b) for b in np.random.default_rng(1).integers(0, 2, nbits)]
    te = TensorEngine(cfg, rng=np.random.default_rng(7), purification=purification)
    pe = PointerEngine(cfg, rng=np.random.default_rng(7))
    configuration = list(range(cfg.qubits))
    for t, b in enumerate(bits):
        assert te.predict(b) == pytest.approx(pe.predict(b), abs=1e-12)
        assert te.step(b) == pytest.approx(pe.step(b), abs=1e-12)
        assert te.last_output == pe.last_output
        assert np.allclose(te.w, pe.w, atol=1e-12, rtol=0)
        if (t + 1) % full_every == 0:          # the whole configuration distribution, from one contraction
            assert np.allclose(te.lp.probabilities(configuration), pe.p, atol=1e-11, rtol=0)
    assert te.discarded < 1e-20
    return te


@pytest.mark.slow
def test_equals_the_pointer_engine_for_300_bits():
    """Fully einselected, theta = 0.5, equal generators: 300 bits at c = k = 3 in the global purification mode.

    The requested c = 6, k = 4 run is not feasible exactly: the local purification outgrows memory after a few
    events (next test) and the pointer support, hence rank(rho), reaches 2304 there (measured), which makes
    the global concentration sweep far too slow; see the module docstring of vam1_tn."""
    run_against_pointer(Config(3, 3, theta=0.5, einselect="all"), "global", 300, full_every=50)


def test_equals_the_pointer_engine_at_six_cells_while_exact_storage_is_affordable():
    """c = 6, k = 4, theta = 0.5: exact for the first four events, including the full distribution over all
    8192 configurations; the purification then outgrows any test (bond 170 after five events, measured)."""
    te = run_against_pointer(Config(6, 4, theta=0.5, einselect="all"), "local", 4, full_every=1)
    assert max(te.bond_dims) <= 64


def test_massless_engine_equals_the_bit_kernel_short():
    run_against_bit_kernel(300)


@pytest.mark.slow
def test_massless_engine_equals_the_bit_kernel_for_2000_bits():
    run_against_bit_kernel(2000)


def run_against_bit_kernel(nbits):
    """theta = 0, c = 16, k = 8: predictions and weights equal the bit kernel; the state stays a product."""
    cfg = Config(16, 8)
    te, bm = TensorEngine(cfg, rng=np.random.default_rng(3)), BitMachine(cfg, use_numba=False)
    for b in np.random.default_rng(2).integers(0, 2, nbits):
        assert te.step(int(b)) == pytest.approx(bm.step(int(b)), abs=1e-12)
        assert np.allclose(te.w, bm.w, atol=1e-12, rtol=0) and np.allclose(te.n, bm.n, atol=1e-12, rtol=0)
    assert max(te.bond_dims) == 1 and max(te.kraus_dims) == 1
    assert te.discarded < 1e-20


# -- generation ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("purification", ["global", "local"])
def test_generation_equals_the_dense_generator(rng, purification):
    """c = 3: read with prescribed emissions, then generate with prescribed outputs and with equal generators."""
    cfg = Config(3, 2, theta=0.6)
    steps = 8 if purification == "global" else 2      # local storage grows with the events (see below)
    reader = DenseMaps(cfg).reader()
    te = TensorEngine(cfg, rng=np.random.default_rng(5), purification=purification)
    for b, tau in zip(rng.integers(0, 2, steps), rng.integers(0, 2, steps), strict=True):
        reader.force(int(b), int(tau))
        te.force(int(b), int(tau))
    w = te.w.copy()
    gen = DenseMaps(cfg).generator(reader.state)
    for tau in rng.integers(0, 2, steps):
        rec = gen.force(None, int(tau))
        assert te.generate(int(tau)) == int(tau)
        assert te.last_output_probability == pytest.approx(rec.p_outcome, abs=1e-12)
        assert_same_as_dense(te, gen)
    g = np.random.default_rng(5)                       # te.rng is unused so far: equal generators
    for _ in range(steps):
        rec = gen.step(None, g)
        assert te.generate() == rec.outcome
        assert_same_as_dense(te, gen)
    assert np.array_equal(te.w, w)                     # plasticity is off during generation


# -- the event cone ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dephasing", [0.0, math.inf])
def test_exact_event_cone_at_twenty_cells(dephasing):
    """Two engines that differ only in the first bit agree, after T events, on every cell >= T and on the port.

    Learning is off and the emissions are prescribed and equal, so the weights stay 0 and both observations
    are proportional to the identity: what separates the engines is the field alone (with learning on, the
    weights would differ after the first bit and the context observation would differ too)."""
    cfg, T = Config(20, 3, theta=0.6, dephasing=dephasing), 8
    rng = np.random.default_rng(4)
    common, taus = [int(b) for b in rng.integers(0, 2, T)], [int(t) for t in rng.integers(0, 2, T)]
    engines = []
    for first in (0, 1):
        te = TensorEngine(cfg)
        for t in range(T):
            te.force(first if t == 0 else common[t], taus[t], learning=False)
        assert te.discarded < 1e-20
        engines.append(te)
    a, b = engines
    for j in range(cfg.cells):
        same = np.allclose(a.reduced_density([2 * j, 2 * j + 1]), b.reduced_density([2 * j, 2 * j + 1]),
                           atol=1e-12, rtol=0)
        if j >= T:
            assert same                                # beyond the cone: exactly unchanged
        if j == T - 1:
            assert not same                            # the first bit itself, T - 1 cells in
    assert np.allclose(a.reduced_density([2 * cfg.cells]), b.reduced_density([2 * cfg.cells]), atol=1e-12)


# -- purification ----------------------------------------------------------------------------------------------

def test_local_purification_keeps_the_history_and_global_purification_is_minimal():
    """The two modes hold the same state; the local one stores far more than the register needs.

    Two cells, theta = 0.4, context einselection: after eight events the local purification has a bond larger
    than the dimension 32 of the whole register (the environment's history: Kraus legs of different sites
    stay entangled although that means nothing for rho), also with a relative cutoff of 1e-12, which then
    discards almost nothing; the global one has Kraus dimension rank(rho) and bonds equal to the ranks of the
    reduced densities of the left parts."""
    cfg = Config(2, 1, theta=0.4)
    rng = np.random.default_rng(8)
    bits, taus = rng.integers(0, 2, 8), rng.integers(0, 2, 8)
    eng = {}
    for mode, kwargs in (("local", {}), ("cutoff", {"cutoff": 1e-12}), ("global", {"purification": "global"})):
        te = TensorEngine(cfg, **kwargs)
        for b, tau in zip(bits, taus, strict=True):
            te.force(int(b), int(tau))
        eng[mode] = te
    R = eng["global"].to_dense()
    assert np.allclose(eng["local"].to_dense(), R, atol=1e-11)
    assert max(eng["local"].bond_dims) > 2 ** cfg.qubits
    assert max(eng["cutoff"].bond_dims) > 2 ** cfg.qubits and eng["cutoff"].discarded < 1e-9
    assert np.allclose(eng["cutoff"].to_dense(), R, atol=1e-8)
    rank = lambda M: int(np.sum(np.linalg.eigvalsh(M) > 1e-12))
    g = eng["global"]
    assert g.kraus_dims[-1] == rank(R) and g.kraus_dims[:-1] == (1,) * cfg.qubits
    for j, D in enumerate(g.bond_dims[:-1]):
        assert D == rank(g.reduced_density(list(range(j + 1))))


def test_meet_in_the_middle_density_equals_the_purification_vector(rng):
    for dims in [(2, 2, 2), (2, 3, 2, 2), (2, 2, 2, 2, 2, 1)]:
        lp = LPDO.product([random_density(d, rng) for d in dims])
        for j in range(len(dims) - 1):
            lp.apply_two(random_unitary(dims[j] * dims[j + 1], rng), j)
            if dims[j] > 1:
                lp.apply_channel(random_channel(dims[j], rng, n_kraus=2), j)
        assert np.allclose(lpdo_density(lp), lp.to_dense(), atol=1e-13)


# -- truncation and scale --------------------------------------------------------------------------------------

@pytest.mark.slow
def test_truncated_local_purification_is_not_a_faithful_approximation_at_six_cells(capsys):
    """Measured, not hoped for: with max_bond = 32 and max_kraus = 4 the truncated engine drifts away from the
    exact pointer engine by more than 1e-3 within 60 bits, and its discarded weight exceeds 1: the truncation
    cuts the stored environment history, not a small tail."""
    cfg = Config(6, 4, theta=0.5, einselect="all")
    te = TensorEngine(cfg, rng=np.random.default_rng(7), max_bond=32, max_kraus=4)
    pe = PointerEngine(cfg, rng=np.random.default_rng(7))
    err = 0.0
    for b in np.random.default_rng(1).integers(0, 2, 60):
        err = max(err, abs(te.step(int(b)) - pe.step(int(b))))
    with capsys.disabled():
        print(f"\n[vam1_tn c=6 truncated] max |q_TN - q_pointer| = {err:.3e}, discarded = {te.discarded:.3e}")
    assert abs(te.trace() - 1.0) < 1e-10
    assert err > 1e-3 and te.discarded > 1.0


@pytest.mark.slow
def test_scale_run_stays_normalized_positive_and_bounded(capsys):
    """c = 24, k = 4, theta = 0.3, context einselection, dephasing = inf, max_bond = 32 (max_kraus = 4).

    Asserted: trace 1 to 1e-10 at every event, positive reduced densities, storage within the declared caps.
    Measured and printed (asserted only finite): events per second and the discarded weight. This checks the
    machinery at scale, not fidelity: the discarded weight is large (about 0.6 per event), and the six-cell
    test above shows that this truncation is not a faithful approximation."""
    cfg = Config(24, 4, theta=0.3, einselect="context")
    max_bond, max_kraus = 32, 4
    te = TensorEngine(cfg, rng=np.random.default_rng(7), max_bond=max_bond, max_kraus=max_kraus)
    bits = np.random.default_rng(1).integers(0, 2, 100)
    cap = cfg.qubits * max_bond * 2 * max_kraus * max_bond * 16 + te.w.nbytes + te.n.nbytes
    worst_trace, worst_bytes = 0.0, 0
    t0 = time.perf_counter()
    for b in bits:
        te.step(int(b))
        worst_trace = max(worst_trace, abs(te.trace() - 1.0))
        worst_bytes = max(worst_bytes, te.nbytes())
    rate = len(bits) / (time.perf_counter() - t0)
    assert worst_trace < 1e-10
    assert max(te.bond_dims) <= max_bond and max(te.kraus_dims) <= max_kraus and worst_bytes <= cap
    smallest = min(float(np.linalg.eigvalsh(te.reduced_density(s))[0])
                   for s in ([0, 1], [10, 11], [46, 47], [0, 2, 4, 6], [20, 21, 22, 23], [48]))
    assert smallest > -1e-12
    assert math.isfinite(te.discarded) and math.isfinite(rate)
    with capsys.disabled():
        print(f"\n[vam1_tn c=24] {rate:.2f} events/s, discarded = {te.discarded:.3e} over {len(bits)} events, "
              f"max state {worst_bytes / 1e6:.2f} MB, smallest reduced eigenvalue {smallest:.2e}")
