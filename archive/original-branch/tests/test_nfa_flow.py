"""N-L1 flow: every VAM-1 event is the unit-time solution of a differential equation dx/dt = f(x, t) on
the machine's whole state, and integrating it with an ODE solver reproduces the machine (reading and
generation, finite and infinite dephasing, both einselections, plasticity included)."""

import math

import numpy as np
import pytest
from scipy.linalg import expm

from vam.nfa.flow import SWAP, EventFlow, embed, exchange, trajectory_basis
from vam.realizations.vam1 import Config, DenseMaps, learn

FAST = [Config(2, 1, theta=0.3, dephasing=0.7), Config(2, 2, theta=0.5, dephasing=math.inf),
        Config(2, 1, theta=0.4, dephasing=0.5, einselect="all", consolidating=True), Config(2, 2, theta=0.0)]
SLOW = [Config(3, 2, theta=0.2, dephasing=1.3), Config(3, 3, theta=1.0, dephasing=math.inf, einselect="all")]


def test_exchange_flow_is_exactly_a_swap():
    H = exchange(0, 2, 3)
    assert np.allclose(H, H.conj().T)
    assert np.abs(expm(-1j * H) - embed(SWAP, (0, 2), 3)).max() < 1e-13


def test_embed_places_the_operator_on_its_qubits(rng):
    A = rng.normal(size=(4, 4))
    assert np.array_equal(embed(A, (1, 2), 4), np.kron(np.kron(np.eye(2), A), np.eye(2)))


def run_against_machine(cfg, events, seed):
    flow, machine = EventFlow(cfg), DenseMaps(cfg).reader()
    x, rng = flow.initial(), np.random.default_rng(seed)
    worst = 0.0
    for _ in range(events):
        b = int(rng.integers(0, 2))
        rec = machine.step(b, rng)
        res = flow.event(x, b, int(rec.outcome))
        x = res.state
        s = machine.state.structural
        worst = max(worst, np.abs(x.R - machine.state.R).max(), abs(res.q_b - rec.q_bit),
                    abs(res.p_tau - rec.p_outcome), np.abs(x.w - s["w"]).max(), np.abs(x.n - s["n"]).max())
    return worst


@pytest.mark.parametrize("cfg", FAST, ids=lambda c: f"c{c.cells}k{c.context}t{c.theta}g{c.dephasing}{c.einselect}")
def test_reading_events_are_integrated_flows(cfg):
    assert run_against_machine(cfg, 10, 3) < 1e-8


@pytest.mark.slow
@pytest.mark.parametrize("cfg", SLOW, ids=lambda c: f"c{c.cells}k{c.context}t{c.theta}g{c.dephasing}{c.einselect}")
def test_reading_events_are_integrated_flows_three_cells(cfg):
    assert run_against_machine(cfg, 4, 5) < 1e-8


def test_generation_events_are_integrated_flows():
    cfg = Config(2, 2, theta=0.6, dephasing=0.4)
    maps = DenseMaps(cfg)
    reader = maps.reader()
    rng = np.random.default_rng(11)
    for b in (1, 0, 1):
        reader.step(b, rng)
    gen = maps.generator(reader.state)
    flow = EventFlow(cfg)
    s = gen.state
    x = flow.initial()
    x.R, x.w, x.n = s.R.copy(), s.structural["w"].copy(), s.structural["n"].copy()
    for tau in (0, 1, 1, 0):
        rec = gen.force(None, tau)
        res = flow.generation_event(x, tau)
        x = res.state
        assert np.abs(x.R - gen.state.R).max() < 1e-8 and abs(res.p_tau - rec.p_outcome) < 1e-9
        assert np.array_equal(x.w, gen.state.structural["w"])                   # outputs are not evidence


def test_loss_is_the_integral_of_the_observation_generator():
    """-ln q_b = int_0^1 Tr(K_b R(t)) dt: the bit's code length is accumulated along the trajectory."""
    cfg = Config(2, 1, theta=0.3, dephasing=0.7)
    flow = EventFlow(cfg)
    x = flow.initial()
    x.w[:] = [0.6, -0.3]
    for b, tau in ((1, 0), (0, 1), (1, 1)):                 # a mixed state: on a basis state R never moves
        x = flow.event(x, b, tau).state
    k = flow._k(x.w, 1)
    R, ln_q = flow._filter(k, x.R, None, "observe", 0)
    O = np.diag(np.exp(-k / 2))
    q = np.real(np.trace(O @ x.R @ O))
    assert np.abs(R - x.R).max() > 1e-3                      # the filter moved the state along the way
    assert abs(np.real(np.trace(np.diag(k) @ x.R)) + math.log(q)) > 1e-3   # so the integral is not Tr(K R(0))
    assert ln_q == pytest.approx(-math.log(q), abs=1e-11)
    assert np.abs(R - O @ x.R @ O / q).max() < 1e-11


def test_plasticity_is_the_solution_of_its_differential_law():
    cfg = Config(2, 1, consolidating=True)
    flow = EventFlow(cfg)
    w, n, a = np.array([0.2, -0.7]), np.array([3.0, 0.5]), np.array([0.25, 0.75])
    fw, fn = flow._plasticity(w, n, a, 1, None, 0)
    rw, rn = learn(cfg, w, n, a, 1)
    assert np.abs(fw - rw).max() < 1e-11 and np.abs(fn - rn).max() < 1e-11


def test_trajectory_records_a_sentence_and_its_basis():
    cfg = Config(2, 2, theta=0.4, dephasing=0.5)
    flow = EventFlow(cfg)
    bits, taus = [1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0]
    _, losses, Xm, labels = flow.trajectory(bits, taus, samples=2)
    assert len(losses) == len(bits) and np.all(losses > 0)
    assert Xm.shape[1] == cfg.dim ** 2 and len(labels) == Xm.shape[0]
    basis = trajectory_basis(Xm)
    assert 1 <= basis["rank"] <= Xm.shape[0]
    recon = basis["coordinates"] @ basis["basis"]
    assert np.abs(recon - Xm).max() < 1e-10                     # the kept vectors span the whole trajectory
    assert basis["singular_values"][basis["rank"]:].max(initial=0.0) <= basis["floor"]
