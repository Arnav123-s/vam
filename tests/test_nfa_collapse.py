"""N-L7 collapse: the path integral over future bit strings, its least-action limit (the macro/micro
"max" planner exactly), its hbar = 1 limit (the machine's own predictions), the literal Feynman form,
and verified-only emission (N-L6)."""

import math

import numpy as np
import pytest

from vam.bitkernel import BitMachine
from vam.bits import bits_to_bytes, bytes_to_bits
from vam.nested import NestedConfig, NestedMachine
from vam.nfa import collapse
from vam.realizations.vam1 import Config

CONS = Config(cells=1, context=1, consolidating=True, dt=1.0)
TEXT = b"one two three four five six seven eight nine ten. the cat sat on the mat. " * 30


@pytest.fixture(scope="module")
def nested():
    nm = NestedMachine(NestedConfig(CONS, depths=(8, 16, 32, 64), table_bits=14, governor=True))
    nm.run(bytes_to_bits(TEXT[:-37]))
    return nm


def test_paths_carry_their_action(nested):
    r = collapse.Readout(nested)
    P, S = collapse.paths(r, r.state0, 4)
    assert len(P) == 16 and np.isclose(np.exp(-S).sum(), 1.0)
    m = nested.copy()
    for path, s in zip(P, S, strict=True):
        probe = m.copy()
        logp = 0.0
        for k in range(3, -1, -1):
            b = int(path >> k) & 1
            logp += math.log(probe.predict(b))
            probe.run(np.array([b], dtype=np.uint8), learning=False)
        assert s == pytest.approx(-logp, abs=1e-12)


def test_hbar_one_is_the_machines_own_prediction(nested):
    r = collapse.Readout(nested)
    for h in (1, 4, 8):
        Z = collapse.first_step(r, r.state0, h, 1.0)["Z"]
        assert Z[0] == pytest.approx(nested.predict(0), abs=1e-12)


def test_hbar_zero_is_least_action_and_equals_the_max_planner(nested):
    a, b = nested.copy(), nested.copy()
    assert a.generate(8 * 30, horizon=8) == collapse.generate(b, 8 * 30, horizon=8, hbar=0.0)
    assert a.reg == b.reg
    bm = BitMachine(Config(cells=16, context=16, consolidating=True, dt=1.0))
    bm.run(bytes_to_bits(TEXT[:-37]))
    c, d = bm.copy(), bm.copy()
    assert c.generate(80, horizon=8) == collapse.generate(d, 80, horizon=8, hbar=0.0)


def test_small_hbar_approaches_least_action(nested):
    r = collapse.Readout(nested)
    least = collapse.first_step(r, r.state0, 8, 0.0)
    first = least["least"][0] >> 7
    Z = collapse.first_step(r, r.state0, 8, 0.02)["Z"]
    assert Z[first] > 1 - 1e-6


def test_feynman_form_is_the_squared_amplitude(nested):
    r = collapse.Readout(nested)
    P, S = collapse.paths(r, r.state0, 6)
    hbar = 0.7
    amp = np.exp(-0.5 * S + 1j * S / hbar)
    A = np.array([amp[(P >> 5) == b].sum() for b in (0, 1)])
    Z = collapse.first_step(r, r.state0, 6, hbar, form="feynman")["Z"]
    assert np.allclose(Z, np.abs(A) ** 2 / (np.abs(A) ** 2).sum(), atol=1e-14)


def test_generation_never_learns_and_continues_the_text(nested):
    m = nested.copy()
    before = m._wn.copy()
    out = collapse.generate(m, 8 * 37, horizon=8, hbar=0.5, mode="sample", rng=np.random.default_rng(1))
    assert np.array_equal(m._wn, before)
    text = bits_to_bytes(np.array(out, dtype=np.uint8))
    assert text == b"ht nine ten. the cat sat on the mat. "


def test_verified_emission_uses_only_proven_sectors(nested):
    r_all, r_ver = collapse.Readout(nested), collapse.Readout(nested, verified=True)
    assert r_all.p0(r_all.state0) == pytest.approx(nested.predict(0), abs=1e-15)
    # with an impossible protection threshold only the coarsest level speaks
    strict = NestedMachine(NestedConfig(CONS, depths=(8, 16, 32, 64), table_bits=14, governor=True, protect=1e9))
    strict._wn[:], strict._tags[:], strict.reg = nested._wn, nested._tags, nested.reg
    coarse = collapse.Readout(strict, verified=True).p0(int(nested.reg))
    w0 = nested.level_table(0)[int(nested.reg) & 0xFF, 0]
    assert coarse == pytest.approx(0.5 * (1 + (1 - 2 * CONS.eps) * w0), abs=1e-15)
    assert 0.0 < r_ver.p0(r_ver.state0) < 1.0


def test_repetition_measures_loops():
    assert collapse.repetition(b"of the children of the children of the children", 8) > 0.4
    assert collapse.repetition(b"abcdefghijklmnopqrstuvwxyz", 8) == 0.0
