"""E6b Amendment 1: the deepest-level loop copy, the top mixture and the mixed machine, on small machines."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from test_wordclass import kmachine, small
from vam.bits import bytes_to_bits
from vam.nfa import hier, mixloop
from vam.nfa import wcmix as MX

SPEC = Path(__file__).resolve().parents[1] / "spec"


@pytest.fixture(scope="module")
def text() -> bytes:
    return (b" ".join(p.read_bytes() for p in sorted(SPEC.glob("*.md"))) * 3)[:420000]


def strip_mix(src: str) -> str:
    out, skip = [], False
    for line in src.splitlines():
        if "# E6b-mix begin" in line:
            skip = True
        if not skip:
            out.append(line.replace("  # E6b-mix signature", ""))
        if "# E6b-mix end" in line:
            skip = False
    return "\n".join(out)


# -- gate (a): the loop copy is hier's loop plus the recording lines ---------------------------------------------------

def test_the_deep_loop_is_the_hier_loop_plus_the_recording_lines():
    a = inspect.getsource(hier._make_loop)
    b = inspect.getsource(mixloop.make_loop_deep)
    a = re.sub(r'""".*?"""', '""""""', a, flags=re.S)
    b = strip_mix(re.sub(r'""".*?"""', '""""""', b, flags=re.S))
    a = a.replace("def _make_loop(jit):", "def make_loop_deep(jit):")
    b = b.replace("door_on,\n             deep):", "door_on):")
    assert a.split() == b.split()
    assert mixloop.CHUNK == hier.CHUNK


def test_the_deep_loop_changes_no_probability_and_records_a_level(text):
    bits = bytes_to_bits(text[:120000])
    ref, off, deep_m = hier.HierMachine(small()), kmachine("off"), kmachine("off")
    q_ref = ref.run(bits, learning=True)
    q_deep, deep = deep_m.run_deep(bits, learning=True)
    assert np.array_equal(q_ref, q_deep) and np.array_equal(q_ref, off.run(bits, learning=True))
    assert deep.min() >= 0 and deep.max() < deep_m.levels and (deep == 0).any() and (deep > 0).any()
    more = bytes_to_bits(text[120000:150000])
    assert np.array_equal(ref.copy().run(more, learning=False), deep_m.copy().run_deep(more, learning=False)[0])


# -- the mixture ------------------------------------------------------------------------------------------------------

def test_the_mixture_of_identical_inputs_is_that_input_and_frozen_reads_update_nothing():
    rng = np.random.default_rng(0)
    n = 20000
    bits = rng.integers(0, 2, n).astype(np.uint8)
    q = np.clip(rng.random(n), 0.01, 0.99)
    deep = rng.integers(0, 14, n).astype(np.int8)
    ev, acc, out = np.zeros(MX.NODES), np.zeros((MX.NODES, 4)), np.empty(n)
    MX.top_mix(q, q, deep, bits, ev, acc, False, 1.0, out)
    assert np.allclose(out, q) and (ev == 0).all() and (acc == 0).all()


def test_the_mixture_costs_at_most_two_bits_per_node_against_the_better_input():
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(40):
        n = 4000
        bits = rng.integers(0, 2, n).astype(np.uint8)
        qa = np.clip(rng.random(n) ** rng.uniform(0.3, 3), 0.02, 0.98)
        qb = np.clip(rng.random(n) ** rng.uniform(0.3, 3), 0.02, 0.98)
        deep = rng.integers(0, 4, n).astype(np.int8)
        ev, acc, out = np.zeros(MX.NODES), np.zeros((MX.NODES, 4)), np.empty(n)
        MX.top_mix(qa, qb, deep, bits, ev, acc, True, 1.0, out)
        excess = acc[:, 2] - np.minimum(acc[:, 0], acc[:, 1])
        worst = max(worst, float(excess.max()))
    assert worst <= 2.0 + 1e-9


# -- the mixed machine -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["real", "random"])
def test_components_equal_the_standalone_machines_frozen_reads_change_nothing(text, mode):
    bits = bytes_to_bits(text[:300000])
    mm = MX.MixMachine(kmachine("off"), kmachine(mode))
    q = mm.run(bits, learning=True)
    solo_e, solo_k = kmachine("off"), kmachine(mode)
    assert np.array_equal(mm.last["e4"], solo_e.run(bits, learning=True))
    assert np.array_equal(mm.last["k"], solo_k.run(bits, learning=True))
    assert np.isfinite(q).all() and (q > 0).all() and mm.regret()["max_excess_bits"] <= 2.0 + 1e-9
    before, snap = mm.table_hashes(), mm.snapshot()
    more = bytes_to_bits(text[300000:340000])
    qm = mm.run(more, learning=False)
    assert mm.table_hashes() == before
    mm.restore(snap)
    assert np.array_equal(mm.run(more, learning=False), qm)
    assert np.array_equal(mm.last["e4"], solo_e.run(more, learning=False))


def test_the_mixture_needs_an_e4_machine_and_a_k_machine():
    with pytest.raises(ValueError):
        MX.MixMachine(kmachine("real"), kmachine("real"))
    with pytest.raises(ValueError):
        MX.MixMachine(kmachine("off"), kmachine("off"))
