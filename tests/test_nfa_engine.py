"""The composed engine: it is the same machine as the experiments build, read, scored and written."""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pytest

from vam.bits import bits_to_bytes, bytes_to_bits
from vam.nfa import collapse, engine

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
import e2_curriculum as e2  # noqa: E402

SPECS = [{"machine": "nested", "law": "counting", "prior": 4.0},
         {"machine": "nested", "law": "counting", "prior": 4.0, "verify": 2, "coherence": 0.5},
         {"machine": "nested", "law": "counting", "prior": 4.0, "governor": True, "protect": 0.5}]


@pytest.fixture(scope="module")
def text() -> bytes:
    root = Path(__file__).resolve().parents[1]
    raw = b" ".join(Path(f).read_bytes() for f in sorted(glob.glob(str(root / "spec" / "*.md"))))
    return raw[: 1 << 16]


def test_constants_are_the_experiments_constants():
    assert engine.LAWS == e2.LAWS and engine.NFA_KEYS == e2.NFA_KEYS
    assert engine.NESTED == {k: e2.NESTED[k] for k in ("depths", "table_bits")}
    assert engine.BLOCK == e2.BLOCK


@pytest.mark.parametrize("spec", SPECS, ids=["plain", "verify_coherent", "governor"])
def test_read_and_bpb_equal_the_experiments_path(text, spec):
    eng = engine.Engine(spec)
    ref = e2.make(spec)
    head, tail = text[:40000], text[40000:]
    assert eng.read(head) == e2.read(ref, head, learning=True)
    assert eng.bpb(tail) == e2.frozen(ref, tail)
    assert eng.bpb(tail) == e2.frozen(ref, tail)               # frozen scoring leaves the engine as it was
    assert np.array_equal(eng.machine._wn, ref._wn) and np.array_equal(eng.machine._tags, ref._tags)


def test_write_equals_collapse_generate_with_the_same_arguments(text):
    eng = engine.Engine()
    eng.read(text)
    prompt = text[100:140]
    for kw in ({}, dict(hbar=0.5, form="feynman", mode="sample", seed=3), dict(hbar=0.0, mode="collapse"),
               dict(verified=True, seed=1)):
        m = eng.machine.copy()
        m.run(bytes_to_bits(prompt), learning=False)
        rng = np.random.default_rng(kw.get("seed", 0)) if kw.get("mode", "sample") == "sample" else None
        bits = collapse.generate(m, 8 * 12, horizon=8, hbar=kw.get("hbar", 1.0), mode=kw.get("mode", "sample"),
                                 form=kw.get("form", "euclidean"), verified=kw.get("verified", False), rng=rng)
        assert eng.write(prompt, 12, **kw) == bits_to_bytes(np.array(bits, dtype=np.uint8))


def test_write_default_is_the_machines_own_instrument_and_never_learns(text):
    eng = engine.Engine()
    eng.read(text)
    wn = eng.machine._wn.copy()
    out = eng.write(text[:30], 8)
    assert len(out) == 8 and np.array_equal(eng.machine._wn, wn)
    assert eng.write(text[:30], 8, seed=5) == eng.write(text[:30], 8, seed=5)


def test_layers_report_bounds_and_options():
    spec = {"machine": "nested", "law": "counting", "prior": 4.0, "verify": 3, "coherence": 0.25}
    out = engine.Engine(spec).layers()
    assert out["register_cap_bits"] == 64 and out["options"]["verify"] == 3 and out["options"]["coherence"] == 0.25
    assert {"max_mpo_bond", "max_density_dim", "counters"} <= out.keys()


def test_cli_reads_and_writes(tmp_path, capsysbinary):
    f = tmp_path / "a.txt"
    f.write_bytes(b"the quick brown fox jumps over the lazy dog. " * 50)
    engine.main(["read", str(f), "--write", "the quick", "--bytes", "6"])
    cap = capsysbinary.readouterr()
    assert b"bits per byte" in cap.err and len(cap.out) == 7
