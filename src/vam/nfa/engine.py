"""The composed engine of N (DESIGN section 10): read, score and write through the seven layers.

A thin facade, with no new mathematics. It builds the machine the way ``experiments/e2_curriculum.make`` does
(the laws and nested sizes below are the same constants, checked by a test), reads a stream, scores a stream
frozen, and writes through ``vam.nfa.collapse``. The layers it composes:

* N-L1/N-L3 bounds (``bounds``) and the options in force, reported by :meth:`Engine.layers`;
* N-L3 coherent readout, N-L5 verified growth and N-L6 governor, as the options of the nested machine;
* N-L7 writing, by default the machine's own output instrument (hbar = 1, euclidean, sample; T10.1). Every
  other setting is a chosen realization of N-L7 (PROTOCOL E3 Amendment 1).

    python -m vam.nfa.engine read FILES... [--coherence C] [--verify V] [--governor] [--write PROMPT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..bitkernel import BitMachine
from ..bits import bits_to_bytes, bytes_to_bits
from ..nested import NestedConfig, NestedMachine
from ..realizations.vam1 import Config
from . import bounds, collapse

LAWS = {"literal_0.02": {"consolidating": False, "dt": 0.02}, "literal_0.05": {"consolidating": False, "dt": 0.05},
        "literal_0.2": {"consolidating": False, "dt": 0.2}, "counting": {"consolidating": True, "dt": 1.0}}
NESTED = {"depths": (8, 16, 24, 32, 48, 64), "table_bits": 22}
NFA_KEYS = ("depths", "coherence", "verify", "governor", "protect", "sink_rate")
BLOCK = 1 << 20                     # bytes per reading block (1 MiB), as E2


def build(spec: dict):
    """The machine of ``spec``: ``{"machine": "nested" | "window", "law": ..., "prior": ..., N options}``."""
    if spec["machine"] == "window":
        return BitMachine(Config(cells=spec["k"], context=spec["k"], **LAWS[spec["law"]]))
    law = Config(cells=1, context=1, **LAWS[spec["law"]])
    extra = {k: spec[k] for k in NFA_KEYS if k in spec}
    depths = tuple(extra.pop("depths", NESTED["depths"]))
    return NestedMachine(NestedConfig(law, depths=depths, table_bits=NESTED["table_bits"], prior=spec["prior"],
                                      **extra))


def _bits(machine, data: bytes, learning: bool) -> float:
    """Total code length (bits) of ``data`` in 1 MiB blocks, as ``experiments/e2_curriculum.read``."""
    total = 0.0
    for start in range(0, len(data), BLOCK):
        total += float(-np.log2(machine.run(bytes_to_bits(data[start:start + BLOCK]), learning=learning)).sum())
    return total


class Engine:
    def __init__(self, spec: dict | None = None) -> None:
        self.spec = dict(spec or {"machine": "nested", "law": "counting", "prior": 4.0})
        self.machine = build(self.spec)

    def read(self, data: bytes, learning: bool = True) -> float:
        """Read ``data`` (learning by default); bits per byte, each bit charged what the machine gave it before it."""
        return _bits(self.machine, data, learning) / max(1, len(data))

    def bpb(self, data: bytes) -> float:
        """Frozen bits per byte on a copy: the machine's storage is untouched."""
        return _bits(self.machine.copy(), data, False) / max(1, len(data))

    def write(self, prompt: bytes, n_bytes: int, hbar: float = 1.0, form: str = "euclidean", mode: str = "sample",
              verified: bool = False, seed: int = 0, horizon: int = 8) -> bytes:
        """Continue ``prompt`` by ``n_bytes`` on a frozen copy through N-L7. The default setting is the machine's own
        output instrument; ``seed`` seeds the draw of ``mode="sample"``."""
        m = self.machine.copy()
        m.run(bytes_to_bits(prompt), learning=False)
        rng = np.random.default_rng(seed) if mode == "sample" else None
        bits = collapse.generate(m, 8 * n_bytes, horizon=horizon, hbar=hbar, mode=mode, form=form,
                                 verified=verified, rng=rng)
        return bits_to_bytes(np.array(bits, dtype=np.uint8))

    def layers(self) -> dict:
        """The bounds and caps of N-L1 and N-L3 and the N options in force."""
        hw = bounds.Hardware.detect()
        out = {"bounds": bounds.summary(hw), "register_cap_bits": bounds.register_cap(hw),
               "max_density_dim": bounds.max_density_dim(hw), "max_mpo_bond": bounds.max_mpo_bond(hw),
               "spec": self.spec}
        m = self.machine
        if isinstance(m, NestedMachine):
            c = m.cfg
            out["options"] = {"depths": list(c.depths), "coherence": c.coherence, "verify": c.verify,
                              "governor": c.governor, "protect": c.protect, "sink_rate": c.sink_rate}
            out["counters"] = {"evictions": m.evictions, "refused": m.refused, "protected": m.protected,
                               "occupancy": {str(k): v for k, v in m.occupancy().items()}}
        return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m vam.nfa.engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rd = sub.add_parser("read", help="read files, report bits per byte, optionally write")
    rd.add_argument("files", nargs="+", type=Path)
    rd.add_argument("--coherence", type=float, default=0.0)
    rd.add_argument("--verify", type=int, default=1)
    rd.add_argument("--governor", action="store_true")
    rd.add_argument("--write", metavar="PROMPT", help="after reading, continue this prompt")
    rd.add_argument("--bytes", type=int, default=48)
    rd.add_argument("--hbar", type=float, default=1.0)
    rd.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    spec = {"machine": "nested", "law": "counting", "prior": 4.0}
    if a.coherence:
        spec["coherence"] = a.coherence
    if a.verify > 1:
        spec["verify"] = a.verify
    if a.governor:
        spec["governor"] = True
    eng = Engine(spec)
    for f in a.files:
        print(f"{f}: {eng.read(f.read_bytes()):.4f} bits per byte", file=sys.stderr)
    if a.write is not None:
        sys.stdout.buffer.write(eng.write(a.write.encode(), a.bytes, hbar=a.hbar, seed=a.seed) + b"\n")


if __name__ == "__main__":
    main()
