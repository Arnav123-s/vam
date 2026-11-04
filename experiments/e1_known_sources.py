"""Experiment E1: VAM-1 on sources whose best possible loss is known (experiments/PROTOCOL.md, E1).

    python experiments/e1_known_sources.py            # full run, writes experiments/results/e1.json
    python experiments/e1_known_sources.py --small    # a quick smoke run, writes e1_small.json
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from _common import losses, provenance, save

from vam.baselines import kt_context_q
from vam.bitkernel import BitMachine
from vam.realizations.vam1 import Config, PointerEngine
from vam.sources import ByteMarkov, DelayedEcho, IncompleteCodes, MarkovBits

SEED = 20261018
KS = (2, 4, 8, 12, 16)
LAWS = {"constant": {"consolidating": False, "dt": 0.05}, "consolidating": {"consolidating": True, "dt": 1.0}}
ECHO_P = 0.05
MASS = {"cells": 6, "context": 4, "thetas": (0.0, 0.2, 0.4, 0.8), "delays": (6, 9)}


def sources() -> dict:
    return {
        "S1 order-2 Markov bits": MarkovBits(),
        "S2 T2.2 incomplete codes": IncompleteCodes(),
        "S3 order-1 bytes over 16 letters": ByteMarkov.random(seed=1234),
        "S4 delayed echo d=6": DelayedEcho(d=6, p=ECHO_P),
        "S4 delayed echo d=9": DelayedEcho(d=9, p=ECHO_P),
    }


def context_sweep(log2_bits: int) -> list[dict]:
    rows = []
    for i, (name, src) in enumerate(sources().items()):
        t0 = time.perf_counter()
        sample = src.sample(1 << log2_bits, np.random.default_rng(SEED + i))
        row = {"source": name, "bits": len(sample.bits), "seed": SEED + i,
               "oracle": losses(sample.oracle_q),
               "entropy_rate": src.entropy_rate() if hasattr(src, "entropy_rate") else None,
               "by_k": []}
        for k in KS:
            entry = {"k": k, "window_floor": src.window_floor(k),
                     "kt": losses(kt_context_q(sample.bits, k))}
            for law, kw in LAWS.items():
                bm = BitMachine(Config(cells=k, context=k, **kw))
                entry[law] = losses(bm.run(sample.bits))
            row["by_k"].append(entry)
        row["seconds"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        print(f"{name}: {row['seconds']} s", flush=True)
    return rows


def mass_sweep(log2_bits: int) -> list[dict]:
    rows = []
    for j, d in enumerate(MASS["delays"]):
        src = DelayedEcho(d=d, p=ECHO_P)
        sample = src.sample(1 << log2_bits, np.random.default_rng(SEED + 100 + j))
        for theta in MASS["thetas"]:
            t0 = time.perf_counter()
            cfg = Config(cells=MASS["cells"], context=MASS["context"], theta=theta, einselect="all",
                         consolidating=True, dt=1.0)
            emit_seed = SEED + 200 + 10 * j + MASS["thetas"].index(theta)
            pe = PointerEngine(cfg, rng=np.random.default_rng(emit_seed))      # draws the emitted bits
            q = np.array([pe.step(int(b)) for b in sample.bits])
            rows.append({"source": f"S4 delayed echo d={d}", "theta": theta, "bits": len(q),
                         "seed": SEED + 100 + j, "emission_seed": emit_seed, "oracle": losses(sample.oracle_q),
                         "window_floor": src.window_floor(MASS["context"]), "vam1": losses(q),
                         "seconds": round(time.perf_counter() - t0, 1)})
            print(f"mass d={d} theta={theta}: {rows[-1]['vam1']} ({rows[-1]['seconds']} s)", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="2^16 bits and 2^12 for the mass sweep")
    args = ap.parse_args()
    log2_bits, log2_mass = (16, 12) if args.small else (22, 18)
    t0 = time.perf_counter()
    payload = {
        "experiment": "E1 known sources",
        "protocol": "experiments/PROTOCOL.md#e1-known-sources-the-test-sources",
        "provenance": provenance(seed=SEED, ks=list(KS), laws=LAWS, mass=MASS, echo_p=ECHO_P,
                                 log2_bits=log2_bits, log2_mass_bits=log2_mass),
        "context_sweep": context_sweep(log2_bits),
        "mass_sweep": mass_sweep(log2_mass),
    }
    payload["seconds"] = round(time.perf_counter() - t0, 1)
    print(save("e1_small" if args.small else "e1", payload))


if __name__ == "__main__":
    main()
