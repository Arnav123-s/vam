"""E4 control added after the result (PROTOCOL E4 Amendment 2): E3's base configuration with sink rate 0.

    python experiments/e4_t.py --manifest experiments/data/pg19/MANIFEST.json
    python experiments/e4_t.py --manifest ... --smoke      # code check: 256 KiB streams

E4's chosen machine also dropped the sinks (rate 0), so E3 (rate 0.1) compared with E4 (R2, R5) mixes the hierarchy with
that change. This control is the E3 machine (upper levels off, bit-identical to E3's by E4 Part 0) with sink rate 0, trained
on A-train and read on the same frozen splits, with the Part 5 decomposition by window class. Reported only as
"added after the result"; it changes no E4 number and selects nothing.
Results go to results/e4_t.json (e4_t_smoke.json for --smoke).
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import e3_nfa as E
import numpy as np
from _common import provenance, save
from e2_curriculum import SEED, frozen, make, read, streams
from e4_hier import counters, decomposition, hspec, window_classes

from vam.bits import bytes_to_bits

SPEC = hspec(hier_levels="", sink_rate=0.0)       # E3's chosen row with the sinks off


def control(S: dict) -> dict:
    evals = ("A_test", "A_heldout", "B_test", "B_heldout")
    m = make(SPEC)
    train = read(m, S["A_train"], learning=True)
    out = {"spec": SPEC, "train_bpb": train, "A_validation": frozen(m, S["A_validation"]),
           **{e: frozen(m, S[e]) for e in evals}, "counters": counters(m)}
    q = m.copy().run(bytes_to_bits(S["A_test"]), learning=False)
    out["decomposition"] = decomposition(q, window_classes(S["A_train"], S["A_test"]))
    del m, q
    gc.collect()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    S = streams(args.manifest, args.smoke)
    payload = {
        "experiment": "E4 control added after the result: E3's base with sink rate 0",
        "protocol": "experiments/PROTOCOL.md, E4 Amendment 2",
        "label": "added after the result",
        "provenance": provenance(seed=SEED, manifest=str(args.manifest), smoke=args.smoke,
                                 manifest_sha256=E.sha256(args.manifest.read_bytes())),
        "E3_base_sink_rate_0": control(S),
    }
    payload["seconds"] = round(time.perf_counter() - t0, 1)
    print(save("e4_t_smoke" if args.smoke else "e4_t", E.plain(payload)))


if __name__ == "__main__":
    main()
