"""E3 additions after the result (experiments/PROTOCOL.md, E3 Amendment 4). Everything here is labelled "added after the result".

    python experiments/e3_t4.py --manifest experiments/data/pg19/MANIFEST.json
    python experiments/e3_t4.py --manifest ... --smoke      # code check: 256 KiB of text, a 5-cell chain

(a) T4.2: N-L2 on text with a count floor. Part 1 found the split-half Page edge at about 1 at every depth on A-train
(rank 0 everywhere, cutoff 1.1): a context seen in only one half gives a whitened difference of order 1 whatever the
sample size. The floor removes contexts and continuations with fewer than ``FLOOR = 256`` samples. Fixed here, before this
script reads PG-19: 256 is the number of values a next byte can take, and a row of the byte Hankel matrix is a
distribution over that alphabet, so a row with fewer samples than the alphabet cannot show a direction above noise. Not
tuned on any text. The result is a measurement only and is never used to cut depths: ranks on a filtered row set fall with
depth because rows are removed. The same profile is computed for a byte-shuffled copy of the same bytes (the null, seeded),
where rank is at most 1.

(b) T4.3: the 8-cell chain of Part 1's blankets (context 2, theta 0.5, 40 bits of English, 5 forced events after
disturbing cell 6, the same seeds), governed with ``trigger="cache"`` so the sinks act before the cache cap (90) binds:
the largest bond against the cap, at the Page fraction of one half (the pre-declared main row) with and without the halt,
and at the cap itself. Part 1 found the Page-point engine discarding 0.371 of the weight before its trigger.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import e3_nfa as E
import numpy as np
from _common import provenance, save
from e2_curriculum import SEED, streams

from vam.nfa import blankets as bl
from vam.nfa import page
from vam.nfa.governor import GovernedEngine, operator_entropies
from vam.realizations.vam1 import Config

FLOOR = 256                  # see the module docstring
MARGIN = 0.1                 # the split-half margin of ``page.page_spectrum``, as Part 1


def floor_profile(data: bytes, floor: int = FLOOR, depths=(1, 2, 3, 4, 5, 6, 7)) -> list[dict]:
    """Page rank of the byte Hankel matrix (previous d bytes, next byte) restricted to rows and columns with at least
    ``floor`` samples."""
    out = []
    for d in depths:
        full, halves = page.byte_hankel_halves(data, d, 1)
        r, c = np.asarray(full.sum(1)).ravel(), np.asarray(full.sum(0)).ravel()
        keep_r, keep_c = np.flatnonzero(r >= floor), np.flatnonzero(c >= floor)
        row = {"depth_bytes": d, "rows": int(len(r)), "cols": int(len(c)), "rows_kept": int(len(keep_r)),
               "cols_kept": int(len(keep_c)), "samples": float(full.sum())}
        if len(keep_r) < 2 or len(keep_c) < 2:
            out.append({**row, "rank": None, "cutoff": None, "samples_kept": 0.0})
            continue
        f = full[keep_r][:, keep_c]
        sp = page.page_spectrum(f, [h[keep_r][:, keep_c] for h in halves], MARGIN)
        out.append({**row, "rank": sp["rank"], "cutoff": sp["cutoff"], "samples_kept": float(f.sum())})
    return out


def text_part(A_train: bytes, smoke: bool) -> dict:
    t0 = time.perf_counter()
    text = A_train[: (1 << 18) if smoke else (4 << 20)]
    null = bytes(np.random.default_rng(SEED).permutation(np.frombuffer(text, dtype=np.uint8)))
    return {"floor": FLOOR, "text_bytes": len(text), "A_train": floor_profile(text),
            "byte_shuffled_null": floor_profile(null), "seconds": round(time.perf_counter() - t0, 1)}


def chain_part(smoke: bool) -> dict:
    """Part 1's blankets setting with the cache trigger; every number comes from the same functions as Part 1."""
    cells, nbits, events, cell = (5, 16, 3, 3) if smoke else (8, 40, 5, 6)
    cap = E.CACHE_BOND
    cfg = Config(cells, 2, theta=0.5)
    bits = E.bytes_to_bits(E.CHAIN_TEXT)[:nbits]
    taus = np.random.default_rng(7).integers(0, 2, nbits)
    engines = {
        "cache_trigger_half_cap": GovernedEngine(cfg, max_bond=cap, trigger="cache"),
        "cache_trigger_half_cap_no_halt": GovernedEngine(cfg, max_bond=cap, trigger="cache", halt=False),
        "cache_trigger_full_cap": GovernedEngine(cfg, max_bond=cap, trigger="cache", threshold=1.0),
    }
    rng = np.random.default_rng(2)
    nb, nt = [int(x) for x in rng.integers(0, 2, events)], [int(x) for x in rng.integers(0, 2, events)]
    out = {}
    for name, e in engines.items():
        run = E.chain_run(e, bits, taus)
        ch = bl.containment(e, cell, nb, nt)
        out[name] = {"discarded": run["discarded"], "bonds": run["bonds"], "seconds": run["seconds"],
                     "renyi2_per_qubit_end": run["renyi2_per_qubit"][-1],
                     "governed_since": run["governed_since"], "cone_leak": bl.cone_violation(ch, cell, cfg.context),
                     "cmi_bits": bl.cmi_profile(e), "operator_entropy_bits": operator_entropies(e)}
        print("  chain", name, {k: out[name][k] for k in ("discarded", "cone_leak", "governed_since", "seconds")},
              flush=True)
    return {"config": {"cells": cells, "context": 2, "theta": 0.5, "bits": nbits, "events": events, "cell": cell,
                       "cache_cap": cap}, "engines": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    S = streams(args.manifest, args.smoke)
    payload = {
        "experiment": "E3 additions after the result (T4.2, T4.3)",
        "protocol": "experiments/PROTOCOL.md, E3 Amendment 4",
        "label": "added after the result",
        "provenance": provenance(seed=SEED, manifest=str(args.manifest), smoke=args.smoke,
                                 manifest_sha256=E.sha256(args.manifest.read_bytes()), cache_bond_used=E.CACHE_BOND),
        "T4_2_text": text_part(S["A_train"], args.smoke),
        "T4_3_chain": chain_part(args.smoke),
        "seconds": None,
    }
    payload["seconds"] = round(time.perf_counter() - t_start, 1)
    print(save("e3_t4_smoke" if args.smoke else "e3_t4", payload))


if __name__ == "__main__":
    main()
