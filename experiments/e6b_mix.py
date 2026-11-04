"""E6b Amendment 1 (the reserved top mixture): the arms M and M random (experiments/PROTOCOL.md, E6b).

    python experiments/e6b_mix.py --manifest experiments/data/pg19/MANIFEST.json --smoke
    python experiments/e6b_mix.py --manifest experiments/data/pg19/MANIFEST.json              # the two arms
    python experiments/e6b_mix.py --manifest ... --resume                                      # skip the units saved

**M** is E4 with E4+K read side by side with the top mixture (one node per class of the deepest level contributing to
the E4+K readout); **M random** is E4 with E4+K random. Each arm reads A-train once with learning, then A-test and
A-heldout frozen. The components reproduce the E4 and K arms of results/e6b.json (gate (d), 1e-9). The criterion is
computed by ``experiments/report.py`` from results/e6b_mix.json, beside the original C-E6b. Run one job at a time.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import e3_nfa as E
import numpy as np
from _common import RESULTS, provenance, save
from e2_curriculum import BLOCK, streams
from e4_hier import counters, decomposition, window_classes
from e6_operator import GateFailure, book_lengths, check_spec
from e6b_classes import make_machine, unseen_context_sets

from vam.bits import bytes_to_bits
from vam.nfa import wordclass as WC
from vam.nfa.wcmix import MixMachine

ARMS = ("M", "M random")
KMODE = {"M": "real", "M random": "random"}
KARM = {"M": "E4+K", "M random": "E4+K random"}
STOP_SECONDS = 45 * 60.0


def read_mix(mm: MixMachine, data: bytes, learning: bool, curve: list | None = None) -> dict:
    """Read ``data`` in 1 MiB blocks (as E6b's reads); the mixture's and the components' bits per byte."""
    tot = {"mix": 0.0, "e4": 0.0, "k": 0.0}
    for a in range(0, len(data), BLOCK):
        chunk = data[a:a + BLOCK]
        q = mm.run(bytes_to_bits(chunk), learning=learning)
        bits = float(-np.log2(q).sum())
        tot["mix"] += bits
        tot["e4"] += float(-np.log2(mm.last["e4"]).sum())
        tot["k"] += float(-np.log2(mm.last["k"]).sum())
        if curve is not None:
            curve.append(bits / len(chunk))
    return {k: v / max(1, len(data)) for k, v in tot.items()}


def arm(name: str, spec: dict, S: dict, cls: np.ndarray, sets: dict, books: list[int], smoke: bool, V: int,
        e6b: dict | None) -> dict:
    compile_bytes = (1 << 15) if smoke else (1 << 20)
    e4 = make_machine(spec, WC.WordClassConfig(mode="off"))
    k = make_machine(spec, WC.WordClassConfig(mode=KMODE[name], V=V, compile_bytes=compile_bytes))
    mm = MixMachine(e4, k)
    out: dict = {"V": V, "compile_bytes": compile_bytes}
    curve: list[float] = []
    t0 = time.perf_counter()
    out["train_bpb"] = read_mix(mm, S["A_train"], learning=True, curve=curve)
    out["train_seconds"] = time.perf_counter() - t0
    if out["train_seconds"] > STOP_SECONDS:
        raise GateFailure(f"arm {name} ran past {STOP_SECONDS:.0f} s while reading A-train")
    out["learning_curve_bpb_per_MiB"] = curve
    out["regret"] = mm.regret()
    if out["regret"]["max_excess_bits"] > 2.0 + 1e-6:
        raise GateFailure(f"gate (b): the mixture costs {out['regret']['max_excess_bits']} bits at a node ({name})")
    out["nodes"] = mm.nodes()
    out["counters"] = {"e4": counters(e4), "k": counters(k)}
    out["compiles"] = k.compiles
    snap, before = mm.snapshot(), mm.table_hashes()
    q = mm.run(bytes_to_bits(S["A_test"]), learning=False)
    comp = {"e4": mm.last["e4"], "k": mm.last["k"]}
    out["A_test_decomposition"] = decomposition(q, cls)
    out["components_A_test"] = {c: decomposition(v, cls)["all"]["bpb"] for c, v in comp.items()}
    per_byte = -np.log2(q).reshape(-1, 8).sum(axis=1)
    masks = {"class0": cls == 0, "U2": sets["U2"], "U1": sets["U1"], "class0_U2": (cls == 0) & sets["U2"]}
    out["set_bpb"] = {kk: float(per_byte[v].mean()) if v.any() else None for kk, v in masks.items()}
    out["components_set_bpb"] = {
        c: {kk: float((-np.log2(v).reshape(-1, 8).sum(axis=1))[m].mean()) for kk, m in masks.items()}
        for c, v in comp.items()}
    ends = np.cumsum(books)
    if len(ends) and int(ends[-1]) != len(per_byte):
        raise GateFailure("the book boundaries do not cover A-test")
    starts = np.concatenate([[0], ends[:-1]]).astype(int) if len(ends) else np.zeros(0, dtype=int)
    spans = list(zip(starts, ends, strict=True))
    out["per_book"] = {kk: {"bits": [float(per_byte[a:b][v[a:b]].sum()) for a, b in spans],
                            "bytes": [int(v[a:b].sum()) for a, b in spans]} for kk, v in masks.items()}
    del q, per_byte, comp
    mm.restore(snap)
    held = read_mix(mm, S["A_heldout"], learning=False)
    out["A_heldout_bpb"] = held["mix"]
    out["components_A_heldout"] = {"e4": held["e4"], "k": held["k"]}
    mm.restore(snap)
    out["tables_unchanged_by_frozen_reads"] = mm.table_hashes() == before
    if not out["tables_unchanged_by_frozen_reads"]:
        raise GateFailure(f"gate (c): frozen reads changed the tables, the maps or the evidences ({name})")
    if e6b is not None and not smoke:
        ref = e6b["arms"]
        diffs = {"e4": abs(out["components_A_test"]["e4"] - ref["E4"]["A_test_decomposition"]["all"]["bpb"]),
                 "k": abs(out["components_A_test"]["k"] - ref[KARM[name]]["A_test_decomposition"]["all"]["bpb"]),
                 "e4_heldout": abs(out["components_A_heldout"]["e4"] - ref["E4"]["A_heldout_bpb"]),
                 "k_heldout": abs(out["components_A_heldout"]["k"] - ref[KARM[name]]["A_heldout_bpb"])}
        out["reproduction"] = diffs
        if max(diffs.values()) > 1e-9:
            raise GateFailure(f"gate (d): the components do not reproduce results/e6b.json to 1e-9 ({diffs})")
    del mm, e4, k
    gc.collect()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--smoke", action="store_true", help="256 KiB streams, compile every 32 KiB")
    ap.add_argument("--resume", action="store_true", help="reuse the units saved in the partial file")
    ap.add_argument("--V", type=int, default=1 << 15, help="rows clustered at a compile (as E6b's run)")
    args = ap.parse_args()
    t_start = time.perf_counter()
    spec = check_spec()
    S = streams(args.manifest, args.smoke)
    name = "e6b_mix_smoke" if args.smoke else "e6b_mix"
    partial = E.Partial(name, args.resume)
    saved = RESULTS / "e6b.json"
    e6b = json.loads(saved.read_text()) if saved.exists() else None
    if e6b is not None and not args.smoke and e6b["provenance"].get("V") not in (None, args.V):
        raise SystemExit("--V differs from the V of results/e6b.json: the components would not reproduce it")

    def unit(key: str, fn, *a):
        got = partial.get(key)
        if got is None:
            print(key, flush=True)
            partial.put(key, E.timed(fn, *a))
            got = partial.get(key)
        return got

    cls = window_classes(S["A_train"], S["A_test"])
    sets = unseen_context_sets(S)
    books = [len(S["A_test"])] if args.smoke else book_lengths(args.manifest)
    arms_out = {a: unit(a, arm, a, spec, S, cls, sets, books, args.smoke, args.V, e6b) for a in ARMS}
    payload = {
        "experiment": "E6b Amendment 1: the top mixture",
        "protocol": "experiments/PROTOCOL.md#e6b-word-class-levels-on-text (E6b Amendment 1)",
        "provenance": provenance(smoke=args.smoke, manifest=str(args.manifest), resumed=args.resume, V=args.V,
                                 e6b_json_sha256=E.sha256(saved.read_bytes()) if saved.exists() else None),
        "spec": spec, "set_shares": {k: float(v.mean()) for k, v in sets.items()},
        "reuses": "results/e6b.json (the E4 arm and the K arms, reproduced as components)",
        "arms": arms_out, "seconds": round(time.perf_counter() - t_start, 1),
    }
    print(save(name, E.plain(payload)))


if __name__ == "__main__":
    main()
