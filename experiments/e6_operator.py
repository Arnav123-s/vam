"""Experiment E6: the shared operator level O (experiments/PROTOCOL.md, E6; DESIGN section 12, V2).

    python experiments/e6_operator.py --manifest experiments/data/pg19/MANIFEST.json --timing-check
    python experiments/e6_operator.py --manifest experiments/data/pg19/MANIFEST.json --smoke
    python experiments/e6_operator.py --manifest experiments/data/pg19/MANIFEST.json            # the three arms
    python experiments/e6_operator.py --manifest ... --resume                                    # skip the units saved
    python experiments/e6_operator.py --manifest ... --arm O --past-bits 14      # R-a -> results/e6_ra.json
    python experiments/e6_operator.py --manifest ... --arm O --k-force 24        # R-b -> results/e6_rb.json

Units (each saved as it ends, in results/e6_partial.json): ``part0`` (gate (a): O off equals E4 bit for bit on the
first 4 MiB of A-train), then the arms ``E4`` (O off), ``E4+O`` (rank by the Page rule) and ``E4+O rank 1`` (the
empirical byte unigram). Every arm reads A-train once with learning, then reads A-test and A-heldout frozen (the state
is restored after each, the tables are never copied). The criterion C1 and everything else reported is computed by
``experiments/report.py`` from the saved file. Run one job at a time.

``--timing-check`` reads 3 MiB of A-train with k forced to 256 (and the same bytes with O off) and writes
results/e6_timing.json: seconds per MiB, the compile seconds, and the extrapolation to A-train.
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
from e2_curriculum import LAWS, read, streams
from e4_hier import counters, decomposition, window_classes

from vam.bits import bytes_to_bits
from vam.corpus import Corpus, split_book
from vam.nfa import hier
from vam.nfa import operator as O
from vam.realizations.vam1 import Config

#: E4's chosen configuration (results/e4.json part2.chosen), checked against the saved file at the start
CHOSEN = {"machine": "hier", "law": "counting", "prior": 4.0, "coherence": 1.0, "governor": True, "protect": 0.5,
          "sink_rate": 0.0, "verify": 3, "hier_levels": "WS", "hier_segment": "text", "hier_verify": 4}
ARMS = ("E4", "E4+O", "E4+O rank 1")
REGRET_BOUND = 2.0
MIB = 1 << 20


class GateFailure(RuntimeError):
    """A gate of PROTOCOL E6.5 failed: nothing is reported."""


def check_spec(path: Path = RESULTS / "e4.json") -> dict:
    chosen = json.loads(path.read_text())["part2"]["chosen"]
    if chosen != CHOSEN:
        raise SystemExit("E6 stopped: results/e4.json part2.chosen differs from the configuration fixed in "
                         f"PROTOCOL E6: {chosen}")
    return dict(CHOSEN)


def ocfg_for(arm: str, smoke: bool, **kw) -> O.OperatorConfig:
    mode = {"E4": "off", "E4+O": "spectral", "E4+O rank 1": "unigram"}[arm]
    base = {"compile_bytes": (1 << 15) if smoke else (1 << 20)}
    return O.OperatorConfig(mode=mode, **{**base, **kw})


def make_machine(spec: dict, ocfg: O.OperatorConfig) -> O.OperatorMachine:
    return O.build_operator(spec, Config(cells=1, context=1, **LAWS[spec["law"]]), ocfg)


def part0(spec: dict, S: dict, smoke: bool) -> dict:
    """Gate (a): with O off, the per-bit probabilities equal ``hier.HierMachine``'s, learning then frozen."""
    n = min(len(S["A_train"]), (1 << 17) if smoke else 4 * MIB)
    train, more = S["A_train"][:n], S["A_train"][n:n + ((1 << 16) if smoke else MIB)]
    law = Config(cells=1, context=1, **LAWS[spec["law"]])
    ref = hier.build(spec, law)
    q1 = ref.run(bytes_to_bits(train), learning=True)
    f1 = ref.copy().run(bytes_to_bits(more), learning=False)
    del ref
    gc.collect()
    off = make_machine(spec, ocfg_for("E4", smoke))
    q2 = off.run(bytes_to_bits(train), learning=True)
    f2 = off.copy().run(bytes_to_bits(more), learning=False)
    same = bool(np.array_equal(q1, q2) and np.array_equal(f1, f2))
    del off
    gc.collect()
    if not same:
        raise GateFailure("with O off the E6 path differs from E4's")
    return {"bytes": n, "frozen_bytes": len(more), "bit_identical": same}


def book_lengths(manifest: Path) -> list[int]:
    """Bytes of each stage-A book's test split, in the order of the A-test stream."""
    corpus = Corpus.load(manifest)
    return [len(split_book(corpus.text(b))[2]) for b in corpus.books if b.stage == "A" and b.role == "train"]


def distinct_contexts(data: bytes, L: int) -> int:
    a = np.frombuffer(data, dtype=np.uint8).astype(np.uint64)
    v = np.zeros(len(a) - L, dtype=np.uint64)
    for i in range(L):
        v |= a[L - 1 - i:len(a) - 1 - i] << np.uint64(8 * i)
    return len(np.unique(v))


def variant(past_bits: int | None, k_force: int | None) -> tuple[str, dict]:
    """E6 Amendment 1: the diagnostic rerun named by the flags and its overrides; ("", {}) when both are off."""
    if past_bits is not None and k_force is not None:
        raise SystemExit("E6 Amendment 1 runs R-a (--past-bits) and R-b (--k-force) one at a time")
    if past_bits is not None:
        return "e6_ra", {"past_bits": past_bits}
    if k_force is not None:
        return "e6_rb", {"k_force": k_force}
    return "", {}


def arm(name: str, spec: dict, S: dict, cls: np.ndarray, books: list[int], smoke: bool, e4_saved: dict | None,
        kw: dict | None = None) -> dict:
    ocfg = ocfg_for(name, smoke, **(kw or {}))
    m = make_machine(spec, ocfg)
    out: dict = {"ocfg": {k: getattr(ocfg, k) for k in ocfg.__dataclass_fields__}}
    curve: list[float] = []
    out["train_bpb"] = read(m, S["A_train"], learning=True, curve=curve)
    out["learning_curve_bpb_per_MiB"] = curve
    out["counters"] = counters(m)
    out["compiles"] = m.compiles
    out["negative_mass_learning"] = m.negative_mass()
    snap, before = m.snapshot(), m.table_hashes()
    m._ostat[:] = 0.0
    q = m.run(bytes_to_bits(S["A_test"]), learning=False)
    out["negative_mass_A_test"] = m.negative_mass()
    out["A_test_decomposition"] = decomposition(q, cls)
    per_byte = -np.log2(q).reshape(-1, 8).sum(axis=1)
    ends = np.cumsum(books)
    if len(ends) and int(ends[-1]) != len(per_byte):
        raise GateFailure("the book boundaries do not cover A-test")
    starts = np.concatenate([[0], ends[:-1]]).astype(int) if len(ends) else np.zeros(0, dtype=int)
    zero = cls == 0
    spans = list(zip(starts, ends, strict=True))
    out["class0_per_book"] = {"bits": [float(per_byte[a:b][zero[a:b]].sum()) for a, b in spans],
                              "bytes": [int(zero[a:b].sum()) for a, b in spans]}
    del q, per_byte
    m.restore(snap)
    m._ostat[:] = 0.0
    out["A_heldout_bpb"] = read(m, S["A_heldout"], learning=False)
    m.restore(snap)
    out["tables_unchanged_by_frozen_reads"] = m.table_hashes() == before
    if not out["tables_unchanged_by_frozen_reads"]:
        raise GateFailure(f"frozen reads changed the tables, the store or the operators ({name})")
    if ocfg.mode != "off":
        r = m.stage_regret()
        out["stage_regret"] = r
        if r["max_excess_bits"] > REGRET_BOUND + 1e-6:
            raise GateFailure(f"the O stage costs {r['max_excess_bits']} bits at a node ({name}), bound {REGRET_BOUND}")
        st = m._store
        out["store"] = {"bytes": st.n, "positions_c2": st.n2, "positions_c3": st.n3, "max_cell": st.max_cell,
                        "occupied_buckets": st.occupied_buckets(), "buckets": 1 << ocfg.past_bits,
                        "distinct_contexts": distinct_contexts(S["A_train"], ocfg.L)}
        out["store"]["contexts_per_occupied_bucket"] = (out["store"]["distinct_contexts"]
                                                        / max(1, out["store"]["occupied_buckets"]))
    if name == "E4" and e4_saved is not None and not smoke:
        ref = e4_saved["part3a"]["A_only"]["A_test"]
        out["e4_reproduction"] = {"saved_A_test": ref, "A_test": out["A_test_decomposition"]["all"]["bpb"],
                                  "abs_difference": abs(ref - out["A_test_decomposition"]["all"]["bpb"])}
        if out["e4_reproduction"]["abs_difference"] > 1e-9:
            raise GateFailure("the E4 arm does not reproduce results/e4.json A_test to 1e-9")
    del m
    gc.collect()
    return out


def timing_check(spec: dict, S: dict) -> dict:
    """PROTOCOL E6.8: 3 MiB of A-train with k forced to 256 and with O off; the extrapolation to A-train."""
    data = S["A_train"][:3 * MIB]
    res: dict = {}
    for label, cfg in (("off", O.OperatorConfig(mode="off")),
                       ("k256", O.OperatorConfig(mode="spectral", k_force=256, compile_bytes=MIB))):
        m = make_machine(spec, cfg)
        t0 = time.perf_counter()
        read(m, data, learning=True)
        sec = time.perf_counter() - t0
        res[label] = {"seconds": sec, "seconds_per_MiB": sec / 3.0, "compiles": m.compiles,
                      "compile_seconds": sum(c["seconds"] for c in m.compiles)}
        del m
        gc.collect()
    mib = len(S["A_train"]) / MIB
    over = (res["k256"]["seconds_per_MiB"] - res["off"]["seconds_per_MiB"]) * mib
    res["A_train_MiB"] = mib
    res["extrapolated_overhead_seconds"] = over
    res["extrapolated_arm_seconds"] = res["off"]["seconds_per_MiB"] * mib + over
    res["stop_seconds"] = 45 * 60
    res["over_the_stop"] = res["extrapolated_arm_seconds"] > res["stop_seconds"]
    return res


def diagnostic(args: argparse.Namespace, tag: str, kw: dict, spec: dict, S: dict, t_start: float) -> None:
    """E6 Amendment 1 (added after the result): the E4+O arm only, against the E4 and rank-1 arms saved in e6.json."""
    if tag == "e6_ra":
        kw = {**kw, "c3_bytes": tuple(sorted(set(S["A_train"])))}
    name = tag + ("_smoke" if args.smoke else "")
    partial = E.Partial(name, args.resume)
    saved = RESULTS / "e6.json"
    cls = window_classes(S["A_train"], S["A_test"])
    books = [len(S["A_test"])] if args.smoke else book_lengths(args.manifest)
    got = partial.get("E4+O")
    if got is None:
        print(name, kw if tag == "e6_rb" else {**kw, "c3_bytes": len(kw["c3_bytes"])}, flush=True)
        partial.put("E4+O", E.timed(arm, "E4+O", spec, S, cls, books, args.smoke, None, kw))
        got = partial.get("E4+O")
    payload = {
        "experiment": "E6 Amendment 1 diagnostic rerun, added after the result",
        "protocol": "experiments/PROTOCOL.md#e6-the-shared-operator-level-o (E6 Amendment 1)",
        "rerun": {"R-a": "past_bits 14 at L = 4, C3 over the observed rows",
                  "R-b": "k forced to 24"}["R-a" if tag == "e6_ra" else "R-b"],
        "provenance": provenance(smoke=args.smoke, manifest=str(args.manifest), resumed=args.resume,
                                 manifest_sha256=E.sha256(args.manifest.read_bytes()), pinv_rcond=O.PINV_RCOND,
                                 e6_json_sha256=E.sha256(saved.read_bytes()) if saved.exists() else None),
        "spec": spec, "overrides": {k: (list(v) if k == "c3_bytes" else v) for k, v in kw.items() if k != "c3_bytes"},
        "c3_rows": len(kw.get("c3_bytes", ())) or None,
        "reuses": {"file": "results/e6.json", "arms": ["E4", "E4+O rank 1"]},
        "stream_bytes": {k: len(v) for k, v in S.items() if k in ("A_train", "A_test", "A_heldout")},
        "arms": {"E4+O": got}, "seconds": round(time.perf_counter() - t_start, 1),
    }
    print(save(name, E.plain(payload)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--smoke", action="store_true", help="256 KiB streams, compile every 32 KiB")
    ap.add_argument("--resume", action="store_true", help="reuse the units saved in the partial file")
    ap.add_argument("--timing-check", action="store_true", help="the k = 256 timing check of PROTOCOL E6.8, then stop")
    ap.add_argument("--arm", choices=["O"], help="E6 Amendment 1: the E4+O arm only (needs --past-bits or --k-force)")
    ap.add_argument("--past-bits", type=int, help="R-a: past buckets 2**N, C3 over the bytes of A-train; e6_ra.json")
    ap.add_argument("--k-force", type=int, help="R-b: this rank at every compile; e6_rb.json")
    args = ap.parse_args()
    t_start = time.perf_counter()
    spec = check_spec()
    S = streams(args.manifest, args.smoke)
    if args.timing_check:
        res = {"provenance": provenance(smoke=args.smoke), **timing_check(spec, S)}
        print(json.dumps(E.plain(res), indent=1))
        print(save("e6_timing", E.plain(res)))
        return
    tag, kw = variant(args.past_bits, args.k_force)
    if bool(tag) != bool(args.arm):
        raise SystemExit("--arm O goes with exactly one of --past-bits and --k-force")
    if tag:
        return diagnostic(args, tag, kw, spec, S, t_start)
    name = "e6_smoke" if args.smoke else "e6"
    partial = E.Partial(name, args.resume)
    print("stream bytes:", {k: len(v) for k, v in S.items() if k in ("A_train", "A_test", "A_heldout")}, flush=True)

    def unit(key: str, fn, *a):
        got = partial.get(key)
        if got is None:
            print(key, flush=True)
            partial.put(key, E.timed(fn, *a))
            got = partial.get(key)
        return got

    e4_saved = json.loads((RESULTS / "e4.json").read_text())
    p0 = unit("part0", part0, spec, S, args.smoke)
    cls = window_classes(S["A_train"], S["A_test"])
    books = [len(S["A_test"])] if args.smoke else book_lengths(args.manifest)
    arms_out = {a: unit(a, arm, a, spec, S, cls, books, args.smoke, e4_saved) for a in ARMS}
    payload = {
        "experiment": "E6 the shared operator level O",
        "protocol": "experiments/PROTOCOL.md#e6-the-shared-operator-level-o",
        "provenance": provenance(smoke=args.smoke, manifest=str(args.manifest), resumed=args.resume,
                                 manifest_sha256=E.sha256(args.manifest.read_bytes()), pinv_rcond=O.PINV_RCOND),
        "spec": spec, "stream_bytes": {k: len(v) for k, v in S.items() if k in ("A_train", "A_test", "A_heldout")},
        "part0": p0, "arms": arms_out, "seconds": round(time.perf_counter() - t_start, 1),
    }
    print(save(name, E.plain(payload)))


if __name__ == "__main__":
    main()
