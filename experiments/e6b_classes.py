"""Experiment E6b: word-class levels on text (experiments/PROTOCOL.md, E6b; DESIGN section 12, V2').

    python experiments/e6b_classes.py --manifest experiments/data/pg19/MANIFEST.json --timing-check
    python experiments/e6b_classes.py --manifest experiments/data/pg19/MANIFEST.json --smoke
    python experiments/e6b_classes.py --manifest experiments/data/pg19/MANIFEST.json              # the three arms
    python experiments/e6b_classes.py --manifest ... --resume                                     # skip saved units
    python experiments/e6b_classes.py --manifest ... --V 16384                                     # the V rule of E6b.8
    python experiments/e6b_classes.py --manifest ... --dev-check                                  # e6b_dev.json

Units (each saved as it ends, in results/e6b_partial.json): ``part0`` (gates (a) (b) (c) (e)), then the arms ``E4`` (K
levels off), ``E4+K`` (the clustered class map) and ``E4+K random`` (the control). Every arm reads A-train once with
learning, then A-test and A-heldout frozen (the state is restored after each, the tables are never copied). The
criterion C-E6b and everything else reported is computed by ``experiments/report.py`` from the saved file. Run one job
at a time.

``--dev-check`` (E6b Amendment 1): four variants (E4; E4+K as built; the map fixed after its first compile; the K
levels after S1) trained on a 3 MiB prefix of A-train (whole books) and read frozen on the validation parts of the same
books; A-test is not read.

``--timing-check`` reads 3 MiB of A-train with the K levels on and off and writes results/e6b_timing.json: seconds per
MiB, seconds per compile, the extrapolation to A-train and the V of the rule of E6b.8.
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
from e6_operator import MIB, GateFailure, book_lengths, check_spec

from vam.bits import bytes_to_bits
from vam.corpus import Corpus, split_book
from vam.nfa import hier
from vam.nfa import wordclass as WC
from vam.realizations.vam1 import Config

ARMS = ("E4", "E4+K", "E4+K random")
MODES = {"E4": "off", "E4+K": "real", "E4+K random": "random"}
STOP_SECONDS = 45 * 60.0


def wcfg_for(arm: str, smoke: bool, V: int) -> WC.WordClassConfig:
    return WC.WordClassConfig(mode=MODES[arm], V=V, compile_bytes=(1 << 15) if smoke else (1 << 20))


def make_machine(spec: dict, wcfg: WC.WordClassConfig) -> WC.KMachine:
    return WC.build_kmachine(spec, Config(cells=1, context=1, **LAWS[spec["law"]]), wcfg)


# -- gates ---------------------------------------------------------------------------------------------------------

def class_stream(n_units: int = 60000, C: int = 8, words: int = 30, seed: int = 0) -> tuple[bytes, dict]:
    """A stream whose words come from ``C`` known classes: each unit's class follows the previous one's by a cycle with
    probability 0.8, the word within a class is uniform (words of a class share their successor distribution)."""
    rng = np.random.default_rng(seed)
    out, truth, cur = [], {}, 0
    for _ in range(n_units):
        cur = (cur + 1) % C if rng.random() < 0.8 else int(rng.integers(C))
        w = f"w{cur}x{int(rng.integers(words))} "
        out.append(w)
        truth[w] = cur
    return "".join(out).encode(), truth


def gate_recovery() -> dict:
    """(e): the compile recovers 8 known classes with purity at least 0.9 (16 clusters, seed 0)."""
    data, truth = class_stream()
    cfg = WC.WordClassConfig(row_bits=12, feats=64, V=1024, C=16, d=8, iters=10, min_count=2, seed=0)
    store = WC.ClassStore(cfg)
    store.update(WC.unit_codes(np.frombuffer(data, dtype=np.uint8)))
    m = WC.compile_classes(store, None, cfg, 0)
    by_code = {int(WC.unit_codes(np.frombuffer(w.encode(), dtype=np.uint8))[0]): c for w, c in truth.items()}
    clusters: dict[int, list[int]] = {}
    for r in m.sel:
        t = by_code.get(int(m.tag[r]))
        if t is not None:
            clusters.setdefault(int(m.cls[r]), []).append(t)
    pur = sum(max(np.bincount(v)) for v in clusters.values()) / sum(len(v) for v in clusters.values())
    ctrl = WC.control_map(m, None, cfg, 0)
    same_sizes = bool(np.array_equal(np.sort(m.cls[m.sel]), np.sort(ctrl.cls[ctrl.sel])))
    if pur < 0.9 or not same_sizes:
        raise GateFailure(f"gate (e)/(c): purity {pur:.3f} (need 0.9), control sizes equal {same_sizes}")
    return {"purity": float(pur), "clusters": len(clusters), "control_row_counts_equal": same_sizes}


def gate_identity(spec: dict, S: dict, smoke: bool) -> dict:
    """(a) with K off the machine equals ``hier.HierMachine`` (learning, then frozen); (b) its E4 keys equal hier's."""
    law = Config(cells=1, context=1, **LAWS[spec["law"]])
    n = min(len(S["A_train"]), (1 << 17) if smoke else 4 * MIB)
    train, more = S["A_train"][:n], S["A_train"][n:n + ((1 << 16) if smoke else MIB)]
    ref = hier.build(spec, law)
    q1 = ref.run(bytes_to_bits(train), learning=True)
    f1 = ref.copy().run(bytes_to_bits(more), learning=False)
    del ref
    gc.collect()
    off = make_machine(spec, WC.WordClassConfig(mode="off"))
    q2 = off.run(bytes_to_bits(train), learning=True)
    f2 = off.copy().run(bytes_to_bits(more), learning=False)
    same = bool(np.array_equal(q1, q2) and np.array_equal(f1, f2))
    del off
    gc.collect()
    bits = bytes_to_bits(train[:1 << 16])
    kinds, dups = np.arange(5, dtype=np.int64), np.zeros(5, dtype=np.int64)
    e1 = np.empty((bits.shape[0], 5), dtype=np.uint64)
    e2 = np.empty_like(e1)
    st1 = np.array(hier.initial_state(), dtype=np.uint64)
    st2 = np.array(hier.initial_state() + [WC.UNK] * 4, dtype=np.uint64)
    hier._fill_keys(bits, st1, e1, kinds, dups, True, False)
    codes, nc = np.empty(len(bits) // 8 + 8, dtype=np.uint64), np.zeros(1, dtype=np.int64)
    WC._fill_keys_k(bits, st2, e2, kinds, dups, True, False, 4, np.zeros(16, dtype=np.uint64),
                    np.full(16, WC.NOMAP, dtype=np.uint16), codes, nc, WC.UNK)
    keys_same = bool(np.array_equal(e1, e2))
    if not (same and keys_same):
        raise GateFailure("gate (a)/(b): with the K levels off the E6b path differs from E4's")
    return {"bytes": n, "frozen_bytes": len(more), "bit_identical": same, "e4_keys_equal": keys_same}


def part0(spec: dict, S: dict, smoke: bool) -> dict:
    return {"gate_a_b": gate_identity(spec, S, smoke), "gate_c_e": gate_recovery()}


# -- the sets --------------------------------------------------------------------------------------------------------

def unseen_context_sets(S: dict) -> dict:
    """U1 and U2 over the bytes of A-test: the W1 / W2 context key of the byte occurs at no byte of A-train."""
    out = {}
    for name, kind in (("U1", hier.W1), ("U2", hier.W2)):
        train = np.unique(WC.context_keys(S["A_train"], kind))
        out[name] = ~np.isin(WC.context_keys(S["A_test"], kind), train)
        del train
        gc.collect()
    return out


# -- one arm ---------------------------------------------------------------------------------------------------------

def class_share(m: WC.KMachine, data: bytes) -> dict:
    """The share of unit tokens of ``data`` in the map in force (the rest are class C)."""
    cm = m.class_map
    codes = WC.unit_codes(np.frombuffer(data, dtype=np.uint8))
    if cm is None:
        return {"tokens": len(codes), "in_map": 0.0}
    rows = (codes >> np.uint64(64 - m.wcfg.row_bits)).astype(np.int64)
    inside = (m._ctag[rows] == codes) & (m._ccls[rows] < 256)
    return {"tokens": len(codes), "in_map": float(inside.mean())}


def arm(name: str, spec: dict, S: dict, cls: np.ndarray, sets: dict, books: list[int], smoke: bool, V: int,
        e4_saved: dict | None) -> dict:
    wcfg = wcfg_for(name, smoke, V)
    m = make_machine(spec, wcfg)
    out: dict = {"wcfg": {k: getattr(wcfg, k) for k in wcfg.__dataclass_fields__}}
    curve: list[float] = []
    t0 = time.perf_counter()
    out["train_bpb"] = read(m, S["A_train"], learning=True, curve=curve)
    out["train_seconds"] = time.perf_counter() - t0
    if out["train_seconds"] > STOP_SECONDS:
        raise GateFailure(f"arm {name} ran past {STOP_SECONDS:.0f} s while reading A-train")
    out["learning_curve_bpb_per_MiB"] = curve
    out["counters"] = counters(m)
    out["compiles"] = m.compiles
    snap, before = m.snapshot(), m.table_hashes()
    q = m.run(bytes_to_bits(S["A_test"]), learning=False)
    out["A_test_decomposition"] = decomposition(q, cls)
    per_byte = -np.log2(q).reshape(-1, 8).sum(axis=1)
    masks = {"class0": cls == 0, "U2": sets["U2"], "U1": sets["U1"], "class0_U2": (cls == 0) & sets["U2"]}
    out["set_bpb"] = {k: float(per_byte[v].mean()) if v.any() else None for k, v in masks.items()}
    out["set_share"] = {k: float(v.mean()) for k, v in masks.items()}
    ends = np.cumsum(books)
    if len(ends) and int(ends[-1]) != len(per_byte):
        raise GateFailure("the book boundaries do not cover A-test")
    starts = np.concatenate([[0], ends[:-1]]).astype(int) if len(ends) else np.zeros(0, dtype=int)
    spans = list(zip(starts, ends, strict=True))
    out["per_book"] = {k: {"bits": [float(per_byte[a:b][v[a:b]].sum()) for a, b in spans],
                           "bytes": [int(v[a:b].sum()) for a, b in spans]} for k, v in masks.items()}
    del q, per_byte
    m.restore(snap)
    out["A_heldout_bpb"] = read(m, S["A_heldout"], learning=False)
    m.restore(snap)
    out["tables_unchanged_by_frozen_reads"] = m.table_hashes() == before
    if not out["tables_unchanged_by_frozen_reads"]:
        raise GateFailure(f"frozen reads changed the tables, the count store or the map ({name})")
    if wcfg.mode != "off":
        out["A_test_units"] = class_share(m, S["A_test"])
        out["store"] = {"units": m._store.n_units, "bytes": m._store.nbytes()}
        real = m._real
        if real is not None:
            out["final_map_rows_per_class"] = np.bincount(real.cls[real.sel].astype(int), minlength=wcfg.C).tolist()
        cm = m.class_map
        if cm is not None:
            out["final_map_rows_per_class_active"] = np.bincount(cm.cls[cm.sel].astype(int),
                                                                 minlength=wcfg.C).tolist()
    if name == "E4" and e4_saved is not None and not smoke:
        ref = e4_saved["part3a"]["A_only"]["A_test"]
        out["e4_reproduction"] = {"saved_A_test": ref, "A_test": out["A_test_decomposition"]["all"]["bpb"],
                                  "abs_difference": abs(ref - out["A_test_decomposition"]["all"]["bpb"])}
        if out["e4_reproduction"]["abs_difference"] > 1e-9:
            raise GateFailure("the E4 arm does not reproduce results/e4.json A_test to 1e-9")
    del m
    gc.collect()
    return out


def timing_check(spec: dict, S: dict, V: int) -> dict:
    """PROTOCOL E6b.8: 3 MiB of A-train with the K levels on and off; the extrapolation to A-train; the V rule."""
    data = S["A_train"][:3 * MIB]
    res: dict = {}
    for label, wcfg in (("off", WC.WordClassConfig(mode="off")), ("K", WC.WordClassConfig(mode="real", V=V))):
        m = make_machine(spec, wcfg)
        t0 = time.perf_counter()
        read(m, data, learning=True)
        sec = time.perf_counter() - t0
        res[label] = {"seconds": sec, "seconds_per_MiB": sec / 3.0, "compiles": m.compiles,
                      "compile_seconds": sum(c["seconds"] for c in m.compiles)}
        del m
        gc.collect()
    mib = len(S["A_train"]) / MIB
    per_compile = res["K"]["compile_seconds"] / max(1, len(res["K"]["compiles"]))
    base = res["K"]["seconds"] - res["K"]["compile_seconds"]
    spm = base / 3.0
    res.update({"A_train_MiB": mib, "V": V, "stop_seconds": STOP_SECONDS})

    def arm_seconds(v: int) -> float:
        return spm * mib + mib // 1 * per_compile * v / V

    v = V
    while arm_seconds(v) > STOP_SECONDS and v > 256:
        v //= 2
    res["extrapolated_arm_seconds_at_V"] = arm_seconds(V)
    res["recommended_V"] = v
    res["extrapolated_arm_seconds_at_recommended_V"] = arm_seconds(v)
    res["over_the_stop"] = arm_seconds(V) > STOP_SECONDS
    return res


DEV_VARIANTS = {
    "E4": dict(mode="off"),
    "E4+K": dict(mode="real"),
    "E4+K fixed map": dict(mode="real", fix_map=True),
    "E4+K after S1": dict(mode="real", order="after"),
}


def dev_streams(manifest: Path, prefix_bytes: int) -> tuple[bytes, bytes, int]:
    """The train parts of the first whole A books that fit in ``prefix_bytes``, and the validation parts of the same
    books (the structure of A-test against A-train, in small); A-test is never read."""
    corpus = Corpus.load(manifest)
    train, val, total = [], [], 0
    for b in corpus.books:
        if b.stage != "A" or b.role != "train":
            continue
        tr, va, _ = split_book(corpus.text(b))
        if total + len(tr) > prefix_bytes:
            break
        train.append(tr)
        val.append(va)
        total += len(tr)
    return b"".join(train), b"".join(val), len(train)


def dev_check(spec: dict, manifest: Path, prefix_mib: int = 3) -> dict:
    """E6b Amendment 1: the development check on A-validation (never A-test): train on a prefix, read frozen."""
    train, val, books = dev_streams(manifest, prefix_mib * MIB)
    cls = window_classes(train, val)
    sets = {"U1": None, "U2": None}
    for name, kind in (("U1", hier.W1), ("U2", hier.W2)):
        sets[name] = ~np.isin(WC.context_keys(val, kind), np.unique(WC.context_keys(train, kind)))
    masks = {"all": np.ones(len(val), dtype=bool), "class0": cls == 0, "class8": cls == 8, "class16": cls == 16,
             "class32": cls == 32, "U1": sets["U1"], "U2": sets["U2"], "class0_U2": (cls == 0) & sets["U2"],
             "class0_outside_U2": (cls == 0) & ~sets["U2"]}
    out: dict = {"prefix_bytes": len(train), "books": books, "validation_bytes": len(val),
                 "shares": {k: float(v.mean()) for k, v in masks.items()}, "variants": {}}
    for name, kw in DEV_VARIANTS.items():
        wcfg = WC.WordClassConfig(compile_bytes=1 << 20, **kw)
        m = make_machine(spec, wcfg)
        curve: list[float] = []
        t0 = time.perf_counter()
        train_bpb = read(m, train, learning=True, curve=curve)
        secs = time.perf_counter() - t0
        snap = m.snapshot()
        q = m.run(bytes_to_bits(val), learning=False)
        m.restore(snap)
        per_byte = -np.log2(q).reshape(-1, 8).sum(axis=1)
        out["variants"][name] = {"train_bpb": train_bpb, "learning_curve_bpb_per_MiB": curve, "train_seconds": secs,
                                 "validation_bpb": {k: float(per_byte[v].mean()) if v.any() else None
                                                    for k, v in masks.items()},
                                 "compiles": len(m.compiles)}
        print(name, round(train_bpb, 4), round(float(per_byte.mean()), 4), flush=True)
        del m, q, per_byte
        gc.collect()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--smoke", action="store_true", help="256 KiB streams, compile every 32 KiB")
    ap.add_argument("--resume", action="store_true", help="reuse the units saved in the partial file")
    ap.add_argument("--timing-check", action="store_true", help="the timing check of PROTOCOL E6b.8, then stop")
    ap.add_argument("--dev-check", action="store_true", help="the development check of E6b Amendment 1, then stop")
    ap.add_argument("--V", type=int, default=1 << 15, help="rows clustered at a compile (the V rule of E6b.8)")
    args = ap.parse_args()
    t_start = time.perf_counter()
    spec = check_spec()
    if args.dev_check:
        res = {"provenance": provenance(smoke=False), **dev_check(spec, args.manifest)}
        print(save("e6b_dev", E.plain(res)))
        return
    S = streams(args.manifest, args.smoke)
    if args.timing_check:
        res = {"provenance": provenance(smoke=args.smoke), **timing_check(spec, S, args.V)}
        print(save("e6b_timing", E.plain(res)))
        print({k: res[k] for k in ("extrapolated_arm_seconds_at_V", "recommended_V", "over_the_stop")})
        return
    name = "e6b_smoke" if args.smoke else "e6b"
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
    sets = unseen_context_sets(S)
    books = [len(S["A_test"])] if args.smoke else book_lengths(args.manifest)
    arms_out = {a: unit(a, arm, a, spec, S, cls, sets, books, args.smoke, args.V, e4_saved) for a in ARMS}
    payload = {
        "experiment": "E6b word-class levels on text",
        "protocol": "experiments/PROTOCOL.md#e6b-word-class-levels-on-text",
        "provenance": provenance(smoke=args.smoke, manifest=str(args.manifest), resumed=args.resume,
                                 manifest_sha256=E.sha256(args.manifest.read_bytes()), V=args.V),
        "spec": spec, "stream_bytes": {k: len(v) for k, v in S.items() if k in ("A_train", "A_test", "A_heldout")},
        "set_shares": {k: float(v.mean()) for k, v in sets.items()},
        "part0": p0, "arms": arms_out, "seconds": round(time.perf_counter() - t_start, 1),
    }
    print(save(name, E.plain(payload)))


if __name__ == "__main__":
    main()
