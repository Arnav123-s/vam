"""Experiment E4: N above one equation, on text (experiments/PROTOCOL.md, E4; DESIGN section 11, decision D11).

    python experiments/e4_hier.py --manifest experiments/data/pg19/MANIFEST.json
    python experiments/e4_hier.py --manifest ... --smoke      # code check: 256 KiB streams, small grids
    python experiments/e4_hier.py --manifest ... --resume     # continue from results/e4_partial.json

Parts (each saved as it ends; Part 2 after every row, Part 3 in the two units 3a and 3b):
    0 identity and reproduction gate     1 designed behaviours, one criterion per layer
    2 configuration on A-validation      3 the arms (3a: A only, A then B; 3b: B only, shuffled A then B)
    4 controls (equal storage without span, order destroyed)     5 decomposition by window class     6 writing
Run one job at a time. Results go to results/e4.json (e4_smoke.json for --smoke).
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import e3_nfa as E
import numpy as np
from _common import RESULTS, provenance, save
from e2_curriculum import BLOCK, SEED, arms, frozen, generations, make, prompts, read, streams

from vam.bits import bits_to_bytes, bytes_to_bits
from vam.nested import NestedConfig, NestedMachine, coherent_mix
from vam.nfa import bounds, collapse, hier, page
from vam.realizations.vam1 import Config

#: E3's chosen row (results/e3.json part2.chosen), held fixed; checked against the saved file at the start
BASE = {"law": "counting", "prior": 4.0, "coherence": 1.0, "governor": True, "protect": 0.5, "sink_rate": 0.1,
        "verify": 3}
ROW_LIMIT = 900.0            # seconds per Part 2 row (PROTOCOL E4, Part 2)
TIE = 0.0005                 # bpb: ties go to the simpler setting
FLOOR = 256                  # the count floor of E3 Amendment 4 (T4.2), here for the upper levels' Hankel matrices
CLASSES = (32, 16, 8)        # window lengths of Part 5


def hspec(**kw) -> dict:
    return {"machine": "hier", **BASE, **kw}


def e3_spec(**kw) -> dict:
    return {"machine": "nested", **BASE, **kw}


def key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True)


def options_on(spec: dict) -> int:
    """The tie rule's measure of simplicity: upper levels, text rule, global scope, raised bar, own verify, sinks."""
    levels = {"WS": 5, "W": 3, "S": 2, "": 0}[spec.get("hier_levels", "WS")]
    return (levels + int(spec.get("hier_segment", "content") == "text") + int(spec.get("hier_scope") == "global")
            + int(bool(spec.get("hier_raise"))) + int(spec.get("hier_verify", BASE["verify"]) != BASE["verify"])
            + int(spec.get("sink_rate", BASE["sink_rate"]) > 0))


def counters(m) -> dict:
    return {"evictions": m.evictions, "refused": m.refused, "protected": m.protected,
            "occupancy": {str(lv): v for lv, v in m.occupancy().items()}, "global_occupancy": m.global_occupancy()}


# ---------------------------------------------------------------------------------------------------
# Part 0: identity and reproduction
# ---------------------------------------------------------------------------------------------------

def e3_saved_row() -> dict:
    d = json.loads((RESULTS / "e3.json").read_text())
    chosen = d["part2"]["chosen"]
    assert {k: chosen[k] for k in BASE} == BASE and chosen["machine"] == "nested", "BASE is not E3's chosen row"
    return next(r for r in d["part2"]["rows"] if r["spec"] == chosen)


def part0(S: dict, smoke: bool) -> dict:
    n = (1 << 18) if smoke else (1 << 20)
    bits = bytes_to_bits(S["A_train"][:n])
    a, b = make(e3_spec()), make(hspec(hier_levels=""))
    same = bool(np.array_equal(a.run(bits), b.run(bits)))
    val = bytes_to_bits(S["A_validation"][: n // 4])
    same &= bool(np.array_equal(a.run(val, learning=False), b.run(val, learning=False)))
    same &= (a.evictions, a.refused, a.protected) == (b.evictions, b.refused, b.protected)
    del a, b
    gc.collect()
    saved = e3_saved_row()
    t0 = time.perf_counter()
    m = make(hspec(hier_levels=""))
    train = read(m, S["A_train"], learning=True)
    got = frozen(m, S["A_validation"])
    del m
    gc.collect()
    ok_repro = smoke or abs(got - saved["A_validation_bpb"]) < 1e-9
    rng = np.random.default_rng(SEED)
    rand = rng.integers(0, 2, 2_000_000 if smoke else 4_000_000).astype(np.uint8)
    law = Config(cells=1, context=1, consolidating=True, dt=1.0)
    cfg = NestedConfig(law, depths=(8, 16, 24, 32, 48, 64), table_bits=22)
    rates = {"nested": [], "hier_off": []}
    mk = {"nested": lambda: NestedMachine(cfg), "hier_off": lambda: hier.HierMachine(hier.HierConfig(cfg, upper=()))}
    for i in range(3 if smoke else 6):
        for name in (("nested", "hier_off") if i % 2 == 0 else ("hier_off", "nested")):
            mm = mk[name]()
            mm.run(rand[:1000])
            t1 = time.perf_counter()
            mm.run(rand)
            rates[name].append(len(rand) / (time.perf_counter() - t1))
            del mm
    ratio = float(np.median(rates["hier_off"]) / np.median(rates["nested"]))
    # Part 0 (b), as built: E3's NestedMachine and its compiled loop are untouched (E4 has a loop of its own in
    # vam/nfa/hier.py), so E3's default path pays nothing; the upper-off path of the E4 loop is timed and reported.
    out = {"identity_bit_for_bit": same, "e3_saved_A_validation_bpb": saved["A_validation_bpb"],
           "recomputed_A_validation_bpb": got, "recomputed_A_train_bpb": train, "reproduced": bool(ok_repro),
           "rates_bits_per_s": rates, "median_rate_ratio_upper_off_to_nested": ratio,
           "criteria": {"identity": bool(same), "reproduction": bool(ok_repro)},
           "seconds": round(time.perf_counter() - t0, 1)}
    if not all(out["criteria"].values()):
        raise RuntimeError(f"Part 0 gate failed: {out['criteria']}")
    return out


# ---------------------------------------------------------------------------------------------------
# Part 1: designed behaviours
# ---------------------------------------------------------------------------------------------------

def level_hankel(keys: np.ndarray, nxt: np.ndarray, block: int = 4096, splits: int = 3, seed: int = 0):
    """Counts of (level context, next byte) at every byte start, and of random halves in blocks of ``block`` positions."""
    from scipy import sparse
    _, rows = np.unique(keys, return_inverse=True)
    shape = (int(rows.max()) + 1, 256)
    ones = np.ones(len(rows))
    full = sparse.csr_matrix((ones, (rows, nxt.astype(np.int64))), shape=shape)
    blocks = np.arange(len(rows)) // block
    nb = int(blocks[-1]) + 1
    rng = np.random.default_rng(seed)
    halves = []
    for _ in range(splits):
        chosen = np.zeros(nb, dtype=bool)
        chosen[rng.permutation(nb)[: nb // 2]] = True
        sel = chosen[blocks[: len(rows)]]
        halves.append(sparse.csr_matrix((np.ones(int(sel.sum())), (rows[sel], nxt[sel].astype(np.int64))), shape=shape))
    return full, halves


def byte_start_keys(m, data: bytes) -> np.ndarray:
    """The key of every upper level at the start of every byte (``u = 1``) of ``data``: shape (bytes, levels)."""
    bits = bytes_to_bits(data)
    st = np.array(hier.initial_state(), dtype=np.uint64)
    ext = np.empty((hier.BLOCK_BITS, max(1, len(m._kinds))), dtype=np.uint64)
    out = []
    for a in range(0, len(bits), hier.BLOCK_BITS):
        blk = bits[a:a + hier.BLOCK_BITS]
        hier._fill_keys(blk, st, ext[: len(blk)], m._kinds, m._dups, m.hcfg.segment == "text", False)
        out.append(ext[: len(blk)][::8].copy())
    return np.concatenate(out)


def floor_rank(full, halves, floor: int = FLOOR):
    r, c = np.asarray(full.sum(1)).ravel(), np.asarray(full.sum(0)).ravel()
    kr, kc = np.flatnonzero(r >= floor), np.flatnonzero(c >= floor)
    if len(kr) < 2 or len(kc) < 2:
        return {"rank": None, "cutoff": None, "rows_kept": int(len(kr)), "rows": int(len(r))}
    f = full[kr][:, kc]
    sp = page.page_spectrum(f, [h[kr][:, kc] for h in halves], 0.1)
    return {"rank": sp["rank"], "cutoff": sp["cutoff"], "rows_kept": int(len(kr)), "rows": int(len(r)),
            "samples_kept_share": float(f.sum() / full.sum())}


def part1_l1(A_train: bytes, smoke: bool) -> dict:
    m = make(hspec())
    hw = bounds.Hardware.detect()
    ok = hier.ARITY <= hw.line_words and bounds.register_cap(hw) == 64 and m.state_bytes() < 128
    n = (1 << 18) if smoke else (4 << 20)
    text = A_train[:n]
    seen: dict[int, bytes] = {}
    st, start, units, coll = hier.initial_state(), 0, 0, 0
    for i, by in enumerate(text):
        st = hier.advance_byte(st, by)
        if st[hier.ST_C] == hier.SEED_C:
            unit, code = text[start:i + 1], st[hier.ST_R0]
            start = i + 1
            units += 1
            coll += seen.setdefault(code, unit) != unit
    before = m.state_bytes()
    m.run(bytes_to_bits(text[: 1 << 20]))
    stable = m.state_bytes() == before
    return {"arity": hier.ARITY, "line_words": hw.line_words, "register_cap_bits": bounds.register_cap(hw),
            "state_bytes": before, "state_size_constant": bool(stable), "units": units,
            "distinct_units": len(seen), "code_collisions": int(coll),
            "collision_share": coll / max(1, units), "mean_unit_bytes": len(text) / max(1, units),
            "criteria": {"bounds_and_state": bool(ok and stable)}}


def part1_l2(A_train: bytes, smoke: bool) -> dict:
    n = (1 << 18) if smoke else (4 << 20)
    text = A_train[:n]
    null = bytes(np.random.default_rng(SEED).permutation(np.frombuffer(text, dtype=np.uint8)))
    m = make(hspec())
    rows, ok = [], True
    for label, data in (("A_train", text), ("byte_shuffled_null", null)):
        keys = byte_start_keys(m, data)
        nxt = np.frombuffer(data, dtype=np.uint8)
        for j, lvl in enumerate(m.hcfg.upper):
            full, halves = level_hankel(keys[:, j], nxt)
            res = floor_rank(full, halves)
            rows.append({"data": label, "level": hier.LEVEL_NAMES[lvl], **res})
            if label != "A_train" and res["rank"] is not None and res["rank"] > 1:
                ok = False
    return {"floor": FLOOR, "text_bytes": len(text), "rows": rows, "criteria": {"null_never_above_rank_1": bool(ok)}}


def part1_l3(smoke: bool) -> dict:
    eps = Config(cells=1, context=1, consolidating=True, dt=1.0).eps
    grid = np.linspace(eps, 1 - eps, 21)
    lo, hi, nest = 1.0, 0.0, 0.0
    for p, o, s, c in itertools.product(grid, grid, np.linspace(0, 1, 6), (0.0, 0.5, 1.0)):
        v = coherent_mix(float(p), float(o), float(s), float(c))
        lo, hi = min(lo, v), max(hi, v)
        assert min(p, o) - 1e-12 <= v <= max(p, o) + 1e-12
    for p, s, c in itertools.product(grid, np.linspace(0, 1, 6), (0.0, 0.5, 1.0)):
        nest = max(nest, abs(coherent_mix(float(p), float(p), float(s), float(c)) - float(p)))
    return {"min": lo, "max": hi, "eps": eps, "nesting_max_deviation": nest,
            "criteria": {"hull_and_bounded": bool(lo >= eps * (1 - 1e-12) and hi <= 1 - eps * (1 - 1e-12)),
                         "nesting": bool(nest < 1e-12)}}


def part1_l4(A_train: bytes, smoke: bool) -> dict:
    """Flip one byte of a frozen run; predictions beyond the span must not change (and stay unchanged)."""
    m = make(hspec())
    m.run(bytes_to_bits(A_train[: (1 << 18) if smoke else (8 << 20)]))
    n = 1 << 15 if smoke else 1 << 18
    seg = A_train[-n:]
    rng = np.random.default_rng(3)
    clean = m.copy().run(bytes_to_bits(seg), learning=False).reshape(len(seg), 8)
    rows, ok = [], True
    for _ in range(5 if smoke else 20):
        pos = int(rng.integers(100, n // 2))
        bad = bytearray(seg)
        bad[pos] ^= 0x20
        out = m.copy().run(bytes_to_bits(bytes(bad)), learning=False).reshape(len(seg), 8)
        diff = np.flatnonzero(np.any(out != clean, axis=1))
        if len(diff) == 0:
            rows.append({"pos": pos, "last_difference_after_bytes": 0})
            continue
        last = int(diff[-1])
        tail_equal = bool(not np.any(out[last + 1:] != clean[last + 1:]))
        ok &= tail_equal and last < n - 1
        rows.append({"pos": pos, "last_difference_after_bytes": last - pos, "tail_equal": tail_equal})
    spans = [r["last_difference_after_bytes"] for r in rows]
    return {"flips": rows, "max_span_bytes": int(max(spans)), "median_span_bytes": float(np.median(spans)),
            "criteria": {"effect_ends_and_stays_ended": bool(ok)}}


def part1_l5(smoke: bool) -> dict:
    nbytes = (1 << 20) if smoke else (8 << 20)
    noise = bytes_to_bits(bytes(np.random.default_rng(1).integers(0, 256, nbytes, dtype=np.uint8)))
    occ, storage = {}, True
    for v in (1, 3):
        m = make(hspec(verify=v, governor=False))
        size = m.nbytes()
        m.run(noise)
        storage &= m.nbytes() == size
        occ[v] = {hier.LEVEL_NAMES[lv - m._Lb]: m.occupancy()[lv] for lv in range(m._Lb, m.levels)}
        del m
        gc.collect()
    ok = all(occ[3][k] < 0.01 for k in occ[3]) and any(occ[1][k] > 0.5 for k in occ[1])
    return {"bytes": nbytes, "occupancy_verify1": occ[1], "occupancy_verify3": occ[3], "storage_constant": bool(storage),
            "criteria": {"noise_stays_out": bool(ok), "storage_constant": bool(storage)}}


def part1_cuts(A_train: bytes, smoke: bool) -> dict:
    m = make(hspec())
    n = (1 << 18) if smoke else (1 << 21)
    m.run(bytes_to_bits(A_train[:n]))
    st, ent = m.state, []
    for by in A_train[n:n + 2000]:
        st = hier.advance_byte(st, by)
        ent += hier.cut_entropies(m, st)
    return {"mean_label_entropy_bits": float(np.mean(ent)) if ent else None, "max_label_entropy_bits": max(ent, default=0.0),
            "samples": len(ent), "note": "bond 2 per cut: the area law S_cut <= 1 bit is vacuous on text"}


def stored(m, lv: int, x: int) -> bool:
    h = int(m._hidx[lv])
    slot = x >> int(m._shifts[h])
    return int(m._tags[m._tagoff[h] + slot]) == ((x & 0xFFFFFFFF) | 1)


def part1_l6(A_train: bytes, smoke: bool) -> dict:
    sentence = b"the quick brown fox jumps over the lazy dog. "
    noise = bytes_to_bits(bytes(np.random.default_rng(2).integers(0, 256, 8000, dtype=np.uint8)))
    kept = {}
    for name, kw in {"plain": dict(verify=1, governor=False), "verify_2": dict(verify=2, governor=False),
                     "governor": dict(verify=1, governor=True)}.items():
        m = make(hspec(**kw))
        small = hier.HierMachine(hier.HierConfig(replace(m.cfg, table_bits=9), upper_bits=9, scope="global"))
        small.run(bytes_to_bits(sentence * 100))
        lv = small._Lb + hier.ALL_UPPER.index(hier.W4)
        st, keys = small.state, []
        for bit in bytes_to_bits(sentence * 3):
            keys.append(hier.level_key(st, hier.W4))
            st = small.advance(st, int(bit))
        held = [k for k in set(keys) if stored(small, lv, k)]
        small.run(noise)
        kept[name] = {"before": len(held), "after": sum(stored(small, lv, k) for k in held),
                      "global_occupancy": small.global_occupancy()}
        del m, small
    surv = (kept["plain"]["after"] < 0.02 * max(1, kept["plain"]["before"]) and kept["plain"]["before"] > 0
            and all(kept[k]["before"] > 0 and kept[k]["after"] >= 0.97 * kept[k]["before"]
                    for k in ("verify_2", "governor")))
    occ_ok = all(v["global_occupancy"] <= 1.0 for v in kept.values())
    nb = (1 << 18) if smoke else (2 << 20)
    bits = bytes_to_bits(A_train[:nb])
    raised = {}
    base_cfg = make(hspec()).cfg
    for v in (2, 3, 4):                 # tables of 2**14 and 2**12 slots, so that 2 MiB reaches the Page point
        counts = []
        for raise_bar in (False, True):
            cfg = hier.HierConfig(replace(base_cfg, table_bits=14, verify=v), upper_bits=12, scope="global",
                                  raise_bar=raise_bar)
            m = hier.HierMachine(cfg)
            m.run(bits)
            counts.append(m.refused)
            del m
            gc.collect()
        raised[str(v)] = {"refused_without_raise": counts[0], "refused_with_raise": counts[1]}
    changes = all(r["refused_without_raise"] != r["refused_with_raise"] for r in raised.values())
    return {"survival": kept, "raised_bar_refusals": raised, "raise_changes_every_verify": bool(changes),
            "criteria": {"survival": bool(surv), "global_occupancy_at_most_1": bool(occ_ok),
                         "raise_changes_every_verify": bool(changes)}}


def part1_l7(A_train: bytes, smoke: bool) -> dict:
    m = make(hspec())
    m.run(bytes_to_bits(A_train[: (1 << 18) if smoke else (1 << 20)]))
    m.run(bytes_to_bits(b"the quick brown fox"), learning=False)
    ro = collapse.Readout(m)
    st, worst = ro.state0, 0.0
    for _ in range(24 if smoke else 64):
        z = collapse.first_step(ro, st, 8, 1.0)["Z"][0]
        worst = max(worst, abs(z - ro.p0(st)))
        st = ro.advance(st, int(z < 0.5))
    a, b = m.copy(), m.copy()
    same = collapse.generate(a, 8 * 4, horizon=8, hbar=0.0) == b.generate(8 * 4, horizon=8, select="max")
    return {"hbar1_Z_minus_prediction": worst, "hbar0_equals_max_planner": bool(same),
            "criteria": {"hbar1": bool(worst < 1e-10), "hbar0": bool(same)}}


# ---------------------------------------------------------------------------------------------------
# Part 2: configuration, chosen on A-validation only
# ---------------------------------------------------------------------------------------------------

class RowTimeout(RuntimeError):
    pass


def read_capped(m, data: bytes, limit: float, t0: float) -> float:
    total = 0.0
    for start in range(0, len(data), BLOCK):
        chunk = data[start:start + BLOCK]
        total += float(-np.log2(m.run(bytes_to_bits(chunk), learning=True)).sum())
        if time.perf_counter() - t0 > limit:
            raise RowTimeout(f"a Part 2 row ran past {limit:.0f} s: write the amendment before the resume")
    return total / max(1, len(data))


def part2(S: dict, partial: E.Partial, smoke: bool) -> dict:
    done = {r["key"]: r for r in partial.get("part2_rows") or []}

    def score(spec: dict) -> float:
        k = key(spec)
        if k not in done:
            t0 = time.perf_counter()
            m = make(spec)
            train = read_capped(m, S["A_train"], ROW_LIMIT, t0)
            val = frozen(m, S["A_validation"])
            done[k] = {"key": k, "spec": spec, "A_train_bpb": train, "A_validation_bpb": val, **counters(m),
                       "seconds": round(time.perf_counter() - t0, 1)}
            del m
            gc.collect()
            partial.put("part2_rows", list(done.values()))
            print("  row", {x: done[k][x] for x in ("spec", "A_validation_bpb", "seconds")}, flush=True)
        return done[k]["A_validation_bpb"]

    def choose(cands: list[dict]) -> dict:
        scored = [(score(c), c) for c in cands]
        best = min(s for s, _ in scored)
        near = [(options_on(c), s, c) for s, c in scored if s <= best + TIE]
        return min(near, key=lambda t: (t[0], t[1]))[2]

    stages, cur = [], hspec(hier_levels="WS")
    # 1. segmentation
    cands = [cur, {**cur, "hier_segment": "text"}]
    cur = choose(cands)
    stages.append({"stage": "segmentation", "candidates": cands, "chosen": cur})
    # 2. level set
    cands = [cur] if smoke else [cur, {**cur, "hier_levels": "W"}, {**cur, "hier_levels": "S"}]
    cur = choose(cands)
    stages.append({"stage": "levels", "candidates": cands, "chosen": cur})
    # 3. governor scope and the raised bar, then the sink ablation
    cands = [cur, {**cur, "hier_scope": "global"}, {**cur, "hier_scope": "global", "hier_raise": True}]
    cands = cands[:2] if smoke else cands
    cur = choose(cands)
    stages.append({"stage": "governor", "candidates": cands, "chosen": cur})
    cands = [cur, {**cur, "sink_rate": 0.0}]
    cur = choose(cands)
    stages.append({"stage": "sink_ablation", "candidates": cands, "chosen": cur})
    # 4. verify of the upper levels only
    cands = [cur] if smoke else [cur, {**cur, "hier_verify": 2}, {**cur, "hier_verify": 4}]
    cur = choose(cands)
    stages.append({"stage": "upper_verify", "candidates": cands, "chosen": cur})
    return {"stages": stages, "chosen": cur, "rows": list(done.values())}


# ---------------------------------------------------------------------------------------------------
# Parts 3 to 6
# ---------------------------------------------------------------------------------------------------

WORD = E.WORD


def write_settings(machine, texts: list[bytes], vocabulary: set, n_bytes: int, settings: list[dict]) -> dict:
    """The writing metrics of E3's Part 4 for the listed settings; copies are released after each prompt (T2.3)."""
    rows = []
    for setting in settings:
        t0 = time.perf_counter()
        outs, rep, words, found, printable, total, valid, own_bits = [], [], 0, 0, 0, 0, 0, 0.0
        for i, prompt in enumerate(texts):
            m = machine.copy()
            m.run(bytes_to_bits(prompt), learning=False)
            after_prompt = m.copy()
            rng = np.random.default_rng(SEED + i) if setting["mode"] == "sample" else None
            bits = collapse.generate(m, 8 * n_bytes, horizon=8, hbar=setting["hbar"], mode=setting["mode"],
                                     form=setting["form"], verified=False, rng=rng)
            data = bits_to_bytes(np.array(bits, dtype=np.uint8))
            outs.append({"prompt": prompt.decode("utf-8", "backslashreplace"),
                         "text": data.decode("utf-8", "backslashreplace")})
            rep.append(collapse.repetition(data))
            tokens = WORD.findall(data)
            words += len(tokens)
            found += sum(t.lower() in vocabulary for t in tokens)
            printable += sum(32 <= c < 127 or c in (9, 10, 13) for c in data)
            total += len(data)
            valid += E.utf8_valid(data)
            own_bits += float(-np.log2(after_prompt.run(np.array(bits, dtype=np.uint8), learning=False)).sum())
            del m, after_prompt
            gc.collect()
        rows.append({**setting, "verified": False, "repeated_8byte_windows": float(np.mean(rep)),
                     "words_in_A_train_vocabulary": found / words if words else None, "words": words,
                     "printable_bytes": printable / total, "valid_utf8": valid / len(texts),
                     "own_bits_per_byte": own_bits / total, "outputs": outs,
                     "seconds": round(time.perf_counter() - t0, 1)})
        print("  write", {k: rows[-1][k] for k in ("hbar", "mode", "repeated_8byte_windows", "own_bits_per_byte",
                                                    "seconds")}, flush=True)
    return {"bytes": n_bytes, "prompts": len(texts), "settings": rows}


WRITING = [{"hbar": 1.0, "form": "euclidean", "mode": "sample"}, {"hbar": 0.0, "form": "euclidean", "mode": "collapse"}]


def window_hashes(buf: np.ndarray, w: int) -> np.ndarray:
    """Polynomial hash (mod 2**64) of every ``w``-byte window of ``buf``."""
    n = len(buf) - w + 1
    h = np.zeros(n, dtype=np.uint64)
    mult = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        for k in range(w):
            h = h * mult + buf[k:k + n].astype(np.uint64) + np.uint64(1)
    return h


def window_classes(train: bytes, test: bytes) -> np.ndarray:
    """Per byte of ``test``: the longest of 32, 16, 8 whose preceding window occurs in ``train``; else 0."""
    tr = np.frombuffer(train, dtype=np.uint8)
    te = np.frombuffer(test, dtype=np.uint8)
    cls = np.zeros(len(te), dtype=np.int64)
    for w in sorted(CLASSES):                       # ascending: the longest match overwrites
        known = np.sort(window_hashes(tr, w))
        th = window_hashes(te, w)                   # th[j] is the window te[j:j+w]; it precedes byte j+w
        pos = np.searchsorted(known, th)
        hit = np.zeros(len(th), dtype=bool)
        inside = pos < len(known)
        hit[inside] = known[pos[inside]] == th[inside]
        cls[w:][hit[: len(te) - w]] = w
        del known, th, pos
        gc.collect()
    return cls


def decomposition(q: np.ndarray, cls: np.ndarray) -> dict:
    """Bits per byte and share of bytes in each window class, from the per-bit probabilities of a frozen reading."""
    per_byte = -np.log2(q).reshape(-1, 8).sum(axis=1)
    out = {}
    for c in (0, *sorted(CLASSES)):
        sel = cls == c
        out[str(c)] = {"share": float(sel.mean()), "bpb": float(per_byte[sel].mean()) if sel.any() else None,
                       "bits": float(per_byte[sel].sum())}
    out["all"] = {"share": 1.0, "bpb": float(per_byte.mean()), "bits": float(per_byte.sum())}
    return out


def arm_a(spec: dict, S: dict, texts, vocabulary, smoke: bool) -> dict:
    """Unit 3a: A only and A then B, with the end-of-A measurements (T12.3, generation, writing, window classes)."""
    evals = ("A_test", "A_heldout", "B_test", "B_heldout")
    out = {}
    bm = make(spec)
    curve_a, drift_a = [], []
    train_a = read(bm, S["A_train"], learning=True, curve=curve_a, drift=drift_a)
    end_of_a = bm.copy()
    out["A_only"] = {"train_bpb": train_a, **{e: frozen(end_of_a, S[e]) for e in evals}, "counters": counters(end_of_a)}
    out["t12_3"] = {"part2_weight_drift_per_MiB": drift_a,
                    "part3": {n: frozen(end_of_a, S[n]) for n in
                              ("A_test", "A_test_word_shuffled", "A_heldout", "A_heldout_word_shuffled")}}
    q = end_of_a.copy().run(bytes_to_bits(S["A_test"]), learning=False)
    cls = window_classes(S["A_train"], S["A_test"])
    out["decomposition_E4"] = decomposition(q, cls)
    out["window_class_shares"] = {str(c): float((cls == c).mean()) for c in (0, *sorted(CLASSES))}
    del q
    curve_b = []
    train_b = read(bm, S["B_train"], learning=True, curve=curve_b)
    out["A_then_B"] = {"train_bpb_B": train_b, **{e: frozen(bm, S[e]) for e in evals}, "counters": counters(bm)}
    out["learning_curves_bpb_per_MiB"] = {"A_then_B": curve_a + curve_b}
    del bm
    gc.collect()
    out["writing"] = write_settings(end_of_a, texts, vocabulary, 8 if smoke else 48, WRITING[:1] if smoke else WRITING)
    out["generation"] = generations(end_of_a, texts[:2] if smoke else prompts(S["A_test"]))
    return out


def arm_b(spec: dict, S: dict, seed_name: str = "A_train_sentence_shuffled") -> dict:
    """Unit 3b (and 3c): B only, and the shuffled A then B arm."""
    evals = ("A_test", "A_heldout", "B_test", "B_heldout")
    out, curves = {}, {}
    bm = make(spec)
    curves["B_only"] = []
    train = read(bm, S["B_train"], learning=True, curve=curves["B_only"])
    out["B_only"] = {"train_bpb_B": train, **{e: frozen(bm, S[e]) for e in evals}, "counters": counters(bm)}
    del bm
    gc.collect()
    bm = make(spec)
    curves["shuffled_A_then_B"] = []
    read(bm, S[seed_name], learning=True, curve=curves["shuffled_A_then_B"])
    out["shuffled_A_then_B"] = {"after_A": {e: frozen(bm, S[e]) for e in evals}}
    read(bm, S["B_train"], learning=True, curve=curves["shuffled_A_then_B"])
    out["shuffled_A_then_B"].update({e: frozen(bm, S[e]) for e in evals})
    out["shuffled_A_then_B"]["counters"] = counters(bm)
    del bm
    gc.collect()
    out["learning_curves_bpb_per_MiB"] = curves
    return out


def control_arm(spec: dict, S: dict) -> dict:
    """Part 4: the A only arm of a control (train on A-train, frozen evaluation)."""
    evals = ("A_test", "A_heldout", "B_test", "B_heldout")
    bm = make(spec)
    train = read(bm, S["A_train"], learning=True)
    out = {"train_bpb": train, **{e: frozen(bm, S[e]) for e in evals}, "counters": counters(bm)}
    del bm
    gc.collect()
    return out


def e3_decomposition(S: dict) -> dict:
    """Part 5, the E3 side: the E3 machine retrained on A-train, frozen on A-test, by window class."""
    bm = make(hspec(hier_levels=""))
    read(bm, S["A_train"], learning=True)
    q = bm.copy().run(bytes_to_bits(S["A_test"]), learning=False)
    cls = window_classes(S["A_train"], S["A_test"])
    out = decomposition(q, cls)
    del bm, q
    gc.collect()
    return out


# ---------------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--smoke", action="store_true", help="first 256 KiB of each stream, a small grid")
    ap.add_argument("--resume", action="store_true", help="reuse the units saved in the partial file")
    ap.add_argument("--seed2", action="store_true", help="also run unit 3c (the shuffled arm, second seed)")
    args = ap.parse_args()
    t_start = time.perf_counter()
    name = "e4_smoke" if args.smoke else "e4"
    partial = E.Partial(name, args.resume)
    S = streams(args.manifest, args.smoke)
    sizes = {k: len(v) for k, v in S.items()}
    print("stream bytes:", sizes, flush=True)

    def unit(key_: str, fn, *a):
        got = partial.get(key_)
        if got is None:
            print(key_, flush=True)
            partial.put(key_, E.timed(fn, *a))
            got = partial.get(key_)
        return got

    p0 = unit("part0", part0, S, args.smoke)
    p1 = {
        "N_L1": unit("N_L1", lambda: part1_l1(S["A_train"], args.smoke)),
        "N_L2": unit("N_L2", lambda: part1_l2(S["A_train"], args.smoke)),
        "N_L3": unit("N_L3", lambda: part1_l3(args.smoke)),
        "N_L4": unit("N_L4", lambda: part1_l4(S["A_train"], args.smoke)),
        "N_L5": unit("N_L5", lambda: part1_l5(args.smoke)),
        "N_L5_cuts": unit("N_L5_cuts", lambda: part1_cuts(S["A_train"], args.smoke)),
        "N_L6": unit("N_L6", lambda: part1_l6(S["A_train"], args.smoke)),
        "N_L7": unit("N_L7", lambda: part1_l7(S["A_train"], args.smoke)),
    }
    p2 = partial.get("part2")
    if p2 is None:
        print("part 2", flush=True)
        partial.put("part2", part2(S, partial, args.smoke))
        p2 = partial.get("part2")
    spec = p2["chosen"]
    vocabulary = {w.lower() for w in WORD.findall(S["A_train"])}
    texts = prompts(S["A_test"])[: 2 if args.smoke else 8]
    p3a = unit("part3a", arm_a, spec, S, texts, vocabulary, args.smoke)
    p5_e3 = unit("part5_e3", e3_decomposition, S)
    p3b = unit("part3b", arm_b, spec, S)
    p3c = None
    if args.seed2:                  # a second sentence shuffle of A-train, drawn from its own generator
        from vam.corpus import Corpus, shuffle_sentences
        extra = Corpus.load(args.manifest).stream("A", "train", control=shuffle_sentences,
                                                  rng=np.random.default_rng(SEED + 1))
        S["A_train_sentence_shuffled_seed2"] = extra[: len(S["A_train"])]
        p3c = unit("part3c", arm_b, spec, S, "A_train_sentence_shuffled_seed2")
    p4 = {"C1_equal_storage_no_extra_span": unit("C1", control_arm, {**spec, "hier_control": "dup"}, S),
          "C2_order_destroyed": unit("C2", control_arm, {**spec, "hier_control": "comm"}, S)}
    payload = {
        "experiment": "E4 N above one equation, on text",
        "protocol": "experiments/PROTOCOL.md#e4-n-above-one-equation-on-text-design-section-11-decision-d11",
        "provenance": provenance(seed=SEED, manifest=str(args.manifest), smoke=args.smoke,
                                 manifest_sha256=E.sha256(args.manifest.read_bytes()), resumed=args.resume),
        "stream_bytes": sizes, "part0": p0, "part1": p1, "part2": p2, "part3a": p3a, "part3b": p3b, "part3c": p3c,
        "part4": p4, "part5": {"E3": p5_e3, "E4": p3a["decomposition_E4"], "shares": p3a["window_class_shares"]},
        "part6": p3a["writing"], "seconds": round(time.perf_counter() - t_start, 1),
    }
    print(save(name, E.plain(payload)))


if __name__ == "__main__":
    main()
