"""Experiment E7: rotation families on a synthetic additive relation (experiments/PROTOCOL.md, E7; DESIGN section 12).

    python experiments/e7_families.py --smoke                 # tiny draws, every code path
    python experiments/e7_families.py                         # the evaluation draws of E7.3
    python experiments/e7_families.py --resume                # skip the units saved in results/e7_partial.json

No corpus is needed: the data are synthetic (labelled so). Units, each saved as it ends: ``part0`` (gates (a) (b) (c)
(h)), then one machine per (stage, h, seed, relation, arm). The criterion C-E7 and everything else reported is computed
by ``experiments/report.py`` from the saved file. Run one job at a time.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import time

import e3_nfa as E
import numpy as np
from _common import provenance, save
from e2_curriculum import LAWS
from e6_operator import GateFailure, check_spec
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form

from vam.bits import bytes_to_bits
from vam.nfa import hier
from vam.nfa import operator as O
from vam.nfa.families import (
    CPStage,
    FamilyConfig,
    LatticeStage,
    LatticeTimeout,
    Spelling,
    build_family,
    cp_fit,
    fourier_readout,
)
from vam.realizations.vam1 import Config

EPOCHS = 20
STAGES = {1: {"operands": 10, "results": 19}, 2: {"operands": 100, "results": 199}}
#: evaluation seeds (E7.3, Amendment 1); development seeds 0 to 19 (stage 1) and 0 (stage 2) are never used here
SEEDS = {(1, 0.2): list(range(20, 30)), (1, 0.4): list(range(20, 30)), (2, 0.2): [1, 2, 3], (2, 0.4): [1]}
SMOKE_SEEDS = {(1, 0.2): [0, 1], (1, 0.4): [0], (2, 0.2): [], (2, 0.4): []}
ARMS = ("E4", "E4+F5", "E4+F4")
CONTROL_SEEDS = [20, 21, 22, 23, 24]
SMOKE_CONTROL_SEEDS = [0, 1]
LATTICE_DEADLINE = 15 * 60.0
STOP_SECONDS = 15 * 60.0
STAGE2_ITEMS = 100
DEFAULT_SLOT = "top"                     # E7 Amendment 1; "after0" is the reported ablation (E7.4 as first written)
_LATTICES: dict = {}                      # (stage, h, seed) -> LatticeStage, or None after a timeout


# -- data ------------------------------------------------------------------------------------------------------------

def pair_list(nop: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(nop) for b in range(a, nop)]


def split(nop: int, h: float, seed: int) -> tuple[list, list]:
    pairs = pair_list(nop)
    idx = np.random.default_rng(seed).permutation(len(pairs))
    k = round(h * len(pairs))
    return [pairs[i] for i in idx[:k]], [pairs[i] for i in idx[k:]]


def random_table(seed: int, nop: int, nres: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 500)
    t = np.zeros((nop, nop), dtype=np.int64)
    for a in range(nop):
        for b in range(a, nop):
            t[a, b] = t[b, a] = rng.integers(0, nres)
    return t


def ordered(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted({p for a, b in pairs for p in ((a, b), (b, a))})


def epoch_bytes(eqs: list[tuple[int, int, int]], seed: int) -> list[bytes]:
    rng = np.random.default_rng(seed + 2000)
    out = []
    for _ in range(EPOCHS):
        out.append(b"".join(f"{a}+{b}={c}\n".encode() for a, b, c in (eqs[i] for i in rng.permutation(len(eqs)))))
    return out


# -- gates -----------------------------------------------------------------------------------------------------------

def hnf_member(rows: list[list[int]], r: list[int]) -> bool:
    return hermite_normal_form(Matrix(rows).T) == hermite_normal_form(Matrix([*rows, r]).T)


def gate_lattice(n_systems: int = 200) -> dict:
    """(b): canonical-form equality equals membership by sympy's Hermite normal form on random small systems."""
    rng = np.random.default_rng(0)
    agree = 0
    for _ in range(n_systems):
        n = int(rng.integers(3, 6))
        rows = [[int(v) for v in rng.integers(-2, 3, size=n)] for _ in range(int(rng.integers(1, 4)))]
        r = [int(v) for v in rng.integers(-2, 3, size=n)]
        if not any(any(row) for row in rows):
            rows[0][0] = 1
        from vam.nfa.families import Lattice
        lat = Lattice(n)
        for row in rows:
            lat.add(row)
        agree += (lat.canon(r) == lat.canon([0] * n)) == hnf_member([row for row in rows if any(row)], r)
    complete = LatticeStage.build(19, [(a, b, a + b) for a in range(10) for b in range(10)])
    right = all(complete.candidates(a, b) == [a + b] for a in range(10) for b in range(10))
    if agree != n_systems or not right:
        raise GateFailure("gate (b): the lattice code disagrees with the Hermite normal form or is wrong when complete")
    return {"systems": n_systems, "agree": agree, "complete_stage1_unique_and_right": right}


def gate_closed_case() -> dict:
    """(c): V5's closed case, F5 on addition modulo 10 and on stage 1 with nothing held out (the fitted readout)."""
    eqs = [(a, b, (a + b) % 10) for a in range(10) for b in range(10)]
    A, B, G, resid = cp_fit(10, 10, eqs, 10, seed=0)
    mod10 = float(np.mean([int(np.argmax(fourier_readout(A, B, G, a, b))) == (a + b) % 10
                           for a in range(10) for b in range(10)]))
    eqs1 = [(a, b, a + b) for a in range(10) for b in range(10)]
    st1 = CPStage.build(10, 19, eqs1, seed=0)
    stage1 = float(np.mean([int(np.argmax(st1.dist(a, b))) == a + b for a in range(10) for b in range(10)]))
    if mod10 < 1.0 or stage1 < 0.99:
        raise GateFailure(f"gate (c): F5 closed case {mod10}, stage 1 complete {stage1}")
    return {"mod10_top1": mod10, "mod10_mean_residual": float(resid.mean()), "stage1_complete_top1": stage1,
            "stage1_complete_mean_residual": float(st1.resid.mean())}


def gate_parser() -> dict:
    """(h): the answer bytes of every line lie in the word unit that E4's text rule closes at the newline."""
    for a, b, c in ((0, 0, 0), (7, 9, 16), (45, 99, 144)):
        line = f"{a}+{b}={c}\n".encode()
        ans = line[line.index(b"=") + 1:-1]
        if not (all(hier.is_word_byte(x) for x in ans) and not hier.is_word_byte(61) and not hier.is_word_byte(10)):
            raise GateFailure("gate (h): the answer is not one word unit closed by the newline")
    return {"checked": True}


def gate_identity(spec: dict, smoke: bool) -> dict:
    """(a): with every family stage off the E7 loop equals E4's (per-bit probabilities, learning then frozen)."""
    law = Config(cells=1, context=1, **LAWS[spec["law"]])
    rng = np.random.default_rng(0)
    n = (1 << 14) if smoke else (1 << 17)
    data = bytes(rng.integers(0, 256, size=n, dtype=np.uint8))
    ref = hier.build(spec, law)
    q1 = ref.run(bytes_to_bits(data), learning=True)
    f1 = ref.copy().run(bytes_to_bits(data[:n // 2]), learning=False)
    del ref
    gc.collect()
    m = build_family(spec, law, FamilyConfig(route="off"))
    q2 = m.run(bytes_to_bits(data), learning=True)
    f2 = m.copy().run(bytes_to_bits(data[:n // 2]), learning=False)
    same = bool(np.array_equal(q1, q2) and np.array_equal(f1, f2))
    del m
    gc.collect()
    if not same:
        raise GateFailure("gate (a): with the family stages off the E7 path differs from E4's")
    return {"bytes": n, "bit_identical": same}


def part0(spec: dict, smoke: bool) -> dict:
    return {"gate_a": gate_identity(spec, smoke), "gate_b": gate_lattice(40 if smoke else 200),
            "gate_c": gate_closed_case(), "gate_h": gate_parser()}


# -- one unit --------------------------------------------------------------------------------------------------------

def costs(m, sp: Spelling, a: int, b: int) -> np.ndarray:
    """Bits to read each candidate result (and its newline) after ``a+b=``, learning off, from the state after the
    prefix (the prefix is read once; a frozen read changes no table)."""
    start = m.snapshot()
    m.run(bytes_to_bits(f"{a}+{b}=".encode()), learning=False)
    mid = m.snapshot()
    out = np.empty(len(sp.spell))
    for c, o in enumerate(sp.spell):
        m.restore(mid)
        out[c] = float(-np.log2(m.run(bytes_to_bits(o), learning=False)).sum())
    m.restore(start)
    return out


def make_machine(spec: dict, arm: str, st: dict, seed: int, cache: dict, deadline: float | None, slot: str = "after0"):
    law = Config(cells=1, context=1, **LAWS[spec["law"]])
    if arm == "E4+O":
        return O.build_operator(spec, law, O.OperatorConfig(mode="spectral", compile_bytes=1 << 12))
    route = {"E4": "off", "E4+F5": "f5", "E4+F4": "f4"}[arm]
    return build_family(spec, law, FamilyConfig(route=route, results=st["results"], seed=seed, cache=cache,
                                                lattice_deadline=deadline, slot=slot))


def unit(spec: dict, stage: int, h: float, seed: int, arm: str, relation: str, smoke: bool,
         slot: str = "after0") -> dict:
    """One machine: read the draw's stream (E epochs), then score the held-out and seen items frozen."""
    st = STAGES[stage]
    nop, nres = st["operands"], st["results"]
    held, seen = split(nop, h, seed)
    table = random_table(seed, nop, nres) if relation == "random" else None

    def target(a: int, b: int) -> int:
        return int(table[a, b]) if table is not None else a + b

    seen_eqs = [(a, b, target(a, b)) for a, b in ordered(seen)]
    rng = np.random.default_rng(seed + 3000)
    if stage == 1:
        items = [(a, b) for a, b in ordered(held)]
        seen_items = [(a, b) for a, b, _ in seen_eqs]
    else:
        pick = rng.choice(len(held), size=min(STAGE2_ITEMS, len(held)), replace=False)
        items = [(held[i][0], held[i][1]) if rng.integers(2) else (held[i][1], held[i][0]) for i in sorted(pick)]
        pick = rng.choice(len(seen_eqs), size=min(STAGE2_ITEMS, len(seen_eqs)), replace=False)
        seen_items = [(seen_eqs[i][0], seen_eqs[i][1]) for i in sorted(pick)]
    cache: dict = {}
    classify = None
    cls_seconds = None
    if relation == "sum" and arm != "E4+O":
        try:
            lkey = (stage, h, seed)                              # one lattice per draw, shared by the arms
            if lkey not in _LATTICES:
                _LATTICES[lkey] = LatticeStage.build(nres, [(a, b, a + b) for a, b in ordered(seen)], LATTICE_DEADLINE)
            lat = _LATTICES[lkey]
            if lat is None:
                raise LatticeTimeout("cached timeout")
            cache[frozenset(seen_eqs)] = lat
            classify = {(a, b): lat.determined(a, b, a + b) for a, b in items}
            for a, b in items:                                   # gate (d): a determined candidate is the true sum
                if classify[(a, b)] and lat.candidates(a, b) != [a + b]:
                    raise GateFailure(f"gate (d): the determined candidate of {a}+{b} is not unique and right")
            cls_seconds = lat.seconds
        except LatticeTimeout:
            _LATTICES[lkey] = None
            classify = None
    m = make_machine(spec, arm, st, seed, cache, LATTICE_DEADLINE, slot)
    t0 = time.perf_counter()
    for chunk in epoch_bytes(seen_eqs, seed):
        m.run(bytes_to_bits(chunk), learning=True)
        if hasattr(m, "end_epoch"):
            m.end_epoch()
        if time.perf_counter() - t0 > STOP_SECONDS:
            raise GateFailure(f"a unit ran past {STOP_SECONDS:.0f} s while reading")
    read_seconds = time.perf_counter() - t0
    before = m.table_hashes()
    sp = Spelling(nres)
    rows = []
    for (a, b), is_seen in [(i, False) for i in items] + [(i, True) for i in seen_items]:
        c_true = target(a, b)
        cs = costs(m, sp, a, b)
        active = None
        if hasattr(m, "stage_dist") and arm != "E4":
            active = bool(m.stage_dist(a, b) is not None)
        rows.append({"a": a, "b": b, "c": c_true, "seen": is_seen, "top1": int(np.argmin(cs) == c_true),
                     "bits_correct": float(cs[c_true]), "active": active,
                     "determined": None if (classify is None or is_seen) else bool(classify[(a, b)])})
        if time.perf_counter() - t0 > STOP_SECONDS + 600:
            raise GateFailure("a unit ran past its time while scoring")
    if m.table_hashes() != before:
        raise GateFailure(f"frozen reads changed the tables, the equations or the evidences ({arm})")
    out = {"stage": stage, "h": h, "seed": seed, "arm": arm, "relation": relation, "read_seconds": read_seconds,
           "items": rows, "held_out_pairs_sha256": hashlib.sha256(repr(sorted(held)).encode()).hexdigest(),
           "classification_seconds": cls_seconds, "classified": classify is not None}
    if hasattr(m, "stage_regret") and arm != "E4":
        r = m.stage_regret()
        out["stage_regret"] = r
        if r["max_excess_bits"] > 2.0 + 1e-6:
            raise GateFailure(f"the family stage costs {r['max_excess_bits']} bits at a node ({arm}), bound 2")
    if hasattr(m, "fits"):
        out["fits"] = m.fits
    seen_top1 = float(np.mean([r["top1"] for r in rows if r["seen"]]))
    out["seen_top1"] = seen_top1
    if arm == "E4" and stage == 1 and relation == "sum" and seen_top1 < 0.99:
        raise GateFailure(f"gate (g): E4's top-1 on its own seen equations is {seen_top1:.3f} < 0.99")
    del m
    gc.collect()
    return out


def unit_key(stage: int, h: float, seed: int, arm: str, relation: str) -> str:
    return f"s{stage}_h{h}_seed{seed}_{arm}_{relation}"


def plan(smoke: bool, stages: tuple = (1, 2)) -> list[tuple]:
    seeds = SMOKE_SEEDS if smoke else SEEDS
    out = []
    for (stage, h), ss in seeds.items():
        if stage not in stages:
            continue
        for seed in ss:
            for arm in ARMS:
                out.append((stage, h, seed, arm, "sum"))
            if stage == 1 and h == 0.2:
                out.append((stage, h, seed, "E4+O", "sum"))
    for seed in (SMOKE_CONTROL_SEEDS if smoke else CONTROL_SEEDS):
        for arm in ARMS:
            out.append((1, 0.2, seed, arm, "random"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="two stage-1 draws at h = 0.2, one at 0.4, no stage 2")
    ap.add_argument("--stages", choices=["1", "1,2"], default="1,2", help="stage 1 only (for the slot ablation)")
    ap.add_argument("--slot", choices=["after0", "top"], default=DEFAULT_SLOT,
                    help="where the family stage enters the chain (E7.4; 'top' only after an amendment)")
    ap.add_argument("--resume", action="store_true", help="reuse the units saved in the partial file")
    args = ap.parse_args()
    t_start = time.perf_counter()
    spec = check_spec()
    name = ("e7_smoke" if args.smoke else "e7") + ("" if args.slot == DEFAULT_SLOT else "_" + args.slot)
    partial = E.Partial(name, args.resume)

    def run_unit(key: str, fn, *a):
        got = partial.get(key)
        if got is None:
            print(key, flush=True)
            partial.put(key, E.timed(fn, *a))
            got = partial.get(key)
        return got

    p0 = run_unit("part0", part0, spec, args.smoke)
    units = {}
    for stage, h, seed, arm, relation in plan(args.smoke, tuple(int(x) for x in args.stages.split(","))):
        key = unit_key(stage, h, seed, arm, relation)
        units[key] = run_unit(key, unit, spec, stage, h, seed, arm, relation, args.smoke, args.slot)
    payload = {
        "experiment": "E7 rotation families on a synthetic relation (synthetic data)",
        "protocol": "experiments/PROTOCOL.md#e7-rotation-families-on-a-synthetic-relation",
        "provenance": provenance(smoke=args.smoke, resumed=args.resume, epochs=EPOCHS),
        "slot": args.slot, "spec": spec,
        "seeds": {f"{k[0]}_{k[1]}": v for k, v in (SMOKE_SEEDS if args.smoke else SEEDS).items()},
        "part0": p0, "units": units, "seconds": round(time.perf_counter() - t_start, 1),
    }
    print(save(name, E.plain(payload)))


if __name__ == "__main__":
    main()
