"""Experiment E3: the owner's nested fractal architecture, layer by layer (experiments/PROTOCOL.md, E3).

    python experiments/e3_nfa.py --manifest experiments/data/pg19/MANIFEST.json
    python experiments/e3_nfa.py --manifest ... --smoke     # code check: small streams, chains and grids
    python experiments/e3_nfa.py --manifest ... --resume    # continue from results/e3_partial.json

Part 1 measures every layer's designed behaviour against the criterion the protocol states. Part 2
chooses the configuration on A-validation only, one stage per layer. Part 3 runs E2's four arms with it.
Part 4 writes from the end-of-A machine with the path integral of N-L7 at every setting of the grid.
Every part is saved to the partial file as soon as it ends (Part 2 after every configuration), so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import codecs
import itertools
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from _common import RESULTS, commit, provenance, save
from e2_curriculum import LAWS, NESTED, SEED, arms, frozen, make, prompts, read, streams

from vam import sources as S_
from vam.bits import bits_to_bytes, bytes_to_bits
from vam.corpus import sha256
from vam.nested import NestedConfig, NestedMachine, coherent_mix
from vam.nfa import blankets as bl
from vam.nfa import bounds, collapse, page
from vam.nfa.flow import EventFlow
from vam.nfa.governor import GovernedEngine, operator_entropies
from vam.realizations.vam1 import Config, DenseMaps
from vam.realizations.vam1_mpo import OperatorEngine

START = {"machine": "nested", "law": "counting", "prior": 4.0}       # E2's best overall configuration
LAW = Config(cells=1, context=1, **LAWS["counting"])                  # its law, for the small text probes
TIE = 0.0005                                                          # bpb: ties go to the simpler setting
CHAIN_TEXT = (b"A textbook states a rule, shows it at work on an example, and then asks the reader to use it "
              b"alone. The machine reads every byte as eight bits, oldest bit first along the chain. ")
WORD = re.compile(rb"[A-Za-z]+")


# ---------------------------------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------------------------------

def plain(x):
    """JSON-ready copy: numpy scalars and arrays become Python numbers and lists, keys become strings."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


class Partial:
    """Results saved after every unit of work; ``--resume`` reloads them."""

    def __init__(self, name: str, resume: bool) -> None:
        self.path = RESULTS / f"{name}_partial.json"
        self.data = json.loads(self.path.read_text()) if resume and self.path.exists() else {}

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        self.data[key] = value = plain(value)
        RESULTS.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1) + "\n")
        tmp.replace(self.path)


def timed(fn, *args, **kw) -> dict:
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return {**out, "seconds": round(time.perf_counter() - t0, 1)}


# ---------------------------------------------------------------------------------------------------
# Part 1: designed behaviours
# ---------------------------------------------------------------------------------------------------

FLOW_GRID = [Config(2, 1, theta=0.3, dephasing=0.7), Config(2, 2, theta=0.5, dephasing=math.inf),
             Config(2, 1, theta=0.4, dephasing=0.5, einselect="all", consolidating=True), Config(2, 2, theta=0.0)]


def flow_against_machine(cfg: Config, events: int, seed: int) -> float:
    """Largest difference between the integrated flow and the machine over reading events."""
    flow, machine = EventFlow(cfg), DenseMaps(cfg).reader()
    x, rng = flow.initial(), np.random.default_rng(seed)
    worst = 0.0
    for _ in range(events):
        b = int(rng.integers(0, 2))
        rec = machine.step(b, rng)
        res = flow.event(x, b, int(rec.outcome))
        x = res.state
        s = machine.state.structural
        worst = max(worst, np.abs(x.R - machine.state.R).max(), abs(res.q_b - rec.q_bit),
                    abs(res.p_tau - rec.p_outcome), np.abs(x.w - s["w"]).max(), np.abs(x.n - s["n"]).max())
    return float(worst)


def flow_generation(cfg: Config, events: int, seed: int) -> float:
    """Largest difference over generation events, after three reading events."""
    maps = DenseMaps(cfg)
    reader = maps.reader()
    rng = np.random.default_rng(seed)
    for b in (1, 0, 1):
        reader.step(b, rng)
    gen = maps.generator(reader.state)
    flow = EventFlow(cfg)
    x = flow.initial()
    x.R, x.w, x.n = gen.state.R.copy(), gen.state.structural["w"].copy(), gen.state.structural["n"].copy()
    worst = 0.0
    for tau in rng.integers(0, 2, events):
        rec = gen.force(None, int(tau))
        res = flow.generation_event(x, int(tau))
        x = res.state
        worst = max(worst, np.abs(x.R - gen.state.R).max(), abs(res.p_tau - rec.p_outcome),
                    np.abs(x.w - gen.state.structural["w"]).max())
    return float(worst)


def loss_integral(cfg: Config, trials: int, rng: np.random.Generator) -> float:
    """``-ln q_b`` against the integral of ``Tr(K_b R)`` along the observation flow."""
    flow = EventFlow(cfg)
    worst = 0.0
    for _ in range(trials):
        x = flow.initial()
        x.w[:] = rng.uniform(-1, 1, x.w.shape)
        for _ in range(3):                         # a mixed state: on a basis state the filter never moves
            x = flow.event(x, int(rng.integers(0, 2)), int(rng.integers(0, 2))).state
        b = int(rng.integers(0, 2))
        k = flow._k(x.w, b)
        _, ln_q = flow._filter(k, x.R, None, "observe", 0)
        O = np.diag(np.exp(-k / 2))
        worst = max(worst, abs(ln_q + math.log(float(np.real(np.trace(O @ x.R @ O))))))
    return worst


def part1_l1(smoke: bool) -> dict:
    hw = bounds.Hardware.detect()
    cap = bounds.register_cap(hw)
    bounds.check_depths(NESTED["depths"], hw)
    events = 4 if smoke else 10
    reading = {f"c{c.cells}k{c.context}t{c.theta}g{c.dephasing}{c.einselect}": flow_against_machine(c, events, 3)
               for c in (FLOW_GRID[:1] if smoke else FLOW_GRID)}
    generation = flow_generation(Config(2, 2, theta=0.6, dephasing=0.4), events, 11)
    integral = loss_integral(Config(2, 1, theta=0.3, dephasing=0.7), 5 if smoke else 20, np.random.default_rng(1))
    worst = max([*reading.values(), generation])
    return {"bounds": bounds.summary(hw), "register_cap_bits": cap, "deepest_level_bits": max(NESTED["depths"]),
            "flow_max_difference_reading": reading, "flow_max_difference_generation": generation,
            "loss_integral_max_difference": integral,
            "criteria": {"a_register": max(NESTED["depths"]) <= cap == 64, "b_flow": worst < 1e-8,
                         "c_integral": integral < 1e-10}}


def part1_l2(A_train: bytes, smoke: bool) -> dict:
    n_bits = 40_000 if smoke else 400_000
    known = {"S1_markov2": (S_.MarkovBits(), 4), "S2_codes": (S_.IncompleteCodes(), 3),
             "S4_echo3": (S_.DelayedEcho(3, 0.1), 8), "S4_echo4": (S_.DelayedEcho(4, 0.15), 16)}
    rows, ok = [], True
    for (name, (src, states)), k in itertools.product(known.items(), (6, 8)):
        h = src.hmm()
        C, _, _, _ = page.whiten(page.exact_hankel(h.T, h.stationary, k, k))
        exact = page.singular_values(C)
        full, halves = page.hankel_halves(src.sample(n_bits, np.random.default_rng(SEED)).bits, k, k)
        sp = page.page_spectrum(full, halves)
        resolvable = bool(exact[states - 1] > sp["cutoff"])
        good = sp["rank"] <= states and (sp["rank"] == states or not resolvable)
        ok &= good
        rows.append({"source": name, "window": k, "states": states, "rank": sp["rank"],
                     "weakest_direction": float(exact[states - 1]), "cutoff": sp["cutoff"],
                     "resolvable": resolvable, "pass": good})
    for k in (6, 8):
        bits = np.random.default_rng(SEED + k).integers(0, 2, n_bits).astype(np.uint8)
        full, halves = page.hankel_halves(bits, k, k)
        rank = page.page_rank(full, halves)
        ok &= rank == 1
        rows.append({"source": "iid", "window": k, "states": 1, "rank": rank, "pass": rank == 1})
    text = A_train[: (1 << 18) if smoke else (4 << 20)]
    depths = (1, 2, 3, 4, 5, 6, 7)
    profile = page.depth_profile(text, depths=depths)
    # how the profile depends on the verification threshold (the turnaround above uses TEXT_MIN_COUNT only)
    others = {}
    for mc in (1, 4):
        prof = page.depth_profile(text, depths=depths, min_count=mc)
        others[str(mc)] = {"ranks": [r["rank"] for r in prof], "turnaround_bytes": page.turnaround(prof),
                           "unverified_share": [r["unverified_share"] for r in prof]}
    return {"known_sources": rows, "text_depth_profile": profile, "turnaround_bytes": page.turnaround(profile),
            "text_bytes": len(text), "text_min_count": page.TEXT_MIN_COUNT, "text_other_min_counts": others,
            "criteria": {"a_known_sources": bool(ok)}}


def part1_l3() -> dict:
    """Every coherent readout probability stays in ``[eps, 1 - eps]`` (proof: DESIGN 10.1, layer L3; checked on a
    grid)."""
    eps = LAW.eps
    grid = np.linspace(eps, 1 - eps, 41)
    lo, hi = 1.0, 0.0
    for p, o, s, c in itertools.product(grid, grid, np.linspace(0, 1, 11), np.linspace(0, 1, 5)):
        v = coherent_mix(float(p), float(o), float(s), float(c))
        lo, hi = min(lo, v), max(hi, v)
    return {"eps": eps, "min": lo, "max": hi, "criteria": {"bounded": lo >= eps * (1 - 1e-12) and
                                                                     hi <= 1 - eps * (1 - 1e-12)}}


def chain_run(engine, bits, taus) -> dict:
    q, s2 = [], []
    t0 = time.perf_counter()
    for b, t in zip(bits, taus, strict=True):
        q.append(engine.force(int(b), int(t)))
        s2.append(-math.log2(engine.x.norm() ** 2) / (2 * engine.cfg.cells + 1))
    return {"q": q, "renyi2_per_qubit": s2, "discarded": engine.discarded, "bonds": list(engine.bond_dims),
            "governed_since": getattr(engine, "governed_since", None), "seconds": round(time.perf_counter() - t0, 1)}


def part1_chain_blankets(smoke: bool) -> dict:
    """N-L4, N-L5, N-L6 on the chain: 8 cells, context 2, theta 0.5, 40 bits of English, 5 forced events."""
    cells, nbits, events, cell = (5, 16, 3, 3) if smoke else (8, 40, 5, 6)
    cfg = Config(cells, 2, theta=0.5)
    bits = bytes_to_bits(CHAIN_TEXT)[:nbits]
    taus = np.random.default_rng(7).integers(0, 2, nbits)
    engines = {
        "plain_bond64": OperatorEngine(cfg, max_bond=64),
        "governed_cache_bond": GovernedEngine(cfg, always=True),
        "governed_page_threshold": GovernedEngine(cfg),
        "governed_entropy_bond": GovernedEngine(cfg, always=True, bond="entropy", margin=4.0),
        # untruncated (no bond cap below 4096, cutoff 0): every qubit einselected, or the sinks at rate 40
        "plain_untruncated_einselect_all": OperatorEngine(Config(cells, 2, theta=0.5, einselect="all"),
                                                          max_bond=1 << 12, cutoff=0.0),
        "governed_untruncated_sink_rate_40": GovernedEngine(cfg, max_bond=1 << 12, cutoff=0.0, always=True,
                                                            sink_rate=40.0),
    }
    rng = np.random.default_rng(2)
    nb, nt = [int(x) for x in rng.integers(0, 2, events)], [int(x) for x in rng.integers(0, 2, events)]
    out = {}
    for name, e in engines.items():
        run = chain_run(e, bits, taus)
        ch = bl.containment(e, cell, nb, nt)
        out[name] = {"discarded": run["discarded"], "bonds": run["bonds"], "seconds": run["seconds"],
                     "renyi2_per_qubit_end": run["renyi2_per_qubit"][-1],
                     "renyi2_per_qubit_max": max(run["renyi2_per_qubit"]),
                     "governed_since": run["governed_since"],
                     "cone_leak": bl.cone_violation(ch, cell, cfg.context), "changes": ch.tolist(),
                     "cmi_bits": bl.cmi_profile(e), "operator_entropy_bits": operator_entropies(e)}
        print("  chain", name, {k: out[name][k] for k in ("discarded", "cone_leak", "seconds")}, flush=True)
    return {"config": {"cells": cells, "context": 2, "theta": 0.5, "bits": nbits, "events": events, "cell": cell},
            "engines": out, "criteria": {"untruncated_cone_leak": all(
                out[k]["cone_leak"] < 1e-12 and out[k]["discarded"] < 1e-20
                for k in ("plain_untruncated_einselect_all", "governed_untruncated_sink_rate_40"))}}


def part1_chain_fidelity(smoke: bool) -> dict:
    """Governed engines at bonds 32, 64, 90 against bond 128 (12 cells, context 4, theta 0.3, 120 bits), and
    the entropy-governed (area-law) bond at margins 2, 4 and 8 under the cache cap."""
    cells, context, nbits, bonds_ = (6, 2, 24, (8, 16, 32)) if smoke else (12, 4, 120, (32, 64, 90, 128))
    cfg = Config(cells, context, theta=0.3)
    bits = bytes_to_bits(CHAIN_TEXT)[:nbits]
    taus = np.random.default_rng(7).integers(0, 2, nbits)
    engines = {str(D): GovernedEngine(cfg, max_bond=D, always=True) for D in bonds_}
    engines |= {f"entropy_margin_{m:g}": GovernedEngine(cfg, always=True, bond="entropy", margin=m) for m in (2, 4, 8)}
    runs = {}
    for name, e in engines.items():
        runs[name] = chain_run(e, bits, taus)
        print("  fidelity", name, runs[name]["seconds"], "s", flush=True)
    ref = np.array(runs[str(bonds_[-1])]["q"])
    names = list(runs)
    return {"config": {"cells": cells, "context": context, "theta": 0.3, "bits": nbits},
            "reference": f"governed, bond {bonds_[-1]}",
            "max_abs_q_difference_vs_reference": {n: float(np.abs(np.array(runs[n]["q"]) - ref).max())
                                                  for n in names if n != str(bonds_[-1])},
            "discarded_per_event": {n: runs[n]["discarded"] / nbits for n in names},
            "largest_bond": {n: max(runs[n]["bonds"]) for n in names},
            "mean_bond": {n: float(np.mean(runs[n]["bonds"])) for n in names},
            "renyi2_per_qubit_max": {n: max(runs[n]["renyi2_per_qubit"]) for n in names},
            "bits_per_bit": {n: float(-np.log2(runs[n]["q"]).mean()) for n in names},
            "seconds_by_engine": {n: runs[n]["seconds"] for n in names},
            "ungoverned_reference": "experiments/diagnostics/context_convergence.txt: 5.6e-4 at bond 64 (93 bits)"}


def part1_area_law(smoke: bool) -> dict:
    """N-L5 area law: the largest operator entanglement of any cut after 40 bits, against chain length, for the
    plain engine (bond 64) and the governed engine (cache bond)."""
    sizes = (3, 4, 5) if smoke else (4, 6, 8, 10)
    bits = bytes_to_bits(CHAIN_TEXT)[:40]
    taus = np.random.default_rng(7).integers(0, 2, 40)
    rows = []
    for cells in sizes:
        cfg = Config(cells, 2, theta=0.5)
        for name, e in (("plain_bond64", OperatorEngine(cfg, max_bond=64)),
                        ("governed_cache_bond", GovernedEngine(cfg, always=True))):
            run = chain_run(e, bits, taus)
            ent = operator_entropies(e)
            rows.append({"cells": cells, "engine": name, "max_cut_entropy_bits": max(ent),
                         "middle_cut_entropy_bits": ent[len(ent) // 2], "largest_bond": max(run["bonds"]),
                         "discarded": run["discarded"], "seconds": run["seconds"]})
            print("  area law", rows[-1], flush=True)
    return {"rows": rows}


#: exact context einselection with two cells or fewer outside the context (the sinks change nothing: no
#: coherence can return to a population), three or more, and finite dephasing
SINK_GRID = [Config(3, 2, theta=0.5), Config(3, 1, theta=1.1), Config(4, 2, theta=0.3), Config(4, 1, theta=0.7),
             Config(5, 2, theta=0.5), Config(3, 1, theta=1.0, dephasing=0.7), Config(4, 3, theta=0.8, dephasing=2.0)]


def part1_sink_cost(A_train: bytes, smoke: bool) -> dict:
    """What the sinks change: the exact (untruncated) machine with and without them on the same English
    stream and the same emissions; prequential bits per bit and the largest change of a prediction."""
    bits = bytes_to_bits(A_train[: 16 if smoke else 256])
    taus = np.random.default_rng(SEED).integers(0, 2, len(bits))
    rows = []
    for cfg in SINK_GRID[:2] if smoke else SINK_GRID:
        t0 = time.perf_counter()
        plain = OperatorEngine(cfg, max_bond=1 << 12, cutoff=0.0)
        sinks = GovernedEngine(cfg, max_bond=1 << 12, cutoff=0.0, always=True)
        qp, qs, dw = [], [], 0.0
        for b, t in zip(bits, taus, strict=True):
            qp.append(plain.force(int(b), int(t)))
            qs.append(sinks.force(int(b), int(t)))
            dw = max(dw, float(np.abs(plain.w - sinks.w).max()))
        qp, qs = np.array(qp), np.array(qs)
        rows.append({"cells": cfg.cells, "context": cfg.context, "theta": cfg.theta, "dephasing": cfg.dephasing,
                     "bits": len(bits), "plain_bits_per_bit": float(-np.log2(qp).mean()),
                     "sinks_bits_per_bit": float(-np.log2(qs).mean()),
                     "max_abs_q_change": float(np.abs(qp - qs).max()), "max_abs_w_change": dw,
                     "discarded": plain.discarded + sinks.discarded,
                     "seconds": round(time.perf_counter() - t0, 1)})
        print("  sink cost", rows[-1], flush=True)
    return {"rows": rows}


def part1_text_survival(smoke: bool) -> dict:
    """A sentence learned in a 512-slot table, then 60,000 random bits: deepest-level sectors kept."""
    text = b"the quick brown fox jumps over the lazy dog. "
    noise = np.random.default_rng(SEED).integers(0, 2, 60_000).astype(np.uint8)
    ctxs, reg = set(), 0
    for b in bytes_to_bits(text * 3):
        reg = ((reg << 1) | int(b)) & ((1 << 64) - 1)
        ctxs.add(reg)
    out = {}
    for name, kw in {"plain": {}, "verify_2": {"verify": 2}, "governor": {"governor": True}}.items():
        nm = NestedMachine(NestedConfig(LAW, depths=(8, 40, 64), table_bits=9, **kw))
        nm.run(bytes_to_bits(text * 100))
        before = float(np.mean([nm.stored(c, 2) for c in ctxs]))
        nm.run(noise)
        out[name] = {"before": before, "after": float(np.mean([nm.stored(c, 2) for c in ctxs])),
                     "refused": nm.refused, "protected": nm.protected, "evictions": nm.evictions}
    crit = out["plain"]["after"] < 0.02 and all(out[k]["after"] >= 0.97 * out[k]["before"]
                                                 for k in ("verify_2", "governor"))
    return {"rows": out, "criteria": {"survival": bool(crit)}}


def part1_l7(A_train: bytes, A_test: bytes, smoke: bool) -> dict:
    """``hbar = 0`` against the ``select="max"`` planner, and ``Z`` at ``hbar = 1`` against the prediction."""
    m = make(START)
    read(m, A_train[: (1 << 16) if smoke else (1 << 20)], learning=True)
    n_bits = 8 * (4 if smoke else 48)
    same, worst_z = True, 0.0
    for prompt in prompts(A_test)[: 2 if smoke else 8]:
        a = m.copy()
        a.run(bytes_to_bits(prompt), learning=False)
        b = a.copy()
        same &= collapse.generate(a, n_bits, horizon=8, hbar=0.0) == b.generate(n_bits, horizon=8, select="max")
        ro = collapse.Readout(b)
        state = int(b.reg)
        for _ in range(64):
            z = collapse.first_step(ro, state, 8, 1.0)["Z"][0]
            worst_z = max(worst_z, abs(z - ro.p0(state)))
            state = ro.advance(state, int(z < 0.5))
    return {"hbar0_equals_max_planner": bool(same), "hbar1_Z_minus_prediction": worst_z,
            "criteria": {"hbar0": bool(same), "hbar1": worst_z < 1e-10}}


# ---------------------------------------------------------------------------------------------------
# Part 2: configuration on A-validation, one stage per layer
# ---------------------------------------------------------------------------------------------------

def options_on(spec: dict) -> int:
    """How many options of N a spec turns on (the tie rule's measure of simplicity)."""
    return (int("depths" in spec and tuple(spec["depths"]) != NESTED["depths"]) + int(spec.get("coherence", 0) > 0)
            + int(spec.get("verify", 1) > 1) + int(bool(spec.get("governor", False))))


def key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True)


def e2_start_bpb() -> float:
    """E2's A-validation bpb of the start configuration, as saved by E2."""
    e2 = json.loads((RESULTS / "e2.json").read_text())
    return next(r["A_validation_bpb"] for r in e2["selection"]
                if all(r.get(k) == v for k, v in START.items()))


def part2(S: dict, turnaround: int, partial: Partial, smoke: bool) -> dict:
    done = {r["key"]: r for r in partial.get("part2_rows") or []}

    def score(spec: dict) -> float:
        k = key(spec)
        if k not in done:
            t0 = time.perf_counter()
            m = make(spec)
            train = read(m, S["A_train"], learning=True)
            val = frozen(m, S["A_validation"])
            occ = m.occupancy()
            done[k] = {"key": k, "spec": spec, "A_train_bpb": train, "A_validation_bpb": val,
                       "evictions": m.evictions, "refused": m.refused, "protected": m.protected,
                       "occupancy": {str(lv): v for lv, v in occ.items()},
                       "seconds": round(time.perf_counter() - t0, 1)}
            partial.put("part2_rows", list(done.values()))
            print("  select", {x: done[k][x] for x in ("spec", "A_validation_bpb", "seconds")}, flush=True)
        return done[k]["A_validation_bpb"]

    def choose(cands: list[dict]) -> dict:
        scored = [(score(c), c) for c in cands]
        best = min(s for s, _ in scored)
        near = [(options_on(c), s, c) for s, c in scored if s <= best + TIE]
        return min(near, key=lambda t: (t[0], t[1]))[2]

    stages = []
    current = dict(START)
    start_bpb, e2_bpb = score(current), e2_start_bpb()
    reproduced = None if smoke else abs(start_bpb - e2_bpb) < 1e-9     # the smoke streams are not E2's
    if reproduced is False:
        raise RuntimeError(f"E2's start configuration gives {start_bpb}, not {e2_bpb}: E3 stops here")

    # 1. N-L2 depths, cut at the turnaround
    cut = tuple(d for d in NESTED["depths"] if d <= 8 * turnaround)
    cands = [current] + ([{**current, "depths": list(cut)}] if turnaround < 8 and cut != NESTED["depths"] else [])
    current = choose(cands)
    stages.append({"stage": "depths", "candidates": cands, "chosen": current})
    # 2. N-L3 coherence
    cs = (0.0, 0.5) if smoke else (0.0, 0.25, 0.5, 0.75, 1.0)
    cands = [current if c == 0 else {**current, "coherence": c} for c in cs]
    current = choose(cands)
    stages.append({"stage": "coherence", "candidates": cands, "chosen": current})
    # 3. N-L5 verify
    vs = (1, 2) if smoke else (1, 2, 3)
    cands = [current if v == 1 else {**current, "verify": v} for v in vs]
    current = choose(cands)
    stages.append({"stage": "verify", "candidates": cands, "chosen": current})
    # 4. N-L6 governor
    protects, sinks = ((1.0,), (1.0,)) if smoke else ((0.5, 1.0, 2.0), (0.1, 0.25, 0.5, 1.0))
    base = {k: v for k, v in current.items() if k != "verify"}
    cands, seen = [current], {key(current)}
    for p, r, v in itertools.product(protects, sinks, sorted({1, current.get("verify", 1)})):
        c = {**base, "governor": True, "protect": p, "sink_rate": r, **({"verify": v} if v > 1 else {})}
        if key(c) not in seen:
            cands.append(c)
            seen.add(key(c))
    current = choose(cands)
    stages.append({"stage": "governor", "candidates": cands, "chosen": current})
    return {"start_A_validation_bpb": start_bpb, "e2_start_bpb": e2_bpb, "reproduced": reproduced,
            "stages": stages, "chosen": current, "rows": list(done.values()), "commit": commit(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


# ---------------------------------------------------------------------------------------------------
# Parts 3 and 4: the arms, and writing with the path integral
# ---------------------------------------------------------------------------------------------------

def counters(m) -> dict:
    if not isinstance(m, NestedMachine):
        return {}
    return {"evictions": m.evictions, "refused": m.refused, "protected": m.protected,
            "occupancy": {str(lv): v for lv, v in m.occupancy().items()}}


def utf8_valid(data: bytes) -> bool:
    """Valid UTF-8, allowing a character cut at the end of the generated span."""
    try:
        codecs.getincrementaldecoder("utf-8")().decode(data, final=False)
        return True
    except UnicodeDecodeError:
        return False


def writing_grid(machine, texts: list[bytes], vocabulary: set[bytes], n_bytes: int, smoke: bool) -> dict:
    hbars = (0.0, 1.0) if smoke else (0.0, 0.25, 0.5, 1.0)
    settings = []
    for hbar, form, mode, verified in itertools.product(hbars, ("euclidean", "feynman"), ("collapse", "sample"),
                                                        (False, True)):
        t0 = time.perf_counter()
        outs, rep, words, found, printable, total, valid, own_bits = [], [], 0, 0, 0, 0, 0, 0.0
        for i, prompt in enumerate(texts):
            m = machine.copy()
            m.run(bytes_to_bits(prompt), learning=False)
            after_prompt = m.copy()
            rng = np.random.default_rng(SEED + i) if mode == "sample" else None
            bits = collapse.generate(m, 8 * n_bytes, horizon=8, hbar=hbar, mode=mode, form=form,
                                     verified=verified, rng=rng)
            data = bits_to_bytes(np.array(bits, dtype=np.uint8))
            outs.append({"prompt": prompt.decode("utf-8", "backslashreplace"),
                         "text": data.decode("utf-8", "backslashreplace")})
            rep.append(collapse.repetition(data))
            tokens = WORD.findall(data)
            words += len(tokens)
            found += sum(t.lower() in vocabulary for t in tokens)
            printable += sum(32 <= c < 127 or c in (9, 10, 13) for c in data)
            total += len(data)
            valid += utf8_valid(data)
            own_bits += float(-np.log2(after_prompt.run(np.array(bits, dtype=np.uint8), learning=False)).sum())
        settings.append({"hbar": hbar, "form": form, "mode": mode, "verified": verified,
                         "repeated_8byte_windows": float(np.mean(rep)),
                         "words_in_A_train_vocabulary": found / words if words else None, "words": words,
                         "printable_bytes": printable / total, "valid_utf8": valid / len(texts),
                         "own_bits_per_byte": own_bits / total, "outputs": outs,
                         "seconds": round(time.perf_counter() - t0, 1)})
        print("  write", {k: settings[-1][k] for k in ("hbar", "form", "mode", "verified", "repeated_8byte_windows",
                                                        "own_bits_per_byte", "seconds")}, flush=True)
    return {"bytes": n_bytes, "prompts": len(texts), "settings": settings}


# ---------------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--smoke", action="store_true", help="first 256 KiB of each stream, small chains and grids")
    ap.add_argument("--resume", action="store_true", help="reuse the parts saved in the partial file")
    args = ap.parse_args()
    t_start = time.perf_counter()
    name = "e3_smoke" if args.smoke else "e3"
    partial = Partial(name, args.resume)
    S = streams(args.manifest, args.smoke)
    sizes = {k: len(v) for k, v in S.items()}
    print("stream bytes:", sizes, flush=True)

    def part(key_: str, fn, *a):
        got = partial.get(key_)
        if got is None:
            print(key_, flush=True)
            partial.put(key_, {**timed(fn, *a), "commit": commit(),
                               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            got = partial.get(key_)
        return got

    p1 = {
        "N_L1": part("N_L1", part1_l1, args.smoke),
        "N_L2": part("N_L2", part1_l2, S["A_train"], args.smoke),
        "N_L3": part("N_L3", part1_l3),
        "N_L4_L5_L6_chain_blankets": part("chain_blankets", part1_chain_blankets, args.smoke),
        "N_L5_L6_chain_fidelity": part("chain_fidelity", part1_chain_fidelity, args.smoke),
        "N_L5_area_law": part("area_law", part1_area_law, args.smoke),
        "N_L6_chain_sink_cost": part("sink_cost", part1_sink_cost, S["A_train"], args.smoke),
        "N_L5_L6_text_survival": part("text_survival", part1_text_survival, args.smoke),
        "N_L7": part("N_L7", part1_l7, S["A_train"], S["A_test"], args.smoke),
    }
    if partial.get("N_L2_first_measurement") is not None:              # kept and reported (protocol amendment)
        p1["N_L2_first_measurement"] = partial.get("N_L2_first_measurement")
    p2 = partial.get("part2")
    if p2 is None:
        print("part 2", flush=True)
        p2 = part2(S, p1["N_L2"]["turnaround_bytes"], partial, args.smoke)
        partial.put("part2", p2)
        p2 = partial.get("part2")
    spec = p2["chosen"]
    p3 = partial.get("part3")
    if p3 is None:
        print("part 3 and 4 with", spec, flush=True)
        vocabulary = {w.lower() for w in WORD.findall(S["A_train"])}
        n_bytes = 8 if args.smoke else 48
        texts = prompts(S["A_test"])[: 2 if args.smoke else 8]
        t3 = time.perf_counter()
        p3 = arms(spec, S, counters=counters,
                  on_end_of_a=lambda m: writing_grid(m, texts, vocabulary, n_bytes, args.smoke))
        p3 |= {"seconds": round(time.perf_counter() - t3, 1), "commit": commit()}
        partial.put("part3", p3)
        p3 = dict(partial.get("part3"))
    p4 = p3.pop("end_of_a", None) or partial.get("part4")
    partial.put("part4", p4)
    payload = {
        "experiment": "E3 nested fractal architecture",
        "protocol": "experiments/PROTOCOL.md#e3-the-nested-fractal-architecture-layer-by-layer",
        "provenance": provenance(seed=SEED, manifest=str(args.manifest), smoke=args.smoke,
                                 manifest_sha256=sha256(args.manifest.read_bytes()), resumed=args.resume),
        "stream_bytes": sizes,
        "part1": p1, "part2": p2, "part3": p3, "part4": p4,
        "seconds_this_run": round(time.perf_counter() - t_start, 1),
        "seconds": round(sum(v.get("seconds", 0.0) for v in p1.values()) + sum(
            r["seconds"] for r in p2["rows"]) + (p3.get("seconds") or 0.0), 1),
    }
    print(save(name, plain(payload)))


if __name__ == "__main__":
    main()
