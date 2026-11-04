"""Experiment E5: does the taught machine use what it was taught? (experiments/PROTOCOL.md, E5)

    python experiments/e5_taught.py --data DIR                 # both arms, T then C (about 45 minutes each)
    python experiments/e5_taught.py --data DIR --smoke         # code check: 256 KiB per stage, whole question set
    python experiments/e5_taught.py --data DIR --resume        # skip the arms saved in results/e5_partial.json
    python experiments/e5_taught.py --data DIR --arm T         # one arm

``DIR`` holds the stage files named in ``data/e5/sources.json`` (build it with ``experiments/e5_restore.py``).
Units, saved as they end: arm T (stages 1 to 5), arm C (stages 1 and 2, then prose in the place of 3 to 5). Each
stage's record is also saved as it ends, but an interrupted arm is read again from its start (no machine state is
saved). Run one job at a time. Results go to results/e5.json (e5_smoke.json for --smoke); the criteria P1 to P3 are
computed by experiments/report.py from that file, never here.

The machine is E4's chosen configuration (results/e4.json part2.chosen), unchanged. Nothing here adds a score, a path
cost, a beam, a softmax or a threshold: the costs are measured outside the machine from the probabilities it gave.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import e3_nfa as E
import numpy as np
from _common import RESULTS, ROOT, provenance, save
from e2_curriculum import make, read

from vam.bits import bytes_to_bits

DATA = ROOT / "experiments" / "data" / "e5"
#: the SHA-256 of data/e5/questions.json, fixed in PROTOCOL E5; the run stops if the file differs
QUESTIONS_SHA = "2706322fb9c371ff0c12cfc302bb5f8a193bf06bb650c091076e9467892d5e62"
#: E4's chosen configuration, checked against results/e4.json part2.chosen at the start
CHOSEN = {"machine": "hier", "law": "counting", "prior": 4.0, "coherence": 1.0, "governor": True, "protect": 0.5,
          "sink_rate": 0.0, "verify": 3, "hier_levels": "WS", "hier_segment": "text", "hier_verify": 4}
#: the neutral preamble read before every question and every stem-free option, identical in both arms (PROTOCOL E5.6)
PRE = b"\n\nQuestion and answer.\n\n"
STAGES = ("S1", "S2", "S3", "S4", "S5")
SMOKE_CAP = 1 << 18            # bytes per stage (and per held-out part) in a smoke run
GATE_ITEMS = 2                 # questions on which the snapshot is compared with a full copy
ARRAYS = ("_wn", "_tags", "_door", "_occ")    # what HierMachine.copy() duplicates, besides the small state


class GateFailure(RuntimeError):
    """A pre-run gate failed: nothing is scored."""


# ---------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def check_questions(path: Path = DATA / "questions.json", expected: str = QUESTIONS_SHA) -> str:
    """Stop (``SystemExit``) unless the question file hashes to the fixed value."""
    got = sha256_file(path)
    if got != expected:
        raise SystemExit(f"E5 stopped: {path} has SHA-256 {got}, PROTOCOL E5 fixes {expected}")
    return got


def check_spec(e4_path: Path = RESULTS / "e4.json") -> dict:
    chosen = json.loads(e4_path.read_text())["part2"]["chosen"]
    if chosen != CHOSEN:
        raise SystemExit("E5 stopped: results/e4.json part2.chosen differs from the configuration fixed in "
                         f"PROTOCOL E5: {chosen}")
    return dict(CHOSEN)


def verify_files(data: Path, sources: dict) -> int:
    """Every file named in ``sources`` exists in ``data`` with its recorded SHA-256 (the stage texts are not in git)."""
    n = 0
    for group in ("T", "C", "heldout"):
        for files in sources[group].values():
            for rec in files:
                p = data / rec["file"]
                if not p.exists() or sha256_file(p) != rec["sha256"]:
                    raise SystemExit(f"E5 stopped: {p} is missing or differs from sources.json (run e5_restore.py)")
                n += 1
    return n


def join(data: Path, files: list[dict], cap: int | None = None) -> bytes:
    out = b"".join((data / rec["file"]).read_bytes() for rec in files)
    return out[:cap] if cap else out


def build_streams(data: Path, sources: dict, smoke: bool) -> dict:
    """The stage streams of both arms and the held-out parts. In arm C, stages 3 to 5 are read as three segments of
    prose cut at the byte lengths of T's stages 3 to 5, so that both arms pause at the same byte offsets."""
    cap = SMOKE_CAP if smoke else None
    T = {s: join(data, sources["T"][s], cap) for s in STAGES}
    HO = {s: join(data, sources["heldout"][s], cap) for s in STAGES}
    extra = join(data, sources["C"]["prose_extra"])
    cuts, pos = [], 0
    for s in ("S3", "S4", "S5"):
        full = sum(rec["bytes"] for rec in sources["T"][s])
        cuts.append(extra[pos:pos + full][:cap] if cap else extra[pos:pos + full])
        pos += full
    if not smoke and pos != len(extra):
        raise SystemExit(f"E5 stopped: C extra prose has {len(extra)} bytes, T's stages 3 to 5 have {pos}")
    C = {"S1": T["S1"], "S2": T["S2"], "C3": cuts[0], "C4": cuts[1], "C5": cuts[2]}
    return {"T": T, "C": C, "heldout": HO}


# ---------------------------------------------------------------------------------------------------
# the machine, outside measurements
# ---------------------------------------------------------------------------------------------------

def snapshot(m) -> tuple:
    """The register and ring state, and the counters; the tables are not copied (frozen reads write none)."""
    return m.state, m._stats.copy()


def restore(m, snap: tuple) -> None:
    m.set_state(list(snap[0]))
    m._stats[:] = snap[1]


def array_hashes(m) -> dict:
    return {name: hashlib.sha256(np.ascontiguousarray(getattr(m, name)).tobytes()).hexdigest() for name in ARRAYS}


def counters(m) -> dict:
    return {"evictions": m.evictions, "refused": m.refused, "protected": m.protected,
            "occupancy": {str(lv): v for lv, v in m.occupancy().items()}, "global_occupancy": m.global_occupancy()}


def frozen_bpb(m, data: bytes) -> float:
    """Bits per byte of ``data`` read with learning off, from the present state; the state is restored afterwards."""
    snap = snapshot(m)
    out = read(m, data, learning=False)
    restore(m, snap)
    return out


def costs_after(m, prefix: bytes, options: list[bytes]) -> list[float]:
    """Bits to read each option after ``prefix``, learning off. The prefix is read once; every option starts from the
    state after it (a frozen read changes no table, so this equals reading prefix and option in sequence)."""
    start = snapshot(m)
    m.run(bytes_to_bits(prefix), learning=False)
    mid = snapshot(m)
    out = []
    for o in options:
        restore(m, mid)
        out.append(float(-np.log2(m.run(bytes_to_bits(o), learning=False)).sum()))
    restore(m, start)
    return out


def score_questions(m, questions: list[dict]) -> dict:
    """Costs of the four options of every question with its stem and without it (PROTOCOL E5.6 steps 1 to 5)."""
    out = {}
    for q in questions:
        opts = [o.encode("utf-8") for o in q["options"]]
        out[q["id"]] = {"with_stem": costs_after(m, PRE + q["stem"].encode("utf-8"), opts),
                        "no_stem": costs_after(m, PRE, opts)}
    return out


def gate_snapshot_equals_copy(m, questions: list[dict], n_items: int = GATE_ITEMS) -> dict:
    """Pre-run gate (a): on a few questions, the cost of each option through the snapshot equals, bit for bit, its cost
    in a full sequential read on a ``m.copy()``."""
    same = True
    for q in questions[:n_items]:
        stem = PRE + q["stem"].encode("utf-8")
        opts = [o.encode("utf-8") for o in q["options"]]
        fast = costs_after(m, stem, opts)
        for o, f in zip(opts, fast, strict=True):
            c = m.copy()
            p = c.run(bytes_to_bits(stem + o), learning=False)
            slow = float(-np.log2(p[8 * len(stem):]).sum())
            same &= slow == f
            del c
        gc.collect()
    return {"items": min(n_items, len(questions)), "bit_identical": bool(same)}


def run_gates(m, questions: list[dict]) -> tuple[dict, dict]:
    """Both pre-run gates on the machine about to be scored, and the scoring itself between the two table hashes.

    Gate (a): snapshot costs equal full-copy costs, bit for bit. Gate (b): the hashes of the four arrays are equal
    before the first and after the last frozen read. Raises ``GateFailure`` if either fails (nothing is kept);
    otherwise returns the gate record and the costs of every question.
    """
    before = array_hashes(m)
    a = gate_snapshot_equals_copy(m, questions)
    scored = score_questions(m, questions)
    after = array_hashes(m)
    out = {"snapshot_equals_copy": a, "tables_unchanged": before == after, "hashes_before": before,
           "hashes_after": after}
    if not (a["bit_identical"] and out["tables_unchanged"]):
        raise GateFailure(json.dumps(out))
    return out, scored


# ---------------------------------------------------------------------------------------------------
# an arm
# ---------------------------------------------------------------------------------------------------

def run_arm(arm: str, spec: dict, S: dict, questions: list[dict], partial: E.Partial | None = None,
            machine=make) -> dict:
    """One arm: its stages in order with no reset, then the held-out parts and the questions at the end.

    After every stage: the capacity counters, and frozen bits per byte on the held-out part of every stage up to and
    including the one just read (arm C: the same held-out texts, paused at the byte offsets where T's stages end).
    """
    stream, heldout = S[arm], S["heldout"]
    names = list(stream)
    m = machine(spec)
    out = {"stages": {}, "order": names}
    for k, name in enumerate(names):
        t0 = time.perf_counter()
        curve: list[float] = []
        bpb = read(m, stream[name], learning=True, curve=curve)
        rec = {"trained_bytes": len(stream[name]), "train_bpb": bpb, "curve_bpb_per_MiB": curve,
               "counters": counters(m)}
        rec["heldout_bpb"] = {s: frozen_bpb(m, heldout[s]) for s in STAGES[:k + 1] if len(heldout[s])}
        rec["seconds"] = round(time.perf_counter() - t0, 1)
        out["stages"][name] = rec
        if partial is not None:
            partial.put(f"{arm}_stage_{name}", rec)
        print(arm, name, f"train {bpb:.4f} bpb", f"heldout {rec['heldout_bpb']}", flush=True)
    out["gates"], out["questions"] = run_gates(m, questions)
    out["end_counters"] = counters(m)
    del m
    gc.collect()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, help="directory with the stage files of sources.json")
    ap.add_argument("--sources", type=Path, default=DATA / "sources.json")
    ap.add_argument("--questions", type=Path, default=DATA / "questions.json")
    ap.add_argument("--smoke", action="store_true", help="256 KiB per stage and held-out part, the whole question set")
    ap.add_argument("--resume", action="store_true", help="reuse the arms saved in the partial file")
    ap.add_argument("--arm", choices=("T", "C"), help="run one arm only")
    args = ap.parse_args()
    t_start = time.perf_counter()
    q_sha = check_questions(args.questions)
    spec = check_spec()
    sources = json.loads(args.sources.read_text())
    n_files = verify_files(args.data, sources)
    questions = json.loads(args.questions.read_text())["questions"]
    name = "e5_smoke" if args.smoke else "e5"
    partial = E.Partial(name, args.resume)
    S = build_streams(args.data, sources, args.smoke)
    print("checked: questions", q_sha[:12], "files", n_files, "stage bytes",
          {k: {s: len(v) for s, v in d.items()} for k, d in S.items()}, flush=True)
    arms = {}
    for arm in ("T", "C") if args.arm is None else (args.arm,):
        got = partial.get(arm)
        if got is None:
            print("arm", arm, flush=True)
            partial.put(arm, E.timed(run_arm, arm, spec, S, questions, partial))
            got = partial.get(arm)
        arms[arm] = got
    payload = {
        "experiment": "E5 does the taught machine use what it was taught?",
        "protocol": "experiments/PROTOCOL.md#e5-does-the-taught-machine-use-what-it-was-taught",
        "provenance": provenance(seed=20261020, smoke=args.smoke, resumed=args.resume, questions_sha256=q_sha,
                                 sources_sha256=sha256_file(args.sources), preamble=PRE.decode()),
        "spec": spec, "stage_bytes": {k: {s: len(v) for s, v in d.items()} for k, d in S.items()},
        "arms": arms, "seconds": round(time.perf_counter() - t_start, 1),
    }
    print(save(name, E.plain(payload)))


if __name__ == "__main__":
    main()
