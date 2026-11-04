"""Experiment E2: textbooks first, then unstructured text (experiments/PROTOCOL.md, E2).

    python experiments/e2_curriculum.py --manifest PATH/MANIFEST.json
    python experiments/e2_curriculum.py --manifest ... --smoke     # code check: tiny budget, no lzma

Every loss is prequential bits per byte (bpb). "Frozen" evaluation reads a
copy of the machine with learning off, so the evaluated text never changes the
weights. Configurations are chosen on A-validation only. The four arms run for
the best configuration under the literal T12.1 law (the specification's
machine) and, when it differs, for the best configuration overall.
"""

from __future__ import annotations

import argparse
import lzma
import time
from pathlib import Path

import numpy as np
from _common import provenance, save

from vam.baselines import contexts
from vam.bitkernel import BitMachine
from vam.bits import bits_to_bytes, bytes_to_bits
from vam.corpus import Corpus, sha256, shuffle_sentences, shuffle_words
from vam.nested import NestedConfig, NestedMachine
from vam.realizations.vam1 import Config

SEED = 20261018
KS = (16, 24)
# the literal T12.1 law (constant friction) at three rates kappa dt / gamma, and the count-dependent friction (D10)
LAWS = {"literal_0.02": {"consolidating": False, "dt": 0.02}, "literal_0.05": {"consolidating": False, "dt": 0.05},
        "literal_0.2": {"consolidating": False, "dt": 0.2}, "counting": {"consolidating": True, "dt": 1.0}}
NESTED = {"depths": (8, 16, 24, 32, 48, 64), "table_bits": 22, "priors": (0.0, 4.0)}
VISIT_K = 24                        # context length for the visitation statistics of T12.3 part 1


def literal(spec: dict) -> bool:
    return not LAWS[spec["law"]]["consolidating"]


def grid(smoke: bool) -> list[dict]:
    """Every configuration tried on A-validation: fixed windows and nested levels, each law. The prior only
    acts through the activity count, which the literal law never reads, so it varies only with counting."""
    laws = ("literal_0.05", "counting") if smoke else tuple(LAWS)
    specs = [{"machine": "window", "k": k, "law": law} for k in (KS[:1] if smoke else KS) for law in laws]
    for law in laws:
        priors = NESTED["priors"] if LAWS[law]["consolidating"] and not smoke else NESTED["priors"][:1]
        specs += [{"machine": "nested", "law": law, "prior": prior} for prior in priors]
    return specs


#: options of the nested fractal architecture (E3); absent from every E2 spec, so E2 is unchanged
NFA_KEYS = ("depths", "coherence", "verify", "governor", "protect", "sink_rate")


def make(spec: dict):
    if spec["machine"] == "hier":                  # E4: the nested machine with levels above one equation
        from vam.nfa.hier import build
        return build(spec, Config(cells=1, context=1, **LAWS[spec["law"]]))
    if spec["machine"] == "window":
        return BitMachine(Config(cells=spec["k"], context=spec["k"], **LAWS[spec["law"]]))
    law = Config(cells=1, context=1, **LAWS[spec["law"]])
    extra = {k: spec[k] for k in NFA_KEYS if k in spec}
    depths = tuple(extra.pop("depths", NESTED["depths"]))
    return NestedMachine(NestedConfig(law, depths=depths, table_bits=NESTED["table_bits"], prior=spec["prior"],
                                      **extra))


BLOCK = 1 << 20                     # bytes per learning-curve block (1 MiB)
FAMILIAR = 8                        # a sector visited at least this often counts as familiar (T12.3 part 1)
PROMPTS, PROMPT_BYTES, GEN_BYTES = 8, 64, 48


def read(bm: BitMachine, data: bytes, learning: bool, curve: list | None = None,
         drift: list | None = None) -> float:
    """Read ``data`` in 1 MiB blocks; returns bpb. Optionally records bpb and weight drift per block."""
    total = 0.0
    for start in range(0, len(data), BLOCK):
        chunk = data[start:start + BLOCK]
        before = bm.w.copy() if drift is not None else None
        bits = float(-np.log2(bm.run(bytes_to_bits(chunk), learning=learning)).sum())
        total += bits
        if curve is not None:
            curve.append(bits / len(chunk))
        if drift is not None:
            drift.append(float(np.linalg.norm(bm.w - before)))
    return total / max(1, len(data))


def frozen(bm: BitMachine, data: bytes) -> float:
    return read(bm.copy(), data, learning=False)


def visitation(data: bytes, k: int = VISIT_K) -> dict:
    """How the stream revisits its ``k``-bit contexts (T12.3 part 1); a property of the data as read."""
    counts = np.zeros(1 << k, dtype=np.int64)
    carry = np.zeros(0, dtype=np.uint8)
    for start in range(0, len(data), BLOCK):
        bits = np.concatenate([carry, bytes_to_bits(data[start:start + BLOCK])])
        ctx = contexts(bits, k)[len(carry):]
        if start == 0:
            ctx = ctx[k:]                                           # skip contexts that reach before the start
        counts += np.bincount(ctx, minlength=1 << k)
        carry = bits[-k:]
    n = counts[counts > 0].astype(float)
    total = n.sum()
    p = n / total
    return {"k": k, "visits": total, "contexts_visited": len(n), "entropy_bits": float(-(p * np.log2(p)).sum()),
            "share_of_visits_to_familiar_contexts": float(np.maximum(n - FAMILIAR, 0).sum() / total)}


def lzma_conditional_bpb(context: bytes, target: bytes) -> float:
    """Reference only: extra lzma output for ``target`` after ``context`` (64 MiB dictionary), in bpb."""
    filters = [{"id": lzma.FILTER_LZMA2, "preset": 6, "dict_size": 1 << 26}]
    size = lambda data: len(lzma.compress(data, format=lzma.FORMAT_RAW, filters=filters))
    return 8.0 * (size(context + target) - size(context)) / len(target)


def prompts(text: bytes) -> list[bytes]:
    """``PROMPTS`` prompts at evenly spaced line starts, cut back to a word end."""
    out = []
    for i in range(PROMPTS):
        start = text.find(b"\n", i * len(text) // PROMPTS) + 1
        piece = text[start:start + PROMPT_BYTES]
        cut = piece.rfind(b" ")
        out.append(piece[:cut] if cut > 0 else piece)
    return out


def generations(bm: BitMachine, texts: list[bytes]) -> list[dict]:
    rows = []
    for i, prompt in enumerate(texts):
        row = {"prompt": prompt.decode("utf-8", "replace")}
        for select in ("max", "sample"):
            m = bm.copy()
            m.run(bytes_to_bits(prompt), learning=False)
            rng = np.random.default_rng(SEED + i) if select == "sample" else None
            bits = m.generate(8 * GEN_BYTES, horizon=8, select=select, rng=rng)
            row[select] = bits_to_bytes(np.array(bits, dtype=np.uint8)).decode("utf-8", "replace")
            assert np.array_equal(m.w, bm.w)                          # generation never learns
        rows.append(row)
    return rows


def arms(spec: dict, S: dict, counters=None, on_end_of_a=None) -> dict:
    """The four arms for one configuration, with transfer, T12.3 parts 2-3, curves and generation.

    E3 hooks (unused by E2): ``counters(machine) -> dict`` is recorded for each arm when its reading ends,
    and ``on_end_of_a(machine)`` receives the end-of-A machine; its return value is kept as ``"end_of_a"``.
    """
    evals = ("A_test", "A_heldout", "B_test", "B_heldout")
    out = {}
    count = (lambda m: {}) if counters is None else (lambda m: {"counters": counters(m)})
    bm = make(spec)
    curve_a, drift_a = [], []
    train_a = read(bm, S["A_train"], learning=True, curve=curve_a, drift=drift_a)
    end_of_a = bm.copy()
    out["A_only"] = {"train_bpb": train_a, **{e: frozen(end_of_a, S[e]) for e in evals}, **count(end_of_a)}
    t123 = {
        "part2_weight_drift_per_MiB": drift_a,
        "part3": {name: frozen(end_of_a, S[name]) for name in
                  ("A_test", "A_test_word_shuffled", "A_heldout", "A_heldout_word_shuffled")},
    }
    curve_b = []
    train_b = read(bm, S["B_train"], learning=True, curve=curve_b)
    out["A_then_B"] = {"train_bpb_B": train_b, **{e: frozen(bm, S[e]) for e in evals}, **count(bm)}
    curves = {"A_then_B": curve_a + curve_b}
    gens = generations(end_of_a, prompts(S["A_test"]))
    hooked = {} if on_end_of_a is None else {"end_of_a": on_end_of_a(end_of_a)}
    del bm, end_of_a

    bm = make(spec)
    curves["B_only"] = []
    train = read(bm, S["B_train"], learning=True, curve=curves["B_only"])
    out["B_only"] = {"train_bpb_B": train, **{e: frozen(bm, S[e]) for e in evals}, **count(bm)}
    del bm

    bm = make(spec)
    curves["shuffled_A_then_B"] = []
    read(bm, S["A_train_sentence_shuffled"], learning=True, curve=curves["shuffled_A_then_B"])
    out["shuffled_A_then_B"] = {"after_A": {e: frozen(bm, S[e]) for e in evals}}
    read(bm, S["B_train"], learning=True, curve=curves["shuffled_A_then_B"])
    out["shuffled_A_then_B"].update({e: frozen(bm, S[e]) for e in evals}, **count(bm))
    del bm

    transfer = {"A_then_B_minus_B_only": out["A_then_B"]["B_test"] - out["B_only"]["B_test"],
                "shuffled_A_then_B_minus_B_only": out["shuffled_A_then_B"]["B_test"] - out["B_only"]["B_test"]}
    return {"spec": spec, "arms": out, "transfer": transfer, "t12_3": t123,
            "learning_curves_bpb_per_MiB": curves, "generation": gens, **hooked}


def streams(manifest: Path, smoke: bool = False) -> dict[str, bytes]:
    """Every stream of the experiment, the controls drawn from one generator seeded ``SEED`` (also used by E3)."""
    corpus = Corpus.load(manifest)
    rng = np.random.default_rng(SEED)
    cap = (lambda b: b[:1 << 18]) if smoke else (lambda b: b)
    S = {
        "A_train": corpus.stream("A", "train"), "A_validation": corpus.stream("A", "validation"),
        "A_test": corpus.stream("A", "test"), "A_heldout": corpus.stream("A", "heldout"),
        "B_train": corpus.stream("B", "train"), "B_test": corpus.stream("B", "test"),
        "B_heldout": corpus.stream("B", "heldout"),
    }
    S["A_train_sentence_shuffled"] = corpus.stream("A", "train", control=shuffle_sentences, rng=rng)
    S["A_test_word_shuffled"] = corpus.stream("A", "test", control=shuffle_words, rng=rng)
    S["A_heldout_word_shuffled"] = corpus.stream("A", "heldout", control=shuffle_words, rng=rng)
    return {k: cap(v) for k, v in S.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--smoke", action="store_true", help="first 256 KiB of each stream, a small grid, no lzma")
    args = ap.parse_args()
    t_start = time.perf_counter()
    S = streams(args.manifest, args.smoke)
    sizes = {k: len(v) for k, v in S.items()}
    print("stream bytes:", sizes, flush=True)

    # 1. configurations, chosen on A-validation only
    selection = []
    for spec in grid(args.smoke):
        bm = make(spec)
        read(bm, S["A_train"], learning=True)
        selection.append({**spec, "A_validation_bpb": frozen(bm, S["A_validation"])})
        print("select", selection[-1], flush=True)
        del bm
    strip = lambda r: {k: v for k, v in r.items() if k != "A_validation_bpb"}
    chosen = {"literal": strip(min((r for r in selection if literal(r)), key=lambda r: r["A_validation_bpb"]))}
    best = strip(min(selection, key=lambda r: r["A_validation_bpb"]))
    if best != chosen["literal"]:
        chosen["best_overall"] = best

    # 2. the arms for each chosen configuration
    runs = {}
    for name, spec in chosen.items():
        runs[name] = arms(spec, S)
        print(name, "done", flush=True)
    part1 = {"A": visitation(S["A_train"]), "sentence_shuffled_A": visitation(S["A_train_sentence_shuffled"])}
    reference = {} if args.smoke else {
        "lzma_A_test_given_A_train": lzma_conditional_bpb(S["A_train"], S["A_test"]),
        "lzma_B_test_given_B_train": lzma_conditional_bpb(S["B_train"], S["B_test"]),
        "lzma_B_test_given_A_and_B_train": lzma_conditional_bpb(S["A_train"] + S["B_train"], S["B_test"]),
    }
    payload = {
        "experiment": "E2 textbook curriculum",
        "protocol": "experiments/PROTOCOL.md#e2-textbook-curriculum-owners-instruction-of-2026-10-18",
        "provenance": provenance(seed=SEED, manifest=str(args.manifest), smoke=args.smoke,
                                 manifest_sha256=sha256(args.manifest.read_bytes()), laws=LAWS),
        "stream_bytes": sizes,
        "nested": {**NESTED, "depths": list(NESTED["depths"])},
        "selection": selection, "chosen": chosen, "runs": runs,
        "t12_3_part1": part1,
        "reference_lzma": reference,
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    print(save("e2_smoke" if args.smoke else "e2", payload))


if __name__ == "__main__":
    main()
