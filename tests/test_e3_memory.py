"""T2.3: the memory fix of experiments/e3_nfa.py changes no result.

``reference_writing_grid`` is the writing grid exactly as it was at 1029e49 (before the fix), only with its
module names qualified. The new one must give the same outputs and the same own bits per byte.
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

from vam.nested import NestedConfig, NestedMachine
from vam.realizations.vam1 import Config

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
import e3_nfa as E  # noqa: E402


def reference_writing_grid(machine, texts: list[bytes], vocabulary: set[bytes], n_bytes: int, smoke: bool) -> dict:
    hbars = (0.0, 1.0) if smoke else (0.0, 0.25, 0.5, 1.0)
    settings = []
    for hbar, form, mode, verified in itertools.product(hbars, ("euclidean", "feynman"), ("collapse", "sample"),
                                                        (False, True)):
        t0 = time.perf_counter()
        outs, rep, words, found, printable, total, valid, own_bits = [], [], 0, 0, 0, 0, 0, 0.0
        for i, prompt in enumerate(texts):
            m = machine.copy()
            m.run(E.bytes_to_bits(prompt), learning=False)
            after_prompt = m.copy()
            rng = np.random.default_rng(E.SEED + i) if mode == "sample" else None
            bits = E.collapse.generate(m, 8 * n_bytes, horizon=8, hbar=hbar, mode=mode, form=form,
                                     verified=verified, rng=rng)
            data = E.bits_to_bytes(np.array(bits, dtype=np.uint8))
            outs.append({"prompt": prompt.decode("utf-8", "backslashreplace"),
                         "text": data.decode("utf-8", "backslashreplace")})
            rep.append(E.collapse.repetition(data))
            tokens = E.WORD.findall(data)
            words += len(tokens)
            found += sum(t.lower() in vocabulary for t in tokens)
            printable += sum(32 <= c < 127 or c in (9, 10, 13) for c in data)
            total += len(data)
            valid += E.utf8_valid(data)
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


def machine_and_texts():
    root = Path(__file__).resolve().parents[1] / "spec"
    text = b" ".join(p.read_bytes() for p in sorted(root.glob("*.md")))[:60000]
    law = Config(cells=1, context=1, consolidating=True, dt=1.0)
    m = NestedMachine(NestedConfig(law, depths=(8, 16, 24, 32), table_bits=12, prior=4.0, verify=2, governor=True,
                                   coherence=0.5))
    m.run(E.bytes_to_bits(text), learning=True)
    return m, [text[1000:1030], text[20000:20030]], {w.lower() for w in E.WORD.findall(text)}


def strip(grid):
    return {**grid, "settings": [{k: v for k, v in s.items() if k != "seconds"} for s in grid["settings"]]}


def test_writing_grid_is_unchanged_by_the_memory_fix():
    m, texts, vocab = machine_and_texts()
    ref = reference_writing_grid(m, texts, vocab, 4, True)
    new = E.writing_grid(m, texts, vocab, 4, True)
    assert strip(new) == strip(ref)
    assert len(new["settings"]) == 16


def test_writing_grid_resumes_after_the_last_saved_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "RESULTS", tmp_path)
    m, texts, vocab = machine_and_texts()
    whole = E.writing_grid(m, texts, vocab, 3, True)
    first = E.Partial("t", False)
    E.writing_grid(m, texts, vocab, 3, True, first)
    first.data["part4_settings"] = first.data["part4_settings"][:5]          # killed after 5 settings
    first.put("part4_settings", first.data["part4_settings"])
    resumed = E.writing_grid(m, texts, vocab, 3, True, E.Partial("t", True))
    assert strip(resumed) == strip(whole)
