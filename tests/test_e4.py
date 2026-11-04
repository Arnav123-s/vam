"""E4 experiment script (experiments/e4_hier.py): its helpers, and the whole smoke run on synthetic streams."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
import e3_nfa  # noqa: E402
import e4_hier as H  # noqa: E402
import report  # noqa: E402

SPEC = Path(__file__).resolve().parents[1] / "spec"


def raw() -> bytes:
    return b" ".join(p.read_bytes() for p in sorted(SPEC.glob("*.md")))


def test_base_is_e3s_chosen_row():
    row = H.e3_saved_row()                       # asserts that BASE equals e3.json's chosen configuration
    assert row["spec"]["verify"] == 3 and abs(row["A_validation_bpb"] - 1.8676) < 1e-4


def test_window_classes_find_the_longest_match():
    rng = np.random.default_rng(0)
    train = bytes(rng.integers(0, 256, 5000, dtype=np.uint8))
    passage = train[1000:1060]
    test = bytes(rng.integers(0, 256, 100, dtype=np.uint8)) + passage + bytes(rng.integers(0, 256, 50, dtype=np.uint8))
    cls = H.window_classes(train, test)
    assert cls[:100].max() == 0                  # random bytes: no window of 8 occurs in 5000 random bytes
    assert cls[100 + 8] == 8 and cls[100 + 16] == 16 and cls[100 + 32] == 32 and cls[100 + 59] == 32
    assert cls[100 + 60] == 32                   # the first byte after the passage still follows 32 matching bytes
    assert cls[100 + 61] == 0 and cls[100 + 60 + 25] == 0      # the next window holds a byte that is not in the passage


def test_decomposition_accounts_for_every_bit():
    cls = np.array([0, 8, 8, 16, 32, 0])
    q = np.random.default_rng(1).uniform(0.2, 0.9, 8 * len(cls))
    out = H.decomposition(q, cls)
    assert abs(sum(out[c]["bits"] for c in ("0", "8", "16", "32")) - out["all"]["bits"]) < 1e-9
    assert abs(sum(out[c]["share"] for c in ("0", "8", "16", "32")) - 1.0) < 1e-12


def test_simpler_settings_have_fewer_options_on():
    base = H.hspec(hier_levels="WS")
    assert H.options_on({**base, "hier_levels": "W"}) < H.options_on(base)
    assert H.options_on(base) < H.options_on({**base, "hier_scope": "global", "hier_raise": True})
    assert H.options_on({**base, "sink_rate": 0.0}) < H.options_on(base)


def test_report_labels_and_dedupes_like_e3():
    st = {"hbar": 1.0, "form": "euclidean", "mode": "sample", "verified": False}
    assert report.instrument_label(st).startswith("the machine's own")


@pytest.mark.slow
def test_the_whole_smoke_run_on_synthetic_streams(tmp_path, monkeypatch):
    txt = raw() * 16
    mb = 1 << 18
    parts = [txt[i * mb:(i + 1) * mb] for i in range(7)]
    streams = {"A_train": parts[0], "A_validation": parts[1], "A_test": parts[2], "A_heldout": parts[3],
               "B_train": parts[4], "B_test": parts[5], "B_heldout": parts[6],
               "A_train_sentence_shuffled": parts[0][::-1], "A_test_word_shuffled": parts[2][::-1],
               "A_heldout_word_shuffled": parts[3][::-1]}
    monkeypatch.setattr(H, "streams", lambda *a, **k: streams)
    monkeypatch.setattr(e3_nfa, "RESULTS", tmp_path)
    saved = {}
    monkeypatch.setattr(H, "save", lambda name, payload: saved.update({name: payload}) or tmp_path / f"{name}.json")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["e4", "--manifest", str(manifest), "--smoke"])
    monkeypatch.setattr(H, "RESULTS", EXPERIMENTS / "results")
    H.main()
    d = json.loads(json.dumps(saved["e4_smoke"]))
    assert all(d["part0"]["criteria"].values())
    for layer in ("N_L1", "N_L3", "N_L4", "N_L5", "N_L6", "N_L7"):
        assert all(d["part1"][layer]["criteria"].values()), layer
    assert d["part2"]["chosen"]["machine"] == "hier" and len(d["part2"]["stages"]) == 5
    assert d["part3a"]["A_only"]["A_test"] > 0 and "shuffled_A_then_B" in d["part3b"]
    assert abs(sum(v["share"] for k, v in d["part5"]["E4"].items() if k != "all") - 1.0) < 1e-9
    e3 = json.loads((EXPERIMENTS / "results" / "e3.json").read_text())
    text = "\n".join(report.e4_section(d, e3, None))
    assert "Reading rules" in text and "R5" in text
    assert "Unit 3c" in text and "pure" in text and "None" not in text.split("N-L2, Page rank")[1].split("Label")[0]
    # the control added after the result (Amendment 2): E3's base with sink rate 0
    import e4_t
    monkeypatch.setattr(e4_t, "streams", lambda *a, **k: streams)
    control = e4_t.control(streams)
    assert control["spec"]["sink_rate"] == 0.0 and control["spec"]["hier_levels"] == ""
    t = {"provenance": {"commit": "x", "utc": "t"},
         "E3_base_sink_rate_0": json.loads(json.dumps(e3_nfa.plain(control)))}
    sect = "\n".join(report.e4_t_section(t, d, e3))
    for needle in ("added after the result", "R2 read cleanly", "R5 read cleanly"):
        assert needle in sect.lower() or needle in sect
    assert report.e4_t_section(None, d, e3) == []
