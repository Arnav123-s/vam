"""E5 (PROTOCOL E5): the run script's checks, the pre-run gates, the measurement and the criteria, on small machines."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
import e5_restore as restore  # noqa: E402
import e5_taught as E5  # noqa: E402
import report  # noqa: E402

from vam.nested import NestedConfig  # noqa: E402
from vam.nfa import hier  # noqa: E402
from vam.realizations.vam1 import Config  # noqa: E402

SPEC = Path(__file__).resolve().parents[1] / "spec"
QUESTIONS = json.loads((E5.DATA / "questions.json").read_text())["questions"]


def small(spec=None) -> hier.HierMachine:
    law = Config(cells=1, context=1, consolidating=True, dt=1.0)
    cfg = NestedConfig(law, depths=(8, 16, 24, 32, 48, 64), table_bits=12, prior=4.0, coherence=1.0, verify=3,
                       governor=True, protect=0.5, sink_rate=0.1)
    return hier.HierMachine(hier.HierConfig(cfg, upper_bits=10))


@pytest.fixture(scope="module")
def text() -> bytes:
    return b" ".join(p.read_bytes() for p in sorted(SPEC.glob("*.md")))[:60000]


@pytest.fixture(scope="module")
def trained(text) -> hier.HierMachine:
    m = small()
    E5.read(m, text, learning=True)
    return m


# -- the checks at the start --------------------------------------------------------------------------------------

def test_the_question_file_matches_the_fixed_hash():
    assert E5.check_questions() == E5.QUESTIONS_SHA
    assert (E5.DATA / "questions.sha256").read_text().split()[0] == E5.QUESTIONS_SHA


def test_a_different_question_file_stops_the_run(tmp_path):
    other = tmp_path / "questions.json"
    other.write_bytes((E5.DATA / "questions.json").read_bytes() + b" ")
    with pytest.raises(SystemExit, match="E5 stopped"):
        E5.check_questions(other)


def test_the_machine_is_e4s_chosen_configuration():
    assert E5.check_spec() == E5.CHOSEN
    wrong = {"part2": {"chosen": {**E5.CHOSEN, "verify": 2}}}
    path = Path(__file__).with_name("_tmp_e4.json")
    try:
        path.write_text(json.dumps(wrong))
        with pytest.raises(SystemExit, match="E5 stopped"):
            E5.check_spec(path)
    finally:
        path.unlink(missing_ok=True)


def test_the_question_set_is_what_the_protocol_says():
    kinds = [q["kind"] for q in QUESTIONS]
    assert kinds.count("recall") == 79 and kinds.count("inference") == 61
    for q in QUESTIONS:
        assert len(q["options"]) == 4 and len({len(o.encode()) for o in q["options"]}) == 1
        assert 0 <= q["answer_index"] < 4


# -- the pre-run gates --------------------------------------------------------------------------------------------

def test_snapshot_costs_equal_a_full_copy_bit_for_bit(trained):
    gate = E5.gate_snapshot_equals_copy(trained, QUESTIONS, n_items=3)
    assert gate == {"items": 3, "bit_identical": True}


def test_frozen_reads_change_none_of_the_four_arrays_and_the_state_is_restored(trained):
    before, st = E5.array_hashes(trained), E5.snapshot(trained)
    E5.score_questions(trained, QUESTIONS[:5])
    E5.frozen_bpb(trained, b"some held-out passage of ordinary text. " * 20)
    assert E5.array_hashes(trained) == before
    assert trained.state == st[0] and np.array_equal(trained._stats, st[1])


def test_the_gates_pass_and_return_the_costs(trained):
    gates, scored = E5.run_gates(trained, QUESTIONS[:4])
    assert gates["tables_unchanged"] and gates["snapshot_equals_copy"]["bit_identical"]
    assert set(scored) == {q["id"] for q in QUESTIONS[:4]}
    assert all(len(v["with_stem"]) == 4 and len(v["no_stem"]) == 4 for v in scored.values())


def test_a_changed_table_fails_the_gate(trained, monkeypatch):
    calls = iter(range(10))
    monkeypatch.setattr(E5, "array_hashes", lambda m: {"_wn": str(next(calls))})
    with pytest.raises(E5.GateFailure):
        E5.run_gates(trained, QUESTIONS[:1])


# -- streams ------------------------------------------------------------------------------------------------------

def make_data(tmp_path: Path, text: bytes) -> tuple[Path, dict]:
    parts = {"S1_train.txt": text[:3000], "S2p_1.txt": text[3000:6000], "S3_1.txt": text[6000:8000],
             "S4_1.txt": text[8000:10500], "S55a_w.txt": text[10500:11500], "C_1.txt": text[20000:28000],
             "S1_heldout.txt": text[30000:30500], "S2p_9.txt": text[30500:31000], "S3_9.txt": text[31000:31500],
             "S4_9.txt": text[31500:32000], "S55a_w_ho.txt": text[32000:32300]}
    for name, b in parts.items():
        (tmp_path / name).write_bytes(b)
    rec = lambda n: {"file": n, "bytes": len(parts[n]), "sha256": hashlib.sha256(parts[n]).hexdigest()}
    src = {"T": {"S1": [rec("S1_train.txt")], "S2": [rec("S2p_1.txt")], "S3": [rec("S3_1.txt")],
                 "S4": [rec("S4_1.txt")], "S5": [rec("S55a_w.txt")]},
           "C": {"prose_extra": [{"file": "C_1.txt", "bytes": 5500, "sha256": ""}]},
           "heldout": {"S1": [rec("S1_heldout.txt")], "S2": [rec("S2p_9.txt")], "S3": [rec("S3_9.txt")],
                       "S4": [rec("S4_9.txt")], "S5": [rec("S55a_w_ho.txt")]}}
    return tmp_path, src


def test_arm_c_pauses_at_the_byte_offsets_where_ts_stages_end(tmp_path, text):
    data, src = make_data(tmp_path, text)
    (data / "C_1.txt").write_bytes(text[20000:25500])
    S = E5.build_streams(data, src, smoke=False)
    assert [len(S["C"][k]) for k in ("C3", "C4", "C5")] == [len(S["T"][k]) for k in ("S3", "S4", "S5")]
    assert S["C"]["S1"] == S["T"]["S1"] and S["C"]["S2"] == S["T"]["S2"]
    assert b"".join(S["C"][k] for k in ("C3", "C4", "C5")) == text[20000:25500]
    (data / "C_1.txt").write_bytes(text[20000:25000])
    with pytest.raises(SystemExit, match="E5 stopped"):
        E5.build_streams(data, src, smoke=False)


def test_a_file_that_differs_from_sources_stops_the_run(tmp_path, text):
    data, src = make_data(tmp_path, text)
    src["C"]["prose_extra"][0]["sha256"] = hashlib.sha256((data / "C_1.txt").read_bytes()).hexdigest()
    assert E5.verify_files(data, src) == 11
    (data / "S3_1.txt").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="E5 stopped"):
        E5.verify_files(data, src)


# -- an arm, and the report ----------------------------------------------------------------------------------------

def test_an_arm_records_counters_heldout_gates_and_costs(tmp_path, text):
    data, src = make_data(tmp_path, text)
    (data / "C_1.txt").write_bytes(text[20000:25500])
    S = E5.build_streams(data, src, smoke=False)
    arm = E5.run_arm("T", {}, S, QUESTIONS[:6], machine=small)
    assert arm["order"] == ["S1", "S2", "S3", "S4", "S5"]
    assert list(arm["stages"]["S1"]["heldout_bpb"]) == ["S1"]
    assert list(arm["stages"]["S5"]["heldout_bpb"]) == ["S1", "S2", "S3", "S4", "S5"]
    assert all(arm["stages"][s]["counters"]["global_occupancy"] >= 0 for s in arm["order"])
    assert arm["gates"]["tables_unchanged"] and len(arm["questions"]) == 6
    c = E5.run_arm("C", {}, S, QUESTIONS[:6], machine=small)
    assert c["order"] == ["S1", "S2", "C3", "C4", "C5"]
    assert c["stages"]["C3"]["trained_bytes"] == arm["stages"]["S3"]["trained_bytes"]


def test_the_thresholds_are_the_protocols():
    assert report.e5_threshold(79) == 28 and report.e5_threshold(61) == 23
    assert report.binom_tail(10, 0, 0.25) == pytest.approx(1.0) and report.binom_tail(3, 4, 0.5) == 0.0


def synthetic(correct_t: set, correct_c: set, ties: set = frozenset()) -> dict:
    def costs(right):
        out = {}
        for q in QUESTIONS:
            c = [2.0, 3.0, 4.0, 5.0]
            order = [q["answer_index"], *[i for i in range(4) if i != q["answer_index"]]]
            v = [0.0] * 4
            for rank, i in enumerate(order):
                v[i] = 1.0 + rank if q["id"] in right else 4.0 - rank
            if q["id"] in ties:
                v = [1.0, 1.0, 1.0, 1.0]
            out[q["id"]] = {"with_stem": v, "no_stem": c}
        return out
    return {"arms": {"T": {"questions": costs(correct_t)}, "C": {"questions": costs(correct_c)}}}


def test_the_cheapest_option_is_chosen_and_a_tie_counts_wrong():
    q = QUESTIONS[0]
    ok = report.e5_correct({q["id"]: {"with_stem": [1.0, 1.0, 5.0, 5.0]}}, [q])
    assert ok == {q["id"]: False}
    v = [3.0] * 4
    v[q["answer_index"]] = 2.0
    assert report.e5_correct({q["id"]: {"with_stem": v}}, [q])[q["id"]]


def test_p1_p2_p3_computed_from_saved_costs():
    recall = [q["id"] for q in QUESTIONS if q["kind"] == "recall"]
    infer = [q["id"] for q in QUESTIONS if q["kind"] == "inference"]
    d = synthetic(set(recall[:50]) | set(infer[:40]), set(recall[:10]) | set(infer[:35]))
    a = report.e5_analysis(d, QUESTIONS)
    assert a["N"] == {"recall": 79, "inference": 61} and a["X"] == {"recall": 28, "inference": 23}
    assert a["P1"]["recall"]["pass"] and a["P2"]["pass"] and a["P2"]["only_t"] == 40 and a["P2"]["only_c"] == 0
    assert a["P3"]["only_t"] == 5 and not a["P3"]["pass"]            # five more items, gap 0.082 < D
    none = report.e5_analysis(synthetic(set(), set()), QUESTIONS)
    assert not none["P1"]["recall"]["pass"] and not none["P2"]["pass"]


def test_the_section_is_written_from_the_saved_file(tmp_path):
    recall = [q["id"] for q in QUESTIONS if q["kind"] == "recall"]
    d = synthetic(set(recall[:50]), set(recall[:10]))
    stage = {"trained_bytes": 10, "train_bpb": 2.0, "counters": {"evictions": 1, "refused": 2, "protected": 3,
                                                                  "global_occupancy": 0.5},
             "heldout_bpb": {"S1": 2.5}, "seconds": 1.0}
    for arm in d["arms"].values():
        arm.update({"order": ["S1"], "stages": {"S1": stage},
                    "gates": {"snapshot_equals_copy": {"items": 2, "bit_identical": True}, "tables_unchanged": True}})
    d.update({"provenance": {"commit": "x", "utc": "t", "smoke": True, "questions_sha256": "0" * 64, "preamble": "\n"},
              "seconds": 3600.0})
    text = "\n".join(report.e5_section(d, QUESTIONS))
    for needle in ("P1: T correct on at least X = 28 of N = 79", "P2: T beats C on recall",
                   "P3: T beats C on inference", "smoke run: not a result", "Pre-registered secondary analyses",
                   "cannot show understanding"):
        assert needle in text


# -- the restore script -------------------------------------------------------------------------------------------

def test_restore_rules_cut_like_the_assembly(text):
    key, _, cut, _ = restore.derive("S55a_notebooks.txt")
    assert key == "5000"
    body, ho = cut(text), restore.derive("S55a_notebooks_ho.txt")[2](text)
    assert body + ho == text
    assert not ho or body.endswith(b"\n") or body[-1:] == text[len(body) - 1:len(body)]
    train = restore.derive("S1_train.txt")[2](text * 100)
    held = restore.derive("S1_heldout.txt")[2](text * 100)
    assert len(train) + len(held) == len(text) * 100 and 0.04 < len(held) / (len(text) * 100) < 0.06
    assert restore.derive("S3_16449.txt")[0] == "16449" and restore.derive("C_57723.txt")[0] == "57723"
    with pytest.raises(SystemExit):
        restore.derive("unknown.txt")
