"""E3 additions after the result (PROTOCOL E3 Amendment 4): the cache trigger, the count-floor profile, the report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vam.bits import bytes_to_bits
from vam.nfa.governor import GovernedEngine
from vam.realizations.vam1 import Config

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
import e3_t4 as T  # noqa: E402
import report  # noqa: E402

TEXT = b"A textbook states a rule, shows it at work on an example, and then asks the reader to use it alone. "


def run(engine, n=24):
    bits, taus = bytes_to_bits(TEXT * 2)[:n], np.random.default_rng(7).integers(0, 2, n)
    return [engine.force(int(b), int(t)) for b, t in zip(bits, taus, strict=True)]


def test_page_trigger_is_the_default_and_unchanged():
    cfg = Config(5, 2, theta=0.5)
    a, b = GovernedEngine(cfg, max_bond=16), GovernedEngine(cfg, max_bond=16, trigger="page")
    assert run(a) == run(b) and a.discarded == b.discarded and a.governed_since == b.governed_since


def test_cache_trigger_acts_before_the_cap_binds_and_copies():
    cfg = Config(5, 2, theta=0.5)
    cap = 16
    page, cache = GovernedEngine(cfg, max_bond=cap), GovernedEngine(cfg, max_bond=cap, trigger="cache")
    run(page)
    run(cache)
    assert cache.governed_since is not None and cache.governed_since <= (page.governed_since or 10**9)
    assert cache.copy().trigger == "cache"
    with pytest.raises(ValueError):
        GovernedEngine(cfg, trigger="entropy")


def test_floor_profile_keeps_only_well_sampled_rows_and_the_null_never_exceeds_rank_one():
    root = Path(__file__).resolve().parents[1] / "spec"
    text = b" ".join(p.read_bytes() for p in sorted(root.glob("*.md")))
    prof = T.floor_profile(text, floor=T.FLOOR, depths=(1, 2))
    assert prof[0]["rows_kept"] < prof[0]["rows"] and prof[0]["samples_kept"] <= prof[0]["samples"]
    null = bytes(np.random.default_rng(0).permutation(np.frombuffer(text, dtype=np.uint8)))
    assert all(r["rank"] is None or r["rank"] <= 1 for r in T.floor_profile(null, depths=(1, 2, 3)))
    assert T.FLOOR == 256


def test_report_subsection_is_labelled_added_after_the_result():
    root = Path(__file__).resolve().parents[1] / "spec"
    text = b" ".join(p.read_bytes() for p in sorted(root.glob("*.md")))
    t4 = {"provenance": {"commit": "x", "utc": "t"}, "T4_2_text": T.text_part(text, True),
          "T4_3_chain": T.chain_part(True)}
    e3 = json.loads((EXPERIMENTS / "results" / "e3_smoke.json").read_text())
    lines = "\n".join(report.e3_t4_section(t4, e3))
    for needle in ("Added after the result", "cache_trigger_half_cap", "Part 1: governed_cache_bond"):
        assert needle in lines
    assert report.e3_t4_section(None, e3) == []
