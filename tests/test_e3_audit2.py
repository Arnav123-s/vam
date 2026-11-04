"""Audit 2 follow-ups (experiments/PROTOCOL.md, E3 Amendment 1): the pinned cache bond and the Part 4 labels."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import e3_nfa  # noqa: E402
import report  # noqa: E402


def test_cache_bond_is_pinned_not_detected():
    assert e3_nfa.CACHE_BOND == 90
    tree = ast.parse((EXPERIMENTS / "e3_nfa.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "GovernedEngine"]
    assert calls
    assert all(any(k.arg == "max_bond" for k in c.keywords) for c in calls)


def test_host_bounds_are_recorded_for_information():
    hb = e3_nfa.host_bounds()
    assert {"max_mpo_bond", "max_density_dim", "l2_bytes", "detected"} <= hb.keys()


def setting(hbar, form="euclidean", mode="sample", verified=False):
    return {"hbar": hbar, "form": form, "mode": mode, "verified": verified}


def test_only_one_setting_is_the_machines_own_instrument():
    grid = [setting(h, f, m, v) for h in (0.0, 0.25, 0.5, 1.0) for f in ("euclidean", "feynman")
            for m in ("collapse", "sample") for v in (False, True)]
    own = [s for s in grid if report.instrument_label(s).startswith("the machine's own")]
    assert own == [setting(1.0)]
    assert all(report.instrument_label(s).startswith("chosen realization") for s in grid if s not in own)


def test_hbar_zero_settings_are_reported_once_per_verified_value():
    grid = [setting(h, f, m, v) for h in (0.0, 1.0) for f in ("euclidean", "feynman")
            for m in ("collapse", "sample") for v in (False, True)]
    shown = report.reported_once(grid)
    assert len(grid) == 16 and len(shown) == 2 + 8
    assert [s for s in shown if s["hbar"] == 0] == [setting(0.0, "euclidean", "collapse", False),
                                                    setting(0.0, "euclidean", "collapse", True)]
