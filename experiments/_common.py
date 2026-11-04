"""Shared helpers for the experiment scripts: provenance, losses and result files."""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
sys.path.insert(0, str(ROOT / "src"))


def commit() -> str:
    """HEAD of the repository, with ``+uncommitted`` when the code under vam/src or vam/experiments differs
    from it (saved results are outputs, not code, so a run that follows another is not marked)."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "experiments",
                                ":!experiments/results"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance(**extra) -> dict:
    import numba
    import scipy
    return {
        "utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "machine": platform.machine(),
        **extra,
    }


def losses(q: np.ndarray) -> dict:
    """Prequential bits per bit over the whole stream and over its last quarter."""
    loss = -np.log2(np.asarray(q, dtype=float))
    n = len(loss)
    return {"all": float(loss.mean()), "last_quarter": float(loss[3 * n // 4:].mean())}


def save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    return path
