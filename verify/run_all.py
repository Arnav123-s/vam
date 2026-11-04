"""One-command verification of the VAM build.

    python verify/run_all.py            # full suite
    python verify/run_all.py --quick    # skip tests marked slow

Runs the whole test suite, recomputes the headline exact results of T15.2
and the agreement of the VAM-1 engines directly from the library, and writes
``verify/RECEIPT.json``: environment, source-document hashes, every test
outcome, every headline value with its error, the hashes of the saved
experiment results, and an overall verdict. The exit status is nonzero if
anything fails.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import scipy
import sympy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vam.bitkernel import BitMachine  # noqa: E402
from vam.gaussian_blanket import interior_drift, radius_multiplier  # noqa: E402
from vam.linalg import Y, Z, kron, purity  # noqa: E402
from vam.nested import NUMBA_AVAILABLE, NestedConfig, NestedMachine  # noqa: E402
from vam.plasticity import CorrelationWeightLaw  # noqa: E402
from vam.realizations import fourier25 as f25  # noqa: E402
from vam.realizations import incomplete_codes as ic  # noqa: E402
from vam.realizations import two_mode as tm  # noqa: E402
from vam.realizations.vam1 import Config, DenseMaps, PointerEngine  # noqa: E402
from vam.semiclassical import free_action, thread_intermediate  # noqa: E402

SEED = 20261018


class _Collector:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def pytest_runtest_logreport(self, report) -> None:  # pytest hook
        if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
            self.results.append({"test": report.nodeid, "outcome": report.outcome,
                                 "seconds": round(report.duration, 4)})


def run_tests(quick: bool) -> tuple[int, list[dict]]:
    collector = _Collector()
    args = [str(ROOT / "tests"), "-q", "-p", "no:cacheprovider", "--rootdir", str(ROOT)]
    if quick:
        args += ["-m", "not slow"]
    code = pytest.main(args, plugins=[collector])
    return int(code), collector.results


def headline() -> list[dict]:
    """The representative exact results of T15.2, recomputed through the library."""
    rows: list[dict] = []

    def check(name: str, source: str, computed: float, expected: float, tol: float = 1e-12) -> None:
        err = abs(complex(computed) - complex(expected))
        rows.append({"result": name, "source": source, "computed": _num(computed), "expected": _num(expected),
                     "abs_error": err, "tolerance": tol, "pass": bool(err <= tol)})

    # T2.2: completion probability after 0, 01, 010
    m = ic.machine()
    for prefix, expected in (("0", 1 / 2), ("01", 3 / 4), ("010", 2 / 3)):
        m = ic.machine()
        for ch in prefix:
            m.force(int(ch), None)
        check(f"completion probability after {prefix}", "T2.2",
              float(np.real(np.trace(ic.P_COMPLETE @ m.state.R))), expected)

    # T9.1-9.2: joint evolution
    psi = tm.U_S @ np.array([1, 0, 0, 0], dtype=complex)
    R = np.outer(psi, psi.conj())
    RQ, RW = tm.SPACE.reduce(R, "Q"), tm.SPACE.reduce(R, "W")
    check("marginal purity after the joint interaction", "T9.1", purity(RQ), 7 / 8)
    check("exact vector derivative 2<Y_Q Z_W>", "T9.2", float(np.real(np.trace(R @ kron(2 * Y, Z)))), -np.sqrt(2))
    check("independent-marginal derivative", "T9.2",
          float(np.real(2 * np.trace(RQ @ Y) * np.trace(RW @ Z))), -1 / np.sqrt(2))

    # T9.4: next output after re-entry
    mt = tm.machine()
    for tau, _q, _p, st in mt.branches(mt.state, None):
        R_b, _ = mt.input_stage(st, None)
        nxt = mt.outcome_probabilities(st, R_b)[0]
        check(f"next zero probability after re-entered {tau}", "T9.4", nxt, {0: 11 / 12, 1: 1 / 4}[tau])
    dist = tm.machine().output_distribution([None, None])
    for k, v in {(0, 0): 11 / 16, (0, 1): 1 / 16, (1, 0): 1 / 16, (1, 1): 3 / 16}.items():
        check(f"two-output probability {k[0]}{k[1]}", "T9.4", dist[k], v)

    # T8.3: 25-configuration interference
    worst_needle, worst_wrong = 0.0, 0.0
    for r in range(25):
        a = f25.amplitudes(r)
        worst_needle = max(worst_needle, abs(abs(a[r]) ** 2 - 1))
        worst_wrong = max(worst_wrong, float(np.max(np.abs(np.delete(a, r)))))
    check("25-configuration needle probability (worst r)", "T8.3", 1 - worst_needle, 1.0, 1e-14)
    check("25-configuration wrong-channel amplitude (worst r)", "T8.3", worst_wrong, 0.0, 5e-15)
    check("wrong-channel residual at r = 7 (document: about 1.3e-15)", "T15.2",
          float(np.max(np.abs(np.delete(f25.amplitudes(7), 7)))), 0.0, 1.5e-15)

    # T8.4: macro-micro threading
    th = thread_intermediate(lambda z: free_action(0.0, z, 0.25), lambda z: free_action(z, 2.0, 0.75), (-5, 5))
    check("free trajectory location at the first quarter time", "T8.4", th.z, 0.5, 1e-9)
    check("split action 1/2 + 3/2", "T8.4", th.first_action + th.second_action, 2.0, 1e-9)
    check("unsplit action", "T8.4", free_action(0.0, 2.0, 1.0), 2.0)

    # T5: spiral blanket
    ev = sorted(np.linalg.eigvals(interior_drift()), key=lambda z: z.imag)
    check("spiral eigenvalue (lower)", "T5", ev[0], -2 - 2j)
    check("spiral eigenvalue (upper)", "T5", ev[1], -2 + 2j)
    check("spiral radius multiplier at log 2", "T5", radius_multiplier(np.log(2)), 0.25)

    # T12.1: correlation-driven weight
    law = CorrelationWeightLaw(g=1.0, kappa=1.0, gamma=1.0, kT=0.02)
    check("weight mean at log 2", "T12.1", law.mean(np.log(2), 0.0, 0.8), 0.4)
    check("weight variance at log 2", "T12.1", law.variance(np.log(2)), 0.015)

    # D8, D9: the VAM-1 engines are one machine (worst prediction difference on a seeded stream)
    rng = np.random.default_rng(SEED)
    bits = [int(b) for b in rng.integers(0, 2, 200)]
    cfg = Config(cells=3, context=2, dephasing=0.0)
    m = DenseMaps(cfg).reader()
    q_dense = []
    for b in bits:
        q_dense.append(m.predict(b))
        m.step(b, rng)
    q_bits = BitMachine(cfg).run(np.array(bits, dtype=np.uint8))
    check("dense coherent engine against the bit kernel", "D8", float(np.max(np.abs(q_bits - q_dense))), 0.0)
    cfg = Config(cells=4, context=3, consolidating=True, dt=1.0)
    pe = PointerEngine(cfg)
    q_pointer = np.array([pe.step(b) for b in bits])
    q_bits = BitMachine(cfg).run(np.array(bits, dtype=np.uint8))
    check("pointer engine against the bit kernel", "D8", float(np.max(np.abs(q_bits - q_pointer))), 0.0)
    if NUMBA_AVAILABLE:
        law = Config(cells=1, context=1, consolidating=True, dt=1.0)
        stream = rng.integers(0, 2, 20_000).astype(np.uint8)
        q_nested = NestedMachine(NestedConfig(law, depths=(9,), table_bits=12)).run(stream)
        q_bits = BitMachine(Config(cells=9, context=9, consolidating=True, dt=1.0)).run(stream)
        check("one nested level against the bit kernel", "D9", float(np.max(np.abs(q_nested - q_bits))), 0.0, 1e-13)
    return rows


def _num(x):
    z = complex(x)
    return z.real if z.imag == 0 else [z.real, z.imag]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", ".", ":!verify/RECEIPT.json"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="skip tests marked slow")
    ap.add_argument("--out", default=str(ROOT / "verify" / "RECEIPT.json"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    code, tests = run_tests(args.quick)
    rows = headline()
    elapsed = time.perf_counter() - t0
    counts: dict[str, int] = {}
    for r in tests:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    ok = code == 0 and all(r["pass"] for r in rows) and counts.get("failed", 0) == 0
    receipt = {
        "verdict": "PASS" if ok else "FAIL",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seconds": round(elapsed, 2),
        "mode": "quick" if args.quick else "full",
        "commit": git_commit(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "sympy": sympy.__version__, "pytest": pytest.__version__, "platform": platform.platform()},
        "specification": {p.name: sha256(p) for p in sorted((ROOT / "spec").glob("*.md"))},
        "tests": {"exit_code": code, "counts": counts, "results": tests},
        "headline_results": rows,
        "experiment_results": {p.name: sha256(p) for p in sorted((ROOT / "experiments" / "results").glob("*.json"))},
    }
    Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n")
    width = max(len(r["result"]) for r in rows)
    print()
    for r in rows:
        mark = "ok " if r["pass"] else "BAD"
        print(f"  {mark}  {r['result']:<{width}}  {r['source']:<6}  error {r['abs_error']:.1e}")
    print(f"\n{receipt['verdict']}: {counts.get('passed', 0)} tests passed, {counts.get('failed', 0)} failed, "
          f"{sum(r['pass'] for r in rows)}/{len(rows)} headline results; receipt written to {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
