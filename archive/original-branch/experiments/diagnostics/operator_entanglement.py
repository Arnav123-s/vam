"""Operator entanglement of the VAM-1 density: how large would a matrix-product-operator engine need to be?

    python experiments/diagnostics/operator_entanglement.py

Run once on 2026-10-18, before vam1_mpo was written; output in operator_entanglement.txt. The dense
four-cell density after 200 random bits, split at every cell cut: the bond that keeps the discarded
squared operator-Schmidt weight below 1e-6 and 1e-10 of the total, against the register's maximum.
(Five cells were tried too and dropped: about 10 s per dense event.)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vam.realizations.vam1 import Config, DenseMaps


def op_schmidt(R, nq, cut):
    """Operator Schmidt values of R across the qubit cut (first `cut` qubits | rest), Frobenius-normalized."""
    d1, d2 = 2 ** cut, 2 ** (nq - cut)
    T = R.reshape(d1, d2, d1, d2).transpose(0, 2, 1, 3).reshape(d1 * d1, d2 * d2)
    s = np.linalg.svd(T, compute_uv=False)
    return s / np.linalg.norm(s)

def needed(s, tol):
    """Smallest D with discarded Frobenius weight sum_{i>=D} s_i^2 <= tol."""
    tail = np.cumsum((s ** 2)[::-1])[::-1]
    ok = np.flatnonzero(np.append(tail[1:], 0.0) <= tol)
    return int(ok[0]) + 1

rng = np.random.default_rng(3)
RUNS = [(4, 3, 0.3, "context"), (4, 3, 0.6, "context"), (4, 3, 0.6, "all"), (4, 4, 0.3, "context"),
        (4, 4, 0.6, "context")]
for cells, k, theta, eins in RUNS:
    cfg = Config(cells=cells, context=k, theta=theta, dephasing=np.inf, einselect=eins, dt=0.05)
    m = DenseMaps(cfg).reader()
    t0 = time.time()
    bits = rng.integers(0, 2, 200)
    for b in bits:
        m.step(int(b), rng)
    R = m.state.R
    nq = cfg.qubits
    ev = np.linalg.eigvalsh(R)
    rank = int((ev > 1e-12).sum())
    row = []
    for cut in range(2, nq - 1, 2):
        s = op_schmidt(R, nq, cut)
        row.append((cut // 2, needed(s, 1e-6), needed(s, 1e-10), min(4 ** cut, 4 ** (nq - cut))))
    print(f"c={cells} k={k} theta={theta} {eins}: rank {rank}/{cfg.dim}; per cell-cut "
          f"(cut, D@1e-6, D@1e-10, max): {row}  [{time.time()-t0:.0f}s]", flush=True)
