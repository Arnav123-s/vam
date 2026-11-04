"""Is a bounded MPS of configuration probabilities a faithful VAM-1 state (einselect="all")?

Two PointerEngines read the same text bits with equal generators. After every event the second one's
probability vector is replaced by its TT-SVD truncation to bond D (cells as sites, port appended to the
last cell). Printed: the largest prediction difference |q_trunc - q_exact| over the run, the largest
l1 truncation error of p per event, and how many emissions differed.

    python experiments/diagnostics/probability_truncation.py BOOK.txt 1200

Run once on 2026-10-18 with the first PG-19 textbook downloaded (ebook 10140, bytes 20000 on), before
vam1_mpo was written; output in probability_truncation.txt. Once an emission differs the two
trajectories separate, so max |dq| then measures the separation, not the truncation.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vam.realizations.vam1 import Config, PointerEngine


def tt_truncate(p, cells, D):
    t = p.reshape((4,) * (cells - 1) + (8,))
    cores, rest, left = [], t.reshape(4, -1), 1
    for _ in range(cells - 1):
        u, s, vt = np.linalg.svd(rest.reshape(left * 4, -1), full_matrices=False)
        r = min(D, int((s > 1e-15 * s[0]).sum()))
        cores.append(u[:, :r].reshape(left, 4, r))
        rest, left = s[:r, None] * vt[:r], r
    out = rest                                             # (left, 8)
    for core in reversed(cores):
        out = np.tensordot(core, out.reshape(core.shape[2], -1), axes=([2], [0])).reshape(core.shape[0], -1)
    return out.reshape(-1)

raw = open(sys.argv[1], "rb").read()[20000:20000 + int(sys.argv[2]) // 8]
bits = np.unpackbits(np.frombuffer(raw, np.uint8))
for cells in (8, 10):
    for theta in (0.3, 0.6):
        for D in (16, 32, 64):
            cfg = Config(cells, 4, theta=theta, einselect="all")
            ex = PointerEngine(cfg, rng=np.random.default_rng(7))
            tr = PointerEngine(cfg, rng=np.random.default_rng(7))
            dq = l1 = 0.0
            diff = 0
            t0 = time.time()
            for b in bits:
                dq = max(dq, abs(ex.step(int(b)) - tr.step(int(b))))
                diff += ex.last_output != tr.last_output
                new = tt_truncate(tr.p, cells, D)
                l1 = max(l1, float(np.abs(new - tr.p).sum()))
                tr.p = new
            print(f"c={cells:2d} theta={theta} D={D:3d}: max |dq| = {dq:.2e}, max l1 truncation = {l1:.2e}, "
                  f"emissions differing {diff}/{len(bits)}  [{time.time() - t0:.0f}s]", flush=True)
