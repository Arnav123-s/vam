"""How large must an MPS of configuration probabilities be to hold the einselected VAM-1 state exactly?

For einselect="all" and infinite dephasing the state after every event is diagonal, p over 2*4^c
configurations (cells, then the port). A density stored as a matrix-product operator then needs, at the
cut after cell j, the rank of p reshaped to (4^j, rest). We run the exact PointerEngine and record, over
all events and cuts, the bond needed to keep the discarded squared singular weight below tol*|p|^2.

    python experiments/diagnostics/pointer_rank.py BOOK.txt 400

Run once on 2026-10-18 (same book as probability_truncation.py), before vam1_mpo was written; output in
pointer_rank.txt.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vam.realizations.vam1 import Config, PointerEngine


def needed(s, tol):
    w = s ** 2 / (s ** 2).sum()
    tail = np.cumsum(w[::-1])[::-1]
    ok = np.flatnonzero(np.append(tail[1:], 0.0) <= tol)
    return int(ok[0]) + 1

def bits_of(path, n):
    raw = open(path, "rb").read()[20000:20000 + n // 8 + 1]
    return np.unpackbits(np.frombuffer(raw, np.uint8))[:n]

text = sys.argv[1]
n = int(sys.argv[2])
streams = {"random": np.random.default_rng(1).integers(0, 2, n), "text": bits_of(text, n)}
for name, bits in streams.items():
    for theta in (0.3, 0.6):
        for cells in (4, 6, 8, 10):
            cfg = Config(cells, 4, theta=theta, einselect="all")
            pe = PointerEngine(cfg, rng=np.random.default_rng(7))
            t0 = time.time()
            worst = {1e-6: 0, 1e-10: 0, 0.0: 0}
            for t, b in enumerate(bits):
                pe.step(int(b))
                if t < 2 * cells + 2 or t % 4:
                    continue
                for j in range(1, cells):
                    s = np.linalg.svd(pe.p.reshape(4 ** j, -1), compute_uv=False)
                    for tol in worst:
                        worst[tol] = max(worst[tol], needed(s, tol) if tol else int((s > 1e-13 * s[0]).sum()))
            print(f"{name:6s} theta={theta} c={cells:2d}: worst bond over events and cuts: "
                  f"tol 1e-6 -> {worst[1e-6]}, tol 1e-10 -> {worst[1e-10]}, exact rank -> {worst[0.0]}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
