"""E7 design-time check (D): per-token relative residual of the F5 fit, sum against random table (seeds 0..9)."""

import sys

sys.path.insert(0, ".")
import numpy as np
from e7_cp import split
from e7_cp_control import cp_fit_t, table

nop, nres, r = 10, 19, 19
for rel in ("sum", "random"):
    for h in (0.2, 0.4):
        mx = []
        mn = []
        fr = []
        for s in range(10):
            held, seen = split(nop, h, s)
            T = table(s, nop, nres)
            tgt = (lambda a, b: a + b) if rel == "sum" else (lambda a, b, T=T: T[a, b])
            A, B, G = cp_fit_t(nop, nres, seen, tgt, r, s, 300)
            W = np.zeros((nop, nop), bool)
            for a, b in seen:
                W[a, b] = W[b, a] = True
            res = []
            for a in range(nop):
                e = 0
                n = 0
                for b in np.flatnonzero(W[a]):
                    row = G @ (A[a] * B[b])
                    y = np.zeros(nres)
                    y[tgt(a, b)] = 1
                    e += np.sum(np.abs(row - y) ** 2)
                    n += 1
                res.append(np.sqrt(e / max(n, 1)))
            res = np.array(res)
            mx.append(res.max())
            mn.append(res.mean())
            fr.append((res > 0.05).mean())
        print(
            rel,
            h,
            "mean per-token rel residual",
            np.mean(mn).round(4),
            "max",
            np.max(mx).round(4),
            "share tokens >0.05",
            np.mean(fr).round(3),
        )
