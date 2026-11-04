"""E7 design-time check (D): fraction of held-out sums determined by the seen ones (integer lattice membership).

python audits/e7_lattice.py 9 60     # stage 1: operands 0..9, 60 draws
python audits/e7_lattice.py 99 3     # stage 2: operands 0..99, 3 draws (about 3 minutes per h)
"""

import sys
import time

import numpy as np


def egcd(a, b):
    if b == 0:
        return (abs(a), (1 if a > 0 else -1), 0)
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a - q * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return (a, x0, y0) if a > 0 else (-a, -x0, -y0)


class Lattice:
    def __init__(s, n):
        s.n = n
        s.piv = {}

    def add(s, r):
        r = np.array(r, dtype=object)
        for c in range(s.n):
            if r[c] == 0:
                continue
            if c not in s.piv:
                if r[c] < 0:
                    r = -r
                s.piv[c] = r
                return
            v = s.piv[c]
            p = v[c]
            if r[c] % p == 0:
                r = r - (r[c] // p) * v
                continue
            g, x, y = egcd(p, r[c])
            nv = x * v + y * r
            r = (r[c] // g) * v - (p // g) * r
            s.piv[c] = nv

    def canon(s, r):
        r = np.array(r, dtype=object)
        for c in sorted(s.piv):
            if r[c] != 0:
                v = s.piv[c]
                q = r[c] // v[c]
                r = r - q * v
        return tuple(r)


def unit(n, i):
    v = [0] * n
    v[i] = 1
    return v


def run(maxop, h, draws, ordered, seed=3):
    n = 2 * maxop + 1
    pairs = [(a, b) for a in range(maxop + 1) for b in range(a, maxop + 1)]
    rng = np.random.default_rng(seed)
    fr = []
    for _ in range(draws):
        idx = rng.permutation(len(pairs))
        k = round(h * len(pairs))
        held = [pairs[i] for i in idx[:k]]
        seen = [pairs[i] for i in idx[k:]]
        L = Lattice(n)
        for a, b in seen:
            r = [0] * n
            r[a] += 1
            r[b] += 1
            r[a + b] -= 1
            L.add(r)
        cv = {c: L.canon(unit(n, c)) for c in range(n)}
        det = 0
        uniq = 0
        for a, b in held:
            r = [0] * n
            r[a] += 1
            r[b] += 1
            u = L.canon(r)
            m = [c for c in range(n) if cv[c] == u]
            det += a + b in m
            uniq += m == [a + b]
        fr.append((det / len(held), uniq / len(held)))
    f = np.array(fr)
    return f.mean(0), f.min(0)


if __name__ == "__main__":
    maxop = int(sys.argv[1])
    draws = int(sys.argv[2])
    for h in (0.2, 0.4):
        t = time.time()
        print(maxop, h, run(maxop, h, draws, False), round(time.time() - t, 1), flush=True)
