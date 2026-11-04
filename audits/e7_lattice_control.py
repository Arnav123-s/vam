"""E7 design-time check (D): F4 on a random commutative table (no additive structure) never has a unique candidate."""

import sys

import numpy as np

sys.path.insert(0, ".")
from e7_cp import split
from e7_lattice import Lattice, unit

nop = 10
n = 19


def table(seed):
    rng = np.random.default_rng(seed + 500)
    T = np.zeros((nop, nop), int)
    for a in range(nop):
        for b in range(a, nop):
            T[a, b] = T[b, a] = rng.integers(0, n)
    return T


def lat_fire(h, seed):
    T = table(seed)
    held, seen = split(nop, h, seed)
    L = Lattice(n)
    for a, b in seen:
        r = [0] * n
        r[a] += 1
        r[b] += 1
        r[T[a, b]] -= 1
        L.add(r)
    cv = {c: L.canon(unit(n, c)) for c in range(n)}
    fire = right = 0
    for a, b in held:
        r = [0] * n
        r[a] += 1
        r[b] += 1
        u = L.canon(r)
        m = [c for c in range(n) if cv[c] == u]
        if len(m) == 1:
            fire += 1
            right += m[0] == T[a, b]
    return fire / len(held), right / max(fire, 1), len(L.piv)


for h in (0.2, 0.4):
    r = [lat_fire(h, s) for s in range(10)]
    print(
        "lattice random table h",
        h,
        "fire",
        np.mean([x[0] for x in r]).round(3),
        "right|fire",
        np.mean([x[1] for x in r]).round(3),
        "rank",
        np.mean([x[2] for x in r]),
    )
