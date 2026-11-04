"""E7 design-time check (D): naive zero-filled Jennrich diagonalization of a random combination of the T_a."""

import numpy as np


def trial(m, h, seed):
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for a in range(m) for b in range(a, m)]
    k = round(h * len(pairs))
    idx = rng.permutation(len(pairs))
    held = [pairs[i] for i in idx[:k]]
    seen = [pairs[i] for i in idx[k:]]
    T = np.zeros((m, m, m))
    for a, b in seen:
        c = (a + b) % m
        T[a][c, b] = 1
        T[b][c, a] = 1
    x = rng.normal(size=m) + 1j * rng.normal(size=m)
    M = np.tensordot(x, T, axes=1)
    _, V = np.linalg.eig(M)
    V = V / V[0:1, :]
    ok = 0
    tot = 0
    for a, b in held:
        for p, q in ((a, b), (b, a)):
            pred = V[p] * V[q]
            sc = np.abs(np.einsum("cj,j->c", np.conj(V), pred)) / m
            ok += np.argmax(sc) == (p + q) % m
            tot += 1
    return ok / max(tot, 1)


for m in (10, 100):
    for h in (0.0, 0.2, 0.4):
        r = [trial(m, h, s) for s in range(5 if m == 100 else 20)]
        print(m, h, np.round(np.mean(r), 3), np.round(np.min(r), 3), flush=True)
