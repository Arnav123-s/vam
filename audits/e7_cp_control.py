"""E7 design-time check (D): F5 internal validation and the random-table relation control at stage 1."""

import sys

import numpy as np

sys.path.insert(0, ".")
from e7_cp import split


def cp_fit_t(nop, nres, seen, tgt, r, seed, iters=300, lam=1e-3, init=0.3):
    rng = np.random.default_rng(seed + 1000)
    W = np.zeros((nop, nop), bool)
    Ct = np.zeros((nop, nop), int)
    for a, b in seen:
        W[a, b] = W[b, a] = True
        Ct[a, b] = Ct[b, a] = tgt(a, b)
    A = init * (rng.normal(size=(nop, r)) + 1j * rng.normal(size=(nop, r)))
    B = A.copy()
    G = init * (rng.normal(size=(nres, r)) + 1j * rng.normal(size=(nres, r)))
    I = np.eye(r)
    for _ in range(iters):
        GG = G.conj().T @ G
        for a in range(nop):
            bs = np.flatnonzero(W[a])
            Bs = B[bs]
            A[a] = np.linalg.solve((Bs.conj().T @ Bs) * GG + lam * I, (Bs.conj() * G[Ct[a, bs]].conj()).sum(0))
        for b in range(nop):
            as_ = np.flatnonzero(W[:, b])
            As = A[as_]
            B[b] = np.linalg.solve((As.conj().T @ As) * GG + lam * I, (As.conj() * G[Ct[as_, b]].conj()).sum(0))
        ia, ib = np.nonzero(W)
        P = A[ia] * B[ib]
        PP = P.conj().T @ P
        R = np.zeros((nres, r), complex)
        np.add.at(R, Ct[ia, ib], P.conj())
        G = np.linalg.solve(PP + lam * I, R.T).T
    return A, B, G


def table(seed, nop, nres):
    rng = np.random.default_rng(seed + 500)
    T = np.zeros((nop, nop), int)
    for a in range(nop):
        for b in range(a, nop):
            T[a, b] = T[b, a] = rng.integers(0, nres)
    return T


def val_acc(nop, h, seed, rel, r=19, iters=300):
    nres = 2 * nop - 1
    held, seen = split(nop, h, seed)
    rng = np.random.default_rng(seed + 77)
    idx = rng.permutation(len(seen))
    k = max(1, round(0.1 * len(seen)))
    val = [seen[i] for i in idx[:k]]
    fit = [seen[i] for i in idx[k:]]
    T = table(seed, nop, nres)
    tgt = (lambda a, b: a + b) if rel == "sum" else (lambda a, b: T[a, b])
    A, B, G = cp_fit_t(nop, nres, fit, tgt, r, seed, iters)
    ok = sum(int(np.argmax(np.abs(G @ (A[a] * B[b]))) == tgt(a, b)) for a, b in val)
    ho = sum(int(np.argmax(np.abs(G @ (A[a] * B[b]))) == tgt(a, b)) for a, b in held)
    return ok / len(val), ho / len(held)


if __name__ == "__main__":
    for rel in ("sum", "random"):
        for h in (0.2, 0.4):
            r = np.array([val_acc(10, h, s, rel) for s in range(10)])
            print(rel, h, "val", r[:, 0].round(2), "heldout mean", r[:, 1].mean().round(3))
