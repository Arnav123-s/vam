"""E7 design-time check (D): the V5 route (masked CP decomposition), determined against undetermined items.

python audits/e7_cp.py 10 19 300 10     # operands 0..9, R = 19, 300 sweeps, 10 draws (run from audits/)
python audits/e7_cp.py 100 199 60 1     # stage 2 size, one draw (about 12 minutes with the lattice)
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
from e7_lattice import Lattice, unit


def split(nop, h, seed):
    pairs = [(a, b) for a in range(nop) for b in range(a, nop)]
    rng = np.random.default_rng(seed)
    k = round(h * len(pairs))
    idx = rng.permutation(len(pairs))
    return [pairs[i] for i in idx[:k]], [pairs[i] for i in idx[k:]]


def cp_fit(nop, seen, r, seed, iters=300, lam=1e-3, init=0.3):
    nres = 2 * nop - 1
    rng = np.random.default_rng(seed + 1000)
    W = np.zeros((nop, nop), bool)
    for a, b in seen:
        W[a, b] = W[b, a] = True
    A = init * (rng.normal(size=(nop, r)) + 1j * rng.normal(size=(nop, r)))
    B = A.copy()
    G = init * (rng.normal(size=(nres, r)) + 1j * rng.normal(size=(nres, r)))
    I = np.eye(r)
    for _ in range(iters):
        GG = G.conj().T @ G
        for a in range(nop):
            bs = np.flatnonzero(W[a])
            Bs = B[bs]
            H = (Bs.conj().T @ Bs) * GG  # r x r  (conj(B)B  o conj(G)G)
            rhs = (Bs.conj() * G[a + bs].conj()).sum(0)  # sum_b conj(B[b,j]) conj(G[a+b,j])
            A[a] = np.linalg.solve(H + lam * I, rhs)  # careful with conj conventions below
        GG = G.conj().T @ G
        for b in range(nop):
            as_ = np.flatnonzero(W[:, b])
            As = A[as_]
            H = (As.conj().T @ As) * GG
            rhs = (As.conj() * G[as_ + b].conj()).sum(0)
            B[b] = np.linalg.solve(H + lam * I, rhs)
        ia, ib = np.nonzero(W)
        P = A[ia] * B[ib]
        PP = P.conj().T @ P
        R = np.zeros((nres, r), complex)
        np.add.at(R, ia + ib, P.conj())  # sum over pairs with sum c of conj(P)
        G = np.linalg.solve(PP + lam * I, R.T).T
    return A, B, G


def predict(A, B, G, a, b):
    return np.abs(G @ (A[a] * B[b]))


def run(nop, h, seeds, r, iters):
    out = []
    for s in seeds:
        held, seen = split(nop, h, s)
        n = 2 * nop - 1
        L = Lattice(n)
        for a, b in seen:
            row = [0] * n
            row[a] += 1
            row[b] += 1
            row[a + b] -= 1
            L.add(row)
        cv = {c: L.canon(unit(n, c)) for c in range(n)}
        A, B, G = cp_fit(nop, seen, r, s, iters)
        det = []
        und = []
        for a, b in held:
            row = [0] * n
            row[a] += 1
            row[b] += 1
            u = L.canon(row)
            isdet = cv[a + b] == u
            ok = int(np.argmax(predict(A, B, G, a, b)) == a + b)
            (det if isdet else und).append(ok)
        out.append((np.mean(det) if det else np.nan, np.mean(und) if und else np.nan, len(det), len(und)))
    return out


if __name__ == "__main__":
    nop = int(sys.argv[1])
    r = int(sys.argv[2])
    iters = int(sys.argv[3])
    n = int(sys.argv[4])
    for h in (0.2, 0.4):
        t = time.time()
        o = run(nop, h, range(n), r, iters)
        d = np.array([x[0] for x in o])
        u = np.array([x[1] for x in o])
        print(
            nop,
            r,
            h,
            "det",
            np.nanmean(d).round(3),
            np.nanmin(d).round(3),
            "und",
            np.nanmean(u).round(3),
            round(time.time() - t),
            flush=True,
        )
