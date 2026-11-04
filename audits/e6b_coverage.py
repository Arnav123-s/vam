"""E6b design-time check (D): word-class coverage on A-validation against A-train, with the declared store.

    cd experiments && python ../audits/e6b_coverage.py [ROWS_BITS FEATURES]     # defaults 17 512; about 4 minutes

Follows PROTOCOL E6b.3: 2**ROWS_BITS hashed rows with a majority-vote tag, FEATURES hashed neighbor counters per side,
the 2**15 rows with the largest tag counts, positive PMI, the 64 leading right singular vectors (Gram eigenvectors),
spherical frequency-weighted k-means with 256 classes. One clustering from the counts of all of A-train: an upper bound
for the maps in force while A-train was read (the machine reads it under 31 evolving maps). Reads A-validation, not
A-test.
"""

import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, ".")
from e2_curriculum import streams
from e4_hier import window_classes

M1, M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)
GOLD, SEEDC = np.uint64(0x9E3779B97F4A7C15), np.uint64(0x243F6A8885A308D3)
UNK = 256


@njit(cache=True)
def mix(x):
    x = (x ^ (x >> np.uint64(30))) * M1
    x = (x ^ (x >> np.uint64(27))) * M2
    return x ^ (x >> np.uint64(31))


@njit(cache=True)
def unit_codes(data):
    n = data.shape[0]
    out = np.empty(n, dtype=np.uint64)
    k = 0
    c = SEEDC
    for i in range(n):
        b = np.int64(data[i])
        c = mix(c ^ ((np.uint64(b) + np.uint64(1)) * GOLD))
        if not ((48 <= b <= 57) or (65 <= b <= 90) or (97 <= b <= 122) or b == 39):
            out[k] = c
            k += 1
            c = SEEDC
    return out[:k]


@njit(cache=True)
def store_update(codes, rbits, feats, tag, tcnt, counts):
    rows = np.int64(1) << rbits
    for i in range(1, codes.shape[0] - 1):
        code = codes[i]
        r = np.int64(code >> np.uint64(64 - rbits))
        if tag[r] == code:
            tcnt[r] += 1
        elif tcnt[r] == 0:
            tag[r] = code
            tcnt[r] = 1
        else:
            tcnt[r] -= 1
        counts[r, np.int64(codes[i + 1] % np.uint64(feats))] += 1
        counts[r, feats + np.int64(codes[i - 1] % np.uint64(feats))] += 1
    return rows


@njit(cache=True)
def keys(data, rbits, tag, sel, cls):
    n = data.shape[0]
    w1 = np.empty(n, dtype=np.uint64)
    w2 = np.empty(n, dtype=np.uint64)
    k1 = np.empty(n, dtype=np.uint64)
    k2 = np.empty(n, dtype=np.uint64)
    r0 = np.uint64(0)
    r1 = np.uint64(0)
    q0 = np.uint64(UNK)
    q1 = np.uint64(UNK)
    c = SEEDC
    for i in range(n):
        w1[i] = mix(mix(r0 ^ SEEDC) ^ mix(c + np.uint64(7)))
        w2[i] = mix(mix(mix(r0 ^ SEEDC) ^ r1) ^ mix(c + np.uint64(7)))
        k1[i] = mix(mix(q0 ^ np.uint64(99)) ^ mix(c + np.uint64(7)))
        k2[i] = mix(mix(mix(q0 ^ np.uint64(99)) ^ q1) ^ mix(c + np.uint64(7)))
        b = np.int64(data[i])
        c = mix(c ^ ((np.uint64(b) + np.uint64(1)) * GOLD))
        if not ((48 <= b <= 57) or (65 <= b <= 90) or (97 <= b <= 122) or b == 39):
            r1 = r0
            r0 = c
            q1 = q0
            r = np.int64(c >> np.uint64(64 - rbits))
            if tag[r] == c and sel[r]:
                q0 = np.uint64(cls[r])
            else:
                q0 = np.uint64(UNK)
            c = SEEDC
    return w1, w2, k1, k2


def main(rbits=17, feats=512, V=1 << 15, d=64, C=256, iters=15):
    t0 = time.time()
    S = streams(Path("data/pg19/MANIFEST.json"), False)
    tr = np.frombuffer(S["A_train"], np.uint8)
    va = np.frombuffer(S["A_validation"], np.uint8)
    codes = unit_codes(tr)
    vcodes = unit_codes(va)
    rows = 1 << rbits
    tag = np.zeros(rows, dtype=np.uint64)
    tcnt = np.zeros(rows, dtype=np.int64)
    counts = np.zeros((rows, 2 * feats), dtype=np.uint32)
    store_update(codes, rbits, feats, tag, tcnt, counts)
    u, cnt = np.unique(codes, return_counts=True)
    top = u[np.argsort(-cnt)[:V]]
    r_top = (top >> np.uint64(64 - rbits)).astype(np.int64)
    lost = int((tag[r_top] != top).sum())
    order = np.argsort(-tcnt)[:V]
    order = order[tcnt[order] >= 4]
    sel = np.zeros(rows, dtype=bool)
    sel[order] = True
    # token share in the map: a unit token is in the map iff its row is selected and its tag is its code
    rr = (codes >> np.uint64(64 - rbits)).astype(np.int64)
    in_map = sel[rr] & (tag[rr] == codes)
    rv = (vcodes >> np.uint64(64 - rbits)).astype(np.int64)
    in_map_v = sel[rv] & (tag[rv] == vcodes)
    print(f"units {len(codes)} distinct {len(u)}; rows {rows}, features {feats} per side")
    print(f"top-{V} units that lose their row: {lost} ({lost / V:.3f}); selected rows {len(order)}")
    print(
        f"unit tokens in the map: A-train {in_map.mean():.4f}, A-validation {in_map_v.mean():.4f} (class C otherwise)"
    )
    P = counts[order].astype(np.float64)
    tot = P.sum()
    rs, cs = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.log(P * tot / (rs * cs))
    P[~np.isfinite(P)] = 0.0
    P = np.maximum(P, 0.0)
    w, vec = np.linalg.eigh(P.T @ P)
    X = P @ vec[:, np.argsort(-w)[:d]]
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    wt = tcnt[order].astype(np.float64)
    wt /= wt.sum()
    rng = np.random.default_rng(0)
    cent = X[rng.choice(len(X), C, replace=False, p=wt)]
    for _ in range(iters):
        a = np.argmax(X @ cent.T, axis=1)
        for c in range(C):
            m = a == c
            if m.any():
                v = (X[m] * wt[m, None]).sum(0)
                cent[c] = v / max(np.linalg.norm(v), 1e-9)
    a = np.argmax(X @ cent.T, axis=1)
    cls = np.full(rows, UNK, dtype=np.int64)
    cls[order] = a
    rows_per = np.bincount(a, minlength=C)
    tok_per = np.bincount(a, weights=tcnt[order], minlength=C)
    perm = np.random.default_rng(0).permutation(len(a))
    tok_rand = np.bincount(a, weights=tcnt[order][perm], minlength=C)
    q = lambda v: np.percentile(v, [0, 25, 50, 75, 100]).round(0)
    print("rows per class (min,q1,med,q3,max):", q(rows_per))
    print("tag-count mass per class, real map:", q(tok_per), " random control (same rows per class):", q(tok_rand))
    print("clustering seconds", round(time.time() - t0))
    wt1, wt2, kt1, kt2 = keys(tr, rbits, tag, sel, cls)
    wv1, wv2, kv1, kv2 = keys(va, rbits, tag, sel, cls)
    cl0 = window_classes(S["A_train"], S["A_validation"]) == 0
    seen = {
        n: np.isin(b, np.unique(a_))
        for n, a_, b in (("W1", wt1, wv1), ("W2", wt2, wv2), ("K1", kt1, kv1), ("K2", kt2, kv2))
    }
    for n, s in seen.items():
        print(n, "A-validation bytes with a seen key", s.mean().round(4))
    u2 = ~seen["W2"]
    print(
        f"U2 share {u2.mean():.4f}; K2 seen on U2 {seen['K2'][u2].mean():.4f}; K1 seen on U2 "
        f"{seen['K1'][u2].mean():.4f}; class 0 share {cl0.mean():.4f}; U2 within class 0 {u2[cl0].mean():.4f}; "
        f"class 0 within U2 {cl0[u2].mean():.4f}"
    )


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:3]))
