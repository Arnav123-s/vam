"""Phase 3b: unitary disentanglers -- the real H4 (MERA-shadow).

H4's distinctive claim is NOT "low-rank beats full" (that's 3a). It's that
knowledge separates REVERSIBLY via LOCAL HIERARCHICAL unitaries (MERA-like
disentanglers between scales): O(n) parameters, scale-respecting, no
information destroyed (unlike truncation).

Test (classical shadow -- honest about it):
- Data: 32-dim. TRUE: 4 slow dims (var 4.0, the shared structure) + 28 fast
  dims (var 1.0, noise), mixed by a random orthogonal (the "entanglement").
- Method A (global SVD / PCA): top-4 PCs. Optimal linear disentangler.
- Method B (hierarchical-local, the MERA-shadow): binary tree of LEARNED 2x2
  rotations; at each pair, rotate to concentrate variance in the "slow"
  output; recurse on slows. Reversible, O(n), scale-respecting.
- Method C (fixed Haar): same tree but FIXED 45-degree rotations (non-learned
  hierarchical baseline).
- Method D (none): first 4 raw dims.
- Metric: fraction of TRUE slow-signal variance captured by the 4 kept dims
  (project kept subspace onto true slow subspace).

Protocol (written before the run): seeds 7,8,9. Expectation: A ~ 1.0
(optimal), B close to A and >> C (learning the disentangler matters),
D ~ 0.125 (chance). If B ≈ D, hierarchical disentangling fails. If B ≈ A,
the MERA-shadow works: local + reversible + cheap ≈ global optimal.
"""

import json
import time

import numpy as np


def gen(seed=7, d=32, k=4, n=20000):
    rng = np.random.default_rng(seed)
    # true slow subspace: first k dims, var 4; fast: var 1
    Z = rng.normal(size=(n, d))
    Z[:, :k] *= 2.0
    # random orthogonal "entanglement"
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    X = Z @ Q.T
    # true slow projector (in observed coords): Q[:, :k] @ Q[:, :k].T
    P_slow = Q[:, :k] @ Q[:, :k].T
    return X, P_slow


def slow_frac(X_kept, P_slow):
    """Fraction of true slow variance captured by kept dims' subspace."""
    # orthonormalize kept dims, project onto slow subspace
    Qk, _ = np.linalg.qr(X_kept.T)  # d x 4
    # variance of slow signal along kept subspace = tr(Qk^T P_slow Qk) / k
    return float(np.trace(Qk.T @ P_slow @ Qk) / P_slow.shape[0] * P_slow.shape[0] / 4)


def pca_disentangle(X, k=4):
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return Vt[:k, :]  # k x d, rows = top PCs


def tree_disentangle(X, k=4, learned=True, adaptive=False):
    """Binary tree of 2x2 rotations. Returns k kept dim directions (k x d).
    adaptive=True: pair dims by max |covariance| instead of adjacent."""
    d = X.shape[1]
    # work with directions; start as identity (each dim)
    dirs = np.eye(d)  # d x d, rows are current basis vectors
    cur = X.copy()
    slows_idx = list(range(d))
    while len(slows_idx) > k:
        nxt = []
        if adaptive:
            pairs = _adaptive_pairs(cur, slows_idx)
        else:
            pairs = [(slows_idx[i], slows_idx[i + 1])
                     for i in range(0, len(slows_idx) - 1, 2)]
            leftover = slows_idx[-1] if len(slows_idx) % 2 == 1 else None
        for a, b in pairs:
            xa, xb = cur[:, a], cur[:, b]
            if learned:
                # 2D PCA: angle maximizing variance of first output
                C = np.cov(np.stack([xa, xb]))
                w, V = np.linalg.eigh(C)
                v = V[:, -1]  # top eigenvector
            else:
                v = np.array([1, 1]) / np.sqrt(2)  # Haar
            # slow output direction in original coords
            s_dir = v[0] * dirs[a] + v[1] * dirs[b]
            # store: reuse slot a for slow, drop b (keep f implicitly)
            dirs[a] = s_dir
            # project data onto new pair for next level
            cur[:, a] = xa * v[0] + xb * v[1]
            nxt.append(a)
        # odd one out carries forward
        if not adaptive and leftover is not None:
            nxt.append(leftover)
        elif adaptive:
            used = {a for a, _ in pairs} | {b for _, b in pairs}
            nxt += [i for i in slows_idx if i not in used]
        slows_idx = nxt
    return dirs[slows_idx[:k]]


def _adaptive_pairs(cur, slows_idx):
    C = np.cov(cur[:, slows_idx].T)
    np.fill_diagonal(C, 0)
    remaining = set(range(len(slows_idx)))
    pairs = []
    while len(remaining) >= 2:
        i, j = np.unravel_index(np.argmax(np.abs(C)), C.shape)
        if i not in remaining or j not in remaining:
            C[i, j] = C[j, i] = 0
            continue
        pairs.append((slows_idx[i], slows_idx[j]))
        remaining -= {i, j}
        C[i, :] = C[:, i] = C[j, :] = C[:, j] = 0
    return pairs


def run(seed=7):
    X, P_slow = gen(seed)
    out = {}
    # A: PCA
    D = pca_disentangle(X)
    out["pca"] = round(slow_frac(D, P_slow), 4)
    # B: learned hierarchical (adjacent pairing)
    D = tree_disentangle(X, learned=True)
    out["hier_learned"] = round(slow_frac(D, P_slow), 4)
    # B2: learned hierarchical (adaptive pairing by |cov|)
    D = tree_disentangle(X, learned=True, adaptive=True)
    out["hier_adaptive"] = round(slow_frac(D, P_slow), 4)
    # C: fixed Haar
    D = tree_disentangle(X, learned=False)
    out["hier_haar"] = round(slow_frac(D, P_slow), 4)
    # D: none
    D = np.eye(32)[:4]
    out["none"] = round(slow_frac(D, P_slow), 4)
    return out


def main():
    t0 = time.time()
    out = {"protocol": "phase3b_disentanglers", "seeds": {}}
    for seed in (7, 8, 9):
        r = run(seed)
        out["seeds"][seed] = r
        print(f"seed {seed}: " + " ".join(f"{k}={v:.3f}" for k, v in r.items()))
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase3b.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase3b.json")


if __name__ == "__main__":
    main()
