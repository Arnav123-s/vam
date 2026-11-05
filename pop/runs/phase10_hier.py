"""Phase 10: soft routing -- the fix for Phase 9's bottleneck.

Phase 9: hard K-way identification collapses at K=500 (0.1% accuracy).
The fix is NOT better identification -- it's NO identification. Use the full
posterior as weights (soft routing, i.e. attention) instead of argmax.

Source: K=500 regimes in C=10 clusters of 50 (same as Phase 9 setup).
Methods (all use the same joint posterior over 500 regimes):
  hard: argmax regime -> its table. (Phase 9's approach.)
  soft: p(bit) = sum over regimes of posterior * p(bit | regime's table).
  oracle: true regime's table.
Expectation: soft >> hard, soft ≈ oracle. If soft also collapses, the
posterior itself is broken. If soft works, the bottleneck was the hard
decision, not the inference.
"""

import json
import time

import numpy as np


def gen_hier(n_bits, C=10, R=50, p_flip=0.004, seed=0):
    rng = np.random.default_rng(seed)
    K = C * R
    # cluster bias levels: 0.2 .. 0.74
    cbias = np.linspace(0.2, 0.74, C)
    tables = {}
    for c in range(C):
        for r in range(R):
            k = c * R + r
            # regime table: 8 contexts, biased toward cluster mean + unique pattern
            base = rng.random(8) < 0.5
            t = np.where(base, 0.9, 0.1)
            # shift toward cluster bias: mix
            t = 0.7 * t + 0.3 * cbias[c]
            tables[k] = np.clip(t, 0.05, 0.95)
    bits, regs = [], []
    s = 0
    prev = (0, 0, 0)
    for _ in range(n_bits):
        if rng.random() < p_flip:
            s = rng.integers(K)
        m = (prev[0] << 2) | (prev[1] << 1) | prev[2]
        b = 1 if rng.random() < tables[s][m] else 0
        bits.append(b)
        regs.append(s)
        prev = (prev[1], prev[2], b)
    return np.array(bits), np.array(regs), tables, C, R


class HierTracker:
    """Proper hierarchical Bayes: joint posterior p(c,r), marginalized."""

    def __init__(self, C, R, p_flip=0.004):
        self.C, self.R = C, R
        self.f = p_flip
        self.joint = np.ones((C, R)) / (C * R)

    def predict(self):
        c, r = np.unravel_index(np.argmax(self.joint), self.joint.shape)
        return int(c) * self.R + int(r)

    def update(self, bit, rlegend):
        # transition: stay 1-f, spread f uniformly over ALL (c,r)
        p = (1 - self.f) * self.joint + self.f / (self.C * self.R)
        # likelihood: p(bit | c,r)
        lik = rlegend[:, :, bit] / rlegend.sum(axis=2)
        self.joint = p * lik
        self.joint /= self.joint.sum()


def main():
    t0 = time.time()
    C, R = 10, 50
    K = C * R
    bits, regs, tables, _, _ = gen_hier(40000, C, R, seed=7)
    train, test = bits[:30000], bits[30000:]
    te_reg = regs[30000:]
    # legends + tables (oracle tables for isolation)
    clegend = np.ones((C, 2))   # p(bit | cluster) = mean over regimes
    rlegend = np.ones((C, R, 2))
    tabs = np.ones((K, 8, 2))
    for t in range(3, 30000):
        m = (int(train[t-3]) << 2) | (int(train[t-2]) << 1) | int(train[t-1])
        k = int(regs[t])
        c, r = k // R, k % R
        tabs[k, m, int(train[t])] += 1
        rlegend[c, r, int(train[t])] += 1
    for c in range(C):
        clegend[c] = rlegend[c].sum(axis=0)
    # flat tracker (Phase 9 style)
    flat_post = np.ones(K) / K
    flat_legend = np.ones((K, 2))
    for t in range(3, 30000):
        flat_legend[int(regs[t]), int(train[t])] += 1
    ht = HierTracker(C, R)
    acc_h, lps_h, lps_s, lps_o = 0, [], [], []
    f = 0.004
    for t in range(3, len(test)):
        m = (int(test[t-3]) << 2) | (int(test[t-2]) << 1) | int(test[t-1])
        b = int(test[t])
        true_k = int(te_reg[t])
        # hard: argmax
        sh = ht.predict()
        acc_h += (sh == true_k)
        p = tabs[sh, m, 1] / tabs[sh, m].sum()
        p = min(max(p, 1e-6), 1 - 1e-6)
        lps_h.append(-np.log2(p if b else 1 - p))
        # soft: posterior-weighted
        pt = tabs[:, m, 1] / tabs[:, m].sum(axis=1)
        p_soft = float((ht.joint.reshape(-1) * pt).sum())
        p_soft = min(max(p_soft, 1e-6), 1 - 1e-6)
        lps_s.append(-np.log2(p_soft if b else 1 - p_soft))
        # oracle
        po = tabs[true_k, m, 1] / tabs[true_k, m].sum()
        po = min(max(po, 1e-6), 1 - 1e-6)
        lps_o.append(-np.log2(po if b else 1 - po))
        ht.update(b, rlegend)
        fp = (1 - f) * flat_post + f / K
        lik = flat_legend[:, b] / flat_legend.sum(axis=1)
        flat_post = fp * lik
        flat_post /= flat_post.sum()
    n = len(test) - 3
    out = {
        "protocol": "phase10_soft_routing",
        "K": K, "C": C, "R": R,
        "hard_acc": round(acc_h / n, 4),
        "hard_bpb": round(float(np.mean(lps_h)), 4),
        "soft_bpb": round(float(np.mean(lps_s)), 4),
        "oracle_bpb": round(float(np.mean(lps_o)), 4),
        "elapsed_s": round(time.time() - t0, 1),
    }
    for k, v in out.items():
        if k not in ("protocol", "elapsed_s"):
            print(f"{k:10s} {v}")
    with open("runs/results_phase10.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase10.json")


if __name__ == "__main__":
    main()
