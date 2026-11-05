"""Phase 14: unified synthetic -- soft routing + governor purge.

Combines Phase 10 (soft routing) + Phase 4 (governor purge).
Source: K=10 regimes, strong signatures (Phase 9 fixed setup).
Machines:
  A: soft routing, tables accumulate forever.
  B: soft routing + purge: a regime's table resets to prior if its
     posterior stays < 0.01 for 500 bits (stale -> forgotten).
Question: does purge help on top of soft routing, or is soft weighting
already enough to handle staleness?
Protocol (written before the run): seed 7, 30k train / 10k test.
"""

import json
import time

import numpy as np

from runs.phase9_scale import gen_k_regimes


def run(seed=7, K=10, purge=False, stale_thresh=0.01, stale_window=500):
    bits, regs, _ = gen_k_regimes(40000, K, seed=seed)
    train, test = bits[:30000], bits[30000:]
    te_reg = regs[30000:]
    # tables + legend (oracle tables for isolation)
    tables = np.ones((K, 4, 2))
    legend = np.ones((K, 2))
    for t in range(2, 30000):
        m = (int(train[t-2]) << 1) | int(train[t-1])
        tables[int(regs[t]), m, int(train[t])] += 1
        legend[int(regs[t]), int(train[t])] += 1
    # soft posterior
    post = np.ones(K) / K
    stale_count = np.zeros(K, dtype=int)
    f = 0.004
    lps = []
    for t in range(2, len(test)):
        m = (int(test[t-2]) << 1) | int(test[t-1])
        b = int(test[t])
        # soft prediction
        pt = tables[:, m, 1] / tables[:, m].sum(axis=1)
        p = float((post * pt).sum())
        p = min(max(p, 1e-6), 1 - 1e-6)
        lps.append(-np.log2(p if b else 1 - p))
        # update posterior
        fp = (1 - f) * post + f / K
        lik = legend[:, b] / legend.sum(axis=1)
        post = fp * lik
        post /= post.sum()
        # online table update (soft-weighted)
        for k in range(K):
            tables[k, m, b] += post[k]
        # governor purge
        if purge:
            stale_count = np.where(post < stale_thresh, stale_count + 1, 0)
            for k in range(K):
                if stale_count[k] >= stale_window:
                    tables[k] = np.ones((4, 2))
                    stale_count[k] = 0
    return float(np.mean(lps))


def main():
    t0 = time.time()
    a = run(purge=False)
    b = run(purge=True)
    out = {
        "protocol": "phase14_unified_synthetic",
        "soft_no_purge_bpb": round(a, 4),
        "soft_purge_bpb": round(b, 4),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"soft no-purge: {a:.4f}")
    print(f"soft + purge:  {b:.4f}")
    with open("runs/results_phase14.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase14.json")


if __name__ == "__main__":
    main()
