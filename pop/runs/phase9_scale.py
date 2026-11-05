"""Phase 9: scale -- does the population hold up with many regimes?

The F spec says "millions or billions of vectors." Phases 1-8 used 2-8
nodes. This tests whether the architecture survives at scale.

Source: K regimes, each with a random order-2 table (bias varies per regime
so the slow key can track). Regime flips p=0.004. K in {10, 100, 500}.

Population: K nodes (one per regime), each an order-2 table.
Tracker: K-way Bayesian filter over regimes (generalized Phase 2).
  - Prior: uniform. Transition: stay 1-f, spread f/(K-1) to others.
  - Likelihood: p(bit | regime) from frozen legend.
  - Predict: argmax regime -> its table.

Metrics per K: bpb (vs flat order-2 and vs oracle-true-regime), seconds.
Expectation: bpb holds (population scales statistically); time grows
~linearly in K (each bit touches the tracker O(K) + one node O(1)).
If bpb degrades with K, the tracker can't keep up at scale. If time
explodes, the architecture doesn't scale computationally.

Protocol (written before the run): seeds 7. Train 30k, test 10k bits.
"""

import json
import time

import numpy as np


def gen_k_regimes(n_bits, K, p_flip=0.004, seed=0):
    rng = np.random.default_rng(seed)
    # STRONG regime signatures: each regime k has a distinct peak context.
    # p(1|m) = 0.9 if m == k%4, else 0.1, plus small noise. Knowing the regime
    # is highly informative (unlike random similar tables).
    tables = {}
    for k in range(K):
        peak = k % 4
        t = np.full(4, 0.1)
        t[peak] = 0.9
        t += rng.normal(0, 0.02, 4)
        tables[k] = np.clip(t, 0.05, 0.95)
    bits, regs = [], []
    s = 0
    prev = (0, 0)
    for _ in range(n_bits):
        if rng.random() < p_flip:
            s = rng.integers(K)
        m = (prev[0] << 1) | prev[1]
        b = 1 if rng.random() < tables[s][m] else 0
        bits.append(b)
        regs.append(s)
        prev = (prev[1], b)
    return np.array(bits), np.array(regs), tables


class KTracker:
    def __init__(self, K, p_flip=0.004):
        self.K = K
        self.f = p_flip
        self.post = np.ones(K) / K

    def predict(self):
        return int(np.argmax(self.post))

    def update(self, bit, legend):
        # transition: stay 1-f, spread f
        p = (1 - self.f) * self.post + self.f / self.K
        # likelihood: p(bit | regime) from legend
        lik = legend[:, bit] / legend.sum(axis=1)
        self.post = p * lik
        self.post /= self.post.sum()


def run(K, seed=7):
    t0 = time.time()
    bits, regs, _ = gen_k_regimes(40000, K, seed=seed)
    train, test = bits[:30000], bits[30000:]
    # legend + tables from majority-keyed training (consistent labeling)
    # For K regimes, majority key doesn't work directly. Use a simple proxy:
    # cluster by recent-bit mean into K bins? Instead: train tables per TRUE
    # regime is cheating for the tracker test. Use ORACLE tables for the
    # tracker comparison (isolate tracker scaling), and note it.
    # Honest approach: tables per true regime (oracle tables), tracker must
    # infer. The bpb gap vs oracle-regime shows tracker cost at scale.
    tables = np.ones((K, 4, 2))
    legend = np.ones((K, 2))
    tr_reg = regs[:30000]
    for t in range(2, 30000):
        m = (int(train[t-2]) << 1) | int(train[t-1])
        tables[int(tr_reg[t]), m, int(train[t])] += 1
        legend[int(tr_reg[t]), int(train[t])] += 1
    # flat baseline
    flat = np.ones((4, 2))
    for t in range(2, 30000):
        m = (int(train[t-2]) << 1) | int(train[t-1])
        flat[m, int(train[t])] += 1
    # evaluate
    tr = KTracker(K)
    lps_tr, lps_flat, lps_orc = [], [], []
    te_reg = regs[30000:]
    for t in range(2, len(test)):
        m = (int(test[t-2]) << 1) | int(test[t-1])
        b = int(test[t])
        s = tr.predict()
        p = tables[s, m, 1] / tables[s, m].sum()
        lps_tr.append(-np.log2(min(max(p, 1e-6), 1-1e-6) if b else 1-min(max(p, 1e-6), 1-1e-6)))
        pf = flat[m, 1] / flat[m].sum()
        lps_flat.append(-np.log2(min(max(pf, 1e-6), 1-1e-6) if b else 1-min(max(pf, 1e-6), 1-1e-6)))
        so = int(te_reg[t])
        po = tables[so, m, 1] / tables[so, m].sum()
        lps_orc.append(-np.log2(min(max(po, 1e-6), 1-1e-6) if b else 1-min(max(po, 1e-6), 1-1e-6)))
        tr.update(b, legend)
    dt = time.time() - t0
    return {
        "K": K,
        "tracker_bpb": round(float(np.mean(lps_tr)), 4),
        "flat_bpb": round(float(np.mean(lps_flat)), 4),
        "oracle_bpb": round(float(np.mean(lps_orc)), 4),
        "seconds": round(dt, 1),
    }


def main():
    t0 = time.time()
    out = {"protocol": "phase9_scale", "results": []}
    for K in (10, 100, 500):
        r = run(K)
        out["results"].append(r)
        print(f"K={K:3d}: tracker={r['tracker_bpb']:.4f} flat={r['flat_bpb']:.4f} "
              f"oracle={r['oracle_bpb']:.4f} time={r['seconds']}s")
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase9.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase9.json")


if __name__ == "__main__":
    main()
