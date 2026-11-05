"""Phase 3a experiment: shared cross-scale factors (H4) vs independent nodes.

Protocol (written before the run):
- Source: factor-structured bits. slow s in {0,1}, P(s=1)=0.05 (rare regime);
  micro m in {0..31} (order-5 patterns, same distribution in both regimes);
  TRUE logit p(1|s,m) = A_s + B_m  (exactly rank-1/additive).
- Models, all online prequential:
    I: independent (slow x micro) count tables -- 64 params, no sharing.
    F: factorized rank-1, logit = A_s + B_m -- 18 params, B_m shared across slow.
    P: Phase-1 population (nodes + legend backoff) -- marginal sharing only.
- Metrics: bpb overall (excl. first 1000), bpb on slow=1 bits (transfer zone),
  burn-in bpb.  100k bits, seeds 7, 8, 9.
- Expectation: F wins clearly on slow=1 bits by borrowing B_m from slow=0;
  I and P starve there.  Steady-state on slow=0 should tie.
"""

import json
import time

import numpy as np

from proto.factors import FactorModel
from proto.population import Population


def gen_bits(n, p_rare=0.05, order=5, seed=0):
    rng = np.random.default_rng(seed)
    n_micro = 2 ** order
    A = np.array([-0.4, 0.4])
    B = rng.permutation(np.linspace(-1.5, 1.5, n_micro))
    # micro from a fixed order-`order` Markov chain, independent of slow
    T = rng.uniform(0.3, 0.7, size=(n_micro, 2))
    T = T / T.sum(axis=1, keepdims=True)
    bits, ss, ms = [], [], []
    m = 0
    for _ in range(n):
        s = 1 if rng.random() < p_rare else 0
        p = 1 / (1 + np.exp(-(A[s] + B[m])))
        b = 1 if rng.random() < p else 0
        bits.append(b); ss.append(s); ms.append(m)
        m = ((m << 1) | b) & (n_micro - 1)
        # occasionally reseed micro from the chain to keep it mixing
        if rng.random() < 0.02:
            m = rng.integers(0, n_micro)
    return np.array(bits), np.array(ss), np.array(ms), n_micro


def run_independent(bits, ss, ms, n_pairs, burn=1000):
    counts = np.zeros((n_pairs, 2))
    losses, burn_losses, rare = [], [], []
    for t, (b, s, m) in enumerate(zip(bits, ss, ms)):
        pid = s * (n_pairs // 2) + m
        tot = counts[pid].sum()
        p = counts[pid, 1] / tot if tot > 0 else 0.5
        p = min(max(p, 1e-6), 1 - 1e-6)
        l = -np.log2(p if b else 1 - p)
        (losses if t >= burn else burn_losses).append(l)
        if s == 1 and t >= burn:
            rare.append(l)
        counts[pid, b] += 1
    return float(np.mean(losses)), float(np.mean(burn_losses)), float(np.mean(rare))


def run_factorized(bits, ss, ms, n_micro, burn=1000):
    fm = FactorModel(2, n_micro, lr=0.3)
    losses, burn_losses, rare = [], [], []
    for t, (b, s, m) in enumerate(zip(bits, ss, ms)):
        p = fm.predict(s, m)
        l = -np.log2(p if b else 1 - p)
        (losses if t >= burn else burn_losses).append(l)
        if s == 1 and t >= burn:
            rare.append(l)
        fm.observe(s, int(m), int(b))
    return float(np.mean(losses)), float(np.mean(burn_losses)), float(np.mean(rare))


def run_population(bits, ss, ms, n_micro, burn=1000):
    n_pairs = 2 * n_micro
    pop = Population(n_pairs, 2, n_micro, key_fn=lambda h: (0, 0, 0), dephase=1.0)
    losses, burn_losses, rare = [], [], []
    for t, (b, s, m) in enumerate(zip(bits, ss, ms)):
        pid = s * n_micro + m
        node = pop.nodes[pid]
        p_node = node.predict(m)
        p_leg = pop.legend.predict(s)
        w = node.n / (node.n + 8.0)
        p = min(max(w * p_node + (1 - w) * p_leg, 1e-6), 1 - 1e-6)
        l = -np.log2(p if b else 1 - p)
        (losses if t >= burn else burn_losses).append(l)
        if s == 1 and t >= burn:
            rare.append(l)
        node.observe(m, b)
        pop.legend.observe(s, b)
    return float(np.mean(losses)), float(np.mean(burn_losses)), float(np.mean(rare))


def main():
    out = {"protocol": "phase3a_factors", "seeds": {}}
    t0 = time.time()
    for seed in (7, 8, 9):
        bits, ss, ms, n_micro = gen_bits(100_000, seed=seed)
        i, ib, ir = run_independent(bits, ss, ms, 2 * n_micro)
        f, fb, fr = run_factorized(bits, ss, ms, n_micro)
        p, pb, pr = run_population(bits, ss, ms, n_micro)
        out["seeds"][seed] = {
            "independent": [round(i, 4), round(ir, 4)],
            "factorized": [round(f, 4), round(fr, 4)],
            "population": [round(p, 4), round(pr, 4)],
            "burnin": [round(ib, 4), round(fb, 4), round(pb, 4)],
        }
        print(f"seed {seed}: overall I={i:.4f} F={f:.4f} P={p:.4f} | "
              f"rare-zone I={ir:.4f} F={fr:.4f} P={pr:.4f}")
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase3a.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase3a.json")


if __name__ == "__main__":
    main()
