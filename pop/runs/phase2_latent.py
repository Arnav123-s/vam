"""Phase 2 experiment: latent regime inference vs noisy majority key.

Protocol (written before the run):
- Source: regime_flip, 200k bits, p_flip=0.004, seeds 7, 8, 9.
- Model: population, 8 nodes (slow x micro-2) + legend, dephase=1.0.
- Keys: (a) majority-32 (Phase 1 baseline); (b) majority for the first 2000 bits
  (bootstrap that trains the legend and breaks the label symmetry), then the
  latent Bayes filter fed by the legend's p(bit|slow) prequentially.
- Metrics: bpb excl. first 1000; key-vs-true-regime agreement; burn-in bpb.
- Expectation: agreement 94% -> ~98%+, bpb 0.753 -> ~0.67.
- Lesson logged 2026-11-04: from a symmetric start (odds=1, untrained legend) the
  filter never moves -- likelihood ratio stays ~1, agreement 50%, bpb 1.0.
  Latent inference needs a bootstrap or symmetry-break; it cannot lift itself
  by its own posteriors.
"""

import json
import time

import numpy as np

from proto.population import Population
from proto.slow import RegimeTracker
from proto.sources import regime_flip, slow_key


def run(bits, regimes, latent, burn=1000, bootstrap=2000):
    n_micro = 4
    # stateful slow inference; micro from raw history as before
    pop = Population(8, 2, n_micro, key_fn=lambda h: (0, 0, 0), dephase=1.0)
    tracker = RegimeTracker(p_flip=0.004)
    losses, burn_losses, agree = [], [], []
    hist = []
    for t, b in enumerate(bits):
        b = int(b)
        m = int(hist[-2]) * 2 + int(hist[-1]) if len(hist) >= 2 else 0
        if latent and t >= bootstrap:
            s = tracker.predict()
        else:
            # Majority key: baseline, and the bootstrap that trains the legend
            # before the latent tracker takes over (breaks the symmetry).
            s = slow_key(hist)
        agree.append((s == 1) == (regimes[t] == 0))
        pid = s * n_micro + m
        node = pop.nodes[pid]
        p_node = node.predict(m)
        p_leg = pop.legend.predict(s)
        w = node.n / (node.n + 8.0)
        p = min(max(w * p_node + (1 - w) * p_leg, 1e-6), 1 - 1e-6)
        l = -np.log2(p if b else 1 - p)
        (losses if t >= burn else burn_losses).append(l)
        node.observe(m, b)
        pop.legend.observe(s, b)
        if latent and t >= bootstrap:
            tracker.update(b, pop.legend.predict(0), pop.legend.predict(1))
        if latent and t == bootstrap:
            # Freeze the legend: it becomes the immutable likelihood model
            # (bedrock, H3) that the tracker infers against.  Nodes stay plastic.
            pop.legend.frozen = True
        hist.append(b)
    return float(np.mean(losses)), float(np.mean(burn_losses)), float(np.mean(agree))


def main():
    out = {"protocol": "phase2_latent_key", "seeds": {}}
    t0 = time.time()
    for seed in (7, 8, 9):
        bits, regimes = regime_flip(200_000, seed=seed)
        a, a_burn, a_agr = run(bits, regimes, latent=False)
        b, b_burn, b_agr = run(bits, regimes, latent=True)
        out["seeds"][seed] = {
            "majority_bpb": round(a, 4), "majority_burnin": round(a_burn, 4),
            "majority_agree": round(a_agr, 4),
            "latent_bpb": round(b, 4), "latent_burnin": round(b_burn, 4),
            "latent_agree": round(b_agr, 4),
        }
        print(f"seed {seed}: majority bpb={a:.4f} agree={a_agr:.4f} | "
              f"latent bpb={b:.4f} agree={b_agr:.4f}")
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase2.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase2.json")


if __name__ == "__main__":
    main()
