"""Phase 1b: data-starvation probe (loop iteration 2).

Gap from Phase 1: steady-state population == flat 8-context.  The population's
distinctive machinery (legend = cross-node sharing) should matter when nodes are
data-starved: rare (slow, micro) pairs where an independent context has almost no
counts but the legend's slow-marginal backoff still informs.

Protocol (written before the run):
- Source: regime_flip, 200k bits, seeds 7, 8, 9 (same as Phase 1).
- Micro orders 2, 4, 6 -> 8, 32, 128 (slow, micro) pairs.
- Models: flat monolithic table vs population (nodes + legend), same key.
- Metrics: overall bpb (excl. first 1000), burn-in bpb, bpb on the rarest quartile
  of pairs (by visit count) -- the starvation zone where the legend should win.
"""

import json
import time

import numpy as np

from proto.population import Population
from proto.sources import regime_flip, slow_key


def make_key_order(order, win=32):
    n_micro = 2 ** order
    def key_fn(bits):
        s = slow_key(bits, win)
        if len(bits) >= order:
            m = 0
            for b in bits[-order:]:
                m = (m << 1) | int(b)
        else:
            m = 0
        return s * n_micro + m, s, m
    return key_fn, 2 * n_micro, n_micro


def run_flat_starve(bits, n_pairs, key_fn, burn=1000):
    counts = np.zeros((n_pairs, 2))
    visits = np.zeros(n_pairs, dtype=int)
    losses, burn_losses = [], []
    pair_of_t = []
    for t, b in enumerate(bits):
        if t >= 6:
            pid, _, _ = key_fn(bits[:t])
            pair_of_t.append(pid)
            tot = counts[pid].sum()
            p = counts[pid, 1] / tot if tot > 0 else 0.5
            p = min(max(p, 1e-6), 1 - 1e-6)
            l = -np.log2(p if b else 1 - p)
            (losses if t >= burn else burn_losses).append(l)
            counts[pid, b] += 1
            visits[pid] += 1
    return losses, burn_losses, visits, pair_of_t


def run_pop_starve(bits, n_pairs, n_micro, key_fn, burn=1000):
    pop = Population(n_pairs, 2, n_micro, key_fn, dephase=1.0)
    losses, burn_losses = [], []
    hist = []
    for t, b in enumerate(bits):
        b = int(b)
        if t >= 6:
            l = pop.step(hist, b)
            (losses if t >= burn else burn_losses).append(l)
        else:
            # warm the history without scoring
            pid, s, m = key_fn(hist)
            pop.nodes[pid].observe(m, b)
            pop.legend.observe(s, b)
        hist.append(b)
    return losses, burn_losses


def main():
    out = {"protocol": "phase1b_starvation", "orders": {}}
    t0 = time.time()
    for order in (2, 4, 6):
        key_fn, n_pairs, n_micro = make_key_order(order)
        agg = {"flat": [], "flat_burn": [], "pop": [], "pop_burn": [],
               "flat_rare": [], "pop_rare": []}
        for seed in (7, 8, 9):
            bits, _ = regime_flip(200_000, seed=seed)
            fl, flb, visits, pair_of_t = run_flat_starve(bits, n_pairs, key_fn)
            pl, plb = run_pop_starve(bits, n_pairs, n_micro, key_fn)
            # Rarest quartile of pairs by visit count -> starvation zone.
            thresh = np.quantile(visits, 0.25)
            rare = set(np.where(visits <= thresh)[0])
            fl_rare = [l for l, pid in zip(fl, pair_of_t) if pid in rare]
            # population losses align with fl losses (same t range)
            pl_rare = [l for l, pid in zip(pl, pair_of_t) if pid in rare]
            agg["flat"].append(float(np.mean(fl)))
            agg["flat_burn"].append(float(np.mean(flb)))
            agg["pop"].append(float(np.mean(pl)))
            agg["pop_burn"].append(float(np.mean(plb)))
            agg["flat_rare"].append(float(np.mean(fl_rare)))
            agg["pop_rare"].append(float(np.mean(pl_rare)))
        row = {k: round(float(np.mean(v)), 4) for k, v in agg.items()}
        out["orders"][order] = row
        print(f"order {order} ({n_pairs} pairs): flat={row['flat']} pop={row['pop']} "
              f"| rare: flat={row['flat_rare']} pop={row['pop_rare']} "
              f"| burn-in: flat={row['flat_burn']} pop={row['pop_burn']}")
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase1b.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase1b.json")


if __name__ == "__main__":
    main()
