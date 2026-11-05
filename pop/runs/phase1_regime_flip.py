"""Phase 1 experiment: population of Page-sized density matrices (H1)
on the regime-flip source, vs flat baselines.

Protocol (written before the run):
- Source: regime_flip, 200k bits, p_flip=0.004, seeds 7, 8, 9.
- Models:
    A: flat order-2 counting table (single scale -- the extraction's analog).
    B: flat 8-context table (slow x micro, monolithic -- isolates the slow key).
    C: population, 8 blanket-insulated nodes + legend (H1 machinery).
- Metrics: prequential bits/bit excluding first 1000 (burn-in reported separately);
  per-node Page ranks at end; prediction drift across quartiles (H3 stability);
  legend reliance over time.
- Honesty note: if B ~= C, the win is the slow key, not population-ness; the
  population's distinctive value (legend sharing under data starvation) is then
  probed with a 32-node scale variant.
"""

import json
import time

import numpy as np

from proto.population import Population
from proto.sources import regime_flip, make_key


def run_flat(bits, n_ctx, ctx_fn, burn=1000):
    counts = np.zeros((n_ctx, 2))
    losses, burn_losses = [], []
    for t, b in enumerate(bits):
        if t >= 2:
            c = ctx_fn(bits, t)
            tot = counts[c].sum()
            p = counts[c, 1] / tot if tot > 0 else 0.5
            p = min(max(p, 1e-6), 1 - 1e-6)
            l = -np.log2(p if b else 1 - p)
            (losses if t >= burn else burn_losses).append(l)
            counts[c, b] += 1
    return float(np.mean(losses)), float(np.mean(burn_losses))


def ctx_flat2(bits, t):
    return int(bits[t - 2]) * 2 + int(bits[t - 1])


def make_ctx_slow_micro(win=32):
    def ctx(bits, t):
        s = int(np.mean(bits[max(0, t - win):t]) >= 0.5) if t > 0 else 0
        m = int(bits[t - 2]) * 2 + int(bits[t - 1])
        return s * 4 + m
    return ctx


def run_population(bits, burn=1000, win=32):
    key_fn = make_key(win)
    pop = Population(8, 2, 4, key_fn, dephase=1.0)
    losses, burn_losses = [], []
    hist = []
    snaps = {}
    for t, b in enumerate(bits):
        b = int(b)
        l = pop.step(hist, b)
        (losses if t >= burn else burn_losses).append(l)
        hist.append(b)
        frac = t / len(bits)
        for q in (0.25, 0.5, 0.75, 1.0):
            if frac >= q and q not in snaps:
                snaps[q] = pop.drift_snapshot()
    # Drift: mean abs change of per-node predictions between quartile snapshots.
    keys = sorted(snaps)
    drifts = []
    for a, bq in zip(keys[:-1], keys[1:]):
        d = np.mean([abs(np.array(snaps[bq][i]) - np.array(snaps[a][i])).mean()
                     for i in snaps[a]])
        drifts.append(float(d))
    return (float(np.mean(losses)), float(np.mean(burn_losses)),
            pop.ranks(), drifts)


def main():
    out = {"protocol": "phase1_regime_flip", "seeds": {}}
    t0 = time.time()
    for seed in (7, 8, 9):
        bits, _ = regime_flip(200_000, seed=seed)
        a, a_burn = run_flat(bits, 4, ctx_flat2)
        b, b_burn = run_flat(bits, 8, make_ctx_slow_micro())
        c, c_burn, ranks, drifts = run_population(bits)
        out["seeds"][seed] = {
            "flat_order2_bpb": round(a, 4),
            "flat_order2_burnin_bpb": round(a_burn, 4),
            "flat_8ctx_bpb": round(b, 4),
            "flat_8ctx_burnin_bpb": round(b_burn, 4),
            "population_bpb": round(c, 4),
            "population_burnin_bpb": round(c_burn, 4),
            "node_page_ranks": ranks,
            "quartile_drifts": [round(d, 4) for d in drifts],
        }
        print(f"seed {seed}: flat2={a:.3f} flat8={b:.3f} pop={c:.3f} "
              f"(burn-in {a_burn:.3f}/{b_burn:.3f}/{c_burn:.3f}) drifts={drifts}")
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase1.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase1.json")


if __name__ == "__main__":
    main()
