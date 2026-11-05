"""Phase 5b: H5 x H2 latent navigation (the transform becomes load-bearing).

Phase 5a's transform was implemented but not decisive (choice made at scope
time). Here validity depends on a LATENT regime, so transform = re-inference
per emission is load-bearing: apply the wrong regime's constraint and the
sequence is invalid.

Source: 12-bit sequences. Latent R in {0,1}, P(R=0)=0.6.
  R=0: prefix bits i.i.d. p=0.7; valid iff sum(full)%3==0.
  R=1: prefix bits i.i.d. p=0.3; valid iff sum(full)%4==0.
Given a 6-bit prefix, complete 6 bits.

Methods:
  greedy: sample completion i.i.d. from mixture marginal (0.54).
  nav_plain: scope completions valid under R=0 (majority), thread max-prob
             under p=0.7. No re-inference.
  nav_transform: TRANSFORM = exact Bayes infer R from prefix (binomial
             likelihoods + priors); scope completions valid under inferred R;
             thread max-prob under p_R.

Protocol (written before the run): 2000 prefixes, seed 7. Expectation:
validity greedy < nav_plain (~0.6, fails R=1) < nav_transform (~1.0).
If nav_transform does not beat nav_plain, the H5xH2 fusion claim fails.
"""

import json
import time
from itertools import product

import numpy as np
from scipy.special import comb


P0, P1 = 0.7, 0.3          # p(1) per regime
PRIOR0 = 0.6
K = {0: 3, 1: 4}           # validity modulus per regime
N_PRE, N_POST = 6, 6


def infer_regime(prefix):
    """Exact Bayes: P(R|prefix) via binomial likelihoods."""
    n1 = int(prefix.sum())
    n = len(prefix)
    l0 = PRIOR0 * comb(n, n1) * P0**n1 * (1 - P0)**(n - n1)
    l1 = (1 - PRIOR0) * comb(n, n1) * P1**n1 * (1 - P1)**(n - n1)
    return 0 if l0 >= l1 else 1


def best_completion(prefix, regime):
    """Scope: completions valid under regime's constraint. Thread: max prob."""
    p = P0 if regime == 0 else P1
    k = K[regime]
    base = int(prefix.sum())
    best, best_lp = None, -1e18
    for bits in product((0, 1), repeat=N_POST):
        if (base + sum(bits)) % k != 0:
            continue
        lp = sum(np.log(p) if b else np.log(1 - p) for b in bits)
        if lp > best_lp:
            best_lp, best = lp, bits
    return np.array(best), best_lp


def run(seed=7, n=2000):
    rng = np.random.default_rng(seed)
    p_mix = PRIOR0 * P0 + (1 - PRIOR0) * P1
    res = {m: {"valid": 0, "lp": []} for m in ("greedy", "nav_plain", "nav_transform")}
    agree = 0
    for _ in range(n):
        r = 0 if rng.random() < PRIOR0 else 1
        p_true = P0 if r == 0 else P1
        prefix = (rng.random(N_PRE) < p_true).astype(int)
        # greedy
        g = (rng.random(N_POST) < p_mix).astype(int)
        full = np.concatenate([prefix, g])
        res["greedy"]["valid"] += (int(full.sum()) % K[r] == 0)
        res["greedy"]["lp"].append(float(np.sum(np.where(g, np.log(p_true), np.log(1 - p_true)))))
        # nav_plain (assume R=0)
        c, lp = best_completion(prefix, 0)
        full = np.concatenate([prefix, c])
        res["nav_plain"]["valid"] += (int(full.sum()) % K[r] == 0)
        # logprob under TRUE model: recompute with p_true
        lp_true = float(np.sum(np.where(c, np.log(p_true), np.log(1 - p_true))))
        res["nav_plain"]["lp"].append(lp_true)
        # nav_transform (infer R, then navigate)
        r_hat = infer_regime(prefix)
        agree += (r_hat == r)
        c, lp = best_completion(prefix, r_hat)
        full = np.concatenate([prefix, c])
        res["nav_transform"]["valid"] += (int(full.sum()) % K[r] == 0)
        lp_true = float(np.sum(np.where(c, np.log(p_true), np.log(1 - p_true))))
        res["nav_transform"]["lp"].append(lp_true)
    out = {"n": n, "seed": seed, "regime_agreement": round(agree / n, 4)}
    for m, d in res.items():
        out[m] = {"validity": round(d["valid"] / n, 4),
                  "mean_logprob": round(float(np.mean(d["lp"])), 3)}
    return out


def main():
    t0 = time.time()
    out = {"protocol": "phase5b_latent_navigation"}
    out.update(run())
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase5b.json", "w") as f:
        json.dump(out, f, indent=2)
    for m in ("greedy", "nav_plain", "nav_transform"):
        d = out[m]
        print(f"{m:13s} validity={d['validity']:.3f} mean_logprob={d['mean_logprob']:.2f}")
    print(f"regime agreement: {out['regime_agreement']:.3f}")
    print("saved runs/results_phase5b.json")


if __name__ == "__main__":
    main()
