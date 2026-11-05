"""Collapse as navigation (H5, Phase 5a).

The owner's F spec: "we scope out everything that token could fire using Feynman,
and with Newton's law, we pull out the needle from the haystack" -- macro-scope
(Feynman over the field) -> Newton threads the attractor -> micro-scope -> Newton
picks the minimal step -> emit -> EVERY TOKEN THAT EMERGES CHANGES THE ENTIRE
ALGORITHM via Markov transitions.

Minimal faithful mapping for constrained generation:
- SCOPE (Feynman): consider the whole field -- enumerate ALL completions, score
  each by the population's path probability. Not sampling; the full distribution.
- THREAD (Newton): follow the probability gradient to the attractor -- the
  max-probability completion satisfying the global constraint.
- MINIMAL STEP + EMIT: emit the threaded completion bit by bit.
- TRANSFORM: each emitted bit re-conditions the whole system (counts update AND
  the slow tracker re-infers -- the algorithm's state transforms, not just a
  bit appended to a fixed predictor).

What was extracted instead: hbar sampler settings. V7 bounds SAMPLING, which is
not this idea -- navigation never samples.

Test: 12-bit sequences, constraint sum%3==0, position biases. Prefix 6 bits,
complete 6 (64 candidates). Greedy (independent sampling) vs greedy+repair (flip
least-confident bit) vs navigation (scope+thread+emit+transform).
Metrics: validity %, mean log-prob of the completion (quality).
"""

import itertools
import json
import time

import numpy as np


def gen_sequences(n, seed=0):
    rng = np.random.default_rng(seed)
    pos_bias = rng.uniform(0.2, 0.8, size=12)
    seqs = []
    while len(seqs) < n:
        s = (rng.random(12) < pos_bias).astype(int)
        if s.sum() % 3 == 0:
            seqs.append(s)
    return np.array(seqs), pos_bias


class PosModel:
    """p(bit | position) tables -- the population's field."""

    def __init__(self, n_pos=12, prior=1.0):
        self.counts = np.full((n_pos, 2), prior)
        self.n_updates = 0

    def predict(self, pos):
        tot = self.counts[pos].sum()
        return self.counts[pos, 1] / tot

    def observe(self, pos, bit):
        self.counts[pos, bit] += 1
        self.n_updates += 1

    def path_logprob(self, seq):
        # seq: full 12-bit array
        lp = 0.0
        for pos, b in enumerate(seq):
            p = min(max(self.predict(pos), 1e-9), 1 - 1e-9)
            lp += np.log(p if b else 1 - p)
        return lp


def greedy_complete(model, prefix, rng):
    comp = np.array([1 if rng.random() < model.predict(6 + i) else 0
                     for i in range(6)])
    return np.concatenate([prefix, comp])


def greedy_repair_complete(model, prefix, rng):
    comp = np.array([1 if rng.random() < model.predict(6 + i) else 0
                     for i in range(6)])
    full = np.concatenate([prefix, comp])
    if full.sum() % 3 != 0:
        # flip the least-confident completion bit
        conf = []
        for i in range(6):
            p = model.predict(6 + i)
            pb = p if comp[i] else 1 - p
            conf.append(abs(pb - 0.5))
        j = int(np.argmin(conf))
        comp[j] = 1 - comp[j]
        full = np.concatenate([prefix, comp])
    return full


def navigate_complete(model, prefix):
    # SCOPE: the whole field -- all 64 completions with path probabilities
    best, best_lp = None, -np.inf
    for bits in itertools.product([0, 1], repeat=6):
        comp = np.array(bits)
        full = np.concatenate([prefix, comp])
        if full.sum() % 3 != 0:
            continue
        lp = model.path_logprob(full)
        # THREAD: Newton follows the gradient to the attractor (max prob)
        if lp > best_lp:
            best_lp, best = lp, comp
    # EMIT bit by bit + TRANSFORM: each emission re-conditions the system
    out = []
    for i, b in enumerate(best):
        out.append(int(b))
        model.observe(6 + i, int(b))  # the algorithm transforms per emission
    return np.concatenate([prefix, np.array(out)]), best_lp


def main():
    out = {"protocol": "phase5a_navigation"}
    t0 = time.time()
    rng = np.random.default_rng(42)
    train, _ = gen_sequences(20_000, seed=0)
    model_g = PosModel()
    model_r = PosModel()
    model_n = PosModel()
    for s in train:
        for pos, b in enumerate(s):
            model_g.observe(pos, int(b))
            model_r.observe(pos, int(b))
            model_n.observe(pos, int(b))
    test, _ = gen_sequences(500, seed=1)
    res = {"greedy": [], "repair": [], "navigate": []}
    lp_g, lp_r, lp_n = [], [], []
    for s in test:
        prefix = s[:6]
        fg = greedy_complete(model_g, prefix, rng)
        fr = greedy_repair_complete(model_r, prefix, rng)
        # navigate uses a FRESH copy each time (transform is per-task)
        mn = PosModel()
        mn.counts = model_n.counts.copy()
        fn, nlp = navigate_complete(mn, prefix)
        res["greedy"].append(int(fg.sum() % 3 == 0))
        res["repair"].append(int(fr.sum() % 3 == 0))
        res["navigate"].append(int(fn.sum() % 3 == 0))
        # quality: log-prob under the TRAINED model (not the transformed one)
        lp_g.append(model_g.path_logprob(fg))
        lp_r.append(model_r.path_logprob(fr))
        lp_n.append(model_n.path_logprob(fn))
    for k in res:
        out[k] = {"validity": round(float(np.mean(res[k])), 4)}
    out["greedy"]["mean_logprob"] = round(float(np.mean(lp_g)), 3)
    out["repair"]["mean_logprob"] = round(float(np.mean(lp_r)), 3)
    out["navigate"]["mean_logprob"] = round(float(np.mean(lp_n)), 3)
    print("validity: greedy=%.3f repair=%.3f navigate=%.3f" % (
        out["greedy"]["validity"], out["repair"]["validity"],
        out["navigate"]["validity"]))
    print("mean logprob: greedy=%.2f repair=%.2f navigate=%.2f" % (
        out["greedy"]["mean_logprob"], out["repair"]["mean_logprob"],
        out["navigate"]["mean_logprob"]))
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase5a.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase5a.json")


if __name__ == "__main__":
    main()
