"""Phase 18: the faithful present-state learner.

The divergence: the track built batch count tables (growing history), but
VAM specifies a FIXED-SIZE present state updated by an online event loop
(T10.1, T12.1). This tests whether that distinction matters.

Machine:
  - State: fixed-size weight matrix W[context_hash, char] -- the state IS
    the memory. Size does not grow with data (hashing trick, fixed buckets).
  - Prediction: softmax over W[context] -- from current state only.
  - Plasticity (T12.1 shadow): on each observed char, W += eta * (onehot -
    predicted) * 1 -- correlation law, from the external bit only.
  - No n-gram tables, no stored history, no replay. Pure online learning.
Compare on the same 150KB P&P split as Phase 11:
  - count-based population (Phase 11): 2.75
  - flat5: 2.81
  - present-state learner: ???
If the present-state learner is competitive with the population, the idea
was misunderstood and the scale limit was my artifact. If much worse, the
count tables were doing the heavy lifting.
"""

import json
import time

import numpy as np

from runs.phase7b_realtext import load_corpus


class PresentStateLearner:
    """Fixed-size state, online plasticity. No growing tables."""

    def __init__(self, n_buckets=4096, ctx_len=4, eta=0.1, alphabet="abcdefghijklmnopqrstuvwxyz "):
        self.n_buckets = n_buckets
        self.ctx_len = ctx_len
        self.eta = eta
        self.alpha = list(alphabet)
        self.a2i = {c: i for i, c in enumerate(self.alpha)}
        self.n_chars = len(self.alpha)
        # STATE: fixed-size weight matrix. This is the whole memory.
        self.W = np.zeros((n_buckets, self.n_chars))

    def _bucket(self, ctx):
        h = 2166136261
        for ch in ctx:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        return h % self.n_buckets

    def predict_dist(self, hist):
        ctx = hist[-self.ctx_len:]
        b = self._bucket(ctx)
        logits = self.W[b]
        # softmax
        m = logits.max()
        e = np.exp(logits - m)
        p = e / e.sum()
        return {self.alpha[i]: float(p[i]) for i in range(self.n_chars)}

    def learn(self, hist, ch):
        """Plasticity: correlation law from the observed char only."""
        if ch not in self.a2i:
            return
        ctx = hist[-self.ctx_len:]
        b = self._bucket(ctx)
        p = self.predict_dist(hist)
        for i, c in enumerate(self.alpha):
            target = 1.0 if c == ch else 0.0
            self.W[b, i] += self.eta * (target - p[c])

    def bpb(self, text, train_first=True):
        lps = []
        hist = ""
        for ch in text:
            if ch not in self.a2i:
                continue
            d = self.predict_dist(hist)
            lps.append(-np.log2(max(d.get(ch, 1e-6), 1e-6)))
            self.learn(hist, ch)
            hist = (hist + ch)[-self.ctx_len:]
        return float(np.mean(lps))


def main():
    t0 = time.time()
    train, test = load_corpus(n_train=150000, n_test=20000)
    out = {"protocol": "phase18_present_state"}
    for buckets, eta in [(4096, 0.1), (16384, 0.1), (4096, 0.3), (16384, 0.3)]:
        m = PresentStateLearner(n_buckets=buckets, eta=eta)
        # online: train then test (state carries over, as VAM specifies)
        m.bpb(train)  # learn from train (online, no separate fit)
        pb = m.bpb(test)
        out[f"buckets_{buckets}_eta_{eta}"] = round(pb, 4)
        print(f"buckets={buckets} eta={eta}: test bpb {pb:.4f}")
    out["phase11_population_bpb"] = 2.7506
    out["phase11_flat5_bpb"] = 2.8111
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase18.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase18.json")


if __name__ == "__main__":
    main()
