"""Phase 6: integrated machine -- the pieces wired together.

Combines: population tables (H1) + latent regime tracker (H2/H3) + frozen
legend (H3 bedrock). The loop per bit: PERCEIVE (tracker infers regime) ->
PREDICT (regime's table) -> LEARN (update regime's table + tracker).

Source: regime_flip (slow flips p=0.004; order-2 tables REVERSED per regime).
Train 50k bits (tables + legend, then FREEZE legend), test 10k bits.

Methods:
  flat: single order-2 table (no regime). Expected ~1.0 (Phase 1).
  pop_majority: population keyed by noisy majority (Phase 1). Expected ~0.75.
  integrated: tracker-inferred regime -> regime's table. Expected ~0.72,
              approaching oracle.
  oracle: true regime -> regime's table. Expected ~0.66 (Phase 1).

Protocol (written before the run): seed 7. If integrated does not beat
pop_majority, the integration adds nothing. If it matches Phase 2's latent
(0.723), the pieces compose.
"""

import json
import time

import numpy as np

from proto.sources import regime_flip
from proto.slow import RegimeTracker


class IntegratedMachine:
    def __init__(self, p_flip=0.004, bootstrap=2000):
        # 2 regimes x 4 micros (order-2) x 2 counts
        self.tables = np.ones((2, 4, 2))
        # legend: p(1 | slow) for the tracker likelihoods; frozen after bootstrap
        self.legend = np.ones((2, 2))
        self.tracker = RegimeTracker(p_flip=p_flip)
        self.bootstrap = bootstrap
        self.n = 0
        self.frozen = False

    def _micro(self, bits):
        return (int(bits[-2]) << 1) | int(bits[-1]) if len(bits) >= 2 else 0

    def step(self, bits, bit=None):
        """If bit is None: predict. Else: learn. Returns p(1) prediction."""
        m = self._micro(bits)
        if not self.frozen:
            # bootstrap: use majority key
            s = int(np.mean(bits[-32:]) >= 0.5) if len(bits) >= 32 else 0
        else:
            s = self.tracker.predict()
        p = self.tables[s, m, 1] / self.tables[s, m].sum()
        p = min(max(p, 1e-6), 1 - 1e-6)
        if bit is not None:
            self.tables[s, m, bit] += 1
            if not self.frozen:
                self.legend[s, bit] += 1
            else:
                p1_0 = self.legend[0, 1] / self.legend[0].sum()
                p1_1 = self.legend[1, 1] / self.legend[1].sum()
                self.tracker.update(bit, p1_0, p1_1)
            self.n += 1
            if self.n >= self.bootstrap and not self.frozen:
                self.frozen = True
        return p


def bpb(bits, model_fn):
    lps = []
    for t in range(2, len(bits)):
        p = model_fn(bits[:t])
        b = int(bits[t])
        lps.append(-np.log2(p if b else 1 - p))
    return float(np.mean(lps))


def main():
    t0 = time.time()
    bits, regimes = regime_flip(60000, seed=7)
    train, test = bits[:50000], bits[50000:]
    test_reg = regimes[50000:]

    # train models
    flat_counts = np.ones((4, 2))
    pop_counts = np.ones((2, 4, 2))  # majority-keyed
    mach = IntegratedMachine()
    for t in range(2, len(train)):
        b = int(train[t])
        m = (int(train[t-2]) << 1) | int(train[t-1])
        flat_counts[m, b] += 1
        s_maj = int(np.mean(train[max(0, t-32):t]) >= 0.5)
        pop_counts[s_maj, m, b] += 1
        mach.step(train[:t], b)

    def p_flat(h):
        m = (int(h[-2]) << 1) | int(h[-1])
        p = flat_counts[m, 1] / flat_counts[m].sum()
        return min(max(p, 1e-6), 1 - 1e-6)

    def p_pop(h):
        m = (int(h[-2]) << 1) | int(h[-1])
        s = int(np.mean(h[-32:]) >= 0.5)
        p = pop_counts[s, m, 1] / pop_counts[s, m].sum()
        return min(max(p, 1e-6), 1 - 1e-6)

    def p_int(h):
        return mach.step(h)  # predict only

    # oracle: train regime tables on train with TRUE regimes
    orc_counts = np.ones((2, 4, 2))
    tr_reg = regimes[:50000]
    for t in range(2, len(train)):
        b = int(train[t])
        m = (int(train[t-2]) << 1) | int(train[t-1])
        orc_counts[int(tr_reg[t]), m, b] += 1

    def p_orc(h, t):
        # t = index into test; use true regime
        m = (int(h[-2]) << 1) | int(h[-1])
        s = int(test_reg[t])
        p = orc_counts[s, m, 1] / orc_counts[s, m].sum()
        return min(max(p, 1e-6), 1 - 1e-6)

    # evaluate on test (prequential for learned models; oracle uses true reg)
    lps = {k: [] for k in ("flat", "pop_majority", "integrated", "oracle")}
    for t in range(2, len(test)):
        h = test[:t]
        b = int(test[t])
        for k, fn in (("flat", p_flat), ("pop_majority", p_pop), ("integrated", p_int)):
            p = fn(h)
            lps[k].append(-np.log2(p if b else 1 - p))
        p = p_orc(h, t)
        lps["oracle"].append(-np.log2(p if b else 1 - p))
        # online update for learned models (fair: they keep learning)
        m = (int(test[t-2]) << 1) | int(test[t-1])
        flat_counts[m, b] += 1
        s_maj = int(np.mean(test[max(0, t-32):t]) >= 0.5)
        pop_counts[s_maj, m, b] += 1
        mach.step(h, b)

    out = {"protocol": "phase6_integrated", "seed": 7}
    for k, v in lps.items():
        out[k] = round(float(np.mean(v)), 4)
        print(f"{k:13s} {out[k]:.4f}")
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase6.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase6.json")


if __name__ == "__main__":
    main()
