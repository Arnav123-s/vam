"""Phase 8: quantum density-matrix tracker -- coherence, for real.

The classical shadows (Phases 1-7) used probability vectors. VAM's actual
claim uses DENSITY MATRICES: off-diagonal elements (coherence) encode
superposition. The question: does coherence buy anything on classical data?

Model: 2x2 density matrix rho for the latent regime.
  - Diagonal: regime probabilities (like classical).
  - Off-diagonal: coherence (superposition amplitude).
  - Evolution: unitary mixing U (models regime flip) + Kraus measurement
    update on each observed bit.
  - Prediction: Born rule -- p(regime) = diagonal elements.

Kraus update for bit b with likelihoods p(b|0), p(b|1):
  K_b = diag(sqrt(p(b|0)), sqrt(p(b|1)))
  rho -> K_b rho K_b^dagger / Tr(K_b rho K_b^dagger)
The diagonal evolves like Bayes; the coherence scales by
sqrt(p(b|0) p(b|1)) and feeds future evolution through U. Classical Bayes
has no coherence term -- that's the difference.

Comparison (regime_flip, Phase 2 setup):
  classical: exact Bayes tracker (Phase 2).
  quantum: density-matrix tracker above.
  Both use the same frozen legend likelihoods (fair).
  Test at full training and at SHORT training (500 bits) -- coherence as a
  prior might help sample efficiency.

Protocol (written before the run): seeds 7,8. Expectation (honest): tie at
full data (Bayes is optimal on classical data). The question is whether
coherence helps at small data, or is just overhead. If quantum ties
everywhere, coherence is not the differentiator on classical data.
"""

import json
import time

import numpy as np

from proto.sources import regime_flip
from proto.slow import RegimeTracker


class QuantumTracker:
    """2x2 density matrix regime tracker."""

    def __init__(self, p_flip=0.004):
        f = p_flip
        # unitary mixing: rotation by flip probability
        c, s = np.sqrt(1 - f), np.sqrt(f)
        self.U = np.array([[c, s], [s, c]], dtype=complex)
        # start: maximally coherent superposition |+><+|
        self.rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)

    def predict(self):
        """p(regime=1) = rho[1,1] (Born rule). Regime estimate."""
        return 1 if self.rho[1, 1].real > 0.5 else 0

    def p1(self):
        return float(self.rho[1, 1].real)

    def update(self, bit, p1_0, p1_1):
        # unitary evolution (flip mixing)
        self.rho = self.U @ self.rho @ self.U.conj().T
        # Kraus measurement update
        pb0 = p1_0 if bit == 1 else 1 - p1_0
        pb1 = p1_1 if bit == 1 else 1 - p1_1
        K = np.diag([np.sqrt(pb0), np.sqrt(pb1)]).astype(complex)
        self.rho = K @ self.rho @ K.conj().T
        self.rho /= np.trace(self.rho).real
        # keep Hermitian, trace 1
        self.rho = (self.rho + self.rho.conj().T) / 2


def run(seed=7, n_train=50000, n_test=10000, bootstrap=2000):
    bits, regimes = regime_flip(n_train + n_test, seed=seed)
    train, test = bits[:n_train], bits[n_train:]
    # frozen legend from training (majority key) + tables keyed the SAME way
    # (consistent labeling -- the bug was tables keyed by TRUE regime while
    # legend used majority key, which is flipped: majority=1 means high mean
    # = TRUE regime 0).
    legend = np.ones((2, 2))
    tables = np.ones((2, 4, 2))
    for t in range(2, n_train):
        s = int(np.mean(train[max(0, t-32):t]) >= 0.5)
        legend[s, int(train[t])] += 1
        m = (int(train[t-2]) << 1) | int(train[t-1])
        tables[s, m, int(train[t])] += 1
    p1_0 = legend[0, 1] / legend[0].sum()
    p1_1 = legend[1, 1] / legend[1].sum()

    def eval_tracker(tracker, is_quantum):
        lps = []
        # reset tracker state
        if is_quantum:
            tracker.rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        else:
            tracker.odds = 1.0
        for t in range(2, len(test)):
            m = (int(test[t-2]) << 1) | int(test[t-1])
            s = tracker.predict()
            p = tables[s, m, 1] / tables[s, m].sum()
            p = min(max(p, 1e-6), 1 - 1e-6)
            b = int(test[t])
            lps.append(-np.log2(p if b else 1 - p))
            if is_quantum:
                tracker.update(b, p1_0, p1_1)
            else:
                tracker.update(b, p1_0, p1_1)
        return float(np.mean(lps))

    ct = RegimeTracker()
    qt = QuantumTracker()
    return {
        "classical_bpb": round(eval_tracker(ct, False), 4),
        "quantum_bpb": round(eval_tracker(qt, True), 4),
    }


def main():
    t0 = time.time()
    out = {"protocol": "phase8_quantum_tracker", "full": {}, "short": {}}
    for seed in (7, 8):
        r = run(seed, n_train=50000)
        out["full"][seed] = r
        print(f"full seed {seed}: classical={r['classical_bpb']:.4f} quantum={r['quantum_bpb']:.4f}")
    for seed in (7, 8):
        r = run(seed, n_train=500)
        out["short"][seed] = r
        print(f"short seed {seed}: classical={r['classical_bpb']:.4f} quantum={r['quantum_bpb']:.4f}")
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase8.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase8.json")


if __name__ == "__main__":
    main()
