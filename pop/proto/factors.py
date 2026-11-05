"""Shared cross-scale factors (H4, Phase 3a).

The owner's H4: "Coarse-grained macroscopic concepts entangle with fine-grained
microscopic features using unitary disentanglers and isometric coarse-graining
operators" -- a density matrix at one layer representing higher layers in
superposition (MERA-like).

Phase 3a tests the principle in its minimal classical shadow: the population's
conditional table p(bit | slow, micro) FACTORIZED through shared factors,
logit p(1|s,m) = A_s + B_m.  The micro factors B_m are "entangled" across slow
states -- learned mostly from the common regime, transferred to the rare one.
Independent leaves (32 params) vs factorized (18 params): the factorization IS
the coarse-graining operator (32 leaves -> shared factors).

Full unitary disentanglers on density matrices are Phase 3b; this phase asks
whether the *principle* (shared cross-scale structure beats independent nodes)
holds.  Online SGD on log-loss is just the learning rule, not the idea.
"""

import numpy as np


class FactorModel:
    """Rank-1 factorization of p(bit|slow, micro): logit = A_s + B_m."""

    def __init__(self, n_slow, n_micro, lr=0.1):
        self.A = np.zeros(n_slow)
        self.B = np.zeros(n_micro)
        self.lr = lr
        self.n = 0

    def predict(self, s, m):
        p = 1.0 / (1.0 + np.exp(-(self.A[s] + self.B[m])))
        return min(max(p, 1e-6), 1 - 1e-6)

    def observe(self, s, m, bit):
        p = self.predict(s, m)
        err = bit - p  # d(log-loss)/d(logit) = p - bit
        # Slightly higher lr early (burn-in), decaying to a floor.
        lr = self.lr / (1 + self.n / 2000.0) + 0.01
        self.A[s] += lr * err
        self.B[m] += lr * err
        self.n += 1
