"""True disentangling (H2 x H4, Phase 4c).

Phase 4b's lesson: shape transfer fails when shape is entangled with mean and
scale. The disentangler's job is to SEPARATE invariant from variant.

Disentangled representation of a slot's table p(1|micro):
    p(1|m) = sigmoid(alpha + beta * s[m])
where s = zero-mean, unit-std SHAPE (the invariant), alpha = mean (logit),
beta = scale (amplitude).  Radiate s only.  Transfer = SGD on (alpha, beta)
with s FIXED: 2 params vs 32.  Sign ambiguity (s,beta) <-> (-s,-beta) resolves
via SGD (beta can go negative) -- no shootout needed.

This is the classical shadow of H4's "unitary disentanglers": separate the
degrees of freedom into shared (radiated) and idiosyncratic (relearned).
"""

import numpy as np


def disentangle(counts, prior=1.0, eps=1e-6):
    """Factor a (M,2) count table into (alpha, beta, shape)."""
    p = (counts[:, 1] + prior) / (counts.sum(axis=1) + 2 * prior)
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    alpha = float(logit.mean())
    beta = float(logit.std())
    beta = max(beta, 0.1)
    shape = (logit - alpha) / beta
    return alpha, beta, shape


class TransferSlot:
    """A slot with RADIATED shape; learns only (alpha, beta) via SGD."""

    def __init__(self, shape, lr=0.3):
        self.s = np.asarray(shape, dtype=float)
        self.alpha = 0.0
        self.beta = 1.0
        self.lr = lr
        self.n = 0

    def predict(self, m):
        z = self.alpha + self.beta * self.s[m]
        p = 1.0 / (1.0 + np.exp(-z))
        return min(max(p, 1e-6), 1 - 1e-6)

    def observe(self, m, bit):
        p = self.predict(m)
        err = bit - p
        lr = self.lr / (1 + self.n / 1000.0) + 0.02
        self.alpha += lr * err
        self.beta += lr * err * self.s[m]
        self.n += 1
