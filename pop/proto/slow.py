"""Latent slow-state inference (Phase 2).

Gap from Phase 1/1b: the slow key (majority of last 32 bits) agrees with the true
regime only 94% of the time; the oracle test showed the remaining loss is almost
entirely key noise (0.753 -> 0.656 with a perfect key).

Fix: stop *observing* a noisy proxy and *infer* the latent regime with the exact
binary Bayes filter.  The Legend (overlying layer) doubles as the regime estimator:
it already tracks p(bit | slow).  Strict prequential order per bit:

    predict() -> slow_id from pre-bit odds
    update(bit, p1_0, p1_1): o_pred = ((1-f)*o + f)/(f*o + (1-f)); then multiply
        by the likelihood ratio OF THE ACTUAL BIT.

The Legend must be FROZEN before the tracker takes over (bedrock, H3): a latent
tracker that trains its own likelihood model on its own outputs collapses to a
degenerate fixed point (logged 2026-11-04).

Lessons logged 2026-11-04:
- From a symmetric start the filter never moves (likelihood ratio ~1 forever).
  Latent inference needs a bootstrap or symmetry-break.
- The likelihood ratio must be for the observed bit, not p(1|slow) unconditionally
  (bug found 2026-11-04: unconditional ratio pins odds at a fixed point).
"""

import numpy as np


class RegimeTracker:
    """Exact binary Bayes filter for a flipping latent regime."""

    def __init__(self, p_flip=0.004):
        self.f = p_flip
        self.odds = 1.0  # P(slow=1) / P(slow=0), pre-bit

    def predict(self):
        """Slow_id estimate BEFORE seeing the bit."""
        return 1 if self.odds > 1.0 else 0

    def update(self, bit, p1_0, p1_1):
        """Fold in the observed bit; p1_s = p(bit=1 | slow=s) from the legend."""
        f = self.f
        o = self.odds
        o_pred = ((1 - f) * o + f) / (f * o + (1 - f))
        if bit:
            ratio = p1_1 / p1_0
        else:
            ratio = (1 - p1_1) / (1 - p1_0)
        ratio = min(max(ratio, 1e-6), 1e6)
        self.odds = min(max(o_pred * ratio, 1e-9), 1e9)
