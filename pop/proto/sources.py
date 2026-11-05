"""Synthetic hierarchical sources for the interpretation track.

regime_flip: a slow regime S in {0,1} flips with small probability per bit; within a
regime, bits come from an order-2 Markov chain whose transition table is REVERSED
between regimes.  Any flat order-2 model sees each 2-bit context at 50/50 and scores
1.0 bit/bit there -- the structure is invisible without the slow scale.  A population
keyed by (slow feature, fast micro) resolves it.

Regime marginals differ (0.75 vs 0.25) so a slow observable (majority of recent bits)
tracks the regime; the flip itself must be learned per node.
"""

import numpy as np


def regime_flip(n_bits, p_flip=0.004, win=32, seed=0):
    """Yield bits.  Returns (bits, regimes)."""
    rng = np.random.default_rng(seed)
    # P(1 | last2) per regime: regime 0 high-marginal, regime 1 low-marginal,
    # fast pattern reversed.
    T = {
        0: {(0, 0): 0.55, (0, 1): 0.95, (1, 0): 0.95, (1, 1): 0.55},
        1: {(0, 0): 0.45, (0, 1): 0.05, (1, 0): 0.05, (1, 1): 0.45},
    }
    bits = []
    regimes = []
    s = 0
    prev = (rng.integers(0, 2), rng.integers(0, 2))
    for _ in range(n_bits):
        if rng.random() < p_flip:
            s = 1 - s
        b = 1 if rng.random() < T[s][prev] else 0
        bits.append(b)
        regimes.append(s)
        prev = (prev[1], b)
    return np.array(bits, dtype=int), np.array(regimes, dtype=int)


def slow_key(bits, win=32):
    """Slow observable: majority of the last `win` bits (0/1)."""
    if len(bits) < win:
        return 0
    return int(np.mean(bits[-win:]) >= 0.5)


def make_key(win=32):
    """key_fn(bits) -> (node_id, slow_id, micro_id).  8 nodes."""
    def key_fn(bits):
        s = slow_key(bits, win)
        if len(bits) >= 2:
            micro = int(bits[-2]) * 2 + int(bits[-1])
        else:
            micro = 0
        return s * 4 + micro, s, micro
    return key_fn
