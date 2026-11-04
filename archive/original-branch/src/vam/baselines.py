"""Reference learners for the experiments (not part of the VAM machine).

``kt_context_q`` is the textbook sequential Bayesian predictor for a fixed
``k``-bit context (Krichevsky-Trofimov estimate per context). It sees exactly
what a massless VAM-1 with ``context = k`` sees, so the gap between the two
measures the learning law alone, not the representation.
"""

from __future__ import annotations

import numpy as np


def contexts(bits: np.ndarray, k: int) -> np.ndarray:
    """Context before each bit: the previous ``k`` bits with the newest in bit 0, zeros before the start.

    This is the register of ``vam.bitkernel.BitMachine`` (and the sector of a massless VAM-1).
    """
    x = np.asarray(bits, dtype=np.int64)
    s = np.zeros(len(x), dtype=np.int64)
    for j in range(1, k + 1):
        s[j:] |= x[:-j] << (j - 1)
    return s


def kt_context_q(bits: np.ndarray, k: int, alpha: float = 0.5) -> np.ndarray:
    """Probability given to each observed bit by per-context counts ``(c_b + alpha) / (n + 2 alpha)``.

    Prequential: each bit is scored with the counts of earlier bits only.
    Vectorized by grouping positions of equal context (a stable sort keeps time order).
    """
    x = np.asarray(bits, dtype=np.int64)
    s = contexts(x, k)
    order = np.argsort(s, kind="stable")
    xs, ss = x[order], s[order]
    start = np.ones(len(x), dtype=bool)
    start[1:] = ss[1:] != ss[:-1]
    group_first = np.maximum.accumulate(np.where(start, np.arange(len(x)), 0))
    ones_before = np.cumsum(xs) - xs
    c1 = ones_before - ones_before[group_first]
    n = np.arange(len(x)) - group_first
    p1 = (c1 + alpha) / (n + 2 * alpha)
    q_sorted = np.where(xs == 1, p1, 1.0 - p1)
    q = np.empty(len(x))
    q[order] = q_sorted
    return q
