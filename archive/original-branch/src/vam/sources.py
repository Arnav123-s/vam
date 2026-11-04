"""Known bit sources with exact reference losses (experiment E1, experiments/PROTOCOL.md).

A known source answers two questions about any learner exactly:

* ``Sample.oracle_q``: the probability that the exact source model (knowing
  everything except the future) gave to each observed bit. Its mean
  ``-log2`` is the best loss any predictor can expect on that very stream.
* ``window_floor(k)``: the conditional entropy ``H(next bit | last k bits)``
  of the stationary process. No fixed rule from the last ``k`` bits to a
  probability can beat it on average. A learner whose weights keep moving
  can, because its weights then remember recent history (a fast learner
  tracks a slowly changing pattern).

Every stream is MSB-first bits, the same convention as ``vam.bits``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


def h2(p: np.ndarray | float) -> np.ndarray:
    """Binary entropy in bits, elementwise, with ``0 log 0 = 0``."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    out = np.zeros_like(p)
    mid = (p > 0) & (p < 1)
    q = p[mid]
    out[mid] = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    return out


def conditional_entropy(joint: np.ndarray) -> float:
    """``H(next | window)`` in bits from a table ``joint[window, bit]`` of probabilities."""
    p_w = joint.sum(axis=1)
    seen = p_w > 0
    return float(np.sum(p_w[seen] * h2(joint[seen, 1] / p_w[seen])))


@dataclass(frozen=True)
class Sample:
    bits: np.ndarray          # uint8 zeros and ones
    oracle_q: np.ndarray      # probability the exact model gave to each observed bit, before seeing it

    @property
    def oracle_loss(self) -> float:
        """Bits per bit of the exact model on this stream."""
        return float(-np.log2(self.oracle_q).mean())


def _stationary(P: np.ndarray) -> np.ndarray:
    """Stationary distribution of the irreducible row-stochastic matrix ``P`` (periodic chains included)."""
    S = P.shape[0]
    A = P.T - np.eye(S)
    A[-1, :] = 1.0                                   # replace one balance equation by normalization
    rhs = np.zeros(S)
    rhs[-1] = 1.0
    pi = np.linalg.solve(A, rhs)
    if np.any(pi < -1e-10) or not np.allclose(pi @ P, pi, atol=1e-10):
        raise ValueError("no unique stationary distribution")
    return np.clip(pi, 0.0, None) / np.clip(pi, 0.0, None).sum()


@dataclass(frozen=True)
class BitHMM:
    """A hidden Markov bit source. ``T[b][i, j] = P(emit b and move to j | in state i)``."""

    T: np.ndarray

    def __post_init__(self) -> None:
        T = np.asarray(self.T, dtype=float)
        if T.ndim != 3 or T.shape[0] != 2 or T.shape[1] != T.shape[2]:
            raise ValueError("T must have shape (2, S, S)")
        if np.any(T < 0) or not np.allclose(T.sum(axis=(0, 2)), 1.0, atol=1e-12):
            raise ValueError("each state must emit and move with total probability one")
        object.__setattr__(self, "T", T)

    @property
    def states(self) -> int:
        return self.T.shape[1]

    @cached_property
    def stationary(self) -> np.ndarray:
        return _stationary(self.T[0] + self.T[1])

    def block_entropy(self, n: int) -> float:
        """Entropy in bits of ``n`` consecutive bits of the stationary process."""
        V = self.stationary[None, :]
        for _ in range(n):
            V = np.concatenate([V @ self.T[0], V @ self.T[1]])
        p = V.sum(axis=1)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def window_floor(self, k: int) -> float:
        return self.block_entropy(k + 1) - self.block_entropy(k)

    def filter_q(self, bits: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
        """Bayes-optimal prequential probabilities of ``bits`` (forward filter from ``prior``).

        A bit the model makes impossible gets probability 0, and so does everything after it.
        """
        belief = (self.stationary if prior is None else np.asarray(prior, dtype=float)).copy()
        T0, T1 = self.T[0], self.T[1]
        q = np.zeros(len(bits))
        for t, b in enumerate(bits):
            nxt = belief @ (T1 if b else T0)
            q[t] = nxt.sum()
            if q[t] <= 0.0:
                q[t] = 0.0
                break
            belief = nxt / q[t]
        return q


# ---------------------------------------------------------------------------
# S1: order-2 Markov bits
# ---------------------------------------------------------------------------

S1_TABLE = (0.9, 0.2, 0.6, 0.35)


@dataclass(frozen=True)
class MarkovBits:
    """``P(next = 0 | s)`` with ``s = b_{t-1} + 2 b_{t-2}`` (the convention of tests/test_vam1.py)."""

    table: tuple[float, ...] = S1_TABLE

    @property
    def order(self) -> int:
        return int(np.log2(len(self.table)))

    def hmm(self) -> BitHMM:
        m = self.order
        S = 1 << m
        T = np.zeros((2, S, S))
        for s in range(S):
            for b in (0, 1):
                nxt = ((s << 1) | b) & (S - 1)          # newest bit in bit 0
                T[b, s, nxt] += self.table[s] if b == 0 else 1.0 - self.table[s]
        return BitHMM(T)

    def entropy_rate(self) -> float:
        return float(self.hmm().stationary @ h2(np.array(self.table)))

    def window_floor(self, k: int) -> float:
        return self.hmm().window_floor(k)

    def sample(self, n: int, rng: np.random.Generator) -> Sample:
        """Starts from ``s = 0`` (previous bits all zero), which the oracle also knows."""
        u = rng.random(n)
        table = self.table
        mask = (1 << self.order) - 1
        bits = np.empty(n, dtype=np.uint8)
        q = np.empty(n)
        s = 0
        for t in range(n):
            p0 = table[s]
            b = 0 if u[t] < p0 else 1
            bits[t] = b
            q[t] = p0 if b == 0 else 1.0 - p0
            s = ((s << 1) | b) & mask
        return Sample(bits, q)


# ---------------------------------------------------------------------------
# S2: the T2.2 incomplete codes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IncompleteCodes:
    """Units ``A = 0``, ``B = 01``, ``C = 10`` with probability 1/3 each, no delimiters (T2.2)."""

    def hmm(self) -> BitHMM:
        from .realizations.incomplete_codes import M0, M1  # population maps, columns are sources
        return BitHMM(np.stack([M0.T, M1.T]))

    def window_floor(self, k: int) -> float:
        return self.hmm().window_floor(k)

    def sample(self, n: int, rng: np.random.Generator) -> Sample:
        """Starts at a unit boundary (state ``r``); the oracle is the exact forward filter."""
        codes = (np.array([0], np.uint8), np.array([0, 1], np.uint8), np.array([1, 0], np.uint8))
        units = rng.integers(0, 3, n)                  # n units always cover n bits
        bits = np.concatenate([codes[u] for u in units])[:n]
        prior = np.array([1.0, 0.0, 0.0])
        return Sample(bits, self.hmm().filter_q(bits, prior))


# ---------------------------------------------------------------------------
# S3: an order-1 byte source over a small alphabet
# ---------------------------------------------------------------------------

S3_LETTERS = bytes(range(ord("a"), ord("a") + 16))


@dataclass(frozen=True)
class ByteMarkov:
    """Bytes drawn from ``letters`` by a first-order chain ``P[a, b] = P(next = b | previous = a)``.

    The learner sees bits only, with no byte boundaries, so it must find the
    phase itself; the oracle knows the phase and the previous letter.
    """

    letters: bytes
    P: np.ndarray

    @staticmethod
    def random(seed: int = 1234, letters: bytes = S3_LETTERS, concentration: float = 0.5) -> ByteMarkov:
        rng = np.random.default_rng(seed)
        P = rng.dirichlet(np.full(len(letters), concentration), size=len(letters))
        return ByteMarkov(letters, P)

    @property
    def stationary(self) -> np.ndarray:
        return _stationary(self.P)

    def entropy_rate(self) -> float:
        """Bits per bit: the letter entropy rate shared by a byte's eight bits."""
        row_h = -np.sum(np.where(self.P > 0, self.P * np.log2(np.where(self.P > 0, self.P, 1.0)), 0.0), axis=1)
        return float(self.stationary @ row_h) / 8.0

    def _letter_bits(self) -> np.ndarray:
        return np.unpackbits(np.frombuffer(self.letters, np.uint8)).reshape(len(self.letters), 8)

    def window_floor(self, k: int) -> float:
        """Exact ``H(next bit | last k bits)`` for ``k <= 16``, averaged over the eight phases.

        At phase ``f`` (``f`` bits of the current letter seen) the last ``k`` bits lie inside the
        previous two letters and the seen part of the current one, so three letters suffice.
        """
        if not 1 <= k <= 16:
            raise ValueError("the exact enumeration covers 1 <= k <= 16")
        m = len(self.letters)
        L = np.frombuffer(self.letters, np.uint8).astype(np.int64)
        pi = self.stationary
        a, b, c = np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing="ij")
        weight = (pi[a] * self.P[a, b] * self.P[b, c]).ravel() / 8.0
        la, lb, lc = L[a].ravel(), L[b].ravel(), L[c].ravel()
        joint = np.zeros((1 << k, 2))
        for f in range(8):
            history = (la << (8 + f)) | (lb << f) | (lc >> (8 - f))
            window = history & ((1 << k) - 1)
            nxt = (lc >> (7 - f)) & 1
            np.add.at(joint, (window, nxt), weight)
        return conditional_entropy(joint)

    def sample(self, n_bits: int, rng: np.random.Generator) -> Sample:
        if n_bits % 8:
            raise ValueError("n_bits must be a whole number of bytes")
        n = n_bits // 8
        m = len(self.letters)
        cum = np.cumsum(self.P, axis=1)
        u = rng.random(n)
        idx = np.empty(n, dtype=np.int64)
        idx[0] = min(int(np.searchsorted(np.cumsum(self.stationary), u[0], side="right")), m - 1)
        for t in range(1, n):
            idx[t] = min(int(np.searchsorted(cum[idx[t - 1]], u[t], side="right")), m - 1)
        lb = self._letter_bits()
        bits = lb[idx].ravel()
        # oracle: P(bit f of the current letter | previous letter, bits 0..f-1 of the current letter)
        prior = np.vstack([self.stationary[None, :], self.P[idx[:-1]]])          # (n, m)
        q = np.empty((n, 8))
        alive = np.ones((n, m), dtype=bool)                                       # letters still consistent
        for f in range(8):
            mass = np.where(alive, prior, 0.0)
            match = lb[None, :, f] == lb[idx, f][:, None]
            q[:, f] = (mass * match).sum(axis=1) / mass.sum(axis=1)
            alive &= match
        return Sample(bits.astype(np.uint8), q.ravel())


# ---------------------------------------------------------------------------
# S4: a delayed echo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DelayedEcho:
    """``x_t = x_{t-d} XOR n_t`` with ``n_t ~ Bernoulli(p)``; the first ``d`` bits are fair coins.

    The process is uniform on every window shorter than ``d`` bits, so a pure
    ``k``-bit context predicts nothing when ``k < d``.
    """

    d: int
    p: float

    def entropy_rate(self) -> float:
        return float(h2(self.p))

    def window_floor(self, k: int) -> float:
        return 1.0 if k < self.d else self.entropy_rate()

    def hmm(self) -> BitHMM:
        """Exact HMM on the last ``d`` bits (newest in bit 0), for checking ``window_floor``."""
        S = 1 << self.d
        T = np.zeros((2, S, S))
        for s in range(S):
            old = (s >> (self.d - 1)) & 1
            for b in (0, 1):
                T[b, s, ((s << 1) | b) & (S - 1)] = 1.0 - self.p if b == old else self.p
        return BitHMM(T)

    def sample(self, n: int, rng: np.random.Generator) -> Sample:
        d = self.d
        noise = (rng.random(n) < self.p).astype(np.uint8)
        bits = np.empty(n, dtype=np.uint8)
        bits[:d] = rng.integers(0, 2, min(d, n))
        for t in range(d, n, d):
            end = min(t + d, n)
            bits[t:end] = bits[t - d:end - d] ^ noise[t:end]
        q = np.full(n, 0.5)
        q[d:] = np.where(noise[d:] == 1, self.p, 1.0 - self.p)
        return Sample(bits, q)
