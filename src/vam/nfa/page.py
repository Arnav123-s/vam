"""Layer 2 of N: the Page limit decides how many vectors (N-L2, and the growth rule of N-L5).

N-L2 asks for the number of basis vectors that spans a trajectory without redundancy: fewer loses
information, more adds only noise, and "the Page limit establishes the exact dimensional cutoff".
This module makes that cutoff exact and usable.

Page's random state
-------------------
A pure state drawn at random on ``C^m (x) C^n`` (``m <= n``) has, on average, the entanglement
entropy (Page 1993, proved by Foong-Kanno, Sanchez-Ruiz, Sen)

    S(m, n) = sum_{k=n+1}^{mn} 1/k - (m - 1) / (2 n)        (nats)

so the information it shows about its smaller part is only ``ln m - S ~ m / (2 n)``: almost nothing
(:func:`page_entropy`, :func:`page_information`). Over ``N`` qubits split ``k : N - k`` the entropy
rises to its turnaround at ``k = N / 2`` and falls back (:func:`page_curve`). The Schmidt weights of
such a state follow the Marchenko-Pastur law, whose largest value is

    lambda_max = (1 + sqrt(m / n))^2 / m                    (:func:`random_state_edge`)

The cutoff
----------
A direction is information only if it stands above what a random (Page) state of the same shape
shows. For an estimated matrix this is concrete. The whitened count matrix of a stream,
``C[u, v] = N[u, v] / sqrt(N[u] N[v])`` over pasts ``u`` and futures ``v``, has the trivial
singular value 1 (independence) and then the canonical correlations of past and future; its
rank is the number of predictive states (hidden states of a hidden Markov source). Sampling
noise adds a random matrix with entries of variance ``1 / N``, whose top singular value is the
Marchenko-Pastur edge ``(sqrt(m) + sqrt(n)) / sqrt(N)`` for ``m`` observed pasts and ``n``
observed futures (:func:`noise_edge`). This is the random-state edge in count units: it equals
``sqrt(m n / N)`` times the edge of a random state's Schmidt coefficients.

:func:`page_rank` counts singular values above that edge (with a small finite-size margin). It
recovers the number of hidden states of known sources and gives 1 for i.i.d. bits (tested).
Directions under the edge "add zero new mutual information and are pruned", exactly as N-L2 asks.
The edge is the cutoff of N-L2 on estimated matrices. It is not N-L5's growth rule: the nested text machine admits a
context after ``verify`` sightings (a count, a chosen realization), and the chain's bonds follow the cache bond or the
entropy bond (``vam.nfa.governor``, which says why the edge is not used there).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.special import digamma

LN2 = math.log(2.0)


def page_entropy(m: int, n: int, bits: bool = False) -> float:
    """Page's exact average entanglement entropy of the smaller factor of a random state on ``m x n``."""
    if m < 1 or n < 1:
        raise ValueError("dimensions must be positive")
    m, n = min(m, n), max(m, n)
    s = float(digamma(m * n + 1) - digamma(n + 1)) - (m - 1) / (2.0 * n)
    return s / LN2 if bits else s


def page_information(m: int, n: int, bits: bool = False) -> float:
    """Average information about the smaller factor, ``ln m - S(m, n)`` (about ``m / 2n`` nats)."""
    m, n = min(m, n), max(m, n)
    info = math.log(m) - page_entropy(m, n)
    return info / LN2 if bits else info


def page_curve(qubits: int, bits: bool = True) -> np.ndarray:
    """Average entropy of the first ``k`` qubits of a random ``qubits``-qubit state, ``k = 0..qubits``."""
    if qubits < 0:
        raise ValueError("qubits must be nonnegative")
    return np.array([page_entropy(2 ** k, 2 ** (qubits - k), bits) for k in range(qubits + 1)])


def page_turnaround(qubits: int) -> float:
    """The cut where the Page curve turns (half of the system)."""
    return qubits / 2.0


def random_state_edge(m: int, n: int) -> float:
    """Largest Schmidt weight (squared coefficient) of a large random state on ``m x n``."""
    m, n = min(m, n), max(m, n)
    return (1.0 + math.sqrt(m / n)) ** 2 / m


def noise_edge(m: int, n: int, samples: float) -> float:
    """Top singular value of an ``m x n`` noise matrix with entry variance ``1 / samples``."""
    if samples <= 0:
        raise ValueError("need a positive sample count")
    return (math.sqrt(m) + math.sqrt(n)) / math.sqrt(samples)


def tracy_widom_width(m: int, n: int) -> float:
    """Relative width of the top noise singular value's finite-size fluctuation,
    ``(1/sqrt(m) + 1/sqrt(n))^(1/3) / (2 (sqrt(m) + sqrt(n)))`` (Johnstone 2001: the Tracy-Widom
    scale of the top eigenvalue, halved for its square root)."""
    m, n = min(m, n), max(m, n)
    return (1.0 / math.sqrt(m) + 1.0 / math.sqrt(n)) ** (1.0 / 3.0) / (2.0 * (math.sqrt(m) + math.sqrt(n)))


def whiten(counts, marginals=None) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, float]:
    """``C = N[u, v] / sqrt(N[u] N[v])`` over the observed rows and columns.

    ``marginals`` (another count matrix, usually the whole sample) supplies ``N[u]``, ``N[v]`` and the
    observed rows and columns; by default they come from ``counts`` itself. Returns ``(C, rows,
    cols, samples)``.
    """
    Nm = sparse.csr_matrix(counts, dtype=float)
    Mm = Nm if marginals is None else sparse.csr_matrix(marginals, dtype=float)
    r = np.asarray(Mm.sum(axis=1)).ravel()
    c = np.asarray(Mm.sum(axis=0)).ravel()
    rows, cols = np.flatnonzero(r), np.flatnonzero(c)
    C = sparse.diags(1.0 / np.sqrt(r[rows])) @ Nm[rows][:, cols] @ sparse.diags(1.0 / np.sqrt(c[cols]))
    return C.tocsr(), rows, cols, float(r.sum())


GRAM_LIMIT = 4096


def singular_values(C, top: int | None = None) -> np.ndarray:
    """Singular values of a (sparse) matrix, largest first.

    Through the smaller Gram matrix when its side is at most ``GRAM_LIMIT``; otherwise the ``top``
    largest by ARPACK (``top`` is then required).
    """
    C = sparse.csr_matrix(C)
    small = min(C.shape)
    if small <= GRAM_LIMIT:
        gram = (C.T @ C) if C.shape[0] >= C.shape[1] else (C @ C.T)
        ev = np.linalg.eigvalsh(gram.toarray())
        s = np.sqrt(np.clip(ev[::-1], 0.0, None))
        return s if top is None else s[:top]
    if top is None:
        raise ValueError(f"both sides exceed {GRAM_LIMIT}: give top")
    from scipy.sparse.linalg import svds
    return np.sort(svds(C, k=min(top, small - 1), return_singular_vectors=False))[::-1]


def split_edge(counts, half) -> float:
    """Empirical noise edge: the top singular value of ``(N_A - N_B) / sqrt(N[u] N[v])``.

    ``half`` holds the counts of one half ``A`` of the sample and ``counts - half`` the other; a
    list of halves gives the largest of their edges.
    The two halves share every real direction and differ only by sampling noise, so this is the
    random matrix the sample actually carries, with its own variance profile and the dependence
    of overlapping windows (the Marchenko-Pastur edge assumes flat, independent noise and can
    sit well under it, see ``page_spectrum``). It has the variance of the whole sample's noise.
    """
    if isinstance(half, (list, tuple)):
        return max(split_edge(counts, h) for h in half)
    full = sparse.csr_matrix(counts, dtype=float)
    A = sparse.csr_matrix(half, dtype=float)
    B = full - A
    total, a, b = full.sum(), A.sum(), B.sum()
    if a <= 0 or b <= 0:
        raise ValueError("both halves need samples")
    D, _, _, _ = whiten(A * (total / (2 * a)) - B * (total / (2 * b)), marginals=full)   # unbiased if a != b
    return float(singular_values(D, top=1)[0])


def page_cutoff(m: int, n: int, samples: float, split: float | None = None, margin: float = 0.1) -> float:
    """The keep threshold: the Marchenko-Pastur edge plus four Tracy-Widom widths, raised to the
    empirical split-half edge plus ``margin`` when one is given."""
    cut = noise_edge(m, n, samples) * (1.0 + 4.0 * tracy_widom_width(m, n))
    return cut if split is None else max(cut, split * (1.0 + margin))


def page_spectrum(counts, half=None, margin: float = 0.1, top: int | None = None) -> dict:
    """The whitened spectrum, its edges and the Page rank."""
    C, rows, cols, samples = whiten(counts)
    s = singular_values(C, top)
    split = None if half is None else split_edge(counts, half)
    cut = page_cutoff(len(rows), len(cols), samples, split, margin)
    return {"singular_values": s, "rows": len(rows), "cols": len(cols), "samples": samples,
            "edge": noise_edge(len(rows), len(cols), samples), "split_edge": split, "cutoff": cut,
            "rank": int(np.sum(s > cut))}


def page_rank(counts, half=None, margin: float = 0.1, top: int | None = None) -> int:
    """Number of directions of a count matrix that stand above the random-state (Page) edge.

    The trivial direction (singular value 1, independence) is counted, so i.i.d. data gives 1.
    """
    return page_spectrum(counts, half, margin, top)["rank"]


# ---------------------------------------------------------------------------
# Hankel matrices of bit streams
# ---------------------------------------------------------------------------

def windows(bits: np.ndarray, k: int) -> np.ndarray:
    """``k``-bit windows (first bit most significant) starting at every position with room."""
    bits = np.asarray(bits, dtype=np.int64)
    n = len(bits) - k + 1
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    out = np.zeros(n, dtype=np.int64)
    for j in range(k):
        out = (out << 1) | bits[j:j + n]
    return out


def hankel_counts(bits: np.ndarray, past: int, future: int) -> sparse.csr_matrix:
    """Counts ``N[u, v]`` of a ``past``-bit window followed by a ``future``-bit window."""
    w = windows(bits, past + future)
    u = w >> future
    v = w & ((1 << future) - 1)
    data = np.ones(len(w))
    return sparse.csr_matrix((data, (u, v)), shape=(1 << past, 1 << future))


def hankel_halves(bits: np.ndarray, past: int, future: int, block: int = 4096, splits: int = 3,
                  seed: int = 0):
    """Counts of the whole stream and of ``splits`` random halves of it.

    The stream's windows are cut into blocks of ``block`` bits (by start position); each split puts
    a random half of the blocks in ``A``. Blocks keep the dependence of nearby windows inside one
    half, and random blocks put the same mixture of the stream in both halves.
    """
    w = windows(bits, past + future)
    u = w >> future
    v = w & ((1 << future) - 1)
    shape = (1 << past, 1 << future)
    full = sparse.csr_matrix((np.ones(len(w)), (u, v)), shape=shape)
    blocks = np.arange(len(w)) // block
    nb = int(blocks[-1]) + 1 if len(w) else 0
    if nb < 2:
        raise ValueError("the stream needs at least two blocks")
    rng = np.random.default_rng(seed)
    halves = []
    for _ in range(splits):
        chosen = np.zeros(nb, dtype=bool)
        chosen[rng.permutation(nb)[: nb // 2]] = True
        a = chosen[blocks]
        halves.append(sparse.csr_matrix((np.ones(int(a.sum())), (u[a], v[a])), shape=shape))
    return full, halves


def exact_hankel(T: np.ndarray, pi: np.ndarray, past: int, future: int) -> np.ndarray:
    """``P(u v) = pi T_u T_v 1`` of a hidden Markov bit source ``T[b][i, j]`` (rows MSB first)."""
    A = pi[None, :]
    for _ in range(past):
        A = np.concatenate([A @ T[0], A @ T[1]]).reshape(2, -1, A.shape[1]).transpose(1, 0, 2).reshape(-1, A.shape[1])
    B = np.ones((T.shape[1], 1))
    for _ in range(future):
        B = np.stack([T[0] @ B, T[1] @ B], axis=1).reshape(T.shape[1], -1)       # the new bit leads
    return A @ B


def exact_rank(H: np.ndarray, rtol: float = 1e-9) -> int:
    """Rank of an exact probability matrix (whitened, relative tolerance on singular values)."""
    r, c = H.sum(axis=1), H.sum(axis=0)
    rows, cols = r > 0, c > 0
    W = H[rows][:, cols] / np.sqrt(np.outer(r[rows], c[cols]))
    s = np.linalg.svd(W, compute_uv=False)
    return int(np.sum(s > rtol * s[0]))


# ---------------------------------------------------------------------------
# Byte-level Hankel matrices and the depth profile of a text (N-L2 picks the memory depths)
# ---------------------------------------------------------------------------

def byte_hankel_halves(data: bytes, past: int, future: int = 1, block: int = 4096, splits: int = 3,
                       seed: int = 0):
    """Counts of (previous ``past`` bytes, next ``future`` bytes) at every byte position, and of
    ``splits`` random halves of the positions in blocks of ``block`` bytes. Rows and columns are the
    observed contexts and continuations only (``past + future <= 8``)."""
    if past < 1 or future < 1 or past + future > 8:
        raise ValueError("need 1 <= past, 1 <= future and past + future <= 8 bytes")
    a = np.frombuffer(data, dtype=np.uint8).astype(np.uint64)
    n = len(a) - past - future + 1
    if n <= 0:
        raise ValueError("text too short")
    u = np.zeros(n, dtype=np.uint64)
    for j in range(past):
        u = (u << np.uint64(8)) | a[j:j + n]
    v = np.zeros(n, dtype=np.uint64)
    for j in range(future):
        v = (v << np.uint64(8)) | a[past + j:past + j + n]
    _, rows = np.unique(u, return_inverse=True)
    _, cols = np.unique(v, return_inverse=True)
    shape = (int(rows.max()) + 1, int(cols.max()) + 1)
    full = sparse.csr_matrix((np.ones(n), (rows, cols)), shape=shape)
    blocks = np.arange(n) // block
    nb = int(blocks[-1]) + 1
    if nb < 2:
        raise ValueError("the text needs at least two blocks")
    rng = np.random.default_rng(seed)
    halves = []
    for _ in range(splits):
        chosen = np.zeros(nb, dtype=bool)
        chosen[rng.permutation(nb)[: nb // 2]] = True
        sel = chosen[blocks]
        halves.append(sparse.csr_matrix((np.ones(int(sel.sum())), (rows[sel], cols[sel])), shape=shape))
    return full, halves


def depth_profile(data: bytes, depths: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7), future: int = 1,
                  margin: float = 0.1) -> list[dict]:
    """Page rank of the byte Hankel matrix at every context depth (bytes): how many predictive directions
    each depth verifies. The rank rises while deeper contexts add verified directions and turns around
    once the contexts become too rare to verify anything; the turnaround is the deepest useful level."""
    out = []
    for d in depths:
        if d + future > 8:
            break
        full, halves = byte_hankel_halves(data, d, future)
        sp = page_spectrum(full, halves, margin)
        out.append({"depth_bytes": d, "rank": sp["rank"], "rows": sp["rows"], "cols": sp["cols"],
                    "edge": sp["edge"], "split_edge": sp["split_edge"], "cutoff": sp["cutoff"]})
    return out


def turnaround(profile: list[dict]) -> int:
    """The depth (bytes) where the Page rank is largest: deeper levels verify fewer directions."""
    return max(profile, key=lambda r: (r["rank"], -r["depth_bytes"]))["depth_bytes"]
