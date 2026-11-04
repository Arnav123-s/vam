"""Scalable joint states: matrix-product states and locally purified density operators.

Source: T13.1 (a globally positive representation ``R = X X^dagger / Tr`` with
``X`` stored in a tensor network; hierarchical isometries reorganize it
without discarding correlations), T13.4 (an MPS with ``N`` sites, local
dimension ``d`` and bond dimension ``chi`` stores order ``N d chi^2``
entries; exactness needs sufficient rank across each cut), F-Part 9 (the
hierarchical density tensor network).

Conventions
-----------
* Site 0 is the leftmost (most significant) tensor factor, matching
  :class:`vam.space.Space` and :func:`vam.dirac.chain_operator`.
* :class:`MPS` site tensors have shape ``(D_left, d, D_right)``.
* :class:`LPDO` site tensors have shape ``(D_left, d, k, D_right)``; the state
  is ``rho = Tr_K |X><X|`` where ``X`` is the MPS on the combined local
  index ``(d, k)``. Positivity therefore holds by construction, also after
  truncation.
* Truncations are performed in the canonical gauge, so the discarded
  singular values are exactly the removed weight; each truncation's
  discarded fraction is returned and accumulated in ``.discarded``.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from scipy import linalg as sla

from .density import ZeroProbabilityError
from .linalg import as_matrix

#: Singular values below this fraction of the largest are numerically zero and are always dropped.
_RANK_TOL = 1e-14
#: Smallest outcome probability that ``LPDO.condition`` will renormalize by (T10.2).
_ZERO_PROB = 1e-10


# ----------------------------------------------------------------------------
# Shared numerical kernels
# ----------------------------------------------------------------------------

def _svd(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD; falls back to LAPACK ``gesvd`` when the default driver does not converge."""
    try:
        return np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return sla.svd(M, full_matrices=False, lapack_driver="gesvd")


def _truncate(s: np.ndarray, max_keep, cutoff: float) -> tuple[int, float, float]:
    """Choose how many singular values to keep.

    Keeps at most ``max_keep`` values and drops those with ``s^2 < cutoff * sum(s^2)``
    (numerically zero values are always dropped; at least one value is kept). ``max_keep``
    may also be a rule: a callable that receives the normalized weights ``s^2 / sum(s^2)``
    (largest first) and returns the bond for this cut (``vam.nfa.governor.EntropyBond``).
    Returns ``(keep, discarded_fraction, scale)`` where the discarded fraction is
    ``sum(dropped s^2) / sum(all s^2)`` and ``scale`` rescales the kept values so
    that the squared weight before truncation is preserved.
    """
    s2 = s * s
    total = float(s2.sum())
    if total <= 0.0:
        return 1, 0.0, 1.0
    keep = int(np.count_nonzero((s2 >= cutoff * total) & (s > _RANK_TOL * s[0])))
    keep = max(1, keep)
    if callable(max_keep):
        max_keep = max_keep(s2 / total)
    if max_keep is not None:
        keep = min(keep, max(1, int(max_keep)))
    kept = float(s2[:keep].sum())
    return keep, float(s2[keep:].sum()) / total, float(np.sqrt(total / kept))


def _is_isometry(op: np.ndarray) -> bool:
    """True for a square unitary, which preserves every orthonormality condition of a tensor."""
    return op.shape[0] == op.shape[1] and bool(np.allclose(op.conj().T @ op, np.eye(op.shape[1]), atol=1e-12, rtol=0))


def _split_pair(theta: np.ndarray, max_bond: int | None, cutoff: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Split ``theta[a, s, k, t, l, b]`` into ``A[a, s, k, r]`` and ``B[r, t, l, b]`` by a truncated SVD.

    Each site's Kraus index ``k`` / ``l`` stays with its own physical index.
    The caller must hold ``theta`` in canonical gauge (centre on the pair), so the
    discarded fraction is the exact removed weight (T13.1). Singular values are
    absorbed into ``B``; the squared norm of ``theta`` is preserved.
    """
    Dl, d1, k1, d2, k2, Dr = theta.shape
    U, s, Vh = _svd(theta.reshape(Dl * d1 * k1, d2 * k2 * Dr))
    keep, frac, scale = _truncate(s, max_bond, cutoff)
    A = U[:, :keep].reshape(Dl, d1, k1, keep)
    B = ((s[:keep] * scale)[:, None] * Vh[:keep]).reshape(keep, d2, k2, Dr)
    return A, B, frac


def _contract(tensors: Sequence[np.ndarray], keep: Sequence[int] = (), ops: dict[int, np.ndarray] | None = None,
              open_ends: bool = False) -> np.ndarray:
    """Left-to-right transfer contraction of ``<X| ... |X>`` over 4-index tensors ``(Dl, d, k, Dr)``.

    The Kraus leg ``k`` and the physical leg of every site not in ``keep`` are summed
    between bra and ket; sites in ``keep`` keep their physical leg open. ``ops`` are
    inserted on the ket physical leg. Returns the matrix ``M[(ket indices), (bra indices)]``
    over the kept sites in site order. Nothing larger than the requested sites is formed,
    and every step is a BLAS tensor contraction.

    ``open_ends=True`` treats the outer bonds as closed by identity environments: correct
    for a segment whose left neighbours are left-orthonormal and right neighbours
    right-orthonormal (the segment starts at the orthogonality centre).
    """
    ops = ops or {}
    keep_set = set(keep)
    D0 = tensors[0].shape[0]
    E = np.eye(D0, dtype=complex).reshape(1, 1, D0, D0) if open_ends else np.ones((1, 1, 1, 1), dtype=complex)
    for j, T in enumerate(tensors):                         # E[ket open, bra open, bra bond, ket bond]
        Tk = T if j not in ops else np.tensordot(ops[j], T, axes=([1], [1])).transpose(1, 0, 2, 3)
        X = np.tensordot(E, Tk, axes=([3], [0]))             # (I, J, a, s, k, d)
        if j in keep_set:
            Y = np.tensordot(X, T.conj(), axes=([2, 4], [0, 2]))   # (I, J, s, d, t, c)
            I, J, ds, dk, dt, c = Y.shape
            E = Y.transpose(0, 2, 1, 4, 5, 3).reshape(I * ds, J * dt, c, dk)
        else:
            E = np.tensordot(X, T.conj(), axes=([2, 3, 4], [0, 1, 2])).transpose(0, 1, 3, 2)   # (I, J, c, d)
    if open_ends:
        return np.trace(E, axis1=2, axis2=3)
    return E[:, :, 0, 0]


def _check_sites(sites: Sequence[int], n: int) -> list[int]:
    sites = [int(s) for s in sites]
    if not sites or any(b <= a for a, b in pairwise(sites)) or sites[0] < 0 or sites[-1] >= n:
        raise ValueError(f"sites {sites} must be a non-empty strictly increasing list inside 0..{n - 1}")
    return sites


# ----------------------------------------------------------------------------
# Matrix-product states
# ----------------------------------------------------------------------------

class MPS:
    """Matrix-product state ``|psi> = sum A[0]^{s0} A[1]^{s1} ... |s0 s1 ...>`` (T13.4).

    ``tensors[j]`` has shape ``(D_left, d_j, D_right)`` with boundary bonds of size 1.
    ``center`` is the orthogonality centre (every tensor left of it is
    left-orthonormal, every tensor right of it right-orthonormal) or ``None`` when
    the gauge is unknown. ``discarded`` accumulates the exact discarded fractions
    of all truncations performed (see the module conventions).

    The state is not required to be normalized; :meth:`norm` reports the true norm.
    """

    def __init__(self, tensors: Sequence[np.ndarray], center: int | None = None, discarded: float = 0.0) -> None:
        self.tensors: list[np.ndarray] = [np.asarray(t, dtype=complex) for t in tensors]
        if not self.tensors or any(t.ndim != 3 for t in self.tensors):
            raise ValueError("an MPS needs a non-empty list of (D_left, d, D_right) tensors")
        if any(a.shape[2] != b.shape[0] for a, b in pairwise(self.tensors)):
            raise ValueError("neighbouring bond dimensions do not match")
        self.center = center
        self.discarded = float(discarded)

    # -- constructors ---------------------------------------------------------
    @classmethod
    def product(cls, vectors: Sequence[np.ndarray]) -> MPS:
        """Product state ``v_0 (x) v_1 (x) ...`` with bond dimension 1."""
        return cls([np.asarray(v, dtype=complex).reshape(1, -1, 1) for v in vectors])

    @classmethod
    def from_dense(cls, psi: np.ndarray, dims: Sequence[int], max_bond: int | None = None,
                   cutoff: float = 0.0) -> MPS:
        """Decompose a dense vector by successive SVDs from the left.

        Without truncation the representation is exact. ``max_bond`` and ``cutoff``
        act as in :meth:`apply_two`. Each SVD acts on a matrix whose right factor
        is orthonormal in the full space, so the discarded fraction of every cut is exact;
        the fractions are summed into ``discarded`` and the norm of ``psi`` is kept.
        The result has its centre on the last site.
        """
        dims = tuple(int(d) for d in dims)
        psi = np.asarray(psi, dtype=complex).reshape(-1)
        if psi.size != int(np.prod(dims)):
            raise ValueError(f"vector of size {psi.size} does not match dims {dims}")
        tensors, M, Dl, disc = [], psi.reshape(1, -1), 1, 0.0
        for d in dims[:-1]:
            U, s, Vh = _svd(M.reshape(Dl * d, -1))
            keep, frac, scale = _truncate(s, max_bond, cutoff)
            tensors.append(U[:, :keep].reshape(Dl, d, keep))
            M = (s[:keep] * scale)[:, None] * Vh[:keep]
            Dl, disc = keep, disc + frac
        tensors.append(M.reshape(Dl, dims[-1], 1))
        return cls(tensors, center=len(dims) - 1, discarded=disc)

    def copy(self) -> MPS:
        return MPS([t.copy() for t in self.tensors], self.center, self.discarded)

    # -- structure ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tensors)

    @property
    def dims(self) -> tuple[int, ...]:
        return tuple(t.shape[1] for t in self.tensors)

    @property
    def bond_dims(self) -> tuple[int, ...]:
        """Bond dimensions ``D_1 ... D_{N-1}`` (entry ``b`` joins sites ``b`` and ``b + 1``)."""
        return tuple(t.shape[2] for t in self.tensors[:-1])

    def num_entries(self) -> int:
        """Stored complex entries: at most ``N d chi^2`` (T13.4)."""
        return int(sum(t.size for t in self.tensors))

    def _t4(self, j: int) -> np.ndarray:
        return self.tensors[j][:, :, None, :]

    # -- dense form and norms -------------------------------------------------
    def to_dense(self) -> np.ndarray:
        """Full state vector, site 0 most significant. Exponential in ``N``: tests only."""
        v = self.tensors[0].reshape(-1, self.tensors[0].shape[2])
        for T in self.tensors[1:]:
            v = (v @ T.reshape(T.shape[0], -1)).reshape(-1, T.shape[2])
        return v.reshape(-1)

    def inner(self, other: MPS) -> complex:
        """``<self|other>``."""
        if self.dims != other.dims:
            raise ValueError("states have different local dimensions")
        E = np.ones((1, 1), dtype=complex)                      # E[bra bond, ket bond]
        for A, B in zip(self.tensors, other.tensors, strict=True):
            E = np.einsum("ab,asc,bsd->cd", E, A.conj(), B, optimize=True)
        return complex(E[0, 0])

    def norm(self) -> float:
        """True 2-norm ``sqrt(<psi|psi>)`` (the state may be unnormalized after non-unitary operators)."""
        if self.center is not None:
            return float(np.linalg.norm(self.tensors[self.center]))
        return float(np.sqrt(max(self.inner(self).real, 0.0)))

    # -- gauge ------------------------------------------------------------------
    def _shift_right(self, j: int) -> None:
        T = self.tensors[j]
        Q, R = np.linalg.qr(T.reshape(-1, T.shape[2]))
        self.tensors[j] = Q.reshape(T.shape[0], T.shape[1], -1)
        self.tensors[j + 1] = np.tensordot(R, self.tensors[j + 1], axes=(1, 0))

    def _shift_left(self, j: int) -> None:
        T = self.tensors[j]
        Q, R = np.linalg.qr(T.reshape(T.shape[0], -1).conj().T)   # T = R^dagger Q^dagger
        self.tensors[j] = Q.conj().T.reshape(-1, T.shape[1], T.shape[2])
        self.tensors[j - 1] = np.tensordot(self.tensors[j - 1], R.conj().T, axes=(2, 0))

    def canonicalize(self, center: int) -> None:
        """Bring the state to mixed-canonical form with the orthogonality centre on ``center``.

        QR sweeps; the state is unchanged. Starting from a known centre only the tensors
        between the old and new centre are touched.
        """
        n = len(self.tensors)
        if not 0 <= center < n:
            raise ValueError(f"centre {center} outside 0..{n - 1}")
        if self.center is None:
            for j in range(center):
                self._shift_right(j)
            for j in range(n - 1, center, -1):
                self._shift_left(j)
        else:
            while self.center < center:
                self._shift_right(self.center)
                self.center += 1
            while self.center > center:
                self._shift_left(self.center)
                self.center -= 1
        self.center = center

    def _move_to_pair(self, site: int) -> None:
        """Put the centre on one of ``site``, ``site + 1`` (the nearer one when already known)."""
        if not 0 <= site < len(self.tensors) - 1:
            raise ValueError(f"no pair of sites starting at {site}")
        if self.center in (site, site + 1):
            return
        self.canonicalize(site + 1 if self.center is not None and self.center > site + 1 else site)

    def _after_local(self, site: int, op: np.ndarray) -> None:
        """Update the gauge record after ``op`` acted on ``site``."""
        if self.center != site and not _is_isometry(op):
            self.center = None

    # -- operations -------------------------------------------------------------
    def apply_one(self, op: np.ndarray, site: int) -> None:
        """Apply a single-site operator of shape ``(d', d)``; keeps the unnormalized result.

        The operator need not be unitary (projectors) nor square (it may change the local
        dimension). The state is not renormalized, so :meth:`norm` reports the true norm.
        """
        op = as_matrix(op)
        self.tensors[site] = np.einsum("ts,asb->atb", op, self.tensors[site])
        self._after_local(site, op)

    def apply_two(self, op: np.ndarray, site: int, max_bond: int | None = None, cutoff: float = 0.0,
                  out_dims: tuple[int, int] | None = None) -> float:
        """Apply a two-site operator of shape ``(d1' d2', d1 d2)`` to ``(site, site + 1)``.

        The centre is first moved onto the pair, so the SVD split is performed in
        canonical gauge and the discarded weight is exact (T13.1). At most ``max_bond``
        singular values are kept and those with ``s^2 < cutoff * sum(s^2)`` are dropped.
        The result is rescaled to the norm it had before truncation. Returns the
        discarded fraction ``sum(dropped s^2) / sum(all s^2)``, which is also added to
        ``self.discarded``. ``out_dims`` gives ``(d1', d2')`` when the operator changes the
        local dimensions (default: unchanged). The centre ends on ``site + 1``.
        """
        self._move_to_pair(site)
        A, B = self.tensors[site][:, :, None, :], self.tensors[site + 1][:, :, None, :]
        theta = _apply_pair(A, B, as_matrix(op), out_dims)
        A, B, frac = _split_pair(theta, max_bond, cutoff)
        self.tensors[site], self.tensors[site + 1] = A[:, :, 0, :], B[:, :, 0, :]
        self.center = site + 1
        self.discarded += frac
        return frac

    def apply_mpo(self, mpo: Sequence[np.ndarray], first: int = 0, max_bond: int | None = None,
                  cutoff: float = 0.0) -> float:
        """``|psi> -> W |psi>`` for a matrix-product operator ``W`` on consecutive sites from ``first``.

        ``mpo[i]`` has shape ``(D_left, d_out, d_in, D_right)`` with outer bonds of size 1 and
        acts on site ``first + i``. Recompression as in :meth:`LPDO.apply_mpo`: the centre is
        moved to ``first``, the operator is contracted in (the segment's bonds multiply), a QR
        sweep makes the segment left-orthonormal, and SVDs from the right end back to ``first``
        truncate every bond of the segment at the orthogonality centre, so each discarded
        fraction is the exact removed weight; ``max_bond`` and ``cutoff`` act as in
        :meth:`apply_two` and the squared norm is kept. Returns the summed discarded fraction
        (also added to ``discarded``). The centre ends on ``first``.
        """
        mpo = [np.asarray(W, dtype=complex) for W in mpo]
        last = first + len(mpo) - 1
        if not mpo or first < 0 or last >= len(self):
            raise ValueError(f"an operator on {len(mpo)} sites does not fit at site {first}")
        if mpo[0].shape[0] != 1 or mpo[-1].shape[3] != 1:
            raise ValueError("the outer bonds of the operator must have size 1")
        self.canonicalize(first)
        for i, W in enumerate(mpo):
            T = self.tensors[first + i]
            Dl, d, Dr = T.shape
            if W.ndim != 4 or W.shape[2] != d:
                raise ValueError(f"operator tensor of shape {W.shape} does not act on local dimension {d}")
            wl, dout, _, wr = W.shape
            self.tensors[first + i] = np.einsum("ptsq,asb->aptbq", W, T).reshape(Dl * wl, dout, Dr * wr)
        for j in range(first, last):
            self._shift_right(j)
        total = 0.0
        for j in range(last, first, -1):
            T = self.tensors[j]
            Dl, d, Dr = T.shape
            U, s, Vh = _svd(T.reshape(Dl, d * Dr))
            keep, frac, scale = _truncate(s, max_bond, cutoff)
            self.tensors[j] = Vh[:keep].reshape(keep, d, Dr)
            self.tensors[j - 1] = np.tensordot(self.tensors[j - 1], U[:, :keep] * (s[:keep] * scale), axes=(2, 0))
            total += frac
        self.center = first
        self.discarded += total
        return total

    # -- observables ------------------------------------------------------------
    def expectation(self, ops: dict[int, np.ndarray]) -> complex:
        """``<psi| prod_j O_j |psi> / <psi|psi>`` for single-site operators ``{site: O_j}``."""
        return _segment_expectation(self, self._t4, ops)

    def reduced_density(self, sites: Sequence[int]) -> np.ndarray:
        """Reduced density matrix of ``sites`` (sorted, not necessarily contiguous), trace 1.

        Rows and columns are ordered by site index (site first in the list is most
        significant). Computed with transfer environments; only the requested
        sites' dimensions are ever exponentiated.
        """
        sites = _check_sites(sites, len(self.tensors))
        return _segment_density(self, self._t4, sites)

    def entanglement_entropy(self, bond: int) -> float:
        """Von Neumann entropy in bits across the cut between sites ``bond`` and ``bond + 1``.

        Moves the centre to ``bond`` (the state is unchanged); the Schmidt values are
        the singular values of the centre tensor.
        """
        if not 0 <= bond < len(self.tensors) - 1:
            raise ValueError(f"no bond {bond}")
        self.canonicalize(bond)
        T = self.tensors[bond]
        s = _svd(T.reshape(-1, T.shape[2]))[1]
        p = s * s
        p = p[p > 0] / p.sum()
        return float(-np.sum(p * np.log2(p)))


def _apply_pair(A: np.ndarray, B: np.ndarray, op: np.ndarray, out_dims: tuple[int, int] | None,
                kraus_axis: bool = False) -> np.ndarray:
    """Contract two 4-index site tensors and apply ``op[(u v), (s t)]`` on the physical legs.

    With ``kraus_axis`` the operator is a stack ``op[m, (u v), (s t)]`` and the new index ``m``
    is placed in front of the left site's Kraus index (the pair ``(m, k)``, ``m`` major).
    """
    d1, d2 = A.shape[1], B.shape[1]
    d1o, d2o = out_dims if out_dims is not None else (d1, d2)
    theta = np.einsum("askc,ctlb->asktlb", A, B)
    if op.shape[-1] != d1 * d2 or op.shape[-2] != d1o * d2o:
        raise ValueError(f"operator shape {op.shape} does not act on local dimensions ({d1}, {d2}) -> ({d1o}, {d2o})")
    if not kraus_axis:
        return np.einsum("uvst,asktlb->aukvlb", op.reshape(d1o, d2o, d1, d2), theta)
    M = op.shape[0]
    out = np.einsum("muvst,asktlb->aumkvlb", op.reshape(M, d1o, d2o, d1, d2), theta)
    Dl, _, _, k1, _, k2, Dr = out.shape
    return out.reshape(Dl, d1o, M * k1, d2o, k2, Dr)


def _segment_density(x: MPS, t4, sites: list[int]) -> np.ndarray:
    """Reduced density of ``sites``: move the centre to ``sites[0]`` (QR sweep, state unchanged), so only
    the segment ``sites[0] .. sites[-1]`` is contracted; outside it the environments are identities."""
    x.canonicalize(sites[0])
    lo = sites[0]
    seg = [t4(j) for j in range(lo, sites[-1] + 1)]
    rho = _contract(seg, keep=[j - lo for j in sites], open_ends=True)
    return rho / np.trace(rho)


def _segment_expectation(x: MPS, t4, ops: dict[int, np.ndarray]) -> complex:
    """``<prod_j O_j>`` normalized, contracting only between the outermost operator sites."""
    ops = {int(k): as_matrix(v) for k, v in ops.items()}
    if not ops:
        return 1.0 + 0.0j
    lo, hi = min(ops), max(ops)
    x.canonicalize(lo)
    seg = [t4(j) for j in range(lo, hi + 1)]
    num = _contract(seg, ops={j - lo: O for j, O in ops.items()}, open_ends=True)[0, 0]
    den = _contract(seg, open_ends=True)[0, 0]
    return complex(num / den)


# ----------------------------------------------------------------------------
# Locally purified density operators
# ----------------------------------------------------------------------------

class LPDO:
    """Locally purified density operator ``rho = Tr_K |X><X|`` (T13.1, F-Part 9).

    ``tensors[j]`` has shape ``(D_left, d_j, k_j, D_right)``: physical index ``d``, Kraus
    (purification) index ``k``. Internally ``X`` is an :class:`MPS` on the combined local index
    ``(d, k)`` (``d`` major), so all canonical-gauge machinery is shared. Because ``rho`` is
    a Gram matrix of ``X``, positivity holds by construction, also after truncation.
    ``trace()`` equals the squared norm of ``X``.
    """

    def __init__(self, tensors: Sequence[np.ndarray], center: int | None = None, discarded: float = 0.0) -> None:
        ts = [np.asarray(t, dtype=complex) for t in tensors]
        if not ts or any(t.ndim != 4 for t in ts):
            raise ValueError("an LPDO needs a non-empty list of (D_left, d, k, D_right) tensors")
        self._d = [t.shape[1] for t in ts]
        self._k = [t.shape[2] for t in ts]
        self._x = MPS([t.reshape(t.shape[0], -1, t.shape[3]) for t in ts], center)
        self.discarded = float(discarded)

    # -- constructors ---------------------------------------------------------
    @classmethod
    def product(cls, densities: Sequence[np.ndarray]) -> LPDO:
        """Product of local density matrices, each purified as ``rho = sum_k w_k |v_k><v_k|``.

        The Kraus column ``k`` of the local tensor is ``sqrt(w_k) v_k``; numerically zero
        eigenvalues are dropped, so a pure local state has Kraus dimension 1.
        """
        tensors = []
        for rho in densities:
            rho = as_matrix(rho)
            w, V = np.linalg.eigh(0.5 * (rho + rho.conj().T))
            idx = np.nonzero(w > _RANK_TOL * max(float(w[-1]), 0.0))[0] if w[-1] > 0 else np.array([len(w) - 1])
            tensors.append((V[:, idx] * np.sqrt(w[idx]))[None, :, :, None])
        return cls(tensors)

    @classmethod
    def from_mps(cls, mps: MPS) -> LPDO:
        """The pure state ``|psi><psi|`` as an LPDO with Kraus dimension 1."""
        return cls([t[:, :, None, :] for t in mps.tensors], mps.center, mps.discarded)

    def copy(self) -> LPDO:
        return LPDO([t.copy() for t in self.tensors], self._x.center, self.discarded)

    # -- structure ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._d)

    @property
    def tensors(self) -> list[np.ndarray]:
        """Site tensors ``(D_left, d, k, D_right)`` (views of the internal storage)."""
        parts = zip(self._x.tensors, self._d, self._k, strict=True)
        return [t.reshape(t.shape[0], d, k, t.shape[2]) for t, d, k in parts]

    @property
    def center(self) -> int | None:
        return self._x.center

    @property
    def dims(self) -> tuple[int, ...]:
        return tuple(self._d)

    @property
    def kraus_dims(self) -> tuple[int, ...]:
        return tuple(self._k)

    @property
    def bond_dims(self) -> tuple[int, ...]:
        return self._x.bond_dims

    def num_entries(self) -> int:
        return self._x.num_entries()

    def canonicalize(self, center: int) -> None:
        """Move the orthogonality centre of ``X`` (the state ``rho`` is unchanged)."""
        self._x.canonicalize(center)

    def _t4(self, site: int) -> np.ndarray:
        t = self._x.tensors[site]
        return t.reshape(t.shape[0], self._d[site], self._k[site], t.shape[2])

    def _set(self, site: int, T: np.ndarray) -> None:
        self._d[site], self._k[site] = T.shape[1], T.shape[2]
        self._x.tensors[site] = T.reshape(T.shape[0], T.shape[1] * T.shape[2], T.shape[3])

    # -- dense form and observables ---------------------------------------------
    def to_dense(self) -> np.ndarray:
        """Full density matrix ``Tr_K |X><X|``. Exponential in ``N``: tests only."""
        n = len(self)
        inter = [x for pair in zip(self._d, self._k, strict=True) for x in pair]
        X = self._x.to_dense().reshape(inter)
        X = X.transpose(list(range(0, 2 * n, 2)) + list(range(1, 2 * n, 2)))
        X = X.reshape(int(np.prod(self._d)), -1)
        return X @ X.conj().T

    def trace(self) -> float:
        """``Tr rho = <X|X>``."""
        return self._x.norm() ** 2

    def expectation(self, ops: dict[int, np.ndarray]) -> complex:
        """``Tr(rho prod_j O_j) / Tr(rho)`` for single-site operators ``{site: O_j}``."""
        return _segment_expectation(self._x, self._t4, ops)

    def reduced_density(self, sites: Sequence[int]) -> np.ndarray:
        """Reduced density matrix of ``sites`` (sorted, not necessarily contiguous), trace 1."""
        sites = _check_sites(sites, len(self))
        return _segment_density(self._x, self._t4, sites)

    # -- unitary and non-unitary operations on the physical index -------------------
    def apply_one(self, op: np.ndarray, site: int) -> None:
        """``X -> (op on the physical index of site) X``, i.e. ``rho -> op rho op^dagger`` (unnormalized)."""
        op = as_matrix(op)
        self._set(site, np.einsum("ts,askb->atkb", op, self._t4(site)))
        self._x._after_local(site, op)

    def apply_two(self, op: np.ndarray, site: int, max_bond: int | None = None, cutoff: float = 0.0,
                  out_dims: tuple[int, int] | None = None) -> float:
        """Two-site operator on ``(site, site + 1)``; truncation exactly as :meth:`MPS.apply_two`.

        Each site's Kraus index stays with its own physical index when splitting, so the
        truncation of ``X`` (hence of ``rho``) is exact in the canonical gauge. Returns the
        discarded fraction of ``Tr rho``.
        """
        self._x._move_to_pair(site)
        theta = _apply_pair(self._t4(site), self._t4(site + 1), as_matrix(op), out_dims)
        A, B, frac = _split_pair(theta, max_bond, cutoff)
        self._set(site, A)
        self._set(site + 1, B)
        self._x.center = site + 1
        self.discarded += frac
        return frac

    # -- channels ---------------------------------------------------------------
    @staticmethod
    def _kraus_stack(kraus) -> np.ndarray:
        ops = kraus.kraus if hasattr(kraus, "kraus") else kraus
        return np.stack([as_matrix(K) for K in ops])

    def _compress_kraus(self, site: int, max_kraus: int | None, cutoff: float) -> float:
        """Exact SVD compression of the Kraus index of ``site`` (centre must be on ``site``).

        Reshapes the site tensor to ``((D_left d D_right), k)``; the right singular
        vectors are a unitary on the traced-out index, so only the discarded singular
        values change ``rho``. The kept rank is at most ``min(rank, max_kraus)``, hence at
        most ``d D_left D_right``.
        """
        T = self._t4(site)
        Dl, d, k, Dr = T.shape
        if k == 1:
            return 0.0
        U, s, _ = _svd(T.transpose(0, 1, 3, 2).reshape(Dl * d * Dr, k))
        keep, frac, scale = _truncate(s, max_kraus, cutoff)
        new = (U[:, :keep] * (s[:keep] * scale)).reshape(Dl, d, Dr, keep).transpose(0, 1, 3, 2)
        self._set(site, new)
        self.discarded += frac
        return frac

    def apply_channel(self, kraus, site: int, max_kraus: int | None = None, cutoff: float = 0.0) -> float:
        """Apply a channel ``rho -> sum_a K_a rho K_a^dagger`` on one site.

        ``kraus`` is a :class:`vam.channels.KrausMap` or a list of matrices. The new
        Kraus index is the pair ``(a, k)``; it is then compressed exactly by an SVD in
        canonical gauge centred on ``site`` (see :meth:`_compress_kraus`), keeping at
        most ``max_kraus`` vectors and at most ``d D_left D_right``. Returns the
        discarded fraction of ``Tr rho``.
        """
        K = self._kraus_stack(kraus)
        self._x.canonicalize(site)
        T = np.einsum("mts,askb->atmkb", K, self._t4(site))
        self._set(site, T.reshape(T.shape[0], T.shape[1], -1, T.shape[4]))
        return self._compress_kraus(site, max_kraus, cutoff)

    def apply_channel_two(self, kraus, site: int, max_bond: int | None = None, max_kraus: int | None = None,
                          cutoff: float = 0.0) -> float:
        """Apply a two-site channel on ``(site, site + 1)``.

        The pair is merged, every Kraus operator is applied with a new index attached to the
        left site's Kraus index, the pair is split by a truncated SVD (``max_bond``), and
        both Kraus indices are compressed exactly (``max_kraus``). Returns the total discarded
        fraction of ``Tr rho``.
        """
        K = self._kraus_stack(kraus)
        self._x._move_to_pair(site)
        theta = _apply_pair(self._t4(site), self._t4(site + 1), K, None, kraus_axis=True)
        A, B, frac = _split_pair(theta, max_bond, cutoff)
        self._set(site, A)
        self._set(site + 1, B)
        self._x.center = site + 1
        self.discarded += frac
        total = frac + self._compress_kraus(site + 1, max_kraus, cutoff)
        self._x.canonicalize(site)
        return total + self._compress_kraus(site, max_kraus, cutoff)

    def condition(self, kraus_list, site: int) -> float:
        """Condition on an outcome ``rho -> sum_a K_a rho K_a^dagger`` and renormalize (T10.2).

        ``kraus_list`` is the (trace-nonincreasing) outcome map, e.g. ``[P1]``. Returns the
        outcome probability ``p = Tr`` of the unnormalized result and rescales the state to
        trace 1. Raises :class:`vam.density.ZeroProbabilityError` (leaving the state
        untouched) when ``p <= 1e-10``.
        """
        trial = self.copy()
        trial.apply_channel(kraus_list, site)
        p = trial.trace()
        if p <= _ZERO_PROB:
            raise ZeroProbabilityError(f"outcome probability {p:.3e} is zero within tolerance")
        trial._x.tensors[site] = trial._x.tensors[site] / np.sqrt(p)
        self._d, self._k, self._x = trial._d, trial._k, trial._x
        self.discarded = trial.discarded
        return p

    # -- matrix-product operators, basis probabilities, purification moves ----------
    def apply_mpo(self, mpo: Sequence[np.ndarray], first: int = 0, max_bond: int | None = None,
                  cutoff: float = 0.0) -> float:
        """``X -> W X`` for a matrix-product operator ``W`` on the physical legs of consecutive sites.

        ``mpo[i]`` has shape ``(D_left, d_out, d_in, D_right)`` and acts on site ``first + i``;
        the outer bonds of the operator have size 1. Hence ``rho -> W rho W^dagger``
        (unnormalized; :meth:`trace` reports the new trace). The Kraus legs are untouched.

        Recompression in canonical gauge: the centre is first moved to ``first``, the operator
        tensors are contracted into the sites (the segment's bonds multiply), a QR sweep makes
        the segment left-orthonormal, and a sweep of SVDs from the right end back to ``first``
        truncates every bond of the segment. Each SVD is taken at the orthogonality centre, so
        each discarded fraction is the exact removed weight, as in :meth:`apply_two`; ``max_bond``
        and ``cutoff`` act as there and the squared norm is kept. Returns the summed discarded
        fraction (also added to ``discarded``). The centre ends on ``first``.
        """
        mpo = [np.asarray(W, dtype=complex) for W in mpo]
        last = first + len(mpo) - 1
        if not mpo or first < 0 or last >= len(self):
            raise ValueError(f"an operator on {len(mpo)} sites does not fit at site {first}")
        if mpo[0].shape[0] != 1 or mpo[-1].shape[3] != 1:
            raise ValueError("the outer bonds of the operator must have size 1")
        x = self._x
        x.canonicalize(first)
        for i, W in enumerate(mpo):
            T = self._t4(first + i)
            Dl, d, k, Dr = T.shape
            if W.ndim != 4 or W.shape[2] != d:
                raise ValueError(f"operator tensor of shape {W.shape} does not act on local dimension {d}")
            wl, dout, _, wr = W.shape
            self._set(first + i, np.einsum("ptsq,askb->aptkbq", W, T).reshape(Dl * wl, dout, k, Dr * wr))
        for j in range(first, last):
            x._shift_right(j)
        total = 0.0
        for j in range(last, first, -1):
            T = x.tensors[j]
            Dl, dk, Dr = T.shape
            U, s, Vh = _svd(T.reshape(Dl, dk * Dr))
            keep, frac, scale = _truncate(s, max_bond, cutoff)
            x.tensors[j] = Vh[:keep].reshape(keep, dk, Dr)
            x.tensors[j - 1] = np.tensordot(x.tensors[j - 1], U[:, :keep] * (s[:keep] * scale), axes=(2, 0))
            total += frac
        x.center = first
        self.discarded += total
        return total

    def probabilities(self, sites: Sequence[int]) -> np.ndarray:
        """Outcome probabilities of a computational-basis measurement of ``sites`` (sorted).

        The diagonal of :meth:`reduced_density` (trace 1), as a vector of length ``prod(d)``
        with the first listed site most significant. One transfer contraction from
        ``sites[0]`` to ``sites[-1]`` keeps only the diagonal of the listed indices, so its
        cost grows with ``prod(d)`` but never with its square. Moves the centre to ``sites[0]``
        (the state is unchanged).
        """
        sites = _check_sites(sites, len(self))
        x = self._x
        x.canonicalize(sites[0])
        D0 = x.tensors[sites[0]].shape[0]
        E = np.eye(D0, dtype=complex)[None]                      # E[outcome, ket bond, bra bond]
        kept = set(sites)
        for j in range(sites[0], sites[-1] + 1):
            T = self._t4(j)
            Y = np.tensordot(E, T, axes=([1], [0]))              # (outcome, bra a', s, k, ket b)
            if j in kept:
                Z = np.einsum("cpskb,pskq->csbq", Y, T.conj(), optimize=True)
                E = Z.reshape(-1, Z.shape[2], Z.shape[3])
            else:
                E = np.tensordot(Y, T.conj(), axes=([1, 2, 3], [0, 1, 2]))
        p = np.real(np.einsum("cbb->c", E))
        return p / p.sum()

    def swap(self, site: int, max_bond: int | None = None, cutoff: float = 0.0) -> float:
        """Exchange the complete local content of ``site`` and ``site + 1``: physical and Kraus legs.

        This relabels two tensor factors of ``X``, so ``rho -> SWAP rho SWAP^dagger`` on the
        physical legs, and each site's purification (Kraus) leg travels with its physical
        content. The pair is split by a truncated SVD in canonical gauge exactly as in
        :meth:`apply_two`; returns the discarded fraction. The centre ends on ``site + 1``.
        """
        self._x._move_to_pair(site)
        theta = np.einsum("askc,ctlb->asktlb", self._t4(site), self._t4(site + 1)).transpose(0, 3, 4, 1, 2, 5)
        A, B, frac = _split_pair(theta, max_bond, cutoff)
        self._set(site, A)
        self._set(site + 1, B)
        self._x.center = site + 1
        self.discarded += frac
        return frac

    def concentrate_purification(self, max_bond: int | None = None, max_kraus: int | None = None,
                                 cutoff: float = 0.0) -> float:
        """Move every Kraus (purification) leg onto the last site; ``rho`` is unchanged.

        A left-to-right sweep: at each bond the Kraus leg of the left site is handed to the
        right site (a relabelling of the purifying space), the pair is split by an SVD in
        canonical gauge, and the right site's Kraus leg is compressed exactly (see
        :meth:`_compress_kraus`). Without truncation the result is the minimal purification:
        the last site's Kraus dimension equals ``rank(rho)`` and bond ``j`` equals the rank of
        the reduced density of sites ``0..j`` (no site left of a bond holds a Kraus leg, so the
        bond's Schmidt rank is that reduced density's rank). Purification entanglement between
        Kraus legs of different sites, which the local SVD compressions cannot remove, is gone.
        Returns the total discarded fraction. The centre ends on the last site.
        """
        n = len(self)
        self._x.canonicalize(0)
        total = 0.0
        for j in range(n - 1):
            A, B = self._t4(j), self._t4(j + 1)
            Dl, d1, k1, _ = A.shape
            _, d2, k2, Dr = B.shape
            theta = np.einsum("askc,ctlb->asktlb", A, B).transpose(0, 1, 3, 2, 4, 5)
            A, B, frac = _split_pair(theta.reshape(Dl, d1, 1, d2, k1 * k2, Dr), max_bond, cutoff)
            self._set(j, A)
            self._set(j + 1, B)
            self._x.center = j + 1
            self.discarded += frac
            total += frac + self._compress_kraus(j + 1, max_kraus, cutoff)
        return total


def diagonal_mpo(values: np.ndarray, tol: float = _RANK_TOL) -> list[np.ndarray]:
    """Exact matrix-product operator of the diagonal operator ``sum_x values[x] |x><x|``.

    ``values`` has one axis per site (site 0 first). Returns tensors of shape
    ``(D_left, d, d, D_right)`` (output index before input index, outer bonds 1) for
    :meth:`LPDO.apply_mpo`. The cores are a tensor train of the table obtained by successive
    SVDs, so bond ``j`` equals the rank of the table across the cut after site ``j`` and is at
    most ``min(prod(d[:j+1]), prod(d[j+1:]))`` (``2^floor(k/2)`` for ``k`` qubits). Singular
    values below ``tol`` times the largest of their cut are numerically zero and dropped.
    """
    table = np.asarray(values, dtype=complex)
    if table.ndim == 0 or table.size == 0:
        raise ValueError("need a table with one axis per site")
    dims = table.shape
    cores, M, Dl = [], table.reshape(1, -1), 1
    for d in dims[:-1]:
        U, s, Vh = _svd(M.reshape(Dl * d, -1))
        keep = max(1, int(np.count_nonzero(s > tol * s[0]))) if s[0] > 0 else 1
        cores.append(U[:, :keep].reshape(Dl, d, keep))
        M = s[:keep, None] * Vh[:keep]
        Dl = keep
    cores.append(M.reshape(Dl, dims[-1], 1))
    mpo = []
    for A in cores:
        W = np.zeros((A.shape[0], A.shape[1], A.shape[1], A.shape[2]), dtype=complex)
        idx = np.arange(A.shape[1])
        W[:, idx, idx, :] = A
        mpo.append(W)
    return mpo
