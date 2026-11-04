"""Canalization, idempotent locks and retention of future behaviour (T11, T13.1, F master loop).

Source: F's master loop (degenerate canalization, compiled lock ``Pi^2 = Pi``);
T11 develops it.

* T11.2 (derived): ``Pi(rho) = Tr_N rho kron sigma_N`` is CPTP, idempotent, and
  preserves every relational expectation ``Tr[rho (A_R kron I_N)]``.
* T11.3 (derived): if ``Pi K = Pi K Pi`` for every relevant unnormalized event
  map ``K``, compiling after each event preserves all future sequence
  probabilities.
* T11.4: incompatible locks can erase retained structure; an isometry ``V``
  adds distinctions without loss: ``Tr(V rho V^dag V A V^dag) = Tr(rho A)``.
* T13.1: a coarse isometry keeps trace only with an explicit remainder flag.

Construction added by this build (decision D7, derived result with proof in
:func:`invariant_algebra`): the smallest unital *-algebra ``A`` that contains
the identity and is closed under the Heisenberg duals of all event maps. The
Hilbert-Schmidt orthogonal projection ``E_A`` onto it is a trace-preserving
conditional expectation (completely positive, unital, idempotent), and
``Pi = E_A`` satisfies ``Pi K = Pi K Pi`` for every such ``K``. Its dimension
measures how many distinctions the future process can ever reveal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .channels import KrausMap, superop_to_kraus
from .linalg import as_matrix, dag, hermitian_part, is_isometry, unvec, vec
from .space import Space

# ----------------------------------------------------------------------------
# T11.2: relation-preserving many-to-one map
# ----------------------------------------------------------------------------

def relation_preserving_map(space: Space, nuisance: Sequence[str], sigma_N: np.ndarray,
                            label: str = "Pi_rel") -> KrausMap:
    """``Pi(rho) = Tr_N(rho) kron sigma_N`` with ``N`` = the named nuisance factors.

    Kraus operators ``I kron sqrt(q_r)|v_r><k|`` over eigenpairs of ``sigma_N``
    and basis states ``k`` of ``N``.
    """
    nuisance = list(nuisance)
    dN = space.dim_of(nuisance)
    sigma_N = as_matrix(sigma_N)
    if sigma_N.shape != (dN, dN):
        raise ValueError("nuisance state has the wrong dimension")
    q, V = np.linalg.eigh(hermitian_part(sigma_N))
    ks = []
    for r in range(dN):
        if q[r] <= 1e-15:
            continue
        for k in range(dN):
            op = np.zeros((dN, dN), dtype=complex)
            op[:, k] = np.sqrt(q[r]) * V[:, r]
            ks.append(space.embed(op, nuisance))
    return KrausMap(ks, label=label)


# ----------------------------------------------------------------------------
# T11.3: the retention identity
# ----------------------------------------------------------------------------

def retention_defect(Pi: KrausMap | np.ndarray, K: KrausMap | np.ndarray) -> float:
    """``max |Pi K - Pi K Pi|`` over Liouville-matrix entries (0 when T11.3's condition holds)."""
    P = Pi.superop() if hasattr(Pi, "superop") else as_matrix(Pi)
    Ks = K.superop() if hasattr(K, "superop") else as_matrix(K)
    return float(np.max(np.abs(P @ Ks - P @ Ks @ P)))


def idempotence_defect(Pi: KrausMap | np.ndarray) -> float:
    P = Pi.superop() if hasattr(Pi, "superop") else as_matrix(Pi)
    return float(np.max(np.abs(P @ P - P)))


# ----------------------------------------------------------------------------
# Exact canalization discovery (decision D7)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class CanalizationAlgebra:
    """The retained algebra ``A`` and its conditional expectation."""

    basis: np.ndarray          # (D^2, n) orthonormal columns: vec of an HS-orthonormal basis of A
    d: int

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]

    def superop(self) -> np.ndarray:
        """Liouville matrix of ``E_A`` (the HS-orthogonal projector onto ``A``)."""
        return self.basis @ dag(self.basis)

    def channel(self, label: str = "E_A") -> KrausMap:
        return superop_to_kraus(self.superop(), self.d, self.d, label=label)

    def project(self, R: np.ndarray) -> np.ndarray:
        return unvec(self.basis @ (dag(self.basis) @ vec(R)), self.d)

    def compress(self, R: np.ndarray) -> np.ndarray:
        """Coordinates of ``E_A(R)``: ``n`` numbers instead of ``D^2``."""
        return dag(self.basis) @ vec(R)

    def decompress(self, coords: np.ndarray) -> np.ndarray:
        return unvec(self.basis @ coords, self.d)

    def contains(self, X: np.ndarray, atol: float = 1e-9) -> bool:
        v = vec(X)
        r = v - self.basis @ (dag(self.basis) @ v)
        return float(np.linalg.norm(r)) <= atol * max(1.0, float(np.linalg.norm(v)))


class _Span:
    """Incrementally grown orthonormal basis of a subspace of C^{D^2} (preallocated buffer)."""

    def __init__(self, dim: int, tol: float) -> None:
        self.buf = np.zeros((dim, dim), dtype=complex)
        self.n = 0
        self.tol = tol

    @property
    def cols(self) -> list[np.ndarray]:
        return [self.buf[:, k] for k in range(self.n)]

    def matrix(self) -> np.ndarray:
        return self.buf[:, :self.n].copy()

    def residual(self, v: np.ndarray) -> np.ndarray:
        B = self.buf[:, :self.n]
        r = v.astype(complex)
        for _ in range(2):  # classical Gram-Schmidt applied twice is numerically orthogonal
            r = r - B @ (B.conj().T @ r)
        return r

    def add(self, v: np.ndarray) -> bool:
        n0 = float(np.linalg.norm(v))
        if n0 == 0.0 or self.n == self.buf.shape[1]:
            return False
        r = self.residual(v)
        nr = float(np.linalg.norm(r))
        if nr <= self.tol * n0:
            return False
        self.buf[:, self.n] = r / nr
        self.n += 1
        return True


def invariant_algebra(maps: Sequence[KrausMap], d: int, seeds: Sequence[np.ndarray] = (),
                      tol: float = 1e-9, max_dim: int | None = None) -> CanalizationAlgebra:
    """Smallest unital *-algebra containing ``seeds`` and invariant under every ``K^dagger``.

    Proof that ``Pi = E_A`` satisfies T11.3: ``E_A`` is self-adjoint for the
    Hilbert-Schmidt inner product, so ``(Pi K Pi)^dagger = E_A K^dagger E_A``.
    For any ``X``, ``E_A X`` lies in ``A`` and ``K^dagger`` maps ``A`` into
    ``A``, hence ``E_A K^dagger E_A X = K^dagger E_A X``; that is
    ``(Pi K Pi)^dagger = (Pi K)^dagger``. ``E_A`` is completely positive and
    trace preserving because ``A`` is a unital *-subalgebra (the
    trace-preserving conditional expectation onto ``A``).

    Algorithm: grow the span of words in the generators (seeds, their
    adjoints, and images of the basis under each dual map) until the span is
    closed under multiplication by the generators, adjoints and all duals.
    The dimension is nondecreasing and bounded by ``d^2``, so it terminates.
    """
    max_dim = d * d if max_dim is None else max_dim
    span = _Span(d * d, tol)
    span.add(vec(np.eye(d)))
    gens: list[np.ndarray] = []

    def add_gen(G: np.ndarray) -> None:
        for H in (G, dag(G)):
            if np.linalg.norm(H) > 0:
                gens.append(H)
                span.add(vec(H))

    for s in seeds:
        add_gen(as_matrix(s))
    changed = True
    while changed:
        changed = False
        # multiplicative closure: words in the generators
        frontier = list(range(span.n))
        while frontier:
            new_frontier = []
            for idx in frontier:
                b = unvec(span.buf[:, idx], d)
                for G in gens:
                    for prod in (b @ G, G @ b):
                        if span.add(vec(prod)):
                            new_frontier.append(span.n - 1)
                if span.add(vec(dag(b))):
                    new_frontier.append(span.n - 1)
            frontier = new_frontier
            if span.n > max_dim:
                raise RuntimeError("algebra exceeded the declared dimension bound")
        # invariance under the Heisenberg duals
        for idx in range(span.n):
            b = unvec(span.buf[:, idx], d)
            for K in maps:
                img = K.dual(b)
                v = vec(img)
                if np.linalg.norm(span.residual(v)) > tol * max(1.0, float(np.linalg.norm(v))):
                    add_gen(img)
                    changed = True
    return CanalizationAlgebra(span.matrix(), d)


# ----------------------------------------------------------------------------
# T11.4 and T13.1: growth and coarse-graining
# ----------------------------------------------------------------------------

def isometric_nesting(V: np.ndarray, rho: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(V rho V^dagger, V A V^dagger)``; expectations are preserved exactly (T11.4)."""
    V = as_matrix(V)
    if not is_isometry(V):
        raise ValueError("V is not an isometry")
    return V @ as_matrix(rho) @ dag(V), V @ as_matrix(A) @ dag(V)


def append_factor_isometry(d: int, d_new: int, state: int = 0) -> np.ndarray:
    """``V = I_d kron |state>``: adds a new factor without disturbing the old one."""
    e = np.zeros((d_new, 1), dtype=complex)
    e[state, 0] = 1.0
    return np.kron(np.eye(d, dtype=complex), e)


def coarse_channel_with_remainder(V: np.ndarray, label: str = "C") -> KrausMap:
    """``C(R) = V^dagger R V  (+)  Tr[(I - P) R] |perp><perp|`` with ``P = V V^dagger`` (T13.1).

    Output dimension ``d_coarse + 1``; the last basis vector is the remainder
    flag ``|perp>``. Kraus operators: ``V^dagger`` padded with a zero row, and
    ``|perp><f_k|`` for an orthonormal basis ``f_k`` of the complement of ``P``.
    """
    V = as_matrix(V)
    if not is_isometry(V):
        raise ValueError("V is not an isometry")
    d_fine, d_coarse = V.shape
    K0 = np.zeros((d_coarse + 1, d_fine), dtype=complex)
    K0[:d_coarse] = dag(V)
    ks = [K0]
    P = V @ dag(V)
    w, U = np.linalg.eigh(np.eye(d_fine) - P)
    for k in range(d_fine):
        if w[k] > 0.5:
            Kk = np.zeros((d_coarse + 1, d_fine), dtype=complex)
            Kk[d_coarse] = U[:, k].conj()
            ks.append(Kk)
    return KrausMap(ks, label=label)


# ----------------------------------------------------------------------------
# T12.2: consolidation channel
# ----------------------------------------------------------------------------

def consolidation_superop(r: float, Pi_L: KrausMap | np.ndarray) -> np.ndarray:
    """``F = r Id + (1 - r) Pi_L``; ``F^n = Pi_L + r^n (Id - Pi_L)`` when ``Pi_L^2 = Pi_L``."""
    if not 0 <= r < 1:
        raise ValueError("need 0 <= r < 1")
    P = Pi_L.superop() if isinstance(Pi_L, KrausMap) else as_matrix(Pi_L)
    return r * np.eye(P.shape[0]) + (1 - r) * P
