"""Finite-dimensional linear algebra used by every other module.

Conventions (decision log D2-D3 in DESIGN.md)
---------------------------------------------
* All arrays are ``complex128`` unless a function says otherwise.
* Qubit basis: ``|0> = (1, 0)``, ``|1> = (0, 1)``; ``Z|0> = |0>``.
* ``kron(A, B, ...)`` acts with ``A`` on the first tensor factor.
* ``hbar = 1`` unless a caller passes another value.

The spectral helpers here are the only place where eigen-decompositions are
taken, so tolerances are applied consistently.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import reduce

import numpy as np
from scipy import linalg as sla

#: Default absolute tolerance for structural identities on states and maps.
ATOL = 1e-10

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}

#: Projectors onto the qubit computational basis.
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)


class LinalgError(ValueError):
    """Raised when an input violates a structural requirement."""


def as_matrix(A) -> np.ndarray:
    """Return ``A`` as a 2-D complex array (copying only when needed)."""
    M = np.asarray(A, dtype=complex)
    if M.ndim != 2:
        raise LinalgError(f"expected a matrix, got shape {M.shape}")
    return M


def dag(A: np.ndarray) -> np.ndarray:
    """Hermitian conjugate."""
    return np.conjugate(np.asarray(A)).T


def kron(*ops: np.ndarray) -> np.ndarray:
    """Tensor product of the arguments, first argument on the first factor."""
    if not ops:
        return np.ones((1, 1), dtype=complex)
    return reduce(np.kron, [np.asarray(o, dtype=complex) for o in ops])


def ket(index: int, dim: int) -> np.ndarray:
    """Computational-basis column vector ``|index>`` of dimension ``dim``."""
    if not 0 <= index < dim:
        raise LinalgError(f"basis index {index} outside dimension {dim}")
    v = np.zeros(dim, dtype=complex)
    v[index] = 1.0
    return v


def ket_from_bits(bits: str) -> np.ndarray:
    """``ket_from_bits("01")`` is ``|0>|1>``."""
    return kron(*[ket(int(b), 2)[:, None] for b in bits])[:, 0]


def projector(vec: np.ndarray) -> np.ndarray:
    """``|v><v|`` for a (not necessarily normalized) vector ``v``."""
    v = np.asarray(vec, dtype=complex).reshape(-1)
    return np.outer(v, v.conj())


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B + B @ A


def hermitian_part(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + dag(A))


def is_hermitian(A: np.ndarray, atol: float = ATOL) -> bool:
    A = as_matrix(A)
    return A.shape[0] == A.shape[1] and np.allclose(A, dag(A), atol=atol, rtol=0)


def min_eigenvalue(A: np.ndarray) -> float:
    """Smallest eigenvalue of the Hermitian part of ``A``."""
    return float(np.linalg.eigvalsh(hermitian_part(as_matrix(A)))[0])


def is_psd(A: np.ndarray, atol: float = ATOL) -> bool:
    """Hermitian and positive semidefinite up to ``atol``."""
    return is_hermitian(A, atol) and min_eigenvalue(A) >= -atol


def is_unitary(U: np.ndarray, atol: float = ATOL) -> bool:
    U = as_matrix(U)
    n = U.shape[0]
    return U.shape == (n, n) and np.allclose(dag(U) @ U, np.eye(n), atol=atol, rtol=0)


def is_isometry(V: np.ndarray, atol: float = ATOL) -> bool:
    V = as_matrix(V)
    return np.allclose(dag(V) @ V, np.eye(V.shape[1]), atol=atol, rtol=0)


def is_projector(P: np.ndarray, atol: float = ATOL) -> bool:
    P = as_matrix(P)
    return is_hermitian(P, atol) and np.allclose(P @ P, P, atol=atol, rtol=0)


def expm(A: np.ndarray) -> np.ndarray:
    """Matrix exponential (Pade scaling and squaring)."""
    return sla.expm(as_matrix(A))


def unitary_from_hamiltonian(H: np.ndarray, t: float, hbar: float = 1.0) -> np.ndarray:
    """``exp(-i H t / hbar)`` computed from the eigen-decomposition of Hermitian ``H``.

    The spectral route keeps the result unitary to machine precision, which
    the Pade route does not guarantee for long times.
    """
    H = as_matrix(H)
    if not is_hermitian(H, atol=1e-9):
        raise LinalgError("Hamiltonian is not Hermitian")
    evals, evecs = np.linalg.eigh(hermitian_part(H))
    phases = np.exp(-1j * evals * t / hbar)
    return (evecs * phases) @ dag(evecs)


def permute_operator(op: np.ndarray, dims: Sequence[int], perm: Sequence[int]) -> np.ndarray:
    """Reorder the tensor factors of an operator.

    ``op`` acts on factors with dimensions ``dims`` (in that order). The result
    acts on the factors in the order ``perm``: position ``i`` of the result is
    original factor ``perm[i]``.
    """
    dims = list(dims)
    n = len(dims)
    if sorted(perm) != list(range(n)):
        raise LinalgError(f"{perm} is not a permutation of {n} factors")
    op = as_matrix(op)
    D = int(np.prod(dims))
    if op.shape != (D, D):
        raise LinalgError(f"operator shape {op.shape} does not match dims {dims}")
    t = op.reshape(dims + dims)
    axes = list(perm) + [n + p for p in perm]
    return t.transpose(axes).reshape(D, D)


def permute_vector(vec: np.ndarray, dims: Sequence[int], perm: Sequence[int]) -> np.ndarray:
    """Reorder tensor factors of a state vector (same convention as ``permute_operator``)."""
    dims = list(dims)
    v = np.asarray(vec, dtype=complex).reshape(dims)
    return v.transpose(list(perm)).reshape(-1)


def partial_trace(rho: np.ndarray, dims: Sequence[int], keep: Iterable[int]) -> np.ndarray:
    """Trace out every factor not listed in ``keep``.

    The kept factors stay in their original order.
    """
    dims = list(dims)
    n = len(dims)
    keep = sorted(set(keep))
    if any(k < 0 or k >= n for k in keep):
        raise LinalgError(f"keep={keep} outside {n} factors")
    rho = as_matrix(rho)
    D = int(np.prod(dims))
    if rho.shape != (D, D):
        raise LinalgError(f"density shape {rho.shape} does not match dims {dims}")
    t = rho.reshape(dims + dims)
    current = n
    for i in sorted((i for i in range(n) if i not in keep), reverse=True):
        t = np.trace(t, axis1=i, axis2=i + current)
        current -= 1
    d = int(np.prod([dims[k] for k in keep])) if keep else 1
    return t.reshape(d, d)


def purity(rho: np.ndarray) -> float:
    """Purity gauge ``Tr(rho^2)`` (F-Part 4)."""
    rho = as_matrix(rho)
    return float(np.real(np.sum(rho * rho.T)))


def eigvalsh_clipped(rho: np.ndarray, atol: float = ATOL) -> np.ndarray:
    """Eigenvalues of a density matrix with round-off negatives set to zero.

    Raises if a negative eigenvalue exceeds ``atol``: that is a positivity
    violation, not round-off.
    """
    w = np.linalg.eigvalsh(hermitian_part(as_matrix(rho)))
    if w[0] < -atol:
        raise LinalgError(f"matrix has negative eigenvalue {w[0]:.3e}")
    return np.clip(w, 0.0, None)


def von_neumann_entropy(rho: np.ndarray, base: float = 2.0, atol: float = ATOL) -> float:
    """``-Tr rho log rho`` (bits by default)."""
    w = eigvalsh_clipped(rho, atol)
    w = w[w > 0]
    return float(-np.sum(w * np.log(w)) / np.log(base))


def trace_norm(A: np.ndarray) -> float:
    """Schatten-1 norm (sum of singular values)."""
    return float(np.sum(np.linalg.svd(as_matrix(A), compute_uv=False)))


def trace_distance(r1: np.ndarray, r2: np.ndarray) -> float:
    """``D(R, R') = ||R - R'||_1 / 2`` (T12.3)."""
    return 0.5 * trace_norm(as_matrix(r1) - as_matrix(r2))


def expectation(rho: np.ndarray, A: np.ndarray) -> complex:
    """``Tr(rho A)``."""
    return complex(np.trace(as_matrix(rho) @ as_matrix(A)))


def bloch_vector(rho: np.ndarray) -> np.ndarray:
    """``(<X>, <Y>, <Z>)`` of a qubit density matrix."""
    rho = as_matrix(rho)
    if rho.shape != (2, 2):
        raise LinalgError("Bloch vector needs a qubit density matrix")
    return np.array([np.real(expectation(rho, P)) for P in (X, Y, Z)])


def pauli_string(labels: str) -> np.ndarray:
    """``pauli_string("XZ")`` is ``X kron Z``."""
    return kron(*[PAULI[c] for c in labels])


# ----------------------------------------------------------------------------
# Random objects for property tests (seeded generators only)
# ----------------------------------------------------------------------------

def random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary (QR of a complex Ginibre matrix with phase fix)."""
    G = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(2)
    Q, R = np.linalg.qr(G)
    d = np.diag(R)
    return Q * (d / np.abs(d))


def random_hermitian(dim: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    G = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return scale * 0.5 * (G + dag(G))


def random_density(dim: int, rng: np.random.Generator, rank: int | None = None) -> np.ndarray:
    """Random density matrix ``AA^dagger / Tr`` with ``A`` of the given rank."""
    rank = dim if rank is None else rank
    A = rng.standard_normal((dim, rank)) + 1j * rng.standard_normal((dim, rank))
    R = A @ dag(A)
    return R / np.trace(R)


def random_pure_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return v / np.linalg.norm(v)


# ----------------------------------------------------------------------------
# Operator-space helpers (used by canalization discovery)
# ----------------------------------------------------------------------------

def vec(A: np.ndarray) -> np.ndarray:
    """Column-stacking vectorization: ``vec(A X B) = (B^T kron A) vec(X)``."""
    return as_matrix(A).reshape(-1, order="F")


def unvec(v: np.ndarray, rows: int, cols: int | None = None) -> np.ndarray:
    cols = rows if cols is None else cols
    return np.asarray(v, dtype=complex).reshape(rows, cols, order="F")


def hs_inner(A: np.ndarray, B: np.ndarray) -> complex:
    """Hilbert-Schmidt inner product ``Tr(A^dagger B)``."""
    return complex(np.vdot(as_matrix(A).reshape(-1), as_matrix(B).reshape(-1)))


def orthonormal_operator_basis(ops: Sequence[np.ndarray], atol: float = 1e-9) -> list[np.ndarray]:
    """Hilbert-Schmidt orthonormal basis of ``span(ops)`` (rank-revealing SVD)."""
    if not ops:
        return []
    d0, d1 = as_matrix(ops[0]).shape
    M = np.stack([vec(o) for o in ops], axis=1)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(s > atol * max(1.0, s[0])))
    return [unvec(U[:, k], d0, d1) for k in range(r)]
