"""Density operators: validation and constructors (T1.2, T3.1, F-Part 4).

A density operator satisfies ``R = R^dagger``, ``R >= 0`` and ``Tr R = 1``.
Functions here never "repair" a state silently: they either confirm the
three conditions within tolerance or raise :class:`DensityError` naming the
violated condition and its size.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .linalg import ATOL, LinalgError, as_matrix, dag, hermitian_part, min_eigenvalue, projector


class DensityError(LinalgError):
    """A matrix that was required to be a density operator is not one."""


class ZeroProbabilityError(DensityError):
    """Conditioning on an outcome the current model assigns probability zero (T10.2)."""


def density_violations(R: np.ndarray) -> dict[str, float]:
    """Sizes of the three possible violations (0 means satisfied exactly)."""
    R = as_matrix(R)
    return {
        "hermiticity": float(np.max(np.abs(R - dag(R)))) if R.size else 0.0,
        "negativity": max(0.0, -min_eigenvalue(R)),
        "trace": float(abs(np.trace(R) - 1.0)),
    }


def check_density(R: np.ndarray, atol: float = ATOL) -> np.ndarray:
    """Return ``R`` as an array if it is a density operator within ``atol``; raise otherwise."""
    R = as_matrix(R)
    if R.shape[0] != R.shape[1]:
        raise DensityError(f"density must be square, got {R.shape}")
    v = density_violations(R)
    bad = {k: x for k, x in v.items() if x > atol}
    if bad:
        raise DensityError(f"not a density operator: {bad}")
    return R


def is_density(R: np.ndarray, atol: float = ATOL) -> bool:
    try:
        check_density(R, atol)
    except DensityError:
        return False
    return True


def pure(psi: np.ndarray) -> np.ndarray:
    """``|psi><psi|`` for a normalized vector."""
    v = np.asarray(psi, dtype=complex).reshape(-1)
    n = np.linalg.norm(v)
    if not np.isclose(n, 1.0, atol=1e-12, rtol=0):
        raise DensityError(f"state vector has norm {n}")
    return projector(v)


def mixture(probs: Sequence[float], states: Sequence[np.ndarray]) -> np.ndarray:
    """``sum_i p_i rho_i`` with a probability vector ``p`` (F-Part 4)."""
    p = np.asarray(probs, dtype=float)
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0, atol=1e-12, rtol=0):
        raise DensityError(f"weights {p} are not a probability vector")
    mats = [as_matrix(s) if np.ndim(s) == 2 else pure(s) for s in states]
    return sum(pi * m for pi, m in zip(p, mats, strict=True))


def from_factor(A: np.ndarray) -> np.ndarray:
    """``R = A A^dagger / Tr(A A^dagger)``: positive by construction (T3.1, T13.1)."""
    A = np.asarray(A, dtype=complex)
    if A.ndim == 1:
        A = A[:, None]
    R = A @ dag(A)
    tr = np.real(np.trace(R))
    if tr <= 0:
        raise DensityError("factor is zero")
    return R / tr


def maximally_mixed(dim: int) -> np.ndarray:
    return np.eye(dim, dtype=complex) / dim


def condition(unnormalized: np.ndarray, atol: float = ATOL) -> tuple[np.ndarray, float]:
    """Normalize the output of a trace-nonincreasing outcome map.

    Returns ``(R / q, q)`` with ``q = Tr R``. A probability at or below
    ``atol`` raises :class:`ZeroProbabilityError`: the formalism does not
    justify inventing a posterior by dividing by zero (T10.2).
    """
    U = as_matrix(unnormalized)
    q = float(np.real(np.trace(U)))
    if q <= atol:
        raise ZeroProbabilityError(f"outcome probability {q:.3e} is zero within tolerance")
    return hermitian_part(U) / q, q
