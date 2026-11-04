"""An exact whirlpool-like system with a Markov blanket (T5).

State ``X = (x, y, b, e)``: interior ``(x, y)``, blanket ``b``, exterior ``e``.
Stationary Gaussian with precision ``J``; drift ``A = (Q - I) J`` with
antisymmetric ``Q`` (rotation in the interior plane); SDE
``dX = A X dt + sqrt(2) dW``. Everything is a chosen realization from T5,
reproduced exactly here; the module also provides an exact-discretization
sampler so the blanket can be checked on simulated data.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as sla

J = np.array([[2, 0, -1, 0],
              [0, 2, 0, 0],
              [-1, 0, 3, -1],
              [0, 0, -1, 2]], dtype=float)

Q = np.array([[0, -1, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]], dtype=float)

A = (Q - np.eye(4)) @ J
SIGMA = np.linalg.inv(J)
INTERIOR, BLANKET, EXTERIOR = [0, 1], [2], [3]


def leading_minors(M: np.ndarray) -> list[float]:
    return [float(np.linalg.det(M[:k, :k])) for k in range(1, M.shape[0] + 1)]


def lyapunov_residual() -> np.ndarray:
    """``A Sigma + Sigma A^T + 2 I`` (zero for the stationary covariance)."""
    return A @ SIGMA + SIGMA @ A.T + 2 * np.eye(4)


def conditional_interior(b: float, e: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of ``(x, y) | b, e`` from the precision matrix."""
    J_II = J[np.ix_(INTERIOR, INTERIOR)]
    J_IO = J[np.ix_(INTERIOR, BLANKET + EXTERIOR)]
    cov = np.linalg.inv(J_II)
    mean = -cov @ J_IO @ np.array([b, e], dtype=float)
    return mean, cov


def interior_drift() -> np.ndarray:
    """Interior block of ``A`` (dynamics of ``(x, y)`` around ``(b/2, 0)`` for fixed ``b``)."""
    return A[np.ix_(INTERIOR, INTERIOR)]


def planar_vorticity(M: np.ndarray) -> float:
    """``d v_y/dx - d v_x/dy`` for the linear field ``v = M z``."""
    return float(M[1, 0] - M[0, 1])


def radius_multiplier(t: float) -> float:
    """``|z(t)| / |z(0)|`` for the interior drift (a rotation times ``exp(-2t)``)."""
    Mt = sla.expm(interior_drift() * t)
    s = np.linalg.svd(Mt, compute_uv=False)
    if not np.isclose(s[0], s[1], rtol=1e-12):
        raise ArithmeticError("interior flow is not conformal")
    return float(s[0])


def sample_stationary_path(n: int, h: float, rng: np.random.Generator) -> np.ndarray:
    """Exact discretization of the OU process started in its stationary law.

    ``X_{k+1} = exp(A h) X_k + eta_k``, ``Cov(eta) = Sigma - exp(Ah) Sigma exp(Ah)^T``.
    """
    F = sla.expm(A * h)
    Qh = SIGMA - F @ SIGMA @ F.T
    Lq = np.linalg.cholesky(0.5 * (Qh + Qh.T))
    Ls = np.linalg.cholesky(SIGMA)
    out = np.empty((n, 4))
    x = Ls @ rng.standard_normal(4)
    for k in range(n):
        out[k] = x
        x = F @ x + Lq @ rng.standard_normal(4)
    return out
