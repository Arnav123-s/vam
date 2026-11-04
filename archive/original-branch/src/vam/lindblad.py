"""Time-local open dynamics, einselection and dynamical locks (T6.3, T6.4, T7.3, T11.1).

Lindblad form (T6.3, [R5]):

    dR/dt = -(i/hbar)[H, R] + sum_nu ( L_nu R L_nu^dagger - 1/2 {L_nu^dagger L_nu, R} )

Liouville matrices use column stacking (DESIGN D3):

    vec(-i[H,R])        = -i (I kron H - H^T kron I) vec(R)
    vec(L R L^dagger)   = (conj(L) kron L) vec(R)
    vec({M, R})         = (I kron M + M^T kron I) vec(R)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import linalg as sla

from .channels import KrausMap, superop_to_kraus
from .linalg import ATOL, as_matrix, dag, unvec, vec


def liouvillian(H: np.ndarray | None, jumps: Sequence[np.ndarray] = (), hbar: float = 1.0,
                dim: int | None = None) -> np.ndarray:
    """Liouville matrix of the Lindblad generator."""
    if H is None and dim is None and not jumps:
        raise ValueError("need H, jumps or dim")
    d = as_matrix(H).shape[0] if H is not None else (as_matrix(jumps[0]).shape[0] if jumps else dim)
    I = np.eye(d, dtype=complex)
    L = np.zeros((d * d, d * d), dtype=complex)
    if H is not None:
        H = as_matrix(H)
        L += (-1j / hbar) * (np.kron(I, H) - np.kron(H.T, I))
    for Lk in jumps:
        Lk = as_matrix(Lk)
        LdL = dag(Lk) @ Lk
        L += np.kron(Lk.conj(), Lk) - 0.5 * (np.kron(I, LdL) + np.kron(LdL.T, I))
    return L


def lindblad_rhs(R: np.ndarray, H: np.ndarray | None, jumps: Sequence[np.ndarray] = (),
                 hbar: float = 1.0) -> np.ndarray:
    """Right-hand side of the Lindblad equation evaluated directly (no superoperator)."""
    R = as_matrix(R)
    out = np.zeros_like(R)
    if H is not None:
        out += (-1j / hbar) * (as_matrix(H) @ R - R @ as_matrix(H))
    for Lk in jumps:
        Lk = as_matrix(Lk)
        LdL = dag(Lk) @ Lk
        out += Lk @ R @ dag(Lk) - 0.5 * (LdL @ R + R @ LdL)
    return out


def semigroup(L: np.ndarray, t: float) -> np.ndarray:
    """``exp(t L)`` for a Liouville matrix ``L``."""
    return sla.expm(t * as_matrix(L))


def evolve(R: np.ndarray, L: np.ndarray, t: float) -> np.ndarray:
    R = as_matrix(R)
    return unvec(semigroup(L, t) @ vec(R), R.shape[0])


def semigroup_channel(L: np.ndarray, t: float, d: int, label: str = "e^{tL}") -> KrausMap:
    """Kraus form of ``exp(tL)`` (completely positive for a Lindblad ``L``)."""
    return superop_to_kraus(semigroup(L, t), d, d, label=label)


# ----------------------------------------------------------------------------
# Einselection (T6.4)
# ----------------------------------------------------------------------------

def dephasing_jump(A_B: np.ndarray, gamma: float) -> np.ndarray:
    """Jump operator ``sqrt(gamma) A_B``: its dissipator is ``-gamma/2 [A_B, [A_B, R]]``."""
    if gamma < 0:
        raise ValueError("dephasing rate must be nonnegative")
    A = as_matrix(A_B)
    if not np.allclose(A, dag(A), atol=ATOL):
        raise ValueError("pointer observable must be Hermitian")
    return np.sqrt(gamma) * A


def dephasing_closed_form(R: np.ndarray, eigenvalues: np.ndarray, gamma: float, t: float) -> np.ndarray:
    """``R_aa'(t) = exp(-gamma (a - a')^2 t / 2) R_aa'(0)`` in the pointer eigenbasis (T6.4)."""
    a = np.asarray(eigenvalues, dtype=float)
    factor = np.exp(-0.5 * gamma * (a[:, None] - a[None, :]) ** 2 * t)
    return as_matrix(R) * factor


# ----------------------------------------------------------------------------
# Sector dissipator: classical boundary, coherent interior (T7.3)
# ----------------------------------------------------------------------------

def sector_projection_superop(projectors: Sequence[np.ndarray]) -> np.ndarray:
    """Liouville matrix of ``Pi_B(R) = sum_r P_r R P_r``."""
    return sum(np.kron(as_matrix(P).conj(), as_matrix(P)) for P in projectors)


def sector_dissipator(projectors: Sequence[np.ndarray], gamma: float) -> np.ndarray:
    """``D_B = gamma (Pi_B - I)`` as a Liouville matrix.

    It is a Lindblad generator: ``gamma (sum_r P_r R P_r - R)`` equals the
    dissipator with jump operators ``sqrt(gamma) P_r``, because
    ``sum_r P_r^dagger P_r = I``.
    """
    Pi = sector_projection_superop(projectors)
    return gamma * (Pi - np.eye(Pi.shape[0]))


def sector_dissipator_closed_form(R: np.ndarray, projectors: Sequence[np.ndarray], gamma: float,
                                  t: float) -> np.ndarray:
    """``exp(t D_B) R = Pi_B(R) + exp(-gamma t) (R - Pi_B(R))`` (T7.3)."""
    R = as_matrix(R)
    PiR = sum(as_matrix(P) @ R @ as_matrix(P) for P in projectors)
    return PiR + np.exp(-gamma * t) * (R - PiR)


# ----------------------------------------------------------------------------
# Locks from dynamics (T11.1)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class LockReport:
    """Outcome of constructing ``Pi = lim_{t -> inf} exp(tL)``."""

    settled: bool
    gap: float                   # smallest |Re lambda| over the decaying modes
    oscillating_modes: int       # eigenvalues on the imaginary axis other than 0
    kernel_dimension: int
    idempotence_defect: float    # ||Pi^2 - Pi||
    stationarity_defect: float   # ||L Pi||


def spectral_lock(L: np.ndarray, d: int, atol: float = 1e-9) -> tuple[np.ndarray | None, LockReport]:
    """Construct the dynamical lock ``Pi = lim exp(tL)`` if it exists.

    The sufficient spectral condition of T11.1 is checked: zero is the only
    eigenvalue on the imaginary axis and every other eigenvalue has negative
    real part. Nonzero imaginary modes are reported as ongoing dynamics and no
    lock is returned for them (T11.1 forbids calling them a settled lock).
    """
    L = as_matrix(L)
    lam = np.linalg.eigvals(L)
    scale = max(1.0, float(np.max(np.abs(lam))))
    on_axis = np.abs(lam.real) <= atol * scale
    zero = on_axis & (np.abs(lam) <= atol * scale)
    oscillating = int(np.sum(on_axis & ~zero))
    unstable = int(np.sum(lam.real > atol * scale))
    decaying = lam[~on_axis]
    gap = float(np.min(np.abs(decaying.real))) if decaying.size else np.inf
    kernel_dim = int(np.sum(zero))
    if oscillating or unstable:
        return None, LockReport(False, gap, oscillating, kernel_dim, np.nan, np.nan)
    # Riesz projector onto ker(L) along ran(L): Pi = V (W^dagger V)^{-1} W^dagger,
    # with V, W bases of the right and left null spaces. W^dagger V is invertible
    # exactly when the zero eigenvalue is semisimple; then Pi = lim exp(tL)
    # because every other eigenvalue has negative real part.
    V = sla.null_space(L, rcond=atol)
    W = sla.null_space(dag(L), rcond=atol)
    if V.shape[1] == 0 or V.shape[1] != W.shape[1]:
        return None, LockReport(False, gap, oscillating, kernel_dim, np.nan, np.nan)
    G = dag(W) @ V
    if np.linalg.cond(G) > 1e10:
        return None, LockReport(False, gap, oscillating, kernel_dim, np.nan, np.nan)
    Pi = V @ np.linalg.solve(G, dag(W))
    idem = float(np.max(np.abs(Pi @ Pi - Pi)))
    stat = float(max(np.max(np.abs(L @ Pi)), np.max(np.abs(Pi @ L))))
    settled = idem < 1e-8 and stat < 1e-8
    return Pi, LockReport(settled, gap, oscillating, V.shape[1], idem, stat)
