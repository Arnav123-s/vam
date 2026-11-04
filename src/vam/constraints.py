"""Persistent hard constraints (T13.2, F-Part 9 "null space" constraints).

Derived (T13.2): if ``R = PRP``, ``H = PHP`` and ``L_nu = P L_nu P`` for the
projector ``P`` onto allowed configurations, then no probability ever enters
the forbidden complement. A zero eigenvalue of the current density alone is
not a constraint: a generator can populate that direction later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .linalg import ATOL, as_matrix
from .lindblad import lindblad_rhs


@dataclass(frozen=True)
class ConstraintReport:
    state_supported: float     # ||R - PRP||
    hamiltonian_supported: float  # ||H - PHP||
    jumps_supported: float     # max ||L - PLP||

    @property
    def persistent(self) -> bool:
        return max(self.state_supported, self.hamiltonian_supported, self.jumps_supported) <= ATOL


def constraint_report(P: np.ndarray, R: np.ndarray, H: np.ndarray | None = None,
                      jumps: Sequence[np.ndarray] = ()) -> ConstraintReport:
    P = as_matrix(P)
    sandwich = lambda A: float(np.max(np.abs(as_matrix(A) - P @ as_matrix(A) @ P)))
    return ConstraintReport(
        sandwich(R),
        sandwich(H) if H is not None else 0.0,
        max((sandwich(L) for L in jumps), default=0.0),
    )


def forbidden_population(P: np.ndarray, R: np.ndarray) -> float:
    """``Tr((I - P) R)``."""
    P = as_matrix(P)
    return float(np.real(np.trace((np.eye(P.shape[0]) - P) @ as_matrix(R))))


def leakage_rate(P: np.ndarray, R: np.ndarray, H: np.ndarray | None, jumps: Sequence[np.ndarray] = ()) -> float:
    """``d/dt Tr((I - P) R)`` under the Lindblad generator at the current state."""
    P = as_matrix(P)
    return float(np.real(np.trace((np.eye(P.shape[0]) - P) @ lindblad_rhs(R, H, jumps))))
