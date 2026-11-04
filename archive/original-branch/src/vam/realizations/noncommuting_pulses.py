"""Order-sensitive composition of bit-driven generators (T2.3).

Chosen realization: ``U_0 = exp(-i pi Z / 4)`` (mass-axis pulse),
``U_1 = exp(-i pi X / 4)`` (kinetic-axis pulse) in a two-component Dirac-sector
illustration, starting from ``|0>``. Input order ``01`` applies ``U_0`` first.
"""

from __future__ import annotations

import numpy as np

from ..linalg import X, Z, unitary_from_hamiltonian

REALIZATION = {
    "source": "T2.3",
    "status": "chosen realization",
    "U0": "exp(-i pi Z/4)",
    "U1": "exp(-i pi X/4)",
    "initial": "|0>",
}

U = {0: unitary_from_hamiltonian(Z, np.pi / 4), 1: unitary_from_hamiltonian(X, np.pi / 4)}
GENERATORS = {0: -1j * Z, 1: -1j * X}


def final_state(order: str) -> np.ndarray:
    psi = np.array([1, 0], dtype=complex)
    for ch in order:
        psi = U[int(ch)] @ psi
    return psi
