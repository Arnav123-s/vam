"""Vector-weight-output-feedback calculation (T9).

Chosen realization (T9): information spinor ``Q`` and two-state relational
control ``W``; ``hbar = 1``;

    H = X_Q kron Z_W + I_Q kron X_W        (H^2 = 2 I)
    t_s = pi / (4 sqrt 2)

Output: ideal ``Z_Q`` readout after ``U(t_s)``. Re-entry of emitted bit ``b``:
``V_b kron V_b`` with ``V_b = exp(-i (2b - 1) (pi/4) X)``. Each event is a
closed step (the declared identity external injection of T14 step 15).
"""

from __future__ import annotations

import numpy as np

from ..channels import Instrument, KrausMap, identity_channel, unitary_channel
from ..linalg import I2, P0, P1, X, Z, kron, unitary_from_hamiltonian
from ..machine import EventProtocol, Machine, MachineState
from ..space import Space

REALIZATION = {
    "source": "T9",
    "status": "chosen realization",
    "hamiltonian": "X_Q Z_W + I_Q X_W",
    "t_s": "pi/(4 sqrt 2)",
    "readout": "Z_Q after U(t_s)",
    "reentry": "V_b kron V_b, V_b = exp(-i(2b-1)(pi/4)X)",
    "initial": "|00>",
}

SPACE = Space.of(Q=2, W=2)
H = kron(X, Z) + kron(I2, X)
T_S = np.pi / (4 * np.sqrt(2))
U_S = unitary_from_hamiltonian(H, T_S)


def V(b: int) -> np.ndarray:
    return unitary_from_hamiltonian(X, (2 * b - 1) * np.pi / 4)


def output_instrument(_s=None) -> Instrument:
    return Instrument({
        0: KrausMap([kron(P0, I2) @ U_S], label="J0"),
        1: KrausMap([kron(P1, I2) @ U_S], label="J1"),
    }, label="Z_Q readout after U(t_s)")


def feedback(tau: int, _s=None) -> KrausMap:
    return unitary_channel(kron(V(tau), V(tau)), label=f"E{tau}")


def protocol() -> EventProtocol:
    return EventProtocol(
        input_map=lambda b, s: identity_channel(4),
        output_instrument=output_instrument,
        feedback=feedback,
        settle=lambda s: identity_channel(4),
        input_mode="inject",
        label="T9 two-mode",
    )


def initial_state() -> MachineState:
    R = np.zeros((4, 4), dtype=complex)
    R[0, 0] = 1.0
    return MachineState(R)


def machine() -> Machine:
    return Machine(protocol(), initial_state())
