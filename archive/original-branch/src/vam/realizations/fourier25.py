"""Exact 25-configuration interference (T8.3) and a streaming extension.

Chosen realization (T8.3): ``F_jk = exp(2 pi i j k / 25) / 5``,
``(D_r)_jj = exp(2 pi i r j / 25)``. Starting from ``|0>``, ``F^dagger D_r F |0>``
has amplitude 1 at ``k = r`` and 0 elsewhere. Nested readout ``k = 5a + b``
(macro group ``a``, fine configuration ``b``).

Streaming extension (chosen realization of this build, not in T): the machine
reads a binary number most-significant bit first. Each bit applies
``D_b`` after the doubling permutation ``|j> -> |2^{-1} j mod 25>``, which maps
the phase ramp of ``r`` to that of ``2r``. The present state is always the
phase ramp of the prefix value modulo 25, and readout in the conjugate basis
returns that residue with probability one without disturbing the state: the
"haystack" of all 25 configurations is carried coherently and interference
selects the needle at every step, with no stored digits.
"""

from __future__ import annotations

import numpy as np

from ..channels import Instrument, KrausMap, identity_channel, unitary_channel
from ..machine import EventProtocol, Machine, MachineState

N = 25
REALIZATION = {
    "source": "T8.3 (+ streaming extension)",
    "status": "chosen realization",
    "N": N,
    "F": "exp(2 pi i j k/25)/5",
    "D_r": "diag exp(2 pi i r j/25)",
    "nesting": "k = 5a + b",
}

_j = np.arange(N)
F = np.exp(2j * np.pi * np.outer(_j, _j) / N) / np.sqrt(N)


def D(r: int) -> np.ndarray:
    return np.diag(np.exp(2j * np.pi * r * _j / N))


def amplitudes(r: int) -> np.ndarray:
    """``F^dagger D_r F |0>``."""
    e0 = np.zeros(N, dtype=complex)
    e0[0] = 1.0
    return F.conj().T @ D(r) @ F @ e0


def doubling_permutation() -> np.ndarray:
    """``|j> -> |inv2 * j mod 25>`` with ``inv2 = 13`` (``2 * 13 = 26 = 1 mod 25``)."""
    inv2 = pow(2, -1, N)
    P = np.zeros((N, N), dtype=complex)
    for j in range(N):
        P[(inv2 * j) % N, j] = 1.0
    return P


def readout_projector(k: int) -> np.ndarray:
    """Projector onto the phase ramp of ``k`` (``F |k>``), i.e. readout ``k`` after ``F^dagger``."""
    v = F[:, k]
    return np.outer(v, v.conj())


def nested_sectors() -> tuple[dict, dict]:
    coarse = {a: sum(readout_projector(5 * a + b) for b in range(5)) for a in range(5)}
    fine = {a: {b: readout_projector(5 * a + b) for b in range(5)} for a in range(5)}
    return coarse, fine


def protocol() -> EventProtocol:
    Pdbl = doubling_permutation()
    inject = {b: unitary_channel(D(b) @ Pdbl, label=f"bit{b}") for b in (0, 1)}
    readout = Instrument({k: KrausMap([readout_projector(k)], label=f"k={k}") for k in range(N)},
                         label="phase-ramp readout")
    return EventProtocol(
        input_map=lambda b, s: inject[b],
        output_instrument=lambda s: readout,
        feedback=lambda tau, s: identity_channel(N),
        input_mode="inject",
        label="streaming residue mod 25",
    )


def initial_state() -> MachineState:
    """Uniform superposition ``F|0>``: every configuration carried at once (the phase ramp of 0)."""
    v = F[:, 0]
    return MachineState(np.outer(v, v.conj()))


def machine() -> Machine:
    return Machine(protocol(), initial_state())
