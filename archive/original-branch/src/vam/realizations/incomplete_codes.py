"""Incomplete bit configurations as density sectors (T2.2).

Chosen realization (T2.2): three units ``A = 0``, ``B = 01``, ``C = 10``, each
chosen with probability 1/3 when a new unit starts; no delimiters; the codes
are deliberately not prefix-free. Structural states: ``r`` (a unit has just
ended), ``b`` (first bit of B received), ``c`` (first bit of C received).
They are present partial interpretations, not a transcript.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..channels import Instrument, KrausMap, identity_channel
from ..machine import EventProtocol, Machine, MachineState

REALIZATION = {
    "source": "T2.2",
    "status": "chosen realization",
    "units": {"A": "0", "B": "01", "C": "10"},
    "prior": "1/3 each",
    "states": ["r", "b", "c"],
}

R_, B_, C_ = 0, 1, 2
DIM = 3


def _op(to: int, frm: int, amp: float) -> np.ndarray:
    K = np.zeros((DIM, DIM), dtype=complex)
    K[to, frm] = amp
    return K


S3 = 1 / np.sqrt(3)
KRAUS = {
    0: {"A": _op(R_, R_, S3), "B": _op(B_, R_, S3), "C": _op(R_, C_, 1.0)},
    1: {"C": _op(C_, R_, S3), "B": _op(R_, B_, 1.0)},
}

#: Population maps of T2.2 (columns sum to one jointly).
M0 = np.array([[1 / 3, 0, 1], [1 / 3, 0, 0], [0, 0, 0]])
M1 = np.array([[0, 1, 0], [0, 0, 0], [1 / 3, 0, 0]])

P_COMPLETE = np.diag([1.0, 0.0, 0.0]).astype(complex)


def input_map(b: int, _s=None) -> KrausMap:
    return KrausMap(list(KRAUS[b].values()), label=f"B_{b}")


def protocol() -> EventProtocol:
    """Conditioning on an observed bit stream; no output, re-entry or lock in this example."""
    trivial = Instrument({None: identity_channel(DIM)}, label="no readout")
    return EventProtocol(
        input_map=input_map,
        output_instrument=lambda s: trivial,
        feedback=lambda tau, s: identity_channel(DIM),
        input_mode="condition",
        label="T2.2 incomplete codes",
    )


def initial_state() -> MachineState:
    R = np.zeros((DIM, DIM), dtype=complex)
    R[R_, R_] = 1.0
    return MachineState(R)


def machine() -> Machine:
    return Machine(protocol(), initial_state())


def brute_force_parses(observed: str) -> dict[str, float]:
    """Independent check by enumeration of the generative process.

    A *complete* parse is a unit sequence whose concatenation equals
    ``observed``; an *incomplete* parse ends with a unit that started inside
    ``observed`` and overhangs its end (key ``"<units>+<started unit>"``).
    These events are disjoint and exhaustive, and each has probability
    ``(1/3)^k`` for ``k`` units.
    """
    units = REALIZATION["units"]
    n = len(observed)
    out: dict[str, float] = {}
    for k in range(1, n + 1):
        for seq in itertools.product(units, repeat=k):
            s = "".join(units[u] for u in seq)
            prev = "".join(units[u] for u in seq[:-1])
            if s == observed:
                key = "".join(seq)
            elif len(s) > n and s.startswith(observed) and len(prev) < n:
                key = "".join(seq[:-1]) + "+" + seq[-1]
            else:
                continue
            out[key] = out.get(key, 0.0) + (1 / 3) ** k
    return out
