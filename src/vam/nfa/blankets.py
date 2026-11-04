"""Layer 4 of N: Markov blankets around every node, and the global blanket (N-L4, N-L6).

N-L4: every density matrix sits inside its own blanket, ``mu _||_ eta | b``: a sensory interface
that takes inputs, an internal core, and an active interface that passes outputs on; nodes never
bleed into each other directly. On the VAM-1 chain the nodes are the cells ``(R_j, L_j)``; a
cell's blanket is its two neighbours, through which every interaction passes (transport moves one
cell per event, the coin acts inside a cell). Two measurements make the statement exact:

* :func:`cmi` and :func:`cmi_profile`: the quantum conditional mutual information
  ``I(mu : eta | b) = S(mu b) + S(b eta) - S(b) - S(mu b eta)`` for every triple of neighbouring
  cells (``mu = j - 1``, ``b = j``, ``eta = j + 1``). Zero means the blanket screens the two sides
  completely (a quantum Markov chain); its size says how much passes around it.
* :func:`containment`: a local disturbance of one cell, followed by identical events, changes a
  cell at distance ``d`` only after ``d`` events. The one way to reach further at once is the
  machine's own sensory blanket: the observation acts on the whole context region together, so a
  disturbance that has reached the context region is seen by all of it in the same event. The
  function returns the change of every cell, and :func:`cone_violation` the largest change outside
  the cone (zero, to rounding, tested even with learned weights).

The global blanket (N-L6) is the machine's own pair of ports: the sensory port (``R_0``, where the
world's bits enter) and the active port (qubit ``2c``, where the emitted bit leaves). The event
maps touch the world only there (``vam.nfa.flow`` makes the environment units entering and leaving
explicit).
"""

from __future__ import annotations

import numpy as np

from ..realizations.vam1_mpo import OperatorEngine, superoperator


def entropy(rho: np.ndarray) -> float:
    """Von Neumann entropy in bits."""
    ev = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    ev = ev[ev > 1e-14]
    return float(-np.sum(ev * np.log2(ev)))


def cmi(reduced, A, B, C) -> float:
    """``I(A : C | B)`` in bits; ``reduced(sites)`` returns the trace-1 density of a sorted site list."""
    A, B, C = list(A), list(B), list(C)
    s = lambda sites: entropy(reduced(sorted(sites)))
    return s(A + B) + s(B + C) - s(B) - s(A + B + C)


def dense_reduced(R: np.ndarray, n: int):
    """``reduced(sites)`` for a dense density of ``n`` qubits (qubit 0 most significant)."""
    T = R.reshape((2,) * (2 * n))

    def reduced(sites):
        sites = sorted(sites)
        rest = [q for q in range(n) if q not in sites]
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        row = list(letters[:n])
        col = list(letters[n:2 * n])
        for q in rest:
            col[q] = row[q]                                  # traced: the same index on both sides
        out = "".join(row[q] for q in sites) + "".join(col[q] for q in sites)
        M = np.einsum("".join(row) + "".join(col) + "->" + out, T)
        d = 2 ** len(sites)
        return M.reshape(d, d)

    return reduced


def cell_sites(j: int) -> list[int]:
    return [2 * j, 2 * j + 1]


def cmi_profile(engine: OperatorEngine) -> list[float]:
    """``I(cell j-1 : cell j+1 | cell j)`` for ``j = 1 .. c - 2``."""
    c = engine.cfg.cells
    return [cmi(engine.reduced_density, cell_sites(j - 1), cell_sites(j), cell_sites(j + 1)) for j in range(1, c - 1)]


def trace_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(np.linalg.eigvalsh(a - b)).sum())


def containment(engine: OperatorEngine, cell: int, bits, taus, op: np.ndarray | None = None,
                learning: bool = True) -> np.ndarray:
    """Disturb ``cell`` (a unitary ``op`` on its right-mover, a Hadamard by default) in a copy, run the
    same forced events on both, and return the trace distance of every cell's reduced density
    (rows: events, columns: cells), and of the port as the last column."""
    if op is None:
        op = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    a, b = engine.copy(), engine.copy()
    b.x.apply_one(superoperator([np.asarray(op, dtype=complex)]), 2 * cell)
    c = engine.cfg.cells
    rows = []
    for bit, tau in zip(bits, taus, strict=True):
        a.force(int(bit), int(tau), learning)
        b.force(int(bit), int(tau), learning)
        row = [trace_distance(a.reduced_density(cell_sites(j)), b.reduced_density(cell_sites(j))) for j in range(c)]
        row.append(trace_distance(a.reduced_density([2 * c]), b.reduced_density([2 * c])))
        rows.append(row)
    return np.array(rows)


def cone_violation(changes: np.ndarray, cell: int, context: int) -> float:
    """Largest change outside the event cone of ``cell``, before the disturbance reaches the context
    region (the sensory blanket); 0 means fully contained."""
    worst = 0.0
    cells = changes.shape[1] - 1
    for t in range(changes.shape[0]):
        reach = t + 1                                         # cells the disturbance can have reached
        if cell - reach < context:                            # inside the sensory blanket from here on
            break
        for j in range(cells):
            if abs(j - cell) > reach:
                worst = max(worst, float(changes[t, j]))
        if cells - 1 - cell >= reach:                         # the port lies beyond the last cell
            worst = max(worst, float(changes[t, -1]))
    return worst
