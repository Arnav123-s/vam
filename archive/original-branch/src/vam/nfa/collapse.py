"""Layer 7 of N: global path integral, then collapse to the least-action step (N-L7).

N-L7: the engine evaluates ``Z = int D[gamma] e^{i S[gamma] / hbar}`` over its superposed field;
contradictory trajectories cancel; stationary phase leaves the classical trajectory of least
action; and "the collapsed classical path defines the exact, minimal physical step (the next bit),
which is emitted cleanly through the active boundary of the global Markov blanket."

On the quantum chain the evolution is the exact Feynman path sum and its stationary-phase limit
is Newton's law (``vam.paths``, ``vam.semiclassical``; tests T7, T8). This module realizes the
layer for writing text, where a path is a string of future bits.

Paths and action
----------------
From the machine's present state (frozen: generation never learns, T10.3) a path ``gamma`` is a
string of ``h`` future bits. Its probability under the machine is ``p(gamma) = prod_t q(gamma_t |
earlier bits)``, and its action is ``S[gamma] = -ln p(gamma)``: the code length, which is also the
time integral of the observation generator along the event flow (``vam.nfa.flow``).

The path integral, split by the first step ``b``:

* ``form="euclidean"`` (default): ``Z_b = sum_{gamma_1 = b} e^{-S[gamma] / hbar}``, the
  Wick-rotated form of ``e^{i S / hbar}``. ``hbar = 1`` weighs every path by its probability, so
  ``Z_b`` is the machine's own prediction of ``b``; ``hbar -> 0`` keeps only the least-action path
  (``hbar = 0`` is that limit, exactly); ``hbar`` in between weighs paths by ``p^{1/hbar}``.
* ``form="feynman"``: the literal real-time sum ``A_b = sum_{gamma_1 = b} sqrt(p(gamma))
  e^{i S[gamma] / hbar}`` and ``Z_b = |A_b|^2``. Paths of very different action carry rapidly
  turning phases and cancel; paths near the least action add in phase.

Collapse
--------
``mode="collapse"`` emits the ``b`` with the larger ``Z_b`` (the stationary step);
``mode="sample"`` draws ``b`` with probability ``Z_b / (Z_0 + Z_1)``. Only that first bit is
emitted; it re-enters the register and the path integral is taken again (receding horizon).

Emission through the global blanket (N-L6)
------------------------------------------
``verified=True`` lets only verified sectors (evidence above the governor's ``protect`` bits) of
the nested machine drive the output: the completed invariants radiate through the active
boundary, and unproven micro-structure does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from ..bitkernel import BitMachine
from ..nested import EV, MASK64, N, NestedMachine, W, _p0, _share, coherent_mix, context_hash


@dataclass
class Readout:
    """A machine's frozen readout: ``p0(state)`` and ``advance(state, bit)``, with a memo."""

    machine: object
    verified: bool = False

    def __post_init__(self) -> None:
        self.memo: dict[int, float] = {}
        m = self.machine
        if isinstance(m, NestedMachine):
            self.state0 = int(m.reg)
            self.mask = MASK64
        elif isinstance(m, BitMachine):
            self.state0 = int(m.context)
            self.mask = int(m.mask)
            if self.verified:
                raise ValueError("verified emission needs the nested machine's evidence")
        else:
            raise TypeError("a NestedMachine or a BitMachine")

    def p0(self, state: int) -> float:
        got = self.memo.get(state)
        if got is None:
            got = self.memo[state] = self._p0(state)
        return got

    def _p0(self, state: int) -> float:
        m = self.machine
        if isinstance(m, BitMachine):
            w = float(m.w[state])
            return 0.5 * (1.0 + (1.0 - 2.0 * m.cfg.eps) * min(1.0, max(-1.0, w)))
        if not self.verified:
            return m._p0_at(state)
        cfg = m.cfg
        p0 = 0.0
        for lv, d in enumerate(cfg.depths):
            ctx = state & ((1 << d) - 1)
            if cfg.direct(lv):
                row = m._wn[m._offsets[lv] + ctx]
                present = row[N] > 0.0
            else:
                hsh = context_hash(ctx, lv)
                s = hsh >> (64 - cfg.table_bits)
                row = m._wn[m._offsets[lv] + s]
                present = int(m._tags[m._hidx[lv], s]) == ((hsh & 0xFFFFFFFF) | 1)
            if lv == 0:
                p0 = _p0(cfg.law, float(row[W]))
            elif present and row[N] >= cfg.mature and row[EV] > cfg.protect:
                own = _p0(cfg.law, float(row[W]))
                p0 = own if cfg.readout == "mature" else coherent_mix(p0, own, _share(float(row[EV])),
                                                                       cfg.coherence)
        return p0

    def advance(self, state: int, b: int) -> int:
        return ((state << 1) | b) & self.mask


def paths(readout: Readout, state: int, horizon: int, cut: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Every ``horizon``-bit path from ``state`` with its action ``S = -ln p``; a branch whose step has
    probability ``<= cut`` is dropped (its weight is below every ``hbar <= 1`` threshold that matters)."""
    out_paths, out_s = [], []

    def walk(st: int, prefix: int, s: float, depth: int) -> None:
        if depth == horizon:
            out_paths.append(prefix)
            out_s.append(s)
            return
        p0 = readout.p0(st)
        for b, pb in ((0, p0), (1, 1.0 - p0)):
            if pb > cut:
                walk(readout.advance(st, b), (prefix << 1) | b, s - math.log(pb), depth + 1)

    walk(state, 0, 0.0, 0)
    return np.array(out_paths, dtype=np.int64), np.array(out_s)


def first_step(readout: Readout, state: int, horizon: int, hbar: float, form: str = "euclidean") -> dict:
    """The path integral split by the first bit: ``{"Z": (Z_0, Z_1) normalized, "least": (path, S)}``."""
    if hbar < 0:
        raise ValueError("hbar must be nonnegative")
    if form not in ("euclidean", "feynman"):
        raise ValueError("form is 'euclidean' or 'feynman'")
    P, S = paths(readout, state, horizon)
    first = P >> (horizon - 1)
    ties = np.flatnonzero(S == S.min())
    best = int(ties[np.argmax(P[ties])])                         # ties: the largest bit string, as select="max"
    Z = np.zeros(2)
    if hbar == 0.0:
        Z[first[best]] = 1.0                                     # only the least-action path survives
    elif form == "euclidean":
        logs = np.array([logsumexp(-S[first == b] / hbar) if np.any(first == b) else -np.inf for b in (0, 1)])
        Z = np.exp(logs - logs.max())
    else:
        amp = np.exp(-0.5 * S + 1j * S / hbar)
        A = np.array([amp[first == b].sum() for b in (0, 1)])
        Z = np.abs(A) ** 2
    total = Z.sum()
    Z = Z / total if total > 0 else np.array([0.5, 0.5])
    return {"Z": Z, "least": (int(P[best]), float(S[best])), "paths": len(P)}


def generate(machine, n_bits: int, horizon: int = 8, hbar: float = 0.0, mode: str = "collapse",
             form: str = "euclidean", verified: bool = False, rng: np.random.Generator | None = None) -> list[int]:
    """Emit ``n_bits`` bits, each the first step of the path integral taken from the present state.

    The machine's register advances with every emitted bit; its tables never change. ``hbar = 0``
    with ``mode="collapse"`` is the least-action emission (the macro/micro ``select="max"`` of
    ``NestedMachine.generate``); ``hbar = 1``, ``mode="sample"``, ``form="euclidean"`` draws from the
    machine's own predictions.
    """
    if mode not in ("collapse", "sample"):
        raise ValueError("mode is 'collapse' or 'sample'")
    if mode == "sample" and rng is None:
        raise ValueError("sampling needs a random generator")
    readout = Readout(machine, verified)
    state = readout.state0
    out = []
    for _ in range(n_bits):
        Z = first_step(readout, state, horizon, hbar, form)["Z"]
        if mode == "collapse":
            b = int(Z[1] > Z[0])
        else:
            b = int(rng.random() >= Z[0])
        out.append(b)
        state = readout.advance(state, b)
        if len(readout.memo) > 1 << 20:
            readout.memo.clear()
    if isinstance(machine, NestedMachine):
        machine.reg = np.uint64(state)
    else:
        machine.context = state
    return out


def repetition(data: bytes, n: int = 8) -> float:
    """Fraction of the ``n``-byte windows of ``data`` that already occurred earlier in it (loops score high)."""
    seen, rep, total = set(), 0, 0
    for i in range(len(data) - n + 1):
        g = data[i:i + n]
        rep += g in seen
        seen.add(g)
        total += 1
    return rep / total if total else 0.0
