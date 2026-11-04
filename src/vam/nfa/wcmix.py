"""The top mixture of E6b Amendment 1 (experiments/PROTOCOL.md; DESIGN V2a, the fallback).

E4 and E4+K (or E4+K random) read the same bits side by side. Their final per-bit readouts are mixed by the machine's
own two-hypothesis coherent mixture (coherence 1.0), with one node per class of the deepest level that contributes to
the E4+K readout (``KMachine.run_deep``). A node's evidence starts at 0, is updated at every bit read with learning on
and is never drained; frozen reads update nothing. V2a: at most 2 bits per node against the better of the two inputs.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from ..nested import NUMBA_AVAILABLE
from .wordclass import KMachine

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _make_top_mix(jit):
    def top_mix(q_in, q_own, deep, bits, ev, acc, learning, coherence, out):
        """Per bit: ``q_in`` and ``q_own`` are the probabilities the two machines gave the bit actually read;
        ``out`` receives the mixture's. ``ev`` (one entry per node) and ``acc`` (per node: bits of the input, of the
        own readout, of the mixture, and the bit count) are updated only when ``learning``."""
        for i in range(q_in.shape[0]):
            b = bits[i]
            qi, qo = q_in[i], q_own[i]
            p_in = qi if b == 0 else 1.0 - qi
            po = qo if b == 0 else 1.0 - qo
            node = deep[i]
            e = ev[node]
            if e >= 0.0:
                shr = 1.0 / (1.0 + 2.0 ** -e)
            else:
                ee = 2.0 ** e
                shr = ee / (1.0 + ee)
            if coherence == 0.0:
                p0 = p_in + shr * (po - p_in)
            else:
                rr = 1.0 - shr
                a0 = math.sqrt(rr * p_in) + math.sqrt(shr * po)
                a1 = math.sqrt(rr * (1.0 - p_in)) + math.sqrt(shr * (1.0 - po))
                t0 = (1.0 - coherence) * (rr * p_in + shr * po) + coherence * a0 * a0
                t1 = (1.0 - coherence) * (rr * (1.0 - p_in) + shr * (1.0 - po)) + coherence * a1 * a1
                p0 = t0 / (t0 + t1)
            q = p0 if b == 0 else 1.0 - p0
            out[i] = q
            if learning:
                ev[node] += math.log2(qo) - math.log2(qi)
                acc[node, 0] += -math.log2(qi)
                acc[node, 1] += -math.log2(qo)
                acc[node, 2] += -math.log2(q)
                acc[node, 3] += 1.0

    return jit(top_mix) if jit is not None else top_mix


top_mix = _make_top_mix(njit(cache=True)) if NUMBA_AVAILABLE else _make_top_mix(None)

NODES = 16                       # the deepest contributing level is 0 to 13 for the 14 levels of E4+K


class MixMachine:
    """E4 (``KMachine`` with the K levels off) and a K machine, read side by side, with the top mixture."""

    def __init__(self, e4: KMachine, k: KMachine) -> None:
        if e4.wcfg.mode != "off" or k.wcfg.mode == "off":
            raise ValueError("the mixture is of an E4 machine and a K machine")
        self.e4, self.k = e4, k
        self._coherence = float(k.cfg.coherence)
        self._ev = np.zeros(NODES)
        self._acc = np.zeros((NODES, 4))
        self.last: dict[str, np.ndarray] = {}

    # -- state ----------------------------------------------------------------------------------------------
    def snapshot(self) -> tuple:
        return (self.e4.snapshot(), self.k.snapshot())

    def restore(self, snap: tuple) -> None:
        self.e4.restore(snap[0])
        self.k.restore(snap[1])

    def table_hashes(self) -> dict:
        out = {f"e4_{k}": v for k, v in self.e4.table_hashes().items()}
        out.update({f"k_{k}": v for k, v in self.k.table_hashes().items()})
        out["evidence"] = hashlib.sha256(np.ascontiguousarray(self._ev).tobytes()).hexdigest()
        return out

    # -- reading ----------------------------------------------------------------------------------------------
    def run(self, bits: np.ndarray, learning: bool = True) -> np.ndarray:
        """The mixture's probability of every bit read. The components' probabilities of the same bits are kept in
        ``last`` (``e4``, ``k``) for the caller."""
        bits = np.ascontiguousarray(bits, dtype=np.uint8)
        q_e = self.e4.run(bits, learning)
        q_k, deep = self.k.run_deep(bits, learning)
        out = np.empty(bits.shape[0])
        top_mix(q_e, q_k, deep, bits, self._ev, self._acc, learning, self._coherence, out)
        self.last = {"e4": q_e, "k": q_k}
        return out

    # -- measures ---------------------------------------------------------------------------------------------
    def regret(self) -> dict:
        """Per node: the loss of the mixture against the smaller of its two inputs' losses (bits)."""
        a = self._acc
        excess = a[:, 2] - np.minimum(a[:, 0], a[:, 1])
        used = a[:, 3] > 0
        return {"max_excess_bits": float(excess[used].max()) if used.any() else 0.0,
                "total_excess_bits": float(excess.sum()), "nodes_used": int(used.sum()), "bound_per_node": 2.0}

    def nodes(self) -> dict:
        """Per node: bits seen, evidence, the losses of E4, of E4+K and of the mixture (learning reads)."""
        return {"bits": self._acc[:, 3].tolist(), "evidence": self._ev.tolist(), "loss_e4": self._acc[:, 0].tolist(),
                "loss_k": self._acc[:, 1].tolist(), "loss_mix": self._acc[:, 2].tolist()}

    def nbytes(self) -> int:
        return self.e4.nbytes() + self.k.nbytes()
