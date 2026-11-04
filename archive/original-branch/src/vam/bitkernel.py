"""The bit kernel: the massless, einselected limit of VAM-1 on packed bits (F-Part 8, T13.4).

With ``theta = 0`` the Dirac field only transports: every injected bit advances
one cell per event and nothing mixes the right- and left-movers. With the
context sectors einselected, the chain density is then a single pointer
configuration at all times, an exact sliding window over the stream (T13.4).
The complete present state therefore reduces to three things, and nothing else
is needed to reproduce the dense and pointer engines:

* ``context``: one integer register holding the last ``k = cfg.context`` bits,
  newest bit in bit 0. This is vam1's sector convention ``s = sum_j R_j 2^j``
  with ``R_0`` the newest bit (see ``vam1.Layout``), so the register value *is*
  the sector index. After a bit ``b`` the update is
  ``context = ((context << 1) | b) & mask``. It starts at 0 (the vacuum chain).
* ``w``: the sector weights, a float64 array of length ``2**k``.
* ``n``: the sector activity accumulators, a float64 array of length ``2**k``.

Equivalence to :class:`vam.realizations.vam1.DenseMaps` (coherent, no
einselection, still exact when massless) and :class:`vam.realizations.vam1.PointerEngine`
is the defining property of this module and is tested bit for bit, with the
full reading loop (emission at the active port and re-entry after every bit)
running in the dense and pointer engines: at zero mass the emitted bits never
reach the context, so they change no prediction and no weight. The register
is the canalized state: ``DenseMaps.canalization()``, built to hold at every
weight and to contain what plasticity reads, is exactly the algebra of the
context populations when massless (tested for ``k = 1, 2``), so everything the
dense engine carries beyond ``context`` is redundant for the future.

One reading step with external bit ``b``
----------------------------------------
1. Predict: ``q = bit_probabilities(w, eps, b)[context]`` (the floor formula of
   vam1, ``pi_s(0) = (1 + (1 - 2 eps) clip(w_s)) / 2``).
2. Learn (plasticity on): vam1's gated T12.1 law with the one-hot activity
   "sector ``context`` has activity 1, all others 0" (the posterior of a certain
   sector). With ``z = +1`` for ``b = 0`` and ``-1`` for ``b = 1`` and
   ``target = g z / kappa``,
   ``w_s <- target + (w_s - target) exp(-kappa dt / gamma_s)`` and ``n_s <- n_s + 1``
   for ``s = context`` only; ``gamma_s = gamma + kappa dt n_s`` when consolidating.
3. Shift: ``context = ((context << 1) | b) & mask``.

Generation (plasticity off, T10.3) draws each emitted bit from the machine's own
prediction and re-enters it through the same shift; the weights never change.
Planning mirrors :func:`vam.machine.macro_micro_step` and ``Machine.branches``.
"""

from __future__ import annotations

import math

import numpy as np

from .bits import bytes_to_bits
from .linalg import ATOL
from .realizations.vam1 import Config

try:  # numba is an optional accelerator; the reference loop below is always available
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without numba
    NUMBA_AVAILABLE = False


CHUNK = 512   # bits per block in the loop below


def _make_loop(jit):
    """The reading loop, plain or compiled from one source, so both paths agree exactly.

    Mutates ``wn`` (column 0 is ``w``, column 1 is ``n``) and ``out``; returns the new register.
    ``kappa_dt`` is ``kappa dt``.

    The register depends only on the bits, never on ``w`` or ``n``, so each block first walks its
    sectors and touches their weights. Those loads are independent, which lets the memory system
    overlap the cache misses of a large table (k ~ 22) that a dependent read-modify-write loop
    would take one at a time. The second pass then does the arithmetic in stream order, so the
    results are identical to a bit-by-bit loop.
    """

    def loop(bits, wn, context, mask, scale, gamma, g_over_kappa, kappa_dt, consolidating, learning, out):
        size = bits.shape[0]
        sectors = np.empty(min(CHUNK, size), dtype=np.int64)
        sink = 0.0
        for start in range(0, size, CHUNK):
            stop = min(size, start + CHUNK)
            c = context
            for i in range(start, stop):
                sectors[i - start] = c
                sink += wn[c, 0]
                c = ((c << 1) | int(bits[i])) & mask
            for i in range(start, stop):
                b = int(bits[i])
                s = sectors[i - start]
                wc = wn[s, 0]
                if wc > 1.0:
                    wc = 1.0
                elif wc < -1.0:
                    wc = -1.0
                p0 = 0.5 * (1.0 + scale * wc)
                out[i] = p0 if b == 0 else 1.0 - p0
                if learning:
                    target = g_over_kappa if b == 0 else -g_over_kappa
                    gs = gamma + kappa_dt * wn[s, 1] if consolidating else gamma
                    wn[s, 0] = target + (wn[s, 0] - target) * math.exp(-kappa_dt / gs)
                    wn[s, 1] += 1.0
            context = c
        if sink != sink:   # a NaN weight; the check also keeps the touches from being optimized away
            raise ValueError("a sector weight is not a number")
        return context

    return jit(loop) if jit is not None else loop


_reference_loop = _make_loop(None)
_compiled_loop = _make_loop(njit(cache=True)) if NUMBA_AVAILABLE else None


def _resident_zeros(shape: tuple[int, ...]) -> np.ndarray:
    """Zeros whose pages are already mapped, so the first pass over a large table pays no page faults."""
    a = np.empty(shape)
    a.fill(0.0)
    return a


class BitMachine:
    """Massless einselected VAM-1 as a sliding-window register plus ``(w, n)``.

    Reproduces :class:`~vam.realizations.vam1.PointerEngine` and the dense
    engine exactly when ``cfg.theta == 0`` (the defining property, tested).
    Attributes: ``cfg``, ``context`` (int, last ``k`` bits, newest in bit 0),
    ``w`` and ``n`` (float64, length ``2**k``).
    """

    def __init__(self, cfg: Config, use_numba: bool = True) -> None:
        if cfg.theta != 0.0:
            raise ValueError("the bit kernel is the massless limit (theta = 0); a massive field is not a single "
                             "configuration, use PointerEngine or DenseMaps")
        self.cfg = cfg
        self.use_numba = bool(use_numba and NUMBA_AVAILABLE)
        self.mask = cfg.sectors - 1
        self.context = 0
        # w and n of a sector share one cache line: (sectors, 2) row-major, exposed as the views ``w`` and ``n``
        self._wn = _resident_zeros((cfg.sectors, 2))

    # -- state ------------------------------------------------------------------
    @property
    def w(self) -> np.ndarray:
        """Sector weights (a view into the packed table; writes go through)."""
        return self._wn[:, 0]

    @property
    def n(self) -> np.ndarray:
        """Sector activity accumulators (a view into the packed table; writes go through)."""
        return self._wn[:, 1]

    def copy(self) -> BitMachine:
        other = BitMachine(self.cfg, use_numba=self.use_numba)
        other.context = self.context
        other._wn[:] = self._wn
        return other

    def nbytes(self) -> int:
        """Bytes of the complete present state: the register (8) plus ``w`` and ``n``; constant in time."""
        return 8 + int(self._wn.nbytes)

    # -- reading ----------------------------------------------------------------
    def _q(self, b: int) -> float:
        """``bit_probabilities(w, eps, b)[context]`` for the current register."""
        wc = min(1.0, max(-1.0, float(self.w[self.context])))
        p0 = 0.5 * (1.0 + (1.0 - 2.0 * self.cfg.eps) * wc)
        return p0 if b == 0 else 1.0 - p0

    def predict(self, b: int) -> float:
        """Probability the machine assigns to external bit ``b`` now."""
        return self._q(int(b))

    def step(self, b: int, learning: bool = True) -> float:
        """Observe, learn and shift one external bit; returns ``q_b``."""
        return float(self._run(np.array([b], dtype=np.uint8), learning, compiled=False)[0])

    def _run(self, bits: np.ndarray, learning: bool, compiled: bool) -> np.ndarray:
        bits = np.ascontiguousarray(bits, dtype=np.uint8)
        c = self.cfg
        out = np.empty(bits.shape[0])
        loop = _compiled_loop if compiled and self.use_numba else _reference_loop
        self.context = int(loop(bits, self._wn, int(self.context), self.mask, 1.0 - 2.0 * c.eps, c.gamma,
                                c.g / c.kappa, c.kappa * c.dt, c.consolidating, learning, out))
        return out

    def run(self, bits: np.ndarray, learning: bool = True) -> np.ndarray:
        """Process a ``uint8`` bit array; returns ``q`` per bit. The hot loop (compiled when numba is present)."""
        return self._run(bits, learning, compiled=True)

    def bits_per_byte(self, data: bytes) -> float:
        """Online code length ``-sum(log2 q) / len(data)`` of ``data`` (MSB-first bits), learning as it reads."""
        q = self.run(bytes_to_bits(data))
        return float(-np.log2(q).sum() / len(data))

    # -- generation -------------------------------------------------------------
    def continuations(self, horizon: int) -> dict[tuple, float]:
        """Exact distribution over output strings of ``horizon`` bits; plasticity off.

        Each emitted bit is drawn from the machine's own prediction and
        re-enters the register by the shift; the weights do not change. As in
        ``Machine.branches`` a branch is pruned when one step's outcome
        probability is ``<= max(1e-10, ATOL)``.
        """
        cut = max(1e-10, ATOL)
        w, eps, mask = self.w, self.cfg.eps, self.mask
        scale = 1.0 - 2.0 * eps
        dist: dict[tuple, float] = {}

        def p0_of(ctx: int) -> float:
            return 0.5 * (1.0 + scale * min(1.0, max(-1.0, float(w[ctx]))))

        def walk(ctx: int, outs: tuple, prob: float, depth: int) -> None:
            if depth == horizon:
                dist[outs] = dist.get(outs, 0.0) + prob
                return
            p0 = p0_of(ctx)
            for t, pt in ((0, p0), (1, 1.0 - p0)):
                if pt <= cut:
                    continue
                walk(((ctx << 1) | t) & mask, (*outs, t), prob * pt, depth + 1)

        walk(self.context, (), 1.0, 0)
        return dist

    def generate(self, n_bits: int, horizon: int = 8, select: str = "max",
                 rng: np.random.Generator | None = None) -> list[int]:
        """Emit ``n_bits`` bits by macro/micro planning (as :func:`vam.machine.macro_micro_step`).

        Per bit: plan over :meth:`continuations`, choose a continuation
        (``"max"``: largest ``(probability, tuple(map(str, key)))``; ``"sample"``:
        keys sorted by ``tuple(map(str, key))``, drawn with ``rng.choice``), emit
        only its first bit, shift it into the register and re-plan. Weights are
        unchanged; the register ends advanced by the emitted bits.
        """
        if select not in ("max", "sample"):
            raise ValueError("select must be 'max' or 'sample'")
        if select == "sample" and rng is None:
            raise ValueError("sampling needs a random generator")
        emitted: list[int] = []
        for _ in range(n_bits):
            dist = self.continuations(horizon)
            if not dist:
                raise ValueError("no continuation has nonzero probability")
            if select == "max":
                macro = max(dist, key=lambda k: (dist[k], tuple(map(str, k))))
            else:
                keys = sorted(dist, key=lambda k: tuple(map(str, k)))
                pv = np.array([dist[k] for k in keys])
                macro = keys[int(rng.choice(len(keys), p=pv / pv.sum()))]
            first = int(macro[0])
            self.context = ((self.context << 1) | first) & self.mask
            emitted.append(first)
        return emitted
