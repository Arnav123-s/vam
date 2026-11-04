"""Family stages of E7 (experiments/PROTOCOL.md, E7; DESIGN section 12, V3 to V5).

Two ways to make a distribution over the result of ``a+b=`` from the equations read so far, and the machine that mixes
it in:

* **F5** (V5): the masked rank-R canonical (CP) decomposition of the equation tensor ``C[a, b, c]``, read out with
  T8.3's Fourier readout. A compile-time fit to counts, like O's SVD in E6: never fitted to answers or costs, never fed
  the held-out cells, no backpropagation through the machine.
* **F4** (V4): the integer lattice of the seen rows ``e_a + e_b - e_c``. A held-out sum is determined iff its row is
  in the lattice; the stage fires only on a unique determined candidate and abstains otherwise.

``FamilyMachine`` is E6's operator machine with O off and this stage in O's slot (after level 0, nodes keyed by the
partial-byte state, active only at answer bytes); with ``route="off"`` it is E4 bit for bit.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .hier import HierConfig
from .operator import OperatorConfig, OperatorMachine, hier_config

ETA = 2.0 ** -6                 # the mass the stage leaves to the other results (C, fixed in E7.4)
RESIDUAL = 0.05                 # F5: an operand token fitted worse than this (rms, unit cells) is dropped (C)
LAM = 1e-3                      # F5 ridge
INIT = 0.3                      # F5 initial scale
SWEEPS = 300                    # F5 alternating sweeps
EPS_O = 2.0 ** -12              # byte floor of the stage's distribution (as O's)
EPS_BIT = 2.0 ** -12            # the machine's bit epsilon


class LatticeTimeout(RuntimeError):
    """The integer reduction ran past its bound (E7.3: 15 minutes per draw at stage 2)."""


# ------------------------------------------------------------------------------------------------------------------
# F4: the integer lattice
# ------------------------------------------------------------------------------------------------------------------

def _egcd(a: int, b: int) -> tuple[int, int, int]:
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a - q * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return (a, x0, y0) if a > 0 else (-a, -x0, -y0)


class Lattice:
    """The integer span of added rows, in echelon form with canonical remainders: ``canon(r) == canon(s)`` iff
    ``r - s`` is in the span."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.piv: dict[int, np.ndarray] = {}

    def add(self, r) -> None:
        r = np.array(r, dtype=object)
        for c in range(self.n):
            if r[c] == 0:
                continue
            if c not in self.piv:
                self.piv[c] = -r if r[c] < 0 else r
                return
            v = self.piv[c]
            p = v[c]
            if r[c] % p == 0:
                r = r - (r[c] // p) * v
                continue
            g, x, y = _egcd(p, r[c])
            nv = x * v + y * r
            r = (r[c] // g) * v - (p // g) * r
            self.piv[c] = nv

    def canon(self, r) -> tuple:
        r = np.array(r, dtype=object)
        for c in sorted(self.piv):
            if r[c] != 0:
                v = self.piv[c]
                r = r - (r[c] // v[c]) * v
        return tuple(int(t) for t in r)


def unit_row(n: int, i: int) -> list[int]:
    v = [0] * n
    v[i] = 1
    return v


def triple_row(n: int, a: int, b: int, c: int) -> list[int]:
    v = [0] * n
    v[a] += 1
    v[b] += 1
    v[c] -= 1
    return v


def distinct_rows(eqs) -> list[tuple[int, int, int]]:
    """Equations with both orders of a pair merged (they are the same row), sorted."""
    return sorted({(min(a, b), max(a, b), c) for a, b, c in eqs})


@dataclass
class LatticeStage:
    n: int
    lattice: Lattice
    canon_units: dict
    seconds: float = 0.0

    @classmethod
    def build(cls, n: int, eqs, deadline: float | None = None) -> LatticeStage:
        t0 = time.perf_counter()
        lat = Lattice(n)
        for k, (a, b, c) in enumerate(distinct_rows(eqs)):
            if deadline is not None and k % 64 == 0 and time.perf_counter() - t0 > deadline:
                raise LatticeTimeout(f"the lattice reduction passed {deadline:.0f} s")
            lat.add(triple_row(n, a, b, c))
        return cls(n, lat, {c: lat.canon(unit_row(n, c)) for c in range(n)}, time.perf_counter() - t0)

    def candidates(self, a: int, b: int) -> list[int]:
        u = self.lattice.canon([int(a == i) + int(b == i) for i in range(self.n)])
        return [c for c in range(self.n) if self.canon_units[c] == u]

    def determined(self, a: int, b: int, c: int) -> bool:
        return c in self.candidates(a, b)

    def dist(self, a: int, b: int) -> np.ndarray | None:
        cs = self.candidates(a, b)
        if len(cs) != 1:
            return None
        p = np.full(self.n, ETA / (self.n - 1))
        p[cs[0]] = 1.0 - ETA
        return p


# ------------------------------------------------------------------------------------------------------------------
# F5: the masked CP decomposition
# ------------------------------------------------------------------------------------------------------------------

def cp_fit(nop: int, nres: int, eqs, rank: int, seed: int, sweeps: int = SWEEPS, lam: float = LAM,
           init: float = INIT) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``C[a, b, c] ~ sum_j A[a, j] B[b, j] G[c, j]`` on the read cells ``(a, b)`` (a cell not read is not scored).
    Alternating ridge least squares, each update closed form; returns A, B, G and the per-operand rms residual."""
    rng = np.random.default_rng(seed + 1000)
    obs = np.zeros((nop, nop), dtype=bool)
    tgt = np.zeros((nop, nop), dtype=np.int64)
    for a, b, c in eqs:
        obs[a, b] = obs[b, a] = True
        tgt[a, b] = tgt[b, a] = c
    A = init * (rng.normal(size=(nop, rank)) + 1j * rng.normal(size=(nop, rank)))
    B = A.copy()
    G = init * (rng.normal(size=(nres, rank)) + 1j * rng.normal(size=(nres, rank)))
    eye = np.eye(rank)
    ia, ib = np.nonzero(obs)
    for _ in range(sweeps):
        GG = G.conj().T @ G
        for a in range(nop):
            bs = np.flatnonzero(obs[a])
            if len(bs) == 0:
                continue
            Bs = B[bs]
            A[a] = np.linalg.solve((Bs.conj().T @ Bs) * GG + lam * eye, (Bs.conj() * G[tgt[a, bs]].conj()).sum(0))
        for b in range(nop):
            as_ = np.flatnonzero(obs[:, b])
            if len(as_) == 0:
                continue
            As = A[as_]
            B[b] = np.linalg.solve((As.conj().T @ As) * GG + lam * eye, (As.conj() * G[tgt[as_, b]].conj()).sum(0))
        P = A[ia] * B[ib]
        R = np.zeros((nres, rank), dtype=complex)
        np.add.at(R, tgt[ia, ib], P.conj())
        G = np.linalg.solve(P.conj().T @ P + lam * eye, R.T).T
    resid = np.zeros(nop)
    for a in range(nop):
        bs = np.flatnonzero(obs[a])
        if len(bs) == 0:
            resid[a] = np.inf
            continue
        rows = (G @ (A[a] * B[bs]).T).T                      # len(bs) x nres
        y = np.zeros_like(rows)
        y[np.arange(len(bs)), tgt[a, bs]] = 1.0
        resid[a] = math.sqrt(float((np.abs(rows - y) ** 2).sum(1).mean()))
    return A, B, G, resid


def fourier_readout(A: np.ndarray, B: np.ndarray, G: np.ndarray, a: int, b: int) -> np.ndarray:
    """T8.3's readout of the composed rotations: the Born distribution of ``G (A[a] * B[b])`` over results,
    ``(1 - eta)`` of it plus ``eta`` uniform."""
    amp = np.abs(G @ (A[a] * B[b])) ** 2
    tot = float(amp.sum())
    n = amp.shape[0]
    if not tot > 0.0:
        return np.full(n, 1.0 / n)
    return (1.0 - ETA) * amp / tot + ETA / n


@dataclass
class CPStage:
    nop: int
    nres: int
    A: np.ndarray
    B: np.ndarray
    G: np.ndarray
    resid: np.ndarray
    dropped: frozenset
    seconds: float = 0.0
    rank: int = 0

    @classmethod
    def build(cls, nop: int, nres: int, eqs, seed: int, sweeps: int = SWEEPS) -> CPStage:
        t0 = time.perf_counter()
        A, B, G, resid = cp_fit(nop, nres, eqs, nres, seed, sweeps)
        dropped = frozenset(int(a) for a in np.flatnonzero(~(resid <= RESIDUAL)))
        return cls(nop, nres, A, B, G, resid, dropped, time.perf_counter() - t0, nres)

    def dist(self, a: int, b: int) -> np.ndarray | None:
        if not (0 <= a < self.nop and 0 <= b < self.nop) or a in self.dropped or b in self.dropped:
            return None
        return fourier_readout(self.A, self.B, self.G, a, b)


# ------------------------------------------------------------------------------------------------------------------
# the byte distribution at an answer byte, and the line parser
# ------------------------------------------------------------------------------------------------------------------

class Spelling:
    """Results spelled as ``digits\\n``: the exact marginal of a distribution over results for the next byte."""

    def __init__(self, nres: int) -> None:
        self.nres = nres
        self.spell = [(str(c) + "\n").encode() for c in range(nres)]
        nxt: dict[bytes, tuple[list[int], list[int]]] = {}
        for c, sp in enumerate(self.spell):
            for k in range(len(sp)):
                cs, nb = nxt.setdefault(sp[:k], ([], []))
                cs.append(c)
                nb.append(sp[k])
        self.next = {k: (np.array(v[0]), np.array(v[1])) for k, v in nxt.items()}

    def byte_dist(self, p: np.ndarray, prefix: bytes) -> np.ndarray | None:
        got = self.next.get(prefix)
        if got is None:
            return None
        q = np.zeros(256)
        np.add.at(q, got[1], p[got[0]])
        return q


def bit_probs(q: np.ndarray, x: int, eps_o: float = EPS_O, eps: float = EPS_BIT) -> tuple[np.ndarray, np.ndarray]:
    """Per-bit probabilities of bit 0 and the node index (1 << j | the bits read) for byte ``x`` under the byte
    distribution ``q`` (floored at ``eps_o``, renormalized), as O's pass does."""
    q = np.maximum(q, eps_o)
    cs = np.concatenate([[0.0], np.cumsum(q / q.sum())])
    p0 = np.empty(8)
    u = np.empty(8, dtype=np.int64)
    for j in range(8):
        sh = 8 - j
        lo = (x >> sh) << sh
        size = 1 << sh
        v = (cs[lo + size // 2] - cs[lo]) / (cs[lo + size] - cs[lo])
        p0[j] = min(max(v, eps), 1.0 - eps)
        u[j] = (1 << j) | (x >> sh)
    return p0, u


PH_A, PH_B, PH_ANS = 0, 1, 2


# ------------------------------------------------------------------------------------------------------------------
# the machine
# ------------------------------------------------------------------------------------------------------------------

@dataclass
class FamilyConfig:
    route: str = "off"              # "off" (E4), "f5" (V5 route), "f4" (V4 route)
    results: int = 19               # the stage's result count: 19 at stage 1, 199 at stage 2
    seed: int = 0                   # F5's initialization seed offset (the draw's seed)
    sweeps: int = SWEEPS
    lattice_deadline: float | None = None
    slot: str = "top"              # "top": after the top level (E7 Amendment 1); "after0": O's slot (ablation)
    cache: dict = field(default_factory=dict, compare=False, repr=False)   # frozenset(eqs) -> LatticeStage

    def __post_init__(self) -> None:
        if self.route not in ("off", "f5", "f4"):
            raise ValueError("route is 'off', 'f5' or 'f4'")
        if self.slot not in ("after0", "top"):
            raise ValueError("slot is 'after0' or 'top'")

    @property
    def operands(self) -> int:
        return (self.results + 1) // 2


class FamilyMachine(OperatorMachine):
    """E4's machine with a family stage in O's slot. ``route="off"``: E4 bit for bit."""

    def __init__(self, hcfg: HierConfig, fcfg: FamilyConfig) -> None:
        super().__init__(hcfg, OperatorConfig(mode="off"))
        self.fcfg = fcfg
        self._top = int(fcfg.slot == "top")
        self._spelling = Spelling(fcfg.results)
        self._eqs: set[tuple[int, int, int]] = set()
        self._fit_eqs: frozenset | None = None
        self._fam: LatticeStage | CPStage | None = None
        self._dists: dict = {}
        self._ps = (PH_A, 0, 0, b"")                  # parser state: phase, a, b, digits read in this numeral
        self.fits: list[dict] = []

    # -- state ------------------------------------------------------------------------------------------------
    def snapshot(self) -> tuple:
        return (super().snapshot(), self._ps)

    def restore(self, snap: tuple) -> None:
        super().restore(snap[0])
        self._ps = snap[1]

    def copy(self) -> FamilyMachine:
        other = super().copy()
        other._eqs = set(self._eqs)
        other._dists = {}
        other.fits = list(self.fits)
        return other

    def table_hashes(self) -> dict:
        out = super().table_hashes()
        out["equations"] = hashlib.sha256(repr(sorted(self._eqs)).encode()).hexdigest()
        out["evidence"] = hashlib.sha256(np.ascontiguousarray(self._ev_o).tobytes()).hexdigest()
        return out

    def end_epoch(self) -> bool:
        """Rebuild the stage if the set of distinct equations read has changed (E7.4); returns whether it did."""
        if self.fcfg.route == "off" or not self._eqs or self._fit_eqs == frozenset(self._eqs):
            return False
        key = frozenset(self._eqs)
        t0 = time.perf_counter()
        n = self.fcfg.results
        if self.fcfg.route == "f4":
            stage = self.fcfg.cache.get(key)
            if stage is None:
                stage = LatticeStage.build(n, self._eqs, self.fcfg.lattice_deadline)
                self.fcfg.cache[key] = stage
            info = {"route": "f4", "seconds": stage.seconds, "rank": len(stage.lattice.piv)}
        else:
            stage = CPStage.build(self.fcfg.operands, n, self._eqs, self.fcfg.seed, self.fcfg.sweeps)
            info = {"route": "f5", "seconds": stage.seconds, "rank": stage.rank, "dropped": len(stage.dropped),
                    "mean_residual": float(np.mean(stage.resid[np.isfinite(stage.resid)])),
                    "max_residual": float(np.max(stage.resid[np.isfinite(stage.resid)]))}
        self._fam, self._fit_eqs, self._dists = stage, key, {}
        self.fits.append({"equations": len(self._eqs), "wall_seconds": time.perf_counter() - t0, **info})
        return True

    def stage_dist(self, a: int, b: int) -> np.ndarray | None:
        """The stage's distribution over results for ``a+b=`` (None: the stage abstains or is not built)."""
        if self._fam is None:
            return None
        k = (a, b)
        if k not in self._dists:
            self._dists[k] = self._fam.dist(a, b)
        return self._dists[k]

    # -- the stage -------------------------------------------------------------------------------------------
    def _stage(self, data: np.ndarray, extp: np.ndarray, extu: np.ndarray, dummy: tuple, learning: bool) -> tuple:
        if self.fcfg.route == "off":
            return dummy[0], dummy[1], False
        extp[:] = np.nan
        extu[:] = 0
        phase, a, b, digs = self._ps
        on = False
        for t in range(data.shape[0]):
            x = int(data[t])
            if phase == PH_ANS and self._fam is not None:
                p = self.stage_dist(a, b)
                if p is not None:
                    q = self._spelling.byte_dist(p, digs)
                    if q is not None:
                        p0, u = bit_probs(q, x)
                        extp[8 * t:8 * t + 8] = p0
                        extu[8 * t:8 * t + 8] = u
                        on = True
            if 48 <= x <= 57:
                digs += bytes([x])
            elif x == 43 and phase == PH_A and digs:
                a, phase, digs = int(digs), PH_B, b""
            elif x == 61 and phase == PH_B and digs:
                b, phase, digs = int(digs), PH_ANS, b""
            elif x == 10:
                if phase == PH_ANS and digs and learning:
                    self._eqs.add((a, b, int(digs)))
                phase, a, b, digs = PH_A, 0, 0, b""
            else:
                phase, a, b, digs = PH_A, 0, 0, b""
        self._ps = (phase, a, b, digs)
        return extp, extu, on


def build_family(spec: dict, law, fcfg: FamilyConfig) -> FamilyMachine:
    return FamilyMachine(hier_config(spec, law), fcfg)


__all__ = [
    "CPStage",
    "FamilyConfig",
    "FamilyMachine",
    "Lattice",
    "LatticeStage",
    "LatticeTimeout",
    "Spelling",
    "bit_probs",
    "build_family",
    "cp_fit",
    "distinct_rows",
    "fourier_readout",
]
