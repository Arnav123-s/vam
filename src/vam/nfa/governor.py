"""Layers 5 and 6 of N on the quantum chain: area-law bonds and the global Page governor (N-L5, N-L6).

The chain's density is held as a matrix-product operator (``vam.realizations.vam1_mpo``): the
"tensors of density matrices" of N-L5. This module adds the governor of N-L6 to it.

Measures
--------
* The global information measure is the Renyi-2 entropy ``S2 = -log2 Tr rho^2``. For the MPO
  engine ``Tr rho^2 = |vec rho|^2``, the squared norm of the stored vector, so it costs nothing.
  Its capacity (the Bekenstein-Hawking number of this register) is the qubit count: the entropy
  of the maximally mixed state.
* The area law of N-L5 is measured per cut as the operator entanglement of ``vec rho``
  (:func:`operator_entropies`), the quantity that sets each bond.

The governor
------------
* Page threshold: the governor acts once ``S2 >= threshold * qubits`` (``threshold = 1/2``, Page's
  turning point), or always (``always=True``).
* Sinks: unit-rate dephasing in the pointer basis, ``L = sqrt(rate) sigma_z``, on the sink qubits
  after every evolution. Its coherence eigenvalue is ``Re lambda = -2 rate`` (``-2`` at unit rate,
  tested), so each event multiplies every coherence of a sink qubit by ``e^{-2 rate}``.
  ``sinks="interior"`` drains the chain outside the context sectors (the cells the readout does
  not einselect, whose coherence is what makes bonds grow); ``sinks="boundary"`` drains only the
  last cell and the active port, the boundary of the global blanket.
* Verified sectors are protected: no sink acts on a context qubit (the learned, einselected
  readout).
* Halt: when the Page threshold is crossed, no bond may grow past the largest bond held at that
  moment (``halt=True``): the volume stops expanding and the sinks keep the state inside it.
  (With ``always=True`` there is no such moment and the cache cap alone bounds the bonds.)
* ``trigger="cache"`` replaces the Page point by a threshold on the largest bond against the cache cap (off by default).

The bond of N-L5
----------------
The cap is the cache's (``vam.nfa.bounds.max_mpo_bond``, 90 for a 2 MiB L2), and a new singular
direction is kept only above ``cutoff`` of the cut's weight. With ``bond="entropy"`` every cut's bond
is governed by its own entanglement (the area law): after each split the cut keeps
``ceil(margin * 2^S)`` directions, ``S`` the entropy (bits) of its normalized Schmidt weights, and
never more than the cap (:class:`EntropyBond`). ``2^S`` is the effective number of directions the cut
uses; a low-entanglement cut holds a small bond, and only a cut that is really entangled grows.

Why the Page edge is not used to cut the chain's bonds: the edge separates sampling noise from
structure in an estimated matrix (layer 2 on text, and the doorkeeper of the nested machine). The
chain's state is exact; a Schmidt weight below a random state's typical weight is real correlation,
not noise (one context qubit holding 0.9 / 0.1 would lose its 0.1 direction against the edge 1/4).
On the exact chain the unverified part is the coherence that the readout never einselects, and the
sinks are what drain it.
"""

from __future__ import annotations

import math

import numpy as np

from ..channels import FunctionMap
from ..realizations.vam1 import Config, DenseMaps
from ..realizations.vam1_mpo import OperatorEngine
from . import bounds


class EntropyBond:
    """Area-law bond rule of N-L5: ``min(cap, ceil(margin * 2^S))`` with ``S`` the cut's entanglement in bits."""

    def __init__(self, cap: int, margin: float = 2.0) -> None:
        if cap < 1 or margin < 1.0:
            raise ValueError("need cap >= 1 and margin >= 1")
        self.cap, self.margin = int(cap), float(margin)

    def __call__(self, p: np.ndarray) -> int:
        p = p[p > 0.0]
        S = float(-np.sum(p * np.log2(p)))
        return min(self.cap, math.ceil(self.margin * 2.0 ** S - 1e-9))

    def __repr__(self) -> str:
        return f"EntropyBond(cap={self.cap}, margin={self.margin})"


def sink_superoperator(rate: float = 1.0, t: float = 1.0) -> np.ndarray:
    """``exp(t D)`` for ``D(rho) = rate (Z rho Z - rho)`` on one qubit, acting on ``vec`` index ``2 row + col``."""
    f = math.exp(-2.0 * rate * t)
    return np.diag([1.0, f, f, 1.0]).astype(complex)


def sink_liouvillian(rate: float = 1.0) -> np.ndarray:
    """The generator itself, ``rate (Z (x) Z - I)`` on ``vec``: eigenvalues ``0, -2 rate, -2 rate, 0``."""
    Z = np.diag([1.0, -1.0])
    return rate * (np.kron(Z, Z) - np.eye(4)).astype(complex)


class GovernedEngine(OperatorEngine):
    """The MPO engine with the Page governor of N-L6 and the cache bond of N-L5."""

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None, max_bond: int | None = None,
                 cutoff: float = 1e-14, sink_rate: float = 1.0, sinks: str = "interior", threshold: float = 0.5,
                 always: bool = False, halt: bool = True, bond: str = "cache", margin: float = 2.0,
                 trigger: str = "page") -> None:
        if max_bond is None:
            max_bond = bounds.max_mpo_bond()
        if bond not in ("cache", "entropy"):
            raise ValueError("bond is 'cache' or 'entropy'")
        if trigger not in ("page", "cache"):
            raise ValueError("trigger is 'page' or 'cache'")
        self.bond, self.margin, self.trigger = bond, margin, trigger
        super().__init__(cfg, rng=rng, max_bond=self._rule(max_bond), cutoff=cutoff)
        if sinks not in ("interior", "boundary"):
            raise ValueError("sinks are 'interior' or 'boundary'")
        if sink_rate < 0 or not 0.0 <= threshold <= 1.0:
            raise ValueError("need sink_rate >= 0 and 0 <= threshold <= 1")
        self.sink_rate, self.sinks, self.threshold, self.always, self.halt = sink_rate, sinks, threshold, always, halt
        c = cfg.cells
        protected = set(self.context_sites)                   # verified sectors are never drained
        if sinks == "interior":
            sites = [s for s in range(2 * c + 1) if s not in protected and s not in self.sector_sites]
        else:
            sites = [s for s in (2 * c - 2, 2 * c - 1, 2 * c) if s not in protected]
        self.sink_sites = sites
        self._sink = sink_superoperator(sink_rate)
        self.governed_since: int | None = None
        self.events = 0
        self.base_bond = max_bond

    def _rule(self, cap: int):
        return EntropyBond(cap, self.margin) if self.bond == "entropy" else int(cap)

    @property
    def cap(self) -> int:
        """The largest bond any cut may hold now."""
        return self.max_bond.cap if isinstance(self.max_bond, EntropyBond) else int(self.max_bond)

    def copy(self) -> GovernedEngine:
        other = GovernedEngine.__new__(GovernedEngine)
        other.__dict__.update(super().copy().__dict__)
        for k in ("sink_rate", "sinks", "threshold", "always", "halt", "sink_sites", "_sink", "governed_since",
                  "events", "base_bond", "bond", "margin", "trigger"):
            setattr(other, k, getattr(self, k))
        other.max_bond = self._rule(self.cap)                 # a rule of its own: halting one never halts the other
        return other

    # -- measures --------------------------------------------------------------------
    @property
    def qubits(self) -> int:
        return 2 * self.cfg.cells + 1

    def purity(self) -> float:
        """``Tr rho^2 = |vec rho|^2``."""
        return float(self.x.norm() ** 2)

    def renyi2(self) -> float:
        return -math.log2(self.purity())

    def page_fraction(self) -> float:
        """``S2`` over the capacity (the qubit count)."""
        return self.renyi2() / self.qubits

    @property
    def governed(self) -> bool:
        return self.governed_since is not None

    # -- the governor ------------------------------------------------------------------
    def reached(self) -> bool:
        """Has the governor's threshold been crossed? ``trigger="page"`` (default): ``S2`` is at ``threshold`` of the
        register's capacity. ``trigger="cache"`` (added after the E3 result, PROTOCOL E3 Amendment 4): the largest bond
        is at ``threshold`` of the cache cap, so the sinks act before the cache has to discard weight."""
        if self.trigger == "cache":
            return max(self.bond_dims, default=0) >= self.threshold * self.base_bond
        return self.page_fraction() >= self.threshold

    def _evolve(self) -> None:
        super()._evolve()
        self.events += 1
        if not self.governed and (self.always or self.reached()):
            self.governed_since = self.events
            if self.halt and not self.always:                 # at the Page point the volume stops expanding
                self.max_bond = self._rule(min(self.base_bond, max(self.bond_dims, default=1)))
        if self.governed:
            for s in self.sink_sites:
                self.x.apply_one(self._sink, s)


def operator_entropies(engine: OperatorEngine, renyi: int = 1) -> list[float]:
    """Entanglement (bits) of the normalized ``vec rho`` across every cut: the area-law profile of N-L5."""
    x = engine.x.copy()
    out = []
    for bond in range(len(x) - 1):
        x.canonicalize(bond)
        T = x.tensors[bond]
        s = np.linalg.svd(T.reshape(-1, T.shape[-1]), compute_uv=False)
        p = s * s
        p = p[p > 1e-300] / p.sum()
        out.append(float(-np.sum(p * np.log2(p))) if renyi == 1 else float(-np.log2(np.sum(p * p))))
    return out


class SinkMaps(DenseMaps):
    """The dense machine with the governor's sinks after every evolution (always on): an independent
    reference for :class:`GovernedEngine` and the exact machine for measuring what the sinks cost."""

    def __init__(self, cfg: Config, sites, rate: float = 1.0) -> None:
        super().__init__(cfg)
        f = math.exp(-2.0 * rate)
        bits = self.lay.bits
        self.sink_mask = np.ones((cfg.dim, cfg.dim))
        for s in sites:
            self.sink_mask *= np.where(bits[:, s][:, None] != bits[:, s][None, :], f, 1.0)

    def evolve(self) -> FunctionMap:
        base, M = super().evolve(), self.sink_mask
        return FunctionMap(lambda R: base(R) * M, lambda A: base.dual(A * M), self.cfg.dim, label="evolve+sinks")


def interior_sites(cfg: Config) -> list[int]:
    """The sink qubits of ``sinks="interior"``: every qubit outside the einselected sectors."""
    c, k = cfg.cells, cfg.context
    sectors = set(range(0, 2 * k, 2)) if cfg.einselect == "context" else set(range(2 * c + 1))
    return [s for s in range(2 * c + 1) if s not in sectors]
