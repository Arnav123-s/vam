"""VAM-1 on a locally purified density operator: the tensor-network engine (T13.1, T13.4, F-Part 9).

:class:`TensorEngine` runs exactly the reading event of
:meth:`vam.realizations.vam1.DenseMaps.reading_protocol` and the generation step of
:meth:`~vam.realizations.vam1.DenseMaps.generation_protocol` on a :class:`vam.tensornet.LPDO`
``rho = Tr_K |X><X|``. Every map below is the dense engine's map, applied exactly; the only
approximation is the truncation the caller asks for (``max_bond``, ``max_kraus``, ``cutoff``),
always performed in canonical gauge and accumulated in :attr:`TensorEngine.discarded`.
Positivity holds by construction, also after truncation (``rho`` is a Gram matrix of ``X``).

Sites
-----
One qubit per site, in the dense engine's qubit order: site ``2j`` is ``R_j``, site ``2j + 1``
is ``L_j``, site ``2c`` is the active port ``A`` (site 0 most significant), so
:meth:`TensorEngine.to_dense` is directly comparable with the dense state. Qubit sites rather
than four-dimensional cell sites: a two-site update then splits a matrix of side ``2 D K``
instead of ``4 D K`` (eight times cheaper per SVD), and each qubit has its own purification leg,
which lets that leg travel with the qubit (below). With ``purification="global"`` one more site
``E`` of physical dimension 1 follows the port; it holds the environment (see Purification).

One external bit ``b`` (reading)
--------------------------------
1. Observe: ``O_b = sum_s sqrt(pi_s(b)) P_s`` is diagonal in the right-movers of the first
   ``k`` cells. It is applied as an exact matrix-product operator on sites ``0 .. 2k - 2``
   (the tensor train of the table ``sqrt(pi_s(b))`` on the R sites, identities on the L
   sites in between; bond at most ``2^floor(k/2)``) by :meth:`LPDO.apply_mpo`, recompressed in
   canonical gauge, and the state is renormalized. ``q_b = sum_s pi_s(b) Tr(P_s R)`` uses the
   sector probabilities ``Tr(P_s R)`` of all ``2^k`` sectors from one contraction
   (:meth:`LPDO.probabilities` of the R sites).
2. Inject (the dense ``Layout.target`` map): two layers of full-site swaps (physical and Kraus
   legs exchanged, :meth:`LPDO.swap`), first ``(R_j, L_j)`` for every cell, then
   ``(L_j, R_{j+1})`` and ``(L_{c-1}, A)``. Afterwards every right-mover has advanced one cell,
   every left-mover has retreated one, the port's content sits in ``L_{c-1}``, the old
   ``L_0`` sits on site 0 and the old ``R_{c-1}`` on site ``2c``. Those two leave by collisions
   (Kraus ``|v><0|, |v><1|``: the old content is traced out): site 0 is reset to ``|b>``
   (``b`` enters as ``R_0``) and site ``2c`` to ``|0>`` (the port is left empty).
3. Evolve: the coin ``cell_unitary(theta)`` on every pair ``(R_j, L_j)`` (a two-site gate),
   then the dissipator. Supported dephasing values are exactly ``0`` and ``inf``; any other
   value raises ``ValueError``. For finite ``g > 0`` the sector dissipator
   ``R -> e^-g R + (1 - e^-g) sum_s P_s R P_s`` is a correlated (non-local) channel, but at
   ``g = inf`` it is exactly the product of local full dephasings of the R qubits of cells
   ``0 .. k-1`` (``einselect="context"``; the sectors are the joint configurations of those
   qubits) or of every qubit including the port (``einselect="all"``; the mask is the
   identity over configurations), and at ``g = 0`` it is the identity.
4. Emit: ``p(tau) = sum_s pi_s(tau) Tr(P_s R_b)`` with the event's weights; ``tau`` is drawn
   with one ``rng.random()`` exactly as :meth:`vam.machine.Machine.step` and
   :meth:`~vam.realizations.vam1.PointerEngine.step` draw it (equal generators give equal
   emissions), and the state is conditioned on it once by ``J_tau = O_tau`` (as in step 1).
5. Re-enter: a collision writes ``tau`` into the port (site ``2c`` reset to ``|tau>``).
6. Plasticity from the external bit only: ``learn(cfg, w, n, a, b)`` with the posterior sector
   activity ``a_s = Tr(P_s R) pi_s(b) / q_b`` of the state before the event.

Generation (:meth:`TensorEngine.generate`) is ``DenseMaps.generation_protocol``: no external
bit, emit with ``J_tau = O_tau`` and condition once, write ``tau`` into the port, inject ``tau``
(step 2 with ``b = tau``, which also carries the port's ``tau`` into ``L_{c-1}``), evolve; no
plasticity.

Purification
------------
An LPDO stores, besides ``rho``, a purification: every Kraus leg holds the part of the
environment that a channel created at that site. The local SVD compressions remove only what
is linearly redundant at one site; correlations *between* Kraus legs of different sites stay in
the bonds of ``X`` even when they mean nothing for ``rho`` (two records of the same exited bit,
the swap history of the mass coin under einselection). With mass and dissipation this history
grows with every event. It is not a small tail: at two cells the untruncated bond exceeds the
dimension 32 of the whole register within eight events, also with a relative cutoff of
``1e-12`` (tested). Removing it exactly would need non-local unitaries
on the Kraus legs (a minimal-entanglement purification); the SVD-based layer has none. Two exact
modes are offered:

* ``purification="local"`` (default): each qubit's Kraus leg travels with the qubit (full-site
  swaps), so a qubit's dephasing records stay next to it. Storage grows with the number of
  events, so untruncated runs with mass and dissipation are exact but short (the tests run
  two to six events). With ``max_bond`` and ``max_kraus`` the storage is bounded, but the
  truncation then cuts the stored history and the result is not a faithful approximation when
  the field has mass and is einselected: at six cells (``max_bond = 32``, ``max_kraus = 4``)
  the predictions differ from the exact pointer engine by more than ``1e-3`` within 60 bits and
  the discarded weight exceeds 1 (tested, slow).
* ``purification="global"``: after every event all Kraus legs are moved onto the environment
  site ``E`` (:meth:`LPDO.concentrate_purification`). Untruncated this is the minimal
  purification: the Kraus dimension of ``E`` is ``rank(rho)`` and bond ``j`` is the rank of the
  reduced density of sites ``0..j``, both bounded by the register, not by time (tested). This is
  the mode for exact long runs on small registers (300 bits at three cells, tested). The
  concentration sweep merges all Kraus legs before compressing them, so its cost grows with
  ``rank(rho)``: about a minute per event at four cells in the fully einselected case
  (measured, not tested), and for mixed states of long chains those ranks grow exponentially
  with the length.

Both modes apply the same exact maps; without truncation (``max_bond=None``,
``max_kraus=None``, ``cutoff=0``) both are exact (tested against the dense, pointer and bit
engines), and they differ only in storage. In the massless limit the state stays a product
(bond 1, Kraus 1) and the engine is exact over long runs (2000 bits at 16 cells against the bit
kernel, tested).
"""

from __future__ import annotations

import copy
import math

import numpy as np

from ..tensornet import LPDO, diagonal_mpo
from .vam1 import Config, bit_probabilities, cell_unitary, learn

REALIZATION = {
    "source": "vam1 (DESIGN section 5) on vam.tensornet.LPDO (T13.1, T13.4, F-Part 9)",
    "status": "engine for the chosen realization VAM-1; no new modelling choice",
    "sites": "one qubit per site in the dense order R_0, L_0, ..., R_{c-1}, L_{c-1}, A (+ environment E when global)",
    "observation": "O_b as an exact diagonal MPO on the context R sites, canonical-gauge recompression",
    "injection": "two layers of full-site swaps; collisions reset site 0 to |b> and the port to |0>",
    "dissipator": "dephasing in {0, inf} only: identity, or local full dephasing (context R qubits or all qubits)",
    "purification": "local (Kraus legs travel with their qubits) or global (all Kraus legs on E after each event)",
}

_DEPHASE = (np.diag([1.0, 0.0]).astype(complex), np.diag([0.0, 1.0]).astype(complex))


def _collision(v: int) -> list[np.ndarray]:
    """Kraus operators ``|v><0|, |v><1|``: the old content leaves with a fresh unit, ``|v>`` stays."""
    e = np.eye(2, dtype=complex)
    return [np.outer(e[v], e[a]) for a in (0, 1)]


def _half_density(tensors: list[np.ndarray]) -> np.ndarray:
    """``E[ket, bra, ket bond, bra bond]`` of a chain segment, Kraus legs traced site by site (first site most
    significant); the outer bond on the left has size 1."""
    E = np.ones((1, 1, 1, 1), dtype=complex)
    for T in tensors:
        Y = np.tensordot(E, T, axes=([2], [0]))                    # (I, J, a', s, k, b)
        Z = np.tensordot(Y, T.conj(), axes=([2, 4], [0, 2]))       # (I, J, s, b, s', b')
        I, J, d, D = E.shape[0], E.shape[1], T.shape[1], T.shape[3]
        E = Z.transpose(0, 2, 1, 4, 3, 5).reshape(I * d, J * d, D, D)
    return E


def lpdo_density(lp: LPDO) -> np.ndarray:
    """``Tr_K |X><X|`` as a dense matrix, site 0 most significant, by meeting in the middle.

    The two halves are contracted separately (Kraus legs traced site by site) and joined over one bond, so
    the largest intermediate is ``4^(N/2) D^2``, not ``4^N D^2``; the purification vector is never formed.
    Exponential in ``N``: tests only.
    """
    ts = lp.tensors
    m = len(ts) // 2
    left = _half_density(ts[:m])
    right = _half_density([t.transpose(3, 1, 2, 0) for t in ts[:m - 1:-1]])   # mirrored: last site first
    n_right = len(ts) - m
    dims = [t.shape[1] for t in ts[m:]]
    dr = int(np.prod(dims))
    order = list(range(n_right))[::-1]
    right = right.reshape(dims[::-1] + dims[::-1] + list(right.shape[2:]))
    right = right.transpose(order + [n_right + i for i in order] + [2 * n_right, 2 * n_right + 1])
    right = right.reshape(dr, dr, *right.shape[-2:])
    rho = np.einsum("ijab,klab->ikjl", left, right)
    return rho.reshape(left.shape[0] * dr, left.shape[1] * dr)


class TensorEngine:
    """VAM-1 reading and generation on an LPDO; equal to :class:`~vam.realizations.vam1.DenseMaps` when untruncated.

    ``rng`` draws the emitted bits (one ``rng.random()`` per event, as ``Machine.step``); without
    it the engine uses ``default_rng(0)``. ``max_bond``, ``max_kraus`` and ``cutoff`` cap every
    bond split and every Kraus compression (see :mod:`vam.tensornet`); ``None``, ``None``, ``0``
    is exact. ``purification`` is ``"local"`` or ``"global"`` (module docstring).
    """

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None, max_bond: int | None = None,
                 max_kraus: int | None = None, cutoff: float = 0.0, purification: str = "local") -> None:
        if cfg.dephasing not in (0.0, math.inf):
            raise ValueError("the tensor engine supports dephasing = 0 or inf only: for finite g > 0 the sector "
                             "dissipator is a correlated channel, not a product of local channels")
        if purification not in ("local", "global"):
            raise ValueError("purification must be 'local' or 'global'")
        self.cfg = cfg
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.max_bond, self.max_kraus, self.cutoff = max_bond, max_kraus, float(cutoff)
        self.purification = purification
        c, k = cfg.cells, cfg.context
        self.port = 2 * c
        self.context_sites = list(range(0, 2 * k, 2))
        if math.isinf(cfg.dephasing):
            self.dephased_sites = self.context_sites if cfg.einselect == "context" else list(range(2 * c + 1))
        else:
            self.dephased_sites = []
        bits = (np.arange(2 ** k)[:, None] >> (k - 1 - np.arange(k))[None, :]) & 1   # row i: R_0 .. R_{k-1}
        self._sector = (bits << np.arange(k)).sum(axis=1)                             # basis index -> sector
        self._coin = cell_unitary(cfg.theta)
        vacuum = [np.diag([1.0, 0.0])] * (2 * c + 1)
        self.lp = LPDO.product(vacuum + ([np.ones((1, 1))] if purification == "global" else []))
        self.w = np.zeros(cfg.sectors)
        self.n = np.zeros(cfg.sectors)
        self.last_output: int | None = None
        self.last_output_probability: float | None = None

    # -- state ------------------------------------------------------------------
    def copy(self) -> TensorEngine:
        """An independent engine in the same present state (its generator is a copy of this one's)."""
        other = TensorEngine(self.cfg, rng=copy.deepcopy(self.rng), max_bond=self.max_bond,
                             max_kraus=self.max_kraus, cutoff=self.cutoff, purification=self.purification)
        other.lp, other.w, other.n = self.lp.copy(), self.w.copy(), self.n.copy()
        other.last_output, other.last_output_probability = self.last_output, self.last_output_probability
        return other

    @property
    def discarded(self) -> float:
        """Accumulated discarded fraction of ``Tr rho`` over every truncation (0 when exact)."""
        return self.lp.discarded

    @property
    def bond_dims(self) -> tuple[int, ...]:
        return self.lp.bond_dims

    @property
    def kraus_dims(self) -> tuple[int, ...]:
        return self.lp.kraus_dims

    def nbytes(self) -> int:
        """Bytes of the complete present state: the LPDO tensors plus ``w`` and ``n``."""
        return int(sum(t.nbytes for t in self.lp.tensors) + self.w.nbytes + self.n.nbytes)

    def trace(self) -> float:
        return self.lp.trace()

    def to_dense(self) -> np.ndarray:
        """The full density in the dense engine's qubit order. Exponential in ``c``: tests only.

        Not renormalized (its trace is :meth:`trace`). Computed by :func:`lpdo_density`, so the
        purification vector itself is never formed.
        """
        return lpdo_density(self.lp)

    def reduced_density(self, sites) -> np.ndarray:
        """Reduced density of qubit sites (``2j``: ``R_j``, ``2j + 1``: ``L_j``, ``2c``: the port)."""
        return self.lp.reduced_density(sites)

    def sector_probabilities(self) -> np.ndarray:
        """``Tr(P_s R)`` for every context sector ``s = sum_j R_j 2^j``."""
        p = self.lp.probabilities(self.context_sites)
        out = np.empty_like(p)
        out[self._sector] = p
        return out

    def predict(self, b: int) -> float:
        """``q_b = sum_s pi_s(b) Tr(P_s R)``: the probability the machine assigns to external bit ``b``."""
        return float(self.sector_probabilities() @ bit_probabilities(self.w, self.cfg.eps, int(b)))

    # -- elementary maps --------------------------------------------------------
    def observation_mpo(self, pi: np.ndarray) -> list[np.ndarray]:
        """Exact MPO of ``O = sum_s sqrt(pi_s) P_s`` on sites ``0 .. 2k - 2`` (identity on the L sites between).

        The tensor train of the table ``sqrt(pi_s)`` over ``(R_0, ..., R_{k-1})``; its bond is at most
        ``2^floor(k/2)``, and it is 1 when all weights are equal.
        """
        k = self.cfg.context
        table = np.sqrt(np.asarray(pi, dtype=float))[self._sector].reshape((2,) * k)
        mpo = []
        for j, W in enumerate(diagonal_mpo(table)):
            mpo.append(W)
            if j < k - 1:                                  # the L site between R_j and R_{j+1}
                D = W.shape[3]
                mpo.append(np.einsum("ab,ts->atsb", np.eye(D), np.eye(2)).astype(complex))
        return mpo

    def _condition(self, pi: np.ndarray) -> None:
        """``R -> O R O / Tr`` with ``O = sum_s sqrt(pi_s) P_s``, recompressed in canonical gauge."""
        self.lp.apply_mpo(self.observation_mpo(pi), 0, self.max_bond, self.cutoff)
        x = self.lp._x
        x.tensors[x.center] = x.tensors[x.center] / math.sqrt(self.lp.trace())

    def _collide(self, site: int, v: int) -> None:
        self.lp.apply_channel(_collision(v), site, self.max_kraus, self.cutoff)

    def _inject(self, b: int) -> None:
        c, mb, cut = self.cfg.cells, self.max_bond, self.cutoff
        for j in range(c):                                 # (R_j, L_j)
            self.lp.swap(2 * j, mb, cut)
        for j in range(c - 1, -1, -1):                     # (L_j, R_{j+1}); j = c-1 is (L_{c-1}, A)
            self.lp.swap(2 * j + 1, mb, cut)
            if j == c - 1:
                self._collide(self.port, 0)                # the old R_{c-1} leaves; the port is left empty
        self._collide(0, b)                                # the old L_0 leaves; b enters as R_0

    def _evolve(self) -> None:
        if self.cfg.theta != 0.0:
            for j in range(self.cfg.cells - 1, -1, -1):
                self.lp.apply_two(self._coin, 2 * j, self.max_bond, self.cutoff)
        for s in self.dephased_sites:
            self.lp.apply_channel(_DEPHASE, s, self.max_kraus, self.cutoff)

    def _finish(self) -> None:
        if self.purification == "global":
            self.lp.concentrate_purification(self.max_bond, self.max_kraus, self.cutoff)

    def _emission(self) -> np.ndarray:
        sp = self.sector_probabilities()
        return np.array([max(0.0, float(sp @ bit_probabilities(self.w, self.cfg.eps, t))) for t in (0, 1)])

    def _draw(self, p_tau: np.ndarray) -> int:
        """One ``rng.random()``, exactly as ``Machine.step`` and ``PointerEngine.step``."""
        idx = int(np.searchsorted(np.cumsum(p_tau / p_tau.sum()), self.rng.random(), side="right"))
        return min(idx, 1)

    # -- events -------------------------------------------------------------------
    def _read(self, b: int, tau: int | None, learning: bool) -> float:
        if b not in (0, 1) or tau not in (None, 0, 1):
            raise ValueError("bits must be 0 or 1")
        cfg, w = self.cfg, self.w
        sp = self.sector_probabilities()
        pib = bit_probabilities(w, cfg.eps, b)
        q = float(sp @ pib)
        self._condition(pib)                               # 1. observe at the sensory port
        self._inject(b)                                    # 2. inject
        self._evolve()                                     # 3. coin and dissipator
        p_tau = self._emission()                           # 4. emit with the event's weights
        if tau is None:
            tau = self._draw(p_tau)
        self._condition(bit_probabilities(w, cfg.eps, tau))
        self._collide(self.port, tau)                      # 5. re-enter at the active port
        self._finish()
        self.last_output, self.last_output_probability = tau, float(p_tau[tau] / p_tau.sum())
        if learning:                                       # 6. plasticity from the external bit only
            self.w, self.n = learn(cfg, w, self.n, sp * pib / q, b)
        return q

    def step(self, b: int, learning: bool = True) -> float:
        """Observe, inject, evolve, emit, re-enter and learn for one external bit; returns ``q_b``."""
        return self._read(int(b), None, learning)

    def force(self, b: int, tau: int, learning: bool = True) -> float:
        """One reading event with the prescribed emission ``tau`` (exact branch tracking); returns ``q_b``."""
        return self._read(int(b), int(tau), learning)

    def generate(self, tau: int | None = None) -> int:
        """One generation step (``DenseMaps.generation_protocol``); ``tau`` prescribes the output.

        Emit with ``J_tau = O_tau`` (drawn with one ``rng.random()`` unless prescribed) and
        condition once, write ``tau`` into the port, inject ``tau`` at the sensory port, evolve.
        The weights do not change. Returns the emitted bit.
        """
        if tau not in (None, 0, 1):
            raise ValueError("bits must be 0 or 1")
        p_tau = self._emission()
        if tau is None:
            tau = self._draw(p_tau)
        self._condition(bit_probabilities(self.w, self.cfg.eps, tau))
        self._collide(self.port, tau)
        self._inject(tau)
        self._evolve()
        self._finish()
        self.last_output, self.last_output_probability = tau, float(p_tau[tau] / p_tau.sum())
        return tau
