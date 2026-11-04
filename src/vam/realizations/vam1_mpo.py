"""VAM-1 on a density stored as a matrix-product operator: the bounded engine (T13.1, T13.4, F-Part 9).

:class:`OperatorEngine` runs exactly the reading event of
:meth:`vam.realizations.vam1.DenseMaps.reading_protocol` and the generation step of
:meth:`~vam.realizations.vam1.DenseMaps.generation_protocol`, like
:class:`vam.realizations.vam1_tn.TensorEngine`, but it stores the density itself rather than a
purification of it. ``vec(rho)`` is a matrix-product state (:class:`vam.tensornet.MPS`) with one site
per qubit whose local index ``2 s + s'`` labels the matrix element ``|s><s'|`` of that qubit; read as
an operator, this is a matrix-product operator (MPO) for ``rho``. A channel with Kraus operators
``K`` acts on ``vec(rho)`` as its superoperator ``sum_K K (x) conj(K)`` (:func:`superoperator`), so
every local map of the event is a one- or two-site operator on this MPS, and the observation and the
finite-dephasing dissipator are exact MPOs.

Why a second engine
-------------------
A purification records the environment's history: which bit left the chain when, and every swap the
mass coin made under einselection. With mass and einselection its bonds grow with time, and
truncating them is not faithful (``vam1_tn``: at six cells the truncated engine misses the exact one
by up to 0.019 per prediction). The density keeps no history. Its operator Schmidt values across a
cut decay fast, and the bond a given accuracy needs is set by the register, not by the number of
events. Measured before this engine was written (``experiments/diagnostics/``):

* fully einselected chain (exact pointer engine, 400 bits, worst case over events and cell cuts):
  the bond that keeps the discarded squared weight below ``1e-6`` of the total is 6-9, 10-14, 14-18
  and 14-25 at 4, 6, 8 and 10 cells, against an exact rank of 12, 32, 128 and 412;
* the configuration probabilities truncated to bond 64 after every event follow the exact engine at
  8 cells over 1200 bits of English text to ``6e-8`` (theta 0.3) and ``3e-9`` (theta 0.6) in every
  prediction, with every emission equal; bond 16 gives ``3e-4`` and ``4e-5``;
* context einselection (the dense four-cell density after 200 events): bond 3-7 at ``1e-6`` and at
  most 10 at ``1e-10``, where the register allows 256.

Measured limit
--------------
Coherence, not history. With ``einselect="context"`` the ``L`` movers and the ``R`` movers past the
context are never dephased, so that part of the chain is a coherent Dirac walk and its operator
entanglement grows with the length. Exact at six cells, bond 64 follows the untruncated engine to
``2.4e-6`` (tested); at 12 cells, where no exact engine reaches, bonds 16, 32 and 64 differ from bond
128 by up to ``3.3e-3``, ``1.5e-3`` and ``5.6e-4`` per prediction over 93 bits (theta 0.3): the error
roughly halves per doubling of the bond. Fully einselected chains stay cheap (``1.1e-5`` at ten cells
against the pointer engine, bond 64).

What storing the density costs, and what it gains
-------------------------------------------------
* Positivity is not structural. Untruncated the engine is exact; truncated, a reduced density can
  have small negative eigenvalues, so the tests measure the smallest one instead of assuming it.
* Truncation is optimal in the Hilbert-Schmidt norm of ``rho`` (the canonical gauge of
  ``vec(rho)``), not in the trace norm. :attr:`OperatorEngine.discarded` sums Hilbert-Schmidt
  fractions; whether predictions stay faithful is measured against the exact engines.
* Finite dephasing ``0 < g < inf`` is supported exactly: the sector dissipator
  ``e^-g R + (1 - e^-g) sum_s P_s R P_s`` is ``e^-g Id + (1 - e^-g) prod_i Delta_i`` on
  ``vec(rho)``, with ``Delta_i`` the full dephasing of one sector qubit (every context ``R`` qubit, or
  every qubit when ``einselect="all"``): an MPO of bond 2.
* Storage is ``max_bond`` per bond with no Kraus legs.

Sites and the event
-------------------
As in ``vam1_tn``: one qubit per site in the dense engine's order, ``2j`` is ``R_j``, ``2j + 1`` is
``L_j`` and ``2c`` is the active port ``A`` (site 0 most significant).

1. Observe: ``rho -> O_b rho O_b / q_b`` with ``O_b = sum_s sqrt(pi_s(b)) P_s``, applied as the MPO
   ``W (x) conj(W)`` on sites ``0 .. 2k - 2`` where ``W`` is the tensor train of the table
   ``sqrt(pi_s(b))`` (identity on the ``L`` sites in between).
2. Inject: two layers of swaps ``(R_j, L_j)`` and then ``(L_j, R_{j+1})``, ``(L_{c-1}, A)``; the old
   ``R_{c-1}``, now on the port, and the old ``L_0``, now on site 0, leave by collisions: the port is
   reset to ``|0>`` and site 0 to ``|b>``.
3. Evolve: the coin ``cell_unitary(theta)`` on every ``(R_j, L_j)``, then the dissipator.
4. Emit ``tau`` with ``p(tau) = sum_s pi_s(tau) Tr(P_s R)`` (one ``rng.random()``, exactly as
   :meth:`vam.machine.Machine.step`) and condition once by ``O_tau``.
5. Re-enter: a collision writes ``tau`` into the port.
6. Plasticity from the external bit only, with the posterior sector activity of the state before
   the event.

Generation is ``DenseMaps.generation_protocol``: emit and condition, write ``tau`` into the port,
inject ``tau``, evolve; the weights do not change.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from ..tensornet import MPS, diagonal_mpo
from .vam1 import Config, bit_probabilities, cell_unitary, learn

REALIZATION = {
    "source": "vam1 (DESIGN section 5) on vec(rho) as vam.tensornet.MPS, i.e. an MPO density (T13.1, T13.4, "
              "F-Part 9)",
    "status": "engine for the chosen realization VAM-1; no new modelling choice",
    "sites": "one qubit per site in the dense order R_0, L_0, ..., R_{c-1}, L_{c-1}, A; local index 2 s + s'",
    "maps": "superoperators sum_K K (x) conj(K); observation W (x) conj(W) as an exact MPO",
    "dissipator": "any dephasing: identity (0), local full dephasing (inf), or the bond-2 MPO "
                  "e^-g Id + (1 - e^-g) prod_i Delta_i (finite)",
    "truncation": "Hilbert-Schmidt optimal in canonical gauge; positivity measured, not structural",
}

#: ``vec(I)`` of one qubit: contracting a site's local index with it takes the partial trace.
_TRACE = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex)
#: ``vec`` positions of the diagonal elements ``|0><0|`` and ``|1><1|``.
_DIAG = np.array([0, 3])
#: full dephasing of one qubit on ``vec(rho)``.
_DEPHASE = np.diag([1.0, 0.0, 0.0, 1.0]).astype(complex)
_SWAP = np.eye(4, dtype=complex)[[0, 2, 1, 3]]


def superoperator(kraus) -> np.ndarray:
    """``sum_K K (x) conj(K)``: the channel's action on ``vec(rho)`` of its qubits (index ``2 s + s'`` per qubit
    for one qubit)."""
    return sum(np.kron(K, np.conj(K)) for K in (np.asarray(K, dtype=complex) for K in kraus))


def superoperator_two(U: np.ndarray) -> np.ndarray:
    """``rho -> U rho U^dagger`` for a two-qubit ``U``, on the two ``vec`` sites ``(2 t1 + t1', 2 t2 + t2')``."""
    U4 = np.asarray(U, dtype=complex).reshape(2, 2, 2, 2)                  # (t1, t2, s1, s2)
    return np.einsum("abcd,efgh->aebfcgdh", U4, U4.conj()).reshape(16, 16)


def _collision(v: int) -> np.ndarray:
    """Reset to ``|v>``: Kraus ``|v><0|, |v><1|``, the old content leaves."""
    e = np.eye(2, dtype=complex)
    return superoperator([np.outer(e[v], e[a]) for a in (0, 1)])


def _vec_mpo(mpo: list[np.ndarray]) -> list[np.ndarray]:
    """The MPO ``W (x) conj(W)`` on ``vec`` sites for an operator MPO ``W``: ``rho -> W rho W^dagger``."""
    out = []
    for W in mpo:
        a, d, _, b = W.shape
        V = np.einsum("ptsq,PTSQ->pPtTsSqQ", W, W.conj())
        out.append(V.reshape(a * a, d * d, d * d, b * b))
    return out


class OperatorEngine:
    """VAM-1 reading and generation on an MPO density; equal to :class:`~vam.realizations.vam1.DenseMaps` when
    untruncated.

    ``rng`` draws the emitted bits (one ``rng.random()`` per event, as ``Machine.step``); without it
    the engine uses ``default_rng(0)``. ``max_bond`` and ``cutoff`` cap every bond split (see
    :meth:`vam.tensornet.MPS.apply_two`); ``None`` and ``0`` is exact up to numerically zero singular
    values. Every dephasing ``g >= 0`` and both einselections are supported.
    """

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None, max_bond: int | None = None,
                 cutoff: float = 0.0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.max_bond, self.cutoff = max_bond, float(cutoff)
        c, k = cfg.cells, cfg.context
        self.port = 2 * c
        self.context_sites = list(range(0, 2 * k, 2))
        self.sector_sites = self.context_sites if cfg.einselect == "context" else list(range(2 * c + 1))
        bits = (np.arange(2 ** k)[:, None] >> (k - 1 - np.arange(k))[None, :]) & 1   # row i: R_0 .. R_{k-1}
        self._sector = (bits << np.arange(k)).sum(axis=1)                             # basis index -> sector
        self._coin = superoperator_two(cell_unitary(cfg.theta))
        self._swap = superoperator_two(_SWAP)
        self._reset = {v: _collision(v) for v in (0, 1)}
        self._dissipator = self._dissipator_mpo() if 0.0 < cfg.dephasing < math.inf else None
        vacuum = np.array([1.0, 0.0, 0.0, 0.0])
        self.x = MPS.product([vacuum] * (2 * c + 1))
        self.x.center = 0
        self.w = np.zeros(cfg.sectors)
        self.n = np.zeros(cfg.sectors)
        self.last_output: int | None = None
        self.last_output_probability: float | None = None

    # -- state ------------------------------------------------------------------
    def copy(self) -> OperatorEngine:
        """An independent engine in the same present state (its generator is a copy of this one's)."""
        other = OperatorEngine(self.cfg, rng=copy.deepcopy(self.rng), max_bond=self.max_bond, cutoff=self.cutoff)
        other.x, other.w, other.n = self.x.copy(), self.w.copy(), self.n.copy()
        other.last_output, other.last_output_probability = self.last_output, self.last_output_probability
        return other

    @property
    def discarded(self) -> float:
        """Accumulated discarded fraction of ``|rho|_HS^2`` over every truncation (0 when exact)."""
        return self.x.discarded

    @property
    def bond_dims(self) -> tuple[int, ...]:
        return self.x.bond_dims

    def nbytes(self) -> int:
        """Bytes of the complete present state: the MPO tensors plus ``w`` and ``n``."""
        return int(sum(t.nbytes for t in self.x.tensors) + self.w.nbytes + self.n.nbytes)

    def _traced_right(self, start: int) -> np.ndarray:
        """The bond vector left of ``start`` with every site from ``start`` on traced out."""
        v = np.ones(1, dtype=complex)
        for T in reversed(self.x.tensors[start:]):
            v = np.tensordot(T, v, axes=(2, 0)) @ _TRACE
        return v

    def trace(self) -> float:
        return float(self._traced_right(0)[0].real)

    def to_dense(self) -> np.ndarray:
        """The full density in the dense engine's qubit order. Exponential in ``c``: tests only."""
        n = len(self.x)
        R = self.x.to_dense().reshape((2,) * (2 * n))
        return R.transpose(list(range(0, 2 * n, 2)) + list(range(1, 2 * n, 2))).reshape(2 ** n, 2 ** n)

    def reduced_density(self, sites) -> np.ndarray:
        """Reduced density (trace 1) of qubit sites (sorted); the first listed site is most significant."""
        sites = sorted(int(s) for s in sites)
        keep = set(sites)
        E = np.ones((1, 1), dtype=complex)                     # E[open vec indices, bond]
        for j in range(sites[-1] + 1):
            Y = np.tensordot(E, self.x.tensors[j], axes=(1, 0))          # (open, 4, bond)
            E = Y.reshape(-1, Y.shape[2]) if j in keep else np.einsum("osb,s->ob", Y, _TRACE)
        v = (E @ self._traced_right(sites[-1] + 1)).reshape((2,) * (2 * len(sites)))
        m = len(sites)
        R = v.transpose(list(range(0, 2 * m, 2)) + list(range(1, 2 * m, 2))).reshape(2 ** m, 2 ** m)
        return R / np.trace(R)

    def sector_probabilities(self) -> np.ndarray:
        """``Tr(P_s R)`` for every context sector ``s = sum_j R_j 2^j``, from one contraction."""
        last = self.context_sites[-1]
        keep = set(self.context_sites)
        E = np.ones((1, 1), dtype=complex)                     # E[outcome, bond]
        for j in range(last + 1):
            Y = np.tensordot(E, self.x.tensors[j], axes=(1, 0))          # (outcome, 4, bond)
            E = Y[:, _DIAG, :].reshape(-1, Y.shape[2]) if j in keep else np.einsum("osb,s->ob", Y, _TRACE)
        p = np.real(E @ self._traced_right(last + 1))
        out = np.empty_like(p)
        out[self._sector] = p / p.sum()
        return out

    def predict(self, b: int) -> float:
        """``q_b = sum_s pi_s(b) Tr(P_s R)``: the probability the machine assigns to external bit ``b``."""
        return float(self.sector_probabilities() @ bit_probabilities(self.w, self.cfg.eps, int(b)))

    # -- elementary maps --------------------------------------------------------
    def observation_mpo(self, pi: np.ndarray) -> list[np.ndarray]:
        """Exact operator MPO of ``O = sum_s sqrt(pi_s) P_s`` on sites ``0 .. 2k - 2`` (identity on the L sites
        between); bond at most ``2^floor(k/2)``. :meth:`_condition` applies ``O (x) conj(O)`` to ``vec(rho)``."""
        k = self.cfg.context
        table = np.sqrt(np.asarray(pi, dtype=float))[self._sector].reshape((2,) * k)
        mpo = []
        for j, W in enumerate(diagonal_mpo(table)):
            mpo.append(W)
            if j < k - 1:                                  # the L site between R_j and R_{j+1}
                D = W.shape[3]
                mpo.append(np.einsum("ab,ts->atsb", np.eye(D), np.eye(2)).astype(complex))
        return mpo

    def _dissipator_mpo(self) -> list[np.ndarray]:
        """``e^-g Id + (1 - e^-g) prod_{i in sectors} Delta_i`` on sites ``first .. last`` of the sector qubits."""
        keep = math.exp(-self.cfg.dephasing)
        first, last = self.sector_sites[0], self.sector_sites[-1]
        I4 = np.eye(4, dtype=complex)
        mpo = []
        for j in range(first, last + 1):
            W = np.zeros((2, 4, 4, 2), dtype=complex)
            W[0, :, :, 0] = I4
            W[1, :, :, 1] = _DEPHASE if j in self.sector_sites else I4
            mpo.append(W)
        mpo[0] = np.tensordot(np.array([keep, 1.0 - keep]), mpo[0], axes=(0, 0))[None]
        mpo[-1] = np.tensordot(mpo[-1], np.ones(2), axes=(3, 0))[..., None]
        return mpo

    def _normalize(self) -> None:
        """Rescale to trace 1 (on the centre tensor, so a known gauge stays valid)."""
        j = self.x.center if self.x.center is not None else 0
        self.x.tensors[j] = self.x.tensors[j] / self.trace()

    def _condition(self, pi: np.ndarray) -> None:
        """``R -> O R O / Tr`` with ``O = sum_s sqrt(pi_s) P_s``, recompressed in canonical gauge."""
        self.x.apply_mpo(_vec_mpo(self.observation_mpo(pi)), 0, self.max_bond, self.cutoff)
        self._normalize()

    def _collide(self, site: int, v: int) -> None:
        self.x.apply_one(self._reset[v], site)

    def _inject(self, b: int) -> None:
        c, mb, cut = self.cfg.cells, self.max_bond, self.cutoff
        for j in range(c):                                 # (R_j, L_j)
            self.x.apply_two(self._swap, 2 * j, mb, cut)
        for j in range(c - 1, -1, -1):                     # (L_j, R_{j+1}); j = c-1 is (L_{c-1}, A)
            self.x.apply_two(self._swap, 2 * j + 1, mb, cut)
            if j == c - 1:
                self._collide(self.port, 0)                # the old R_{c-1} leaves; the port is left empty
        self._collide(0, b)                                # the old L_0 leaves; b enters as R_0

    def _evolve(self) -> None:
        if self.cfg.theta != 0.0:
            for j in range(self.cfg.cells - 1, -1, -1):
                self.x.apply_two(self._coin, 2 * j, self.max_bond, self.cutoff)
        if math.isinf(self.cfg.dephasing):
            for s in self.sector_sites:
                self.x.apply_one(_DEPHASE, s)
        elif self._dissipator is not None:
            self.x.apply_mpo(self._dissipator, self.sector_sites[0], self.max_bond, self.cutoff)

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

        Emit with ``J_tau = O_tau`` (drawn with one ``rng.random()`` unless prescribed) and condition
        once, write ``tau`` into the port, inject ``tau`` at the sensory port, evolve. The weights do
        not change. Returns the emitted bit.
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
        self._normalize()
        self.last_output, self.last_output_probability = tau, float(p_tau[tau] / p_tau.sum())
        return tau
