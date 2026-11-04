"""VAM-1: the reference machine for arbitrary bit streams (DESIGN section 5, decision D8).

T15.4 says the principles alone do not fix the couplings, so a run needs one
declared realization. Every element below is a chosen realization built only
from mechanisms in T and F; the record ``REALIZATION`` lists them.

Registers
---------
* ``V``: a lattice Dirac field of ``c`` cells. Cell ``j`` holds a right-moving
  mode ``R_j`` (qubit ``2j``) and a left-moving mode ``L_j`` (qubit
  ``2j + 1``); qubit 0 is the leftmost tensor factor. This is the staggered
  (two-sublattice) form of the split-step Dirac walk: its one-excitation
  sector is exactly :func:`vam.dirac.split_step_walk` with coin ``theta``
  (tested), and it has an exact event cone of one cell per event (T13.3).
* Sensory port ``Bs`` at the left boundary, written in its Stinespring-reduced
  form (tested against the explicit port qubit).
* Active port ``A`` at the right boundary: one qubit (qubit ``2c``, the last
  tensor factor) that holds the machine's last emitted bit until the next
  transport carries it into the chain as the entering left-mover ``L_{c-1}``.
  The two blanket ports are distinct, as T10.3 requires: the sensory port
  takes the environment's bits, the active port takes the machine's own.
* Structural coordinates: one slow weight ``w_s`` and one activity
  accumulator ``n_s`` per context sector ``s`` (the right-mover content of the
  first ``k`` cells; ``s = sum_j R_j 2^j`` with ``R_0`` the newest bit).

One external bit ``b`` (reading; conditioning mode, T10.1, T14 steps 2-15)
---------------------------------------------------------------------------
1. Observe at ``Bs``: a weight-controlled port interaction
   ``sum_s P_s kron exp(-i theta_s Y / 2)`` followed by the port readout. The
   reduced Kraus operator is ``O_b = sum_s sqrt(pi_s(b)) P_s`` with
   ``pi_s(0) = (1 + (1 - 2 eps) w_s) / 2``, so the machine's prediction of the
   bit is ``q_b = sum_s pi_s(b) Tr(P_s R)``. Coherence inside a sector
   survives (T7.3). The floor ``eps > 0`` keeps every possible bit possible
   (T13.2: absence of evidence is not a hard constraint), so ``q_b > 0``.
2. Inject (collision model, T6.2): right-movers advance one cell and
   left-movers retreat one cell; ``R_{c-1}`` and ``L_0`` leave with fresh
   environment units (traced out), the bit enters as ``R_0``, and the content
   of the active port enters as ``L_{c-1}`` (the port is left empty).
3. Dirac-Zurek evolution: the mass coin ``exp(-i theta (XX + YY) / 2)`` in every
   cell, then the sector dissipator ``exp(D_B)``,
   ``R -> e^-gamma_D R + (1 - e^-gamma_D) sum_s P_s R P_s`` (T7.3), on the
   context sectors (``einselect="context"``) or on every pointer
   configuration (``einselect="all"``, the fully einselected limit).
4. Emit (T14 step 11): the output instrument is the active-port readout
   ``J_tau = O_tau``, the same weight-controlled port algebra, so the machine
   emits a bit with exactly its own prediction for the next bit, and the state
   is conditioned on that output once.
5. Re-enter (T14 step 12): ``E_tau`` writes ``tau`` into the active port by a
   collision (the port's previous content leaves with an environment unit).
   It is trace preserving and conditions on nothing (T10.3). The emitted bit
   travels into the chain at the next transport and moves leftward; with mass
   the coin mixes it into the right-movers, and so into the context.
6. Plasticity (T12.1, activity gated; ``plasticity.py``), from the external
   bit only: with posterior sector activity ``a_s = Tr(P_s O_b R O_b) / q_b``
   and correlation ``C_s = z_b a_s`` (``z_b = +1`` for 0, ``-1`` for 1),
   ``w_s <- g z_b / kappa + (w_s - g z_b / kappa) exp(-kappa a_s dt / gamma_s)``.
   ``gamma_s = gamma`` is the literal law of T12.1: under one repeated lesson
   its transient shrinks by the fixed factor ``r = exp(-kappa a dt / gamma)``
   per repetition, which is the geometric consolidation ``r^n`` of T12.2
   (tested). With ``consolidating=True``, ``gamma_s = gamma + kappa dt n_s``
   and ``n_s <- n_s + a_s``: a friction that grows with accumulated activity.
   T does not state this law; it is a chosen addition (decision D10). With
   ``g = kappa = gamma = dt = 1`` its step is ``1 - exp(-1 / (n + 1))``, close
   to ``1 / (n + 1)``, so the weight becomes a running average of the
   sector's ``z``: frequency counting, whose transient decays harmonically,
   not geometrically. Under either law with ``g = kappa`` the fixed point is
   ``w_s = E[z | s]``, so ``pi_s`` approaches the source's conditional bit
   statistics: learned forward, nothing replayed.

In the massless limit (``theta = 0``) the context sector is always definite,
so conditioning on the emitted bit changes nothing, and the left-moving
outputs never reach the right-movers: the predictions and weights are exactly
those of a reader that emits nothing (tested against the bit kernel). With
mass, emission and re-entry change the dynamics.

Generation (answering, T14 steps 11-15)
--------------------------------------
No external bit; the settling step is the identity. The machine emits
``tau`` with ``J_tau = O_tau`` and conditions on it once, exactly as in
reading. Re-entry ``E_tau`` writes ``tau`` into the active port and, because
no external stream remains, the sensory port receives the emitted bit too:
``tau`` is injected (step 2, which also carries the port's content into the
chain) and the chain evolves (step 3) before the next emission. Plasticity is
off during generation: re-entered outputs transform the state but are not
evidence (T10.3). Macro/micro selection is :func:`vam.machine.macro_micro_step`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..canalization import CanalizationAlgebra, invariant_algebra
from ..channels import FunctionMap, Instrument, KrausMap
from ..machine import EventContext, EventProtocol, Machine, MachineState

REALIZATION = {
    "source": "DESIGN section 5 (T6.2, T7.3, T10, T12.1, T13.2, T13.3, T14; F master loop, Parts 2, 3, 7, 10)",
    "status": "chosen realization",
    "field": "c cells of (R_j, L_j) qubits; mass coin exp(-i theta (XX+YY)/2); transport one cell per event",
    "observation": "O_b = sum_s sqrt(pi_s(b)) P_s, pi_s(0) = (1 + (1-2 eps) w_s)/2",
    "injection": "collision: R_{c-1}, L_0 leave; b enters as R_0; the active port's content enters as L_{c-1}",
    "einselection": "sector dissipator exp(D_B) on context sectors or on all pointer configurations",
    "emission": "J_tau = O_tau at the active port after every external bit, conditioned once",
    "re-entry": "collision writes tau into the active port qubit; no conditioning (T10.3)",
    "plasticity": "gated T12.1 law with posterior activity, external bits only; optional count-dependent "
                  "friction (D10, not in T)",
    "generation": "J_tau = O_tau, re-entry at the active port and through the sensory injection, plasticity off",
    "initial": "vacuum chain, empty active port, w = 0, n = 0",
}


@dataclass(frozen=True)
class Config:
    cells: int
    context: int
    theta: float = 0.0
    dephasing: float = math.inf
    einselect: str = "context"
    eps: float = 2.0 ** -12
    g: float = 1.0
    kappa: float = 1.0
    gamma: float = 1.0
    dt: float = 0.05
    consolidating: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.context <= self.cells:
            raise ValueError("need 1 <= context <= cells")
        if self.einselect not in ("context", "all"):
            raise ValueError("einselect must be 'context' or 'all'")
        if not 0 < self.eps < 0.5:
            raise ValueError("need 0 < eps < 1/2")
        if self.dephasing < 0 or self.kappa <= 0 or self.gamma <= 0 or self.dt <= 0:
            raise ValueError("rates must be positive")

    @property
    def qubits(self) -> int:
        """The chain's ``2c`` mode qubits plus the active port qubit."""
        return 2 * self.cells + 1

    @property
    def dim(self) -> int:
        return 2 * 4 ** self.cells

    @property
    def sectors(self) -> int:
        return 2 ** self.context


# ----------------------------------------------------------------------------
# Prediction and plasticity (shared by every engine)
# ----------------------------------------------------------------------------

def predictions(w: np.ndarray, eps: float) -> np.ndarray:
    """``pi_s(0)`` for every sector."""
    return 0.5 * (1.0 + (1.0 - 2.0 * eps) * np.clip(w, -1.0, 1.0))


def bit_probabilities(w: np.ndarray, eps: float, b: int) -> np.ndarray:
    p0 = predictions(w, eps)
    return p0 if b == 0 else 1.0 - p0


def learn(cfg: Config, w: np.ndarray, n: np.ndarray, activity: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    """One gated T12.1 update from posterior sector activity ``a_s`` after observing ``b``."""
    z = 1.0 if b == 0 else -1.0
    target = cfg.g * z / cfg.kappa
    gamma_s = cfg.gamma + (cfg.kappa * cfg.dt * n if cfg.consolidating else 0.0)
    w_new = target + (w - target) * np.exp(-cfg.kappa * activity * cfg.dt / gamma_s)
    return w_new, n + activity


# ----------------------------------------------------------------------------
# Pointer-basis bookkeeping
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    """Index tables for the ``2 * 4^c`` pointer configurations (qubit 0 most significant).

    Qubits ``2j`` and ``2j + 1`` are ``R_j`` and ``L_j``; qubit ``2c`` is the active port.
    """

    cfg: Config
    bits: np.ndarray          # (dim, 2c + 1) occupation of each qubit
    sector: np.ndarray        # (dim,) context sector of each configuration
    target: dict              # b -> (dim,) configuration after transport with input b
    exits: np.ndarray         # (dim,) 2 R_{c-1} + L_0 of each configuration
    port: np.ndarray          # (dim,) content of the active port
    write: dict               # tau -> (dim,) configuration with the active port set to tau

    @classmethod
    def build(cls, cfg: Config) -> Layout:
        nq, dim, c = cfg.qubits, cfg.dim, cfg.cells
        idx = np.arange(dim)
        weights = 1 << (nq - 1 - np.arange(nq))
        bits = (idx[:, None] >> (nq - 1 - np.arange(nq))[None, :]) & 1
        R, L, port = bits[:, 0:2 * c:2], bits[:, 1:2 * c:2], bits[:, 2 * c]
        sector = (R[:, :cfg.context] << np.arange(cfg.context)).sum(axis=1)
        target = {}
        for b in (0, 1):
            new = np.empty_like(bits)
            new[:, 0:2 * c:2] = np.concatenate([np.full((dim, 1), b), R[:, :-1]], axis=1)
            new[:, 1:2 * c:2] = np.concatenate([L[:, 1:], port[:, None]], axis=1)
            new[:, 2 * c] = 0
            target[b] = new @ weights
        exits = 2 * R[:, c - 1] + L[:, 0]
        write = {t: (idx & ~1) | t for t in (0, 1)}
        return cls(cfg, bits, sector, target, exits, port, write)

    def sector_probabilities(self, diag: np.ndarray) -> np.ndarray:
        return np.bincount(self.sector, weights=np.real(diag), minlength=self.cfg.sectors)


def cell_unitary(theta: float) -> np.ndarray:
    """``exp(-i theta (XX + YY)/2)`` on ``(R_j, L_j)``: the coin ``exp(-i theta X)`` on one excitation."""
    U = np.eye(4, dtype=complex)
    U[1, 1] = U[2, 2] = np.cos(theta)
    U[1, 2] = U[2, 1] = -1j * np.sin(theta)
    return U


# ----------------------------------------------------------------------------
# Dense engine: the event maps on the full density
# ----------------------------------------------------------------------------

class DenseMaps:
    """The VAM-1 event maps on the ``4^c``-dimensional chain density.

    Kraus forms (for the record and for the tests that check the actions):
    observation and emission ``O_b`` (one diagonal Kraus operator); injection
    ``K_{rl} = sum_{i: exits(i)=rl} |target_b(i)><i|`` (four partial
    permutations); re-entry ``K_a = sum_{i: port(i)=a} |write_tau(i)><i|`` (two
    partial permutations); coin ``U = U_cell^{kron c} kron I_port``;
    dissipator ``sqrt(e^-gamma) I`` and ``sqrt(1 - e^-gamma) P_s``.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.lay = Layout.build(cfg)
        U = np.ones((1, 1), dtype=complex)
        Uc = cell_unitary(cfg.theta)
        for _ in range(cfg.cells):
            U = np.kron(U, Uc)
        self.U = np.kron(U, np.eye(2))                    # the coin acts on the cells, not on the port
        if cfg.einselect == "context":
            s = self.lay.sector
            self.mask = (s[:, None] == s[None, :]).astype(float)
        else:
            self.mask = np.eye(cfg.dim)
        self.keep = 0.0 if math.isinf(cfg.dephasing) else math.exp(-cfg.dephasing)
        self._exit_sets = [np.flatnonzero(self.lay.exits == e) for e in range(4)]
        self._port_sets = [np.flatnonzero(self.lay.port == a) for a in (0, 1)]

    # -- elementary maps --------------------------------------------------------
    def observe(self, w: np.ndarray, b: int) -> FunctionMap:
        d = np.sqrt(bit_probabilities(w, self.cfg.eps, b))[self.lay.sector]
        D = np.outer(d, d)
        return FunctionMap(lambda R: R * D, lambda A: A * D, self.cfg.dim, label=f"O_{b}")

    def inject(self, b: int) -> FunctionMap:
        tgt, sets, dim = self.lay.target[b], self._exit_sets, self.cfg.dim
        same_exit = (self.lay.exits[:, None] == self.lay.exits[None, :])

        def fwd(R: np.ndarray) -> np.ndarray:
            out = np.zeros_like(R)
            for S in sets:
                T = tgt[S]
                out[np.ix_(T, T)] += R[np.ix_(S, S)]
            return out

        def dual(A: np.ndarray) -> np.ndarray:
            return A[np.ix_(tgt, tgt)] * same_exit

        return FunctionMap(fwd, dual, dim, label=f"inject_{b}")

    def reenter(self, tau: int) -> FunctionMap:
        """Write ``tau`` into the active port by a collision; the old content leaves."""
        tgt, sets, dim = self.lay.write[tau], self._port_sets, self.cfg.dim
        same_port = (self.lay.port[:, None] == self.lay.port[None, :])

        def fwd(R: np.ndarray) -> np.ndarray:
            out = np.zeros_like(R)
            for S in sets:
                T = tgt[S]
                out[np.ix_(T, T)] += R[np.ix_(S, S)]
            return out

        def dual(A: np.ndarray) -> np.ndarray:
            return A[np.ix_(tgt, tgt)] * same_port

        return FunctionMap(fwd, dual, dim, label=f"reenter_{tau}")

    def evolve(self) -> FunctionMap:
        U, Ud, M, k = self.U, self.U.conj().T, self.mask, self.keep

        def deph(R: np.ndarray) -> np.ndarray:
            return k * R + (1.0 - k) * (R * M)

        return FunctionMap(lambda R: deph(U @ R @ Ud), lambda A: Ud @ deph(A) @ U, self.cfg.dim, label="evolve")

    # -- protocols --------------------------------------------------------------
    def plasticity(self, ctx: EventContext, s: dict) -> dict:
        p = self.lay.sector_probabilities(np.diag(ctx.R_prior))
        post = p * bit_probabilities(s["w"], self.cfg.eps, ctx.bit)
        a = post / post.sum()
        w, n = learn(self.cfg, s["w"], s["n"], a, ctx.bit)
        return {"w": w, "n": n}

    def active_port(self, w: np.ndarray) -> Instrument:
        """The output instrument ``J_tau = O_tau``: emission with the machine's own prediction."""
        return Instrument({t: self.observe(w, t) for t in (0, 1)}, label="active port")

    def reading_protocol(self, canalization=None, learning: bool = True) -> EventProtocol:
        ev = self.evolve()
        inj = {b: self.inject(b) for b in (0, 1)}
        back = {t: self.reenter(t) for t in (0, 1)}
        return EventProtocol(
            input_map=lambda b, s: ev.compose(inj[b]).compose(self.observe(s["w"], b)),
            output_instrument=lambda s: self.active_port(s["w"]),
            feedback=lambda tau, s: back[tau],
            canalization=canalization,
            plasticity=self.plasticity if learning else None,
            input_mode="condition",
            label="VAM-1 reading",
        )

    def generation_protocol(self) -> EventProtocol:
        ev = self.evolve()
        reenter = {t: ev.compose(self.inject(t)).compose(self.reenter(t)) for t in (0, 1)}
        return EventProtocol(
            input_map=lambda b, s: FunctionMap.identity(self.cfg.dim),
            output_instrument=lambda s: self.active_port(s["w"]),
            feedback=lambda tau, s: reenter[tau],
            settle=lambda s: FunctionMap.identity(self.cfg.dim),
            input_mode="inject",
            label="VAM-1 generation",
        )

    def canalization(self) -> CanalizationAlgebra:
        """The retained algebra of T11.3 for the learning machine (decision D7).

        The event maps depend on the weights, and the weights move, so the
        algebra must be invariant under the maps at every weight, and it must
        contain what plasticity and observation read: the sector projectors
        ``P_s``. Seeding with the ``P_s`` gives both. Every weight-dependent
        factor is ``O_b(w) = sum_s sqrt(pi_s(b)) P_s``, whose dual
        ``A -> sum_{s,s'} c_{ss'} P_s A P_{s'}`` maps the algebra into itself
        because the ``P_s`` belong to it; the remaining factors (injection,
        evolution, re-entry) do not depend on ``w`` and are closed over
        explicitly. Hence ``Pi K(w) = Pi K(w) Pi`` for every event map at every
        ``w``, and the posterior sector activities that drive plasticity are
        unchanged by ``Pi`` (tested, including at ``w = 0``).
        """
        seeds = [np.diag((self.lay.sector == s).astype(complex)) for s in range(self.cfg.sectors)]
        maps = [self.kraus_inject(0), self.kraus_inject(1), self.kraus_evolve(),
                self.kraus_reenter(0), self.kraus_reenter(1)]
        return invariant_algebra(maps, self.cfg.dim, seeds)

    def initial_state(self) -> MachineState:
        R = np.zeros((self.cfg.dim, self.cfg.dim), dtype=complex)
        R[0, 0] = 1.0
        z = np.zeros(self.cfg.sectors)
        return MachineState(R, {"w": z.copy(), "n": z.copy()})

    def reader(self, **kw) -> Machine:
        return Machine(self.reading_protocol(**kw), self.initial_state())

    def generator(self, state: MachineState) -> Machine:
        return Machine(self.generation_protocol(), state.copy())

    # -- explicit Kraus forms (small registers; used by the tests) --------------
    def kraus_observe(self, w: np.ndarray, b: int) -> KrausMap:
        d = np.sqrt(bit_probabilities(w, self.cfg.eps, b))[self.lay.sector]
        return KrausMap([np.diag(d).astype(complex)], label=f"O_{b}")

    def kraus_inject(self, b: int) -> KrausMap:
        ks = []
        for S in self._exit_sets:
            K = np.zeros((self.cfg.dim, self.cfg.dim), dtype=complex)
            K[self.lay.target[b][S], S] = 1.0
            ks.append(K)
        return KrausMap(ks, label=f"inject_{b}")

    def kraus_reenter(self, tau: int) -> KrausMap:
        ks = []
        for S in self._port_sets:
            K = np.zeros((self.cfg.dim, self.cfg.dim), dtype=complex)
            K[self.lay.write[tau][S], S] = 1.0
            ks.append(K)
        return KrausMap(ks, label=f"reenter_{tau}")

    def kraus_evolve(self) -> KrausMap:
        if self.cfg.einselect == "context":
            projs = [np.diag((self.lay.sector == s).astype(complex)) for s in range(self.cfg.sectors)]
        else:
            projs = [np.diag(np.eye(self.cfg.dim)[i]).astype(complex) for i in range(self.cfg.dim)]
        ks = [np.sqrt(1 - self.keep) * P @ self.U for P in projs]
        if self.keep > 0:
            ks.append(np.sqrt(self.keep) * self.U)
        return KrausMap(ks, label="evolve")


# ----------------------------------------------------------------------------
# Pointer engine: the fully einselected limit on probability vectors
# ----------------------------------------------------------------------------

class PointerEngine:
    """Exact VAM-1 dynamics when every pointer configuration is monitored.

    Equal to :class:`DenseMaps` with ``einselect="all"`` and infinite
    dephasing on any diagonal initial state, emission for emission (tested).
    The coin then acts on configuration probabilities as a stochastic swap
    ``R_j <-> L_j`` with probability ``sin^2 theta`` (the telegraph-process
    limit of the Dirac walk). State: a probability vector over the
    ``2 * 4^c`` configurations. Each event emits a bit at the active port,
    drawn with ``rng`` exactly as :meth:`vam.machine.Machine.step` draws it
    (one uniform number per event), so a dense machine and a pointer engine
    given equal generators emit the same bits. Without ``rng`` the engine
    uses ``default_rng(0)``, so every run is reproducible.
    """

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.lay = Layout.build(cfg)
        s2 = np.sin(cfg.theta) ** 2
        self.cell_stochastic = np.array([[1, 0, 0, 0], [0, 1 - s2, s2, 0], [0, s2, 1 - s2, 0], [0, 0, 0, 1]])
        self.p = np.zeros(cfg.dim)
        self.p[0] = 1.0
        self.w = np.zeros(cfg.sectors)
        self.n = np.zeros(cfg.sectors)
        self.last_output: int | None = None

    def predict(self, b: int) -> float:
        sp = self.lay.sector_probabilities(self.p)
        return float(sp @ bit_probabilities(self.w, self.cfg.eps, b))

    def _coin(self, p: np.ndarray) -> np.ndarray:
        if self.cfg.theta == 0.0:
            return p
        c = self.cfg.cells
        t = p.reshape((4,) * c + (2,))                  # cells, then the active port
        for j in range(c):
            t = np.moveaxis(np.tensordot(self.cell_stochastic, t, axes=([1], [j])), 0, j)
        return t.reshape(-1)

    def _transport(self, p: np.ndarray, b: int) -> np.ndarray:
        return np.bincount(self.lay.target[b], weights=p, minlength=len(p))

    def step(self, b: int, learning: bool = True) -> float:
        """Observe, learn, inject, evolve, emit and re-enter for one external bit; returns ``q_b``."""
        w, eps, sector = self.w, self.cfg.eps, self.lay.sector
        sp = self.lay.sector_probabilities(self.p)
        pib = bit_probabilities(w, eps, b)
        q = float(sp @ pib)
        post = self.p * pib[sector] / q
        p = self._coin(self._transport(post, b))
        # emission at the active port with the event's weights, conditioned once, then re-entry
        sp_out = self.lay.sector_probabilities(p)
        p_tau = np.array([sp_out @ bit_probabilities(w, eps, t) for t in (0, 1)])
        tau = min(int(np.searchsorted(np.cumsum(p_tau / p_tau.sum()), self.rng.random(), side="right")), 1)
        p = p * bit_probabilities(w, eps, tau)[sector] / p_tau[tau]
        self.p = np.bincount(self.lay.write[tau], weights=p, minlength=len(p))
        self.last_output = tau
        if learning:                                    # the slow structural update ends the event
            self.w, self.n = learn(self.cfg, w, self.n, sp * pib / q, b)
        return q
