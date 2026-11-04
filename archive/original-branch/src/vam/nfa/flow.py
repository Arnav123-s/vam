"""Layer 1 of N: bits become a differential equation (N-L1), and its trajectory's basis (N-L2).

N-L1: "combinations of words and bits define continuous vector fields and dynamical trajectories:
dx/dt = f(x, t). A complete sentence represents a deterministic or stochastic flow through an
abstract state space." This module writes every VAM-1 event as such a flow, on the machine's
whole state ``x = (R, w, n)``, and solves it with an ODE integrator (``scipy.integrate.solve_ivp``).
A sentence is then one continuous trajectory: the concatenation of its bits' event flows. The
tests check that the integrated flow equals the machine's event maps.

One event as a time-dependent generator
---------------------------------------
Each stage of the event is a constant generator on a unit time interval (seven kinds of interval,
in the order of ``DenseMaps.reading_protocol``):

1. Observe ``b`` (the sensory blanket). ``K_b = -sum_s ln pi_s(b) P_s`` and the normalized
   no-jump (filter) equation ``dR/dt = -1/2 {K_b, R} + Tr(K_b R) R``. At ``t = 1`` it gives
   ``O_b R O_b / q_b`` exactly. The code length ``dl/dt = Tr(K_b R)`` integrates to
   ``l(1) = -ln q_b``: the loss of the bit is the integral of the observation generator along
   the trajectory, which is the action of layer 7 (``S = -ln p``).
2. Transport (collision, T6.2). Two environment units enter: one holding ``b`` and one empty.
   Nearest-neighbour exchanges ``H = (pi/2)(I - SWAP)``, each exactly a SWAP after unit time,
   move the right-movers one cell right and the left-movers one cell left, in ``c + 1`` slots of
   disjoint (commuting) exchanges. ``R_{c-1}`` and ``L_0`` end in the environment units, which
   leave (partial trace): the bit came in through the sensory side and the oldest content left.
3. Coin: ``dR/dt = -i [H_theta, R]`` with ``H_theta = theta/2 sum_j (X X + Y Y)_{(R_j, L_j)}``.
4. Einselection: ``dR/dt = g (Pi_B(R) - R)`` (jump operators ``sqrt(g) P_s``). For ``g = inf``
   the stage is its limit ``Pi_B``, applied exactly.
5. Emit ``tau`` (the active blanket): the same filter equation with ``K_tau``, and
   ``dl_tau/dt = Tr(K_tau R)`` gives ``-ln p_tau``.
6. Re-enter: an environment unit holding ``tau`` is exchanged with the active port and leaves
   with the port's old content.
7. Plasticity: ``dw_s/dt = -(kappa dt a_s / gamma_s)(w_s - g z_b / kappa)``, ``dn_s/dt = a_s``,
   with ``a_s = Tr(P_s R)`` read at the end of stage 1. This is the T12.1 law in its own
   differential form; its unit-time solution is the machine's update.

Only two things are not flows: environment units entering and leaving (the blanket's boundary
crossing: a tensor product and a partial trace), and the infinite-rate limit of stage 4.

The trajectory's basis (N-L2)
-----------------------------
:meth:`EventFlow.trajectory` records the state along the flow. :func:`trajectory_basis` projects
the recorded trajectory on its singular vectors and keeps those above the register's resolution
(N-L1: a direction smaller than ``2**-52`` of the largest one, times the matrix side, is lost in
a 64-bit register anyway). The trajectory is deterministic given the stream; for quantities
estimated from samples (the source's own predictive vectors) the Page edge of ``vam.nfa.page``
is the cutoff.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

from ..realizations.vam1 import Config, DenseMaps, bit_probabilities

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def embed(op: np.ndarray, qubits: tuple[int, ...], n: int) -> np.ndarray:
    """``op`` acting on ``qubits`` (qubit 0 most significant) of an ``n``-qubit register."""
    k = len(qubits)
    T = op.reshape((2,) * (2 * k))
    I = np.eye(2 ** n, dtype=complex).reshape((2,) * (2 * n))
    # out[rows, cols] = sum over the op's input legs of T[out legs, in legs] * I[in legs replaced]
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rows = list(letters[:n])
    cols = list(letters[n:2 * n])
    new = list(letters[2 * n:2 * n + k])
    ident_rows = rows.copy()
    for i, q in enumerate(qubits):
        ident_rows[q] = new[i]
    spec = "".join(rows[q] for q in qubits) + "".join(new) + "," + "".join(ident_rows + cols) + "->" + \
        "".join(rows + cols)
    return np.einsum(spec, T, I).reshape(2 ** n, 2 ** n)


def exchange(q1: int, q2: int, n: int) -> np.ndarray:
    """``(pi/2)(I - SWAP)`` on two qubits: its unit-time flow is exactly the SWAP."""
    return (np.pi / 2.0) * (np.eye(2 ** n, dtype=complex) - embed(SWAP, (q1, q2), n))


@dataclass
class FlowState:
    R: np.ndarray
    w: np.ndarray
    n: np.ndarray

    def copy(self) -> FlowState:
        return FlowState(self.R.copy(), self.w.copy(), self.n.copy())


@dataclass
class EventResult:
    state: FlowState
    q_b: float
    p_tau: float
    activity: np.ndarray
    record: list = field(default_factory=list)      # (stage, t, flat state) samples when recording


class EventFlow:
    """VAM-1 reading events as integrated differential equations (dense; small chains)."""

    def __init__(self, cfg: Config, rtol: float = 1e-11, atol: float = 1e-13, method: str = "DOP853") -> None:
        if cfg.consolidating and cfg.gamma <= 0:
            raise ValueError("gamma must be positive")
        self.cfg = cfg
        self.maps = DenseMaps(cfg)
        self.lay = self.maps.lay
        self.rtol, self.atol, self.method = rtol, atol, method
        c, nq = cfg.cells, cfg.qubits
        self.nq = nq
        # coin Hamiltonian on the cells (the port qubit 2c is untouched)
        H = np.zeros((2 ** nq, 2 ** nq), dtype=complex)
        for j in range(c):
            H += (cfg.theta / 2.0) * (embed(np.kron(X, X), (2 * j, 2 * j + 1), nq)
                                      + embed(np.kron(Y, Y), (2 * j, 2 * j + 1), nq))
        self.H_coin = H
        # transport: system qubits 0..2c, environment units e_in = 2c+1 (holds b), e_out = 2c+2 (empty)
        ne = nq + 2
        e_in, e_out = nq, nq + 1
        right = [(2 * j, 2 * j - 2) for j in range(c - 1, 0, -1)] + [(0, e_in)]      # R_{c-1} bubbles to e_in
        left = [(2 * j + 1, 2 * j + 3) for j in range(c - 1)] + [(2 * c - 1, 2 * c), (2 * c, e_out)]
        slots = max(len(right), len(left))
        self.transport_slots = []
        for k in range(slots):
            Hk = np.zeros((2 ** ne, 2 ** ne), dtype=complex)
            for pairs in (right, left):
                if k < len(pairs):
                    Hk += exchange(*pairs[k], ne)
            self.transport_slots.append(sparse.csr_matrix(Hk))
        self.H_coin = sparse.csr_matrix(self.H_coin)
        self.H_reenter = sparse.csr_matrix(exchange(2 * c, nq, nq + 1))             # port <-> e_tau
        self.mask = self.maps.mask
        self.g = cfg.dephasing

    # -- integration helpers ---------------------------------------------------
    def _solve(self, rhs, y0: np.ndarray, record: list | None, stage: str, samples: int) -> np.ndarray:
        t_eval = np.linspace(0.0, 1.0, samples + 1) if record is not None else None
        sol = solve_ivp(rhs, (0.0, 1.0), y0, method=self.method, rtol=self.rtol, atol=self.atol, t_eval=t_eval)
        if not sol.success:
            raise RuntimeError(f"integration of stage {stage!r} failed: {sol.message}")
        if record is not None:
            record.extend((stage, float(t), sol.y[:, i].copy()) for i, t in enumerate(sol.t))
        return sol.y[:, -1]

    def _hamiltonian(self, H, R: np.ndarray, record, stage, samples) -> np.ndarray:
        """``dR/dt = -i [H, R]`` for unit time (``H`` sparse: exchanges and the coin are local)."""
        d = R.shape[0]
        HT = H.T.tocsr()

        def rhs(t, y):
            M = y.reshape(d, d)
            return (-1j * (H @ M - (HT @ M.T).T)).ravel()

        return self._solve(rhs, R.ravel().astype(complex), record, stage, samples).reshape(d, d)

    def _filter(self, k: np.ndarray, R: np.ndarray, record, stage, samples) -> tuple[np.ndarray, float]:
        """Normalized no-jump flow with diagonal ``K = diag(k)``; returns ``(R(1), -ln q)``."""
        d = R.shape[0]
        ksum = k[:, None] + k[None, :]

        def rhs(t, y):
            M = y[:-1].reshape(d, d)
            tr = np.real(np.sum(k * np.diag(M)))
            dM = -0.5 * ksum * M + tr * M
            return np.concatenate([dM.ravel(), [tr]])

        y = self._solve(rhs, np.concatenate([R.ravel().astype(complex), [0.0]]), record, stage, samples)
        return y[:-1].reshape(d, d), float(np.real(y[-1]))

    def _dephase(self, R: np.ndarray, record, stage, samples) -> np.ndarray:
        if math.isinf(self.g):
            out = R * self.mask
            if record is not None:
                record.append((stage, 1.0, out.ravel().copy()))
            return out
        d, g, M = R.shape[0], self.g, self.mask

        def rhs(t, y):
            Rm = y.reshape(d, d)
            return (g * (Rm * M - Rm)).ravel()

        return self._solve(rhs, R.ravel().astype(complex), record, stage, samples).reshape(d, d)

    def _k(self, w: np.ndarray, b: int) -> np.ndarray:
        return -np.log(bit_probabilities(w, self.cfg.eps, b))[self.lay.sector]

    # -- the event ---------------------------------------------------------------
    def event(self, x: FlowState, b: int, tau: int, record: bool = False, samples: int = 4) -> EventResult:
        """One reading event with external bit ``b`` and emitted bit ``tau``, as seven flows."""
        cfg, rec = self.cfg, ([] if record else None)
        d = cfg.dim
        # 1. observe
        R, ln_q = self._filter(self._k(x.w, b), x.R, rec, "observe", samples)
        activity = self.lay.sector_probabilities(np.diag(R))
        # 2. transport with two environment units entering: |b> and |0>
        env = np.zeros((4, 4), dtype=complex)
        env[2 * b, 2 * b] = 1.0
        Rx = np.kron(R, env)
        for i, Hk in enumerate(self.transport_slots):
            Rx = self._hamiltonian(Hk, Rx, rec, f"transport{i}", samples)
        R = np.einsum("aibi->ab", Rx.reshape(d, 4, d, 4))                           # the units leave
        # 3. coin, 4. einselection
        R = self._hamiltonian(self.H_coin, R, rec, "coin", samples)
        R = self._dephase(R, rec, "einselect", samples)
        # 5. emit
        R, ln_p = self._filter(self._k(x.w, tau), R, rec, "emit", samples)
        # 6. re-enter: the unit |tau> is exchanged with the port and leaves with its old content
        env1 = np.zeros((2, 2), dtype=complex)
        env1[tau, tau] = 1.0
        Rx = self._hamiltonian(self.H_reenter, np.kron(R, env1), rec, "reenter", samples)
        R = np.einsum("aibi->ab", Rx.reshape(d, 2, d, 2))
        R = 0.5 * (R + R.conj().T)
        # 7. plasticity
        w, n = self._plasticity(x.w, x.n, activity, b, rec, samples)
        return EventResult(FlowState(R, w, n), math.exp(-ln_q), math.exp(-ln_p), activity, rec or [])

    def generation_event(self, x: FlowState, tau: int, record: bool = False, samples: int = 4) -> EventResult:
        """One generation event (``DenseMaps.generation_protocol``): emit ``tau``, re-enter it at the
        active port, inject it at the sensory port (no external stream remains), coin, einselection.
        Plasticity is off: re-entered outputs are not evidence (T10.3)."""
        rec = [] if record else None
        d = self.cfg.dim
        R, ln_p = self._filter(self._k(x.w, tau), x.R, rec, "emit", samples)
        env1 = np.zeros((2, 2), dtype=complex)
        env1[tau, tau] = 1.0
        Rx = self._hamiltonian(self.H_reenter, np.kron(R, env1), rec, "reenter", samples)
        R = np.einsum("aibi->ab", Rx.reshape(d, 2, d, 2))
        env = np.zeros((4, 4), dtype=complex)
        env[2 * tau, 2 * tau] = 1.0
        Rx = np.kron(R, env)
        for i, Hk in enumerate(self.transport_slots):
            Rx = self._hamiltonian(Hk, Rx, rec, f"transport{i}", samples)
        R = np.einsum("aibi->ab", Rx.reshape(d, 4, d, 4))
        R = self._hamiltonian(self.H_coin, R, rec, "coin", samples)
        R = self._dephase(R, rec, "einselect", samples)
        R = 0.5 * (R + R.conj().T)
        return EventResult(FlowState(R, x.w.copy(), x.n.copy()), 1.0, math.exp(-ln_p),
                           np.zeros(self.cfg.sectors), rec or [])

    def _plasticity(self, w, n, a, b, record, samples):
        cfg = self.cfg
        target = (cfg.g if b == 0 else -cfg.g) / cfg.kappa
        gamma_s = cfg.gamma + (cfg.kappa * cfg.dt * n if cfg.consolidating else 0.0)
        rate = cfg.kappa * cfg.dt * a / gamma_s
        m = len(w)

        def rhs(t, y):
            return np.concatenate([-rate * (y[:m] - target), a])

        y = self._solve(rhs, np.concatenate([w, n]).astype(float), record, "plasticity", samples)
        return y[:m].copy(), y[m:].copy()

    # -- a sentence ----------------------------------------------------------------
    def initial(self) -> FlowState:
        R = np.zeros((self.cfg.dim, self.cfg.dim), dtype=complex)
        R[0, 0] = 1.0
        z = np.zeros(self.cfg.sectors)
        return FlowState(R, z.copy(), z.copy())

    def trajectory(self, bits, taus, x: FlowState | None = None, samples: int = 4):
        """Integrate a whole stream; returns the final state, the per-bit losses ``-log2 q_b`` and the
        recorded trajectory as a matrix (one row per sample of ``vec R``) with stage labels."""
        x = self.initial() if x is None else x
        rows, labels, losses = [], [], []
        for b, tau in zip(bits, taus, strict=True):
            res = self.event(x, int(b), int(tau), record=True, samples=samples)
            for stage, t, y in res.record:
                if stage in ("observe", "emit"):
                    y = y[:-1]
                if y.size == self.cfg.dim ** 2:
                    rows.append(y)
                    labels.append((stage, t))
            losses.append(-math.log2(res.q_b))
            x = res.state
        return x, np.array(losses), np.array(rows), labels


def trajectory_basis(Xm: np.ndarray, rtol: float | None = None) -> dict:
    """Project a recorded trajectory (rows) onto its singular vectors and keep the ones a register
    can tell apart from zero: above ``max(shape) * 2**-52 * sigma_max`` (N-L1, N-L2).

    The trajectory is deterministic given the stream, so the register's resolution is its only
    floor here; for estimates from samples the Page edge of ``vam.nfa.page`` applies.
    """
    Xm = np.asarray(Xm)
    U, s, Vh = np.linalg.svd(Xm, full_matrices=False)
    floor = (max(Xm.shape) * np.finfo(float).eps if rtol is None else rtol) * (s[0] if s.size else 0.0)
    r = int(np.sum(s > floor))
    return {"rank": r, "singular_values": s, "floor": floor, "basis": Vh[:r], "coordinates": U[:, :r] * s[:r]}
