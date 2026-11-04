"""Coherent path sums and environmental path influence (T7, T13.4).

* T7.1 (derived): inserting complete bases between short-time propagators
  gives the exact finite path sum
  ``<f|U_N...U_1|i> = sum_{s_1..s_{N-1}} prod_k <s_{k+1}|U_{k+1}|s_k>``.
* T7.2 (derived): tracing an environment driven by the system's path weights
  each ket/bra path pair by ``F[g, g'] = Tr_E(U_E[g] sigma_E U_E[g']^dagger)``.
* T7.3 (derived): a which-path record with overlap ``d`` gives the dark-port
  probability ``(1 - Re d)/2``.

The enumerations here are exhaustive and exponential on purpose: they are
the "scope every possibility" reference against which compact operator
propagation is checked.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .linalg import as_matrix, dag


def path_amplitudes(unitaries: Sequence[np.ndarray], i: int, f: int) -> list[tuple[tuple[int, ...], complex]]:
    """Every path ``(i, s_1, ..., s_{N-1}, f)`` with its amplitude ``prod_k <s_{k+1}|U_{k+1}|s_k>``."""
    Us = [as_matrix(U) for U in unitaries]
    d = Us[0].shape[0]
    out = []
    for mids in itertools.product(range(d), repeat=len(Us) - 1):
        path = (i, *mids, f)
        amp = 1.0 + 0j
        for k, U in enumerate(Us):
            amp *= U[path[k + 1], path[k]]
            if amp == 0:
                break
        out.append((path, amp))
    return out


def path_sum(unitaries: Sequence[np.ndarray], i: int, f: int) -> complex:
    """``sum`` over all paths (T7.1); equals ``(U_N ... U_1)[f, i]``."""
    return complex(sum(a for _, a in path_amplitudes(unitaries, i, f)))


@dataclass(frozen=True)
class PathDrivenEnvironment:
    """System paths that drive environment propagators (T7.2).

    At step ``k`` the environment receives ``env_step(k, s)`` controlled by the
    system's current basis configuration ``s``, then the system applies
    ``system_steps[k]``.
    """

    system_steps: tuple[np.ndarray, ...]
    env_step: Callable[[int, int], np.ndarray]
    d_sys: int
    d_env: int

    def joint_unitaries(self) -> list[np.ndarray]:
        out = []
        for k, Us in enumerate(self.system_steps):
            ctrl = sum(np.kron(np.outer(np.eye(self.d_sys)[s], np.eye(self.d_sys)[s]), self.env_step(k, s))
                       for s in range(self.d_sys))
            out.append(np.kron(as_matrix(Us), np.eye(self.d_env)) @ ctrl)
        return out

    def env_propagator(self, path: Sequence[int]) -> np.ndarray:
        """``U_E[g]``: product of the environment steps along the system path (first step rightmost)."""
        U = np.eye(self.d_env, dtype=complex)
        for k in range(len(self.system_steps)):
            U = as_matrix(self.env_step(k, path[k])) @ U
        return U

    def system_amplitude(self, path: Sequence[int], psi0: np.ndarray) -> complex:
        amp = complex(psi0[path[0]])
        for k, Us in enumerate(self.system_steps):
            amp *= as_matrix(Us)[path[k + 1], path[k]]
        return amp

    def reduced_density_by_paths(self, psi0: np.ndarray, sigma_E: np.ndarray) -> np.ndarray:
        """``sum_{g,g'} A[g] conj(A[g']) F[g,g'] |end g><end g'|`` (T7.2)."""
        N = len(self.system_steps)
        paths = list(itertools.product(range(self.d_sys), repeat=N + 1))
        amps = {p: self.system_amplitude(p, psi0) for p in paths}
        props = {p: self.env_propagator(p) for p in paths if amps[p] != 0}
        rho = np.zeros((self.d_sys, self.d_sys), dtype=complex)
        for g, Ag in amps.items():
            if Ag == 0:
                continue
            for gp, Agp in amps.items():
                if Agp == 0:
                    continue
                F = influence_functional(props[g], props[gp], sigma_E)
                rho[g[-1], gp[-1]] += Ag * np.conj(Agp) * F
        return rho

    def reduced_density_by_evolution(self, psi0: np.ndarray, sigma_E: np.ndarray) -> np.ndarray:
        """Same quantity from full joint evolution and a partial trace."""
        R = np.kron(np.outer(psi0, np.conj(psi0)), as_matrix(sigma_E))
        for U in self.joint_unitaries():
            R = U @ R @ dag(U)
        return R.reshape(self.d_sys, self.d_env, self.d_sys, self.d_env).trace(axis1=1, axis2=3)


def influence_functional(U_g: np.ndarray, U_gp: np.ndarray, sigma_E: np.ndarray) -> complex:
    """``F[g, g'] = Tr_E(U_E[g] sigma_E U_E[g']^dagger)`` (T7.2)."""
    return complex(np.trace(as_matrix(U_g) @ as_matrix(sigma_E) @ dag(as_matrix(U_gp))))


def dark_port_probability_from_interferometer(d: complex) -> float:
    """Explicit two-path interferometer with environment marker overlap ``<e_0|e_1> = d``.

    Path qubit ``(|0>|e_0> - |1>|e_1>)/sqrt(2)`` (equal paths with opposite
    phases), recombined by a Hadamard. Port 0 then carries
    ``(|e_0> - |e_1>)/2``: it is dark when the record does not distinguish
    the paths (``d = 1``).
    """
    d = complex(d)
    if abs(d) > 1 + 1e-12:
        raise ValueError("overlap must satisfy |d| <= 1")
    e0 = np.array([1, 0], dtype=complex)
    e1 = np.array([d, np.sqrt(max(0.0, 1 - abs(d) ** 2))], dtype=complex)  # <e0|e1> = d
    psi = (np.kron([1, 0], e0) - np.kron([0, 1], e1)) / np.sqrt(2)
    Hd = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    psi = np.kron(Hd, np.eye(2)) @ psi
    amp_dark = psi.reshape(2, 2)[0]
    return float(np.real(np.vdot(amp_dark, amp_dark)))


def dark_port_probability(d: complex) -> float:
    """Closed form ``(1 - Re d) / 2`` (T7.3)."""
    return 0.5 * (1.0 - float(np.real(d)))


def coherent_sum(amplitudes: np.ndarray, actions: np.ndarray, hbar: float = 1.0) -> complex:
    """``A = sum_j a_j exp(i S_j / hbar)``."""
    return complex(np.sum(np.asarray(amplitudes) * np.exp(1j * np.asarray(actions) / hbar)))


def phase_error_bound(amplitudes: np.ndarray, delta: float, hbar: float = 1.0) -> float:
    """``(delta / hbar) sum_j |a_j|`` bounds ``|A~ - A|`` when ``|S~_j - S_j| <= delta`` (T13.4)."""
    return float(delta / hbar * np.sum(np.abs(amplitudes)))
