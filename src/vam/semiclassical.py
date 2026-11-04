"""Feynman-Newton inference at two scales (T8, F-Part 2).

* T8.1 (derived, leading semiclassical order): the Dirac band gives
  ``H_pos(q, p) = V(q) + sqrt(m^2 c^4 + c^2 |p|^2)``, characteristics
  ``qdot = c^2 p / sqrt(m^2 c^4 + c^2 p^2)``, ``pdot = -grad V``, and Newton's
  law ``m qddot = -grad V`` to order ``p^4 / (m^3 c^2)``.
* T8.2 (derived under stated smoothness assumptions): stationary phase.
* T8.4 (derived): composing propagators threads the macro trajectory through
  a local step where outgoing and incoming momenta match.

Newtonian motion and the Feynman description come from the same generator
here: :func:`dirac_packet_evolution` propagates a positive-energy Dirac wave
packet exactly in momentum space, and its centre is compared with the
characteristic (Newtonian) trajectory.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize

from .dirac import dirac_plane_wave_hamiltonian

# ----------------------------------------------------------------------------
# T8.1: Dirac band -> relativistic characteristics -> Newton
# ----------------------------------------------------------------------------

def plane_wave_energies(p: Sequence[float], m: float, c: float = 1.0, V: float = 0.0) -> np.ndarray:
    """Eigenvalues of ``c alpha.p + beta m c^2 + V`` (each doubly degenerate in 3-D)."""
    return np.linalg.eigvalsh(dirac_plane_wave_hamiltonian(p, m, c, V))


def h_pos(p: np.ndarray, m: float, c: float, V: float = 0.0) -> float:
    p = np.atleast_1d(p).astype(float)
    return float(V + np.sqrt(m ** 2 * c ** 4 + c ** 2 * p @ p))


def relativistic_velocity(p: np.ndarray, m: float, c: float) -> np.ndarray:
    p = np.atleast_1d(p).astype(float)
    return c ** 2 * p / np.sqrt(m ** 2 * c ** 4 + c ** 2 * p @ p)


def characteristics(q0: Sequence[float], p0: Sequence[float], grad_V: Callable[[np.ndarray], np.ndarray],
                    m: float, c: float, t_end: float, n_out: int = 101,
                    rtol: float = 1e-12, atol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the positive-band characteristics; returns ``(t, q, p)``."""
    q0 = np.atleast_1d(np.asarray(q0, dtype=float))
    p0 = np.atleast_1d(np.asarray(p0, dtype=float))
    dim = q0.size

    def rhs(_t, y):
        q, p = y[:dim], y[dim:]
        return np.concatenate([relativistic_velocity(p, m, c), -np.atleast_1d(grad_V(q))])

    ts = np.linspace(0.0, t_end, n_out)
    sol = integrate.solve_ivp(rhs, (0.0, t_end), np.concatenate([q0, p0]), method="DOP853",
                              t_eval=ts, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y[:dim].T, sol.y[dim:].T


def newton_trajectory(q0: Sequence[float], v0: Sequence[float], grad_V: Callable[[np.ndarray], np.ndarray],
                      m: float, t_end: float, n_out: int = 101,
                      rtol: float = 1e-12, atol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ``m qddot = -grad V``; returns ``(t, q)``."""
    q0 = np.atleast_1d(np.asarray(q0, dtype=float))
    v0 = np.atleast_1d(np.asarray(v0, dtype=float))
    dim = q0.size

    def rhs(_t, y):
        return np.concatenate([y[dim:], -np.atleast_1d(grad_V(y[:dim])) / m])

    ts = np.linspace(0.0, t_end, n_out)
    sol = integrate.solve_ivp(rhs, (0.0, t_end), np.concatenate([q0, v0]), method="DOP853",
                              t_eval=ts, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y[:dim].T


# ----------------------------------------------------------------------------
# T8.2: stationary phase
# ----------------------------------------------------------------------------

def bump(x: np.ndarray, center: float = 0.0, width: float = 1.0) -> np.ndarray:
    """Smooth compactly supported amplitude ``exp(-1/(1-u^2))`` on ``|u| < 1``."""
    u = (np.asarray(x, dtype=float) - center) / width
    out = np.zeros_like(u)
    inside = np.abs(u) < 1
    out[inside] = np.exp(-1.0 / (1.0 - u[inside] ** 2))
    return out


def oscillatory_integral(amplitude: Callable[[np.ndarray], np.ndarray], phase: Callable[[np.ndarray], np.ndarray],
                         eps: float, lo: float, hi: float, n: int | None = None) -> complex:
    """``int a(x) exp(i S(x)/eps) dx`` over the support ``[lo, hi]`` of a smooth compact ``a``.

    The trapezoid rule converges faster than any power for smooth integrands
    vanishing with all derivatives at the ends, so a grid resolving the
    oscillation by a fixed factor gives near machine precision.
    """
    if n is None:
        n = int(max(20001, 400 * (hi - lo) / eps))
    x = np.linspace(lo, hi, n)
    f = amplitude(x) * np.exp(1j * phase(x) / eps)
    return complex(np.trapezoid(f, x))


def stationary_phase_leading(a_s: float, S_s: float, S2: np.ndarray | float, eps: float) -> complex:
    """``(2 pi eps)^{d/2} a(x_s) / sqrt|det S''| * exp(i S(x_s)/eps + i pi sgn(S'')/4)`` (T8.2)."""
    H = np.atleast_2d(np.asarray(S2, dtype=float))
    d = H.shape[0]
    evals = np.linalg.eigvalsh(H)
    if np.any(np.abs(evals) < 1e-14):
        raise ValueError("stationary point is degenerate")
    sgn = int(np.sum(np.sign(evals)))
    return complex((2 * np.pi * eps) ** (d / 2) * a_s / np.sqrt(abs(np.prod(evals)))
                   * np.exp(1j * S_s / eps + 1j * np.pi * sgn / 4))


# ----------------------------------------------------------------------------
# T8.4: macro-micro threading
# ----------------------------------------------------------------------------

def free_action(x_from: float, x_to: float, t: float, m: float = 1.0) -> float:
    """Classical action ``m (x_to - x_from)^2 / (2 t)`` of free motion."""
    return m * (x_to - x_from) ** 2 / (2 * t)


@dataclass(frozen=True)
class Threading:
    z: float
    first_action: float
    second_action: float
    total_action: float
    hessian: float
    momentum_mismatch: float


def thread_intermediate(S_first: Callable[[float], float], S_second: Callable[[float], float],
                        bracket: tuple[float, float], h: float = 1e-5) -> Threading:
    """Find ``z`` with ``d/dz (S_first(z) + S_second(z)) = 0`` (outgoing = incoming momentum, T8.4)."""
    total = lambda z: S_first(z) + S_second(z)
    deriv = lambda z: (total(z + h) - total(z - h)) / (2 * h)
    z = optimize.brentq(deriv, *bracket, xtol=1e-14, rtol=1e-14)
    hess = (total(z + h) - 2 * total(z) + total(z - h)) / h ** 2
    p_out = (S_first(z + h) - S_first(z - h)) / (2 * h)
    p_in = -(S_second(z + h) - S_second(z - h)) / (2 * h)
    return Threading(z, S_first(z), S_second(z), total(z), hess, p_out - p_in)


# ----------------------------------------------------------------------------
# Wave packets: the Feynman description whose centre follows Newton
# ----------------------------------------------------------------------------

def gaussian_packet_moments(t: float, x0: float, p0: float, sigma0: float, m: float,
                            hbar: float) -> tuple[float, float]:
    """Centre and variance of a free nonrelativistic Gaussian packet."""
    return x0 + p0 * t / m, sigma0 ** 2 + (hbar * t / (2 * m * sigma0)) ** 2


@dataclass(frozen=True)
class PacketResult:
    mean: float
    variance: float
    norm: float


def dirac_packet_evolution(t: float, x0: float, p0: float, sigma0: float, m: float, c: float, hbar: float,
                           L: float, N: int) -> PacketResult:
    """Exact momentum-space evolution of a positive-energy 1-D Dirac packet.

    The initial spinor field is a Gaussian envelope of width ``sigma0`` and
    mean momentum ``p0`` carried by the positive-energy spinor of each mode.
    For ``H(k) = c hbar k X + m c^2 Z`` that spinor is
    ``(cos(phi/2), sin(phi/2))`` with ``phi = atan2(c hbar k, m c^2)``, a
    smooth gauge. Each mode evolves exactly by ``exp(-i E(k) t / hbar)``.
    Position moments are read from ``|psi_1|^2 + |psi_2|^2``.
    """
    if m <= 0:
        raise ValueError("this packet construction needs m > 0")
    x = np.linspace(x0 - L / 2, x0 + L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    envelope = np.exp(-(x - x0) ** 2 / (4 * sigma0 ** 2) + 1j * p0 * x / hbar)
    env_k = np.fft.fft(envelope)
    phi = np.arctan2(c * hbar * k, m * c ** 2)
    E = np.sqrt((c * hbar * k) ** 2 + (m * c ** 2) ** 2)
    # subtract the rest energy: a global phase that does not affect densities
    phase = np.exp(-1j * (E - m * c ** 2) * t / hbar)
    psi_k = np.stack([np.cos(phi / 2), np.sin(phi / 2)], axis=1) * (env_k * phase)[:, None]
    psi_x = np.fft.ifft(psi_k, axis=0)
    dens = np.sum(np.abs(psi_x) ** 2, axis=1)
    norm = float(np.sum(dens) * dx)
    dens = dens / (np.sum(dens) * dx)
    mean = float(np.sum(x * dens) * dx)
    var = float(np.sum((x - mean) ** 2 * dens) * dx)
    return PacketResult(mean, var, norm)
