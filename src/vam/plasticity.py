"""Forward structural learning (T12, F-Part 3).

* T12.1 (derived, mean-field slow-coordinate approximation): interaction
  energy ``E_int(w, C) = (kappa/2) w^2 - g w C`` gives the vectors the term
  ``-g w C`` and the weight the force ``g C - kappa w``. With friction
  ``gamma`` and a thermal bath,
  ``dw = (g <C> - kappa w)/gamma dt + sqrt(2 k_B T / gamma) dB``.
  For constant ``<C> = c0``:
  ``E w(t) = g c0/kappa + (w0 - g c0/kappa) exp(-kappa t/gamma)``,
  ``Var w(t) = (k_B T / kappa)(1 - exp(-2 kappa t / gamma))``.
* T12.2 (derived): ``F = r Id + (1-r) Pi_L`` gives
  ``F^n = Pi_L + r^n (Id - Pi_L)``.
* T12.3 (derived): ``|p(tau|R) - p(tau|R')| <= D(R, R')`` under one common
  CPTP process; postselection can break it.

Activity gating (chosen realization used by VAM-1, DESIGN section 5): when
the coupling acts only on the support of a sector projector ``P_s``,
``E_int = (kappa/2) <P_s> w^2 - g w <P_s Z>``, whose zero-noise fixed point is
``w* = (g/kappa) <P_s Z> / <P_s>``: a conditional expectation of ``Z`` given
the sector, reached forward with no stored data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrelationWeightLaw:
    """The T12.1 Ornstein-Uhlenbeck law for one slow relational coordinate."""

    g: float = 1.0
    kappa: float = 1.0
    gamma: float = 1.0
    kT: float = 0.0

    def __post_init__(self) -> None:
        if self.kappa <= 0 or self.gamma <= 0 or self.kT < 0:
            raise ValueError("need kappa > 0, gamma > 0, kT >= 0")

    def energy(self, w: float, C: float) -> float:
        return 0.5 * self.kappa * w * w - self.g * w * C

    def force_on_weight(self, w: float, C: float) -> float:
        """``-dE/dw = g C - kappa w``."""
        return self.g * C - self.kappa * w

    def coupling_seen_by_vectors(self, w: float) -> float:
        """Coefficient of ``C`` in the vectors' generator: ``-dE/dC = g w``."""
        return self.g * w

    def target(self, c0: float) -> float:
        return self.g * c0 / self.kappa

    def mean(self, t: float, w0: float, c0: float) -> float:
        mu = self.target(c0)
        return mu + (w0 - mu) * np.exp(-self.kappa * t / self.gamma)

    def variance(self, t: float) -> float:
        return self.kT / self.kappa * (1.0 - np.exp(-2.0 * self.kappa * t / self.gamma))

    def step(self, w: float, c0: float, dt: float, rng: np.random.Generator | None = None) -> float:
        """Exact transition of the OU process over ``dt`` with ``<C> = c0`` held fixed."""
        m = self.mean(dt, w, c0)
        v = self.variance(dt)
        if v == 0.0:
            return float(m)
        if rng is None:
            raise ValueError("a thermal law needs a random generator")
        return float(m + np.sqrt(v) * rng.standard_normal())


@dataclass(frozen=True)
class GatedCorrelationLaw:
    """Activity-gated T12.1 law for sector weights (chosen realization for VAM-1).

    Over a step ``dt`` with sector activity ``a = <P_s>`` and correlation
    ``c = <P_s Z>``, the zero-noise exact solution is
    ``w <- w* + (w - w*) exp(-kappa a dt / gamma)``, ``w* = g c / (kappa a)``.
    Inactive sectors (``a = 0``) do not move.
    """

    g: float = 1.0
    kappa: float = 1.0
    gamma: float = 1.0
    dt: float = 0.05

    def rate(self, activity: np.ndarray) -> np.ndarray:
        """Per-sector relaxation fraction ``1 - exp(-kappa a dt / gamma)``."""
        return 1.0 - np.exp(-self.kappa * np.asarray(activity) * self.dt / self.gamma)

    def update(self, w: np.ndarray, activity: np.ndarray, corr: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        a = np.asarray(activity, dtype=float)
        c = np.asarray(corr, dtype=float)
        target = np.where(a > 0, self.g * c / (self.kappa * np.where(a > 0, a, 1.0)), w)
        return target + (w - target) * np.exp(-self.kappa * a * self.dt / self.gamma)


def consolidation_multiplier(r: float, n: int) -> float:
    """Transient multiplier ``r^n`` of T12.2."""
    return r ** n


def consolidated_state(rho: np.ndarray, Pi_rho: np.ndarray, r: float, n: int) -> np.ndarray:
    """``F^n(rho) = Pi_L(rho) + r^n (rho - Pi_L(rho))``."""
    return Pi_rho + r ** n * (rho - Pi_rho)
