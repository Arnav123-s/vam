"""Density-derived geometry (T3.4, F-Part 7).

Chosen realization (T3.4): ``g_ab(R) = g0 delta_ab + (1/2) Tr(R {A_a, A_b})``
with Hermitian observables ``A_a`` and ``g0 > 0``. For every real ``v``,
``v^T g v = g0 |v|^2 + Tr[R (sum_a v_a A_a)^2] >= g0 |v|^2`` (derived), so the
metric is strictly positive and changes when the density changes.

The two-way loop of F-Part 7 ("density tells vectors how to flow, vectors
tell density where to curve") is realized by
:func:`site_speeds_from_density`: site-local metrics set local Dirac
transport speeds ``v = 1/sqrt(g)`` for
:func:`vam.dirac.variable_speed_dirac_hamiltonian`, and the evolved vectors
change the density that defines the next metric.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .linalg import anticommutator, as_matrix


def density_metric(R: np.ndarray, observables: Sequence[np.ndarray], g0: float) -> np.ndarray:
    if g0 <= 0:
        raise ValueError("g0 must be positive")
    R = as_matrix(R)
    n = len(observables)
    g = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            sym = anticommutator(observables[a], observables[b])
            g[a, b] = (g0 if a == b else 0.0) + 0.5 * float(np.real(np.trace(R @ sym)))
    return g


def site_speeds_from_density(site_populations: np.ndarray, g0: float, coupling: float) -> np.ndarray:
    """One-observable metric per site, ``g_n = g0 + coupling * p_n`` (``p_n`` = local population),
    and speeds ``v_n = 1 / sqrt(g_n)``: more density, slower local transport."""
    p = np.asarray(site_populations, dtype=float)
    g = g0 + coupling * p
    if np.any(g <= 0):
        raise ValueError("metric must stay positive")
    return 1.0 / np.sqrt(g)
