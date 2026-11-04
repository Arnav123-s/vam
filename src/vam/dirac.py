"""Dirac generators: algebra, lattice discretizations, exact light cone, checkerboard paths.

Source statements: F-Part 7 (Dirac causal light cones, Dirac inside Zurek);
T6.1 (``H_D = -i hbar c alpha d_q + beta m c^2 + V``, Clifford relations);
T13.3 (continuous-time lattice Hamiltonians have tails beyond a nominal cone,
shift-and-rotate discretizations have an exact finite event cone).

Three discretizations are provided because T13.3 contrasts them
(decision D4):

* :func:`wilson_dirac_hamiltonian` - continuous-time, first-quantized, with a
  Wilson term that lifts the fermion doublers.
* :func:`staggered_chain_hamiltonian` - second-quantized staggered fermions on
  a qubit chain (Jordan-Wigner), the field used by VAM-1.
* :func:`split_step_walk` - discrete-time shift-and-coin walk with an exact
  event cone; its propagator equals Feynman's checkerboard sum over bit strings
  of chirality choices (:func:`checkerboard_propagator`).

Positions are the first tensor factor and spinor components the second.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

from .linalg import I2, X, Y, Z, as_matrix, dag, kron, unitary_from_hamiltonian

# ----------------------------------------------------------------------------
# Clifford algebra
# ----------------------------------------------------------------------------

def clifford_generators(n: int) -> list[np.ndarray]:
    """The ``2n + 1`` mutually anticommuting Hermitian involutions on ``n`` qubits.

    Jordan-Wigner construction: ``Z..Z X I..I`` and ``Z..Z Y I..I`` for each
    position, then ``Z..Z``.
    """
    gens = []
    for k in range(n):
        for P in (X, Y):
            gens.append(kron(*([Z] * k + [P] + [I2] * (n - k - 1))))
    gens.append(kron(*([Z] * n)))
    return gens


def dirac_matrices(spatial_dim: int) -> tuple[list[np.ndarray], np.ndarray]:
    """``(alphas, beta)`` with ``alpha_i alpha_j + alpha_j alpha_i = 2 delta_ij``,
    ``alpha_i beta + beta alpha_i = 0``, ``beta^2 = I`` (T6.1).

    Uses ``n = ceil(d/2)`` qubits (spinor dimension ``2^n``): ``alpha = X``,
    ``beta = Z`` in one dimension; 2x2 in two dimensions; 4x4 in three.
    """
    n = max(1, int(np.ceil(spatial_dim / 2)))
    gens = clifford_generators(n)
    return gens[:spatial_dim], gens[-1]


def clifford_defect(alphas: Sequence[np.ndarray], beta: np.ndarray) -> float:
    """Largest violation of the Clifford relations (0 when they hold exactly)."""
    mats = [*list(alphas), beta]
    d = mats[0].shape[0]
    I = np.eye(d)
    worst = 0.0
    for i, A in enumerate(mats):
        worst = max(worst, float(np.max(np.abs(A - dag(A)))))
        for j, B in enumerate(mats):
            target = 2 * I if i == j else 0 * I
            worst = max(worst, float(np.max(np.abs(A @ B + B @ A - target))))
    return worst


def dirac_plane_wave_hamiltonian(p: Sequence[float], m: float, c: float = 1.0, V: float = 0.0) -> np.ndarray:
    """``c alpha . p + beta m c^2 + V`` for a plane wave of momentum ``p`` (hbar = 1)."""
    alphas, beta = dirac_matrices(len(p))
    H = sum(c * pi * a for pi, a in zip(p, alphas, strict=True)) + m * c ** 2 * beta
    return H + V * np.eye(beta.shape[0])


# ----------------------------------------------------------------------------
# First-quantized lattice Dirac operator (continuous time)
# ----------------------------------------------------------------------------

def _hop(N: int, periodic: bool) -> np.ndarray:
    """``S |n> = |n+1>`` (shift right by one site)."""
    S = np.zeros((N, N), dtype=complex)
    for n in range(N - 1):
        S[n + 1, n] = 1.0
    if periodic:
        S[0, N - 1] = 1.0
    return S


def wilson_dirac_hamiltonian(N: int, a: float = 1.0, c: float = 1.0, m: float = 0.0,
                             potential: Sequence[float] | None = None, r: float = 1.0,
                             periodic: bool = True, hbar: float = 1.0) -> np.ndarray:
    """1-D lattice Dirac Hamiltonian with central difference and Wilson term.

    ``H = -i hbar c (D kron alpha) + (r hbar c a / 2)(-Lap kron beta) + m c^2 (I kron beta) + V kron I``

    with ``(D psi)_n = (psi_{n+1} - psi_{n-1}) / 2a`` and
    ``(Lap psi)_n = (psi_{n+1} - 2 psi_n + psi_{n-1}) / a^2``. ``alpha = X``,
    ``beta = Z``. Periodic dispersion:
    ``E(k)^2 = (hbar c sin(ka)/a)^2 + (m c^2 + (r hbar c/a)(1 - cos ka))^2``.
    """
    S = _hop(N, periodic)
    D = (dag(S) - S) / (2 * a)          # (D psi)_n = (psi_{n+1} - psi_{n-1}) / 2a
    Lap = (S + dag(S) - 2 * np.eye(N)) / a ** 2
    alpha, beta = X, Z
    H = -1j * hbar * c * np.kron(D, alpha)
    H += -(r * hbar * c * a / 2) * np.kron(Lap, beta)
    H += m * c ** 2 * np.kron(np.eye(N), beta)
    if potential is not None:
        H += np.kron(np.diag(np.asarray(potential, dtype=complex)), I2)
    return H


def wilson_dispersion(k: np.ndarray, a: float = 1.0, c: float = 1.0, m: float = 0.0, r: float = 1.0,
                      hbar: float = 1.0) -> np.ndarray:
    """Positive branch of the Wilson lattice dispersion."""
    k = np.asarray(k, dtype=float)
    kin = hbar * c * np.sin(k * a) / a
    mass = m * c ** 2 + (r * hbar * c / a) * (1 - np.cos(k * a))
    return np.sqrt(kin ** 2 + mass ** 2)


# ----------------------------------------------------------------------------
# Second-quantized staggered field on a qubit chain (VAM-1 field)
# ----------------------------------------------------------------------------

def staggered_chain_terms(n: int, hop: float = 1.0, mass: float = 0.0) -> list[tuple[tuple[int, ...], np.ndarray]]:
    """Local terms of ``H = hop sum_j (X_j X_{j+1} + Y_j Y_{j+1}) / 2 + (mass/2) sum_j (-1)^j Z_j``.

    By the Jordan-Wigner map, the hopping is nearest-neighbour fermion
    hopping with amplitude ``hop`` (no strings for neighbours in 1-D) and the
    staggered term is the Kogut-Susskind mass. In the one-excitation sector,
    with a two-site unit cell, the energies are
    ``mass_shift +- sqrt((mass)^2 + (2 hop cos k)^2)``: a lattice Dirac
    dispersion with speed ``2 hop`` near ``k = pi/2`` and rest energy ``mass``.

    Returns a list of ``(sites, local operator)`` pairs.
    """
    terms = []
    hopping = 0.5 * hop * (np.kron(X, X) + np.kron(Y, Y))
    for j in range(n - 1):
        terms.append(((j, j + 1), hopping))
    if mass != 0.0:
        for j in range(n):
            terms.append(((j,), 0.5 * mass * (-1) ** j * Z))
    return terms


def chain_operator(n: int, terms: Sequence[tuple[tuple[int, ...], np.ndarray]]) -> np.ndarray:
    """Dense ``2^n x 2^n`` operator from local terms on contiguous sites."""
    D = 2 ** n
    H = np.zeros((D, D), dtype=complex)
    for sites, op in terms:
        lo, hi = min(sites), max(sites)
        if list(sites) != list(range(lo, hi + 1)):
            raise ValueError("terms must act on contiguous increasing sites")
        H += kron(np.eye(2 ** lo), op, np.eye(2 ** (n - hi - 1)))
    return H


def staggered_chain_hamiltonian(n: int, hop: float = 1.0, mass: float = 0.0) -> np.ndarray:
    return chain_operator(n, staggered_chain_terms(n, hop, mass))


def one_excitation_block(H: np.ndarray, n: int) -> np.ndarray:
    """Restriction of a number-conserving chain operator to one excitation.

    Basis ``|j>`` = qubit ``j`` in ``|1>``, others in ``|0>`` (qubit 0 is the
    most significant bit of the dense index).
    """
    idx = [1 << (n - 1 - j) for j in range(n)]
    return as_matrix(H)[np.ix_(idx, idx)]


# ----------------------------------------------------------------------------
# Discrete-time split-step walk and the checkerboard path sum
# ----------------------------------------------------------------------------

def coin(theta: float) -> np.ndarray:
    """Mass coin ``exp(-i theta X)``: keeps chirality with amplitude ``cos theta``,
    flips it with amplitude ``-i sin theta``."""
    return np.cos(theta) * I2 - 1j * np.sin(theta) * X


def shift(N: int, periodic: bool = True) -> np.ndarray:
    """Chirality-conditioned shift: component 0 (right-mover) to ``n+1``, component 1 to ``n-1``."""
    S = _hop(N, periodic)
    return np.kron(S, np.diag([1, 0])) + np.kron(dag(S), np.diag([0, 1]))


def split_step_walk(N: int, theta: float, periodic: bool = True) -> np.ndarray:
    """One step ``W = S (I kron C(theta))``: coin, then shift.

    Dispersion (periodic, infinite limit): eigenphases ``omega`` with
    ``cos omega = cos theta cos k``; for small ``k`` and ``theta``,
    ``omega^2 ~ k^2 + theta^2``, the Dirac relation with ``c = 1`` and rest
    energy ``theta`` per step.
    """
    return shift(N, periodic) @ np.kron(np.eye(N), coin(theta))


def walk_dispersion(k: np.ndarray, theta: float) -> np.ndarray:
    return np.arccos(np.cos(theta) * np.cos(np.asarray(k, dtype=float)))


def checkerboard_propagator(steps: int, theta: float, x0: int, chi0: int) -> dict[tuple[int, int], complex]:
    """Feynman checkerboard: sum over all ``2^steps`` chirality bit strings.

    A path is the bit string ``chi_1 ... chi_steps`` of chiralities chosen by
    the coin at each step. Its amplitude is
    ``prod_k C(theta)[chi_k, chi_{k-1}]`` and it ends at
    ``x0 + sum_k (+1 if chi_k == 0 else -1)``. Returns
    ``{(x, chi): amplitude}`` for the infinite line.
    """
    C = coin(theta)
    out: dict[tuple[int, int], complex] = {}
    for bits in itertools.product((0, 1), repeat=steps):
        amp = 1.0 + 0j
        prev = chi0
        x = x0
        for b in bits:
            amp *= C[b, prev]
            x += 1 if b == 0 else -1
            prev = b
        key = (x, prev)
        out[key] = out.get(key, 0.0) + amp
    return out


def light_cone_leakage(U: np.ndarray, N: int, spin: int, x0: int, steps: int, cone: int | None = None) -> float:
    """Total probability found outside ``|x - x0| <= cone`` after ``steps`` applications.

    ``cone`` defaults to ``steps`` (one site per step). Exactly zero for the
    split-step walk; generally nonzero for a continuous-time lattice
    Hamiltonian evolved for a comparable time (T13.3).
    """
    cone = steps if cone is None else cone
    psi = np.zeros(N * spin, dtype=complex)
    psi[x0 * spin] = 1.0
    for _ in range(steps):
        psi = U @ psi
    prob = np.abs(psi.reshape(N, spin)) ** 2
    dist = np.abs(np.arange(N) - x0)
    dist = np.minimum(dist, N - dist)
    return float(np.sum(prob[dist > cone]))


def hamiltonian_light_cone_leakage(H: np.ndarray, N: int, spin: int, x0: int, t: float, cone: int) -> float:
    """Same leakage measure for ``exp(-iHt)`` applied once."""
    U = unitary_from_hamiltonian(H, t)
    return light_cone_leakage(U, N, spin, x0, 1, cone)


# ----------------------------------------------------------------------------
# Position-dependent transport (density-curved geometry, T3.4, F-Part 7)
# ----------------------------------------------------------------------------

def variable_speed_dirac_hamiltonian(speeds: Sequence[float], a: float = 1.0, m: float = 0.0,
                                     periodic: bool = True) -> np.ndarray:
    """Hermitian ``-(i/2){v(q), d_q} alpha + beta m`` with site speeds ``v_n``.

    Symmetric ordering keeps the operator Hermitian when ``v`` varies; it is a
    chosen realization of "density tells vectors how to flow" with
    ``v = 1/sqrt(g)`` for a metric ``g`` (no vielbein or spin connection is
    needed in this 1-D, static, flat-time construction).
    """
    v = np.asarray(speeds, dtype=float)
    N = len(v)
    S = _hop(N, periodic)
    D = (dag(S) - S) / (2 * a)
    Vd = np.diag(v)
    sym = 0.5 * (Vd @ D + D @ Vd)
    return -1j * np.kron(sym, X) + m * np.kron(np.eye(N), Z)
