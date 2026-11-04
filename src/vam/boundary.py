"""Markov boundaries: generator-level screening and statistical blankets (T4, F-Parts 5-6).

Two distinct statements (T4.4 shows they cannot be conflated):

* **Generator-level boundary** (T4.2): no Hamiltonian or dissipative term acts
  on both the interior ``I`` and the exterior ``E``. Then
  ``d<O_I>/dt = (i/hbar) Tr_IB(R_IB [H_IB, O_I kron I_B])``: the exterior
  influences the interior only through ``B``.
* **Statistical blanket** (T4.3): ``I(I:E|B) = S(IB) + S(BE) - S(B) - S(IBE) = 0``.

Boundaries can change with the weight state (T4.5): terms gated by activity
registers ``N_e`` count only on the pointer configurations the current density
still supports.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from .hamiltonian import JointHamiltonian, Term
from .linalg import as_matrix, commutator, von_neumann_entropy
from .space import Space


def accessible_activity_configurations(space: Space, R: np.ndarray, activity: Sequence[str],
                                       atol: float = 1e-12) -> list[dict[str, int]]:
    """Pointer configurations of the activity registers with nonzero probability in ``R``.

    Coherence between configurations does not reduce this set: while the
    boundary interaction has not distinguished them, the union of their
    supports is the retained description (T4.5).
    """
    if not activity:
        return [{}]
    R_act = space.reduce(R, activity)
    names = list(space.sub(activity).names)
    probs = np.real(np.diag(R_act))
    out = []
    for idx, bits in enumerate(itertools.product((0, 1), repeat=len(names))):
        if probs[idx] > atol:
            out.append(dict(zip(names, bits, strict=True)))
    return out


def active_terms(terms: Iterable[Term], config: Mapping[str, int]) -> list[Term]:
    """Terms active in one pointer configuration of the activity registers."""
    return [t for t in terms if t.activity is None or config.get(t.activity, 0) == 1]


def interaction_boundary(terms: Iterable[Term], interior: Iterable[str],
                         configs: Sequence[Mapping[str, int]] = ({},)) -> frozenset[str]:
    """Factors outside ``interior`` that share an active term with it, in any accessible configuration.

    Activity registers gating such terms are included (they are relational
    control factors of the boundary, T4.5).
    """
    I = set(interior)
    terms = list(terms)
    boundary: set[str] = set()
    for cfg in configs:
        for t in active_terms(terms, cfg):
            if t.factors & I:
                boundary |= set(t.factors) - I
    # an inactive term's activity register still controls whether the term can switch on
    for t in terms:
        if t.activity is not None and set(t.support) & I and t.activity not in I:
            boundary.add(t.activity)
    return frozenset(boundary)


def screening_violations(terms: Iterable[Term], interior: Iterable[str], exterior: Iterable[str]) -> list[str]:
    """Labels of terms that act on interior and exterior directly (must be empty for T4.2)."""
    I, E = set(interior), set(exterior)
    return [t.label or str(t.support) for t in terms if (t.factors & I) and (t.factors & E)]


def interior_derivative_via_boundary(H: JointHamiltonian, R: np.ndarray, O_I: np.ndarray,
                                     interior: Sequence[str], boundary: Sequence[str],
                                     hbar: float = 1.0) -> complex:
    """``(i/hbar) Tr_IB(R_IB [H_IB, O_I kron I_B])`` from the terms supported in ``I u B`` (T4.2)."""
    space = H.space
    IB = [n for n in space.names if n in set(interior) | set(boundary)]
    sub = space.sub(IB)
    H_IB = np.zeros((sub.dim, sub.dim), dtype=complex)
    for t in H.terms:
        if t.factors <= set(IB):
            H_IB += t.matrix(sub)
    R_IB = space.reduce(R, IB)
    O = sub.embed(O_I, [n for n in sub.names if n in set(interior)])
    return complex((1j / hbar) * np.trace(R_IB @ commutator(H_IB, O)))


def conditional_mutual_information(R: np.ndarray, space: Space, I: Sequence[str], B: Sequence[str],
                                   E: Sequence[str], base: float = 2.0) -> float:
    """``I(I:E|B) = S(IB) + S(BE) - S(B) - S(IBE)`` in bits (T4.3, [R3])."""
    def S(names: Sequence[str]) -> float:
        if not names:
            return 0.0
        return von_neumann_entropy(space.reduce(R, list(names)), base=base)
    I, B, E = list(I), list(B), list(E)
    return S(I + B) + S(B + E) - S(B) - S(I + B + E)


def pointer_blanket_state(probs: Sequence[float], interiors: Sequence[np.ndarray],
                          exteriors: Sequence[np.ndarray]) -> np.ndarray:
    """``sum_z p_z R_I^(z) kron |z><z|_B kron R_E^(z)`` (T4.3); order I, B, E."""
    nz = len(probs)
    out = 0
    for z, (p, RI, RE) in enumerate(zip(probs, interiors, exteriors, strict=True)):
        Pz = np.zeros((nz, nz), dtype=complex)
        Pz[z, z] = 1.0
        out = out + p * np.kron(np.kron(as_matrix(RI), Pz), as_matrix(RE))
    return out


def classical_permutation(n_bits: int, fn) -> np.ndarray:
    """Permutation matrix of a reversible map on ``n_bits`` classical bits (MSB = bit 0)."""
    D = 2 ** n_bits
    P = np.zeros((D, D), dtype=complex)
    for idx in range(D):
        bits = [(idx >> (n_bits - 1 - k)) & 1 for k in range(n_bits)]
        out = fn(list(bits))
        jdx = sum(b << (n_bits - 1 - k) for k, b in enumerate(out))
        P[jdx, idx] = 1.0
    return P
