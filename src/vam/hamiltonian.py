"""Joint vector-weight-boundary Hamiltonians (T3.3, T4.2, T4.5, T6.2).

A Hamiltonian is kept as a list of local :class:`Term` objects, each knowing
which named factors it acts on. That support information is what the
boundary analysis of T4.2 and T4.5 reads: which factors directly influence
which others is a property of the actual interaction terms.

Reciprocity (T3.3, derived result): with

    H = H_D0 kron I + I kron H_W + sum_a D_a kron W_a

the Heisenberg equations of a vector observable ``A`` and a weight
observable ``B`` both contain the same coupling terms; no separate reverse
update is needed. :func:`heisenberg_derivative` evaluates
``d<A>/dt = (i/hbar) Tr(R [H, A])`` directly, and the tests compare it with
the two expanded formulas of T3.3.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .linalg import ATOL, P1, as_matrix, commutator, dag
from .space import Space


@dataclass(frozen=True)
class Term:
    """A local Hermitian term acting on the named factors ``support`` (in that order)."""

    support: tuple[str, ...]
    op: np.ndarray
    label: str = ""
    #: Optional activity register (T4.5): the term is ``N_e kron h_e`` with
    #: ``N_e = |1><1|`` on this factor. ``support`` and ``op`` describe ``h_e`` only.
    activity: str | None = None

    def __post_init__(self) -> None:
        op = as_matrix(self.op)
        if not np.allclose(op, dag(op), atol=ATOL):
            raise ValueError(f"term {self.label!r} is not Hermitian")
        object.__setattr__(self, "op", op)
        if isinstance(self.support, str):
            object.__setattr__(self, "support", (self.support,))

    @property
    def factors(self) -> frozenset[str]:
        """Every factor the full term touches, including its activity register."""
        s = set(self.support)
        if self.activity is not None:
            s.add(self.activity)
        return frozenset(s)

    def matrix(self, space: Space) -> np.ndarray:
        h = space.embed(self.op, self.support)
        if self.activity is None:
            return h
        return space.embed(P1, self.activity) @ h


@dataclass
class JointHamiltonian:
    """``H = sum_e term_e`` on a named space."""

    space: Space
    terms: list[Term] = field(default_factory=list)

    def add(self, support: str | Sequence[str], op: np.ndarray, label: str = "",
            activity: str | None = None) -> JointHamiltonian:
        support = (support,) if isinstance(support, str) else tuple(support)
        self.terms.append(Term(support, op, label, activity))
        return self

    def matrix(self) -> np.ndarray:
        D = self.space.dim
        H = np.zeros((D, D), dtype=complex)
        for t in self.terms:
            H += t.matrix(self.space)
        return H

    def restricted(self, labels: Iterable[str]) -> JointHamiltonian:
        keep = set(labels)
        return JointHamiltonian(self.space, [t for t in self.terms if t.label in keep])


def reciprocal_hamiltonian(space: Space, vector: str, weight: str, H_D0: np.ndarray, H_W: np.ndarray,
                           couplings: Sequence[tuple[np.ndarray, np.ndarray]]) -> JointHamiltonian:
    """``H_D0 kron I + I kron H_W + sum_a D_a kron W_a`` (T3.3)."""
    H = JointHamiltonian(space)
    H.add(vector, H_D0, "H_D0")
    H.add(weight, H_W, "H_W")
    for a, (Da, Wa) in enumerate(couplings):
        H.add((vector, weight), np.kron(as_matrix(Da), as_matrix(Wa)), f"D{a}W{a}")
    return H


def heisenberg_derivative(H: np.ndarray, A: np.ndarray, R: np.ndarray, hbar: float = 1.0) -> complex:
    """``d<A>/dt = (i/hbar) Tr(R [H, A])`` for unitary evolution ``dR/dt = -(i/hbar)[H, R]``."""
    return complex((1j / hbar) * np.trace(as_matrix(R) @ commutator(as_matrix(H), as_matrix(A))))


def mean_field_vector_hamiltonian(H_D0: np.ndarray, couplings: Sequence[tuple[np.ndarray, np.ndarray]],
                                  R_W: np.ndarray) -> np.ndarray:
    """``H_D0 + sum_a <W_a> D_a``: exact at an instant only for a product state (T3.3)."""
    out = as_matrix(H_D0).copy()
    for Da, Wa in couplings:
        out = out + np.real(np.trace(as_matrix(R_W) @ as_matrix(Wa))) * as_matrix(Da)
    return out
