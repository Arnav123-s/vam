"""Named tensor-product spaces.

The retained organization R of T1.2 lives on a tensor product of named
factors: information-bearing vectors ``V``, relational controls ``W``,
boundary degrees of freedom ``B`` and, when it is kept explicitly, an
environment ``E``. A :class:`Space` records the names and dimensions in a
fixed canonical order, embeds local operators, and takes reduced states by
name, so no caller has to track factor positions by hand.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .linalg import LinalgError, as_matrix, ket, kron, partial_trace, permute_operator, permute_vector


@dataclass(frozen=True)
class Space:
    """An ordered tensor product of named finite-dimensional factors."""

    names: tuple[str, ...]
    dims: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.dims):
            raise LinalgError("names and dims differ in length")
        if len(set(self.names)) != len(self.names):
            raise LinalgError(f"duplicate factor names in {self.names}")
        if any(d < 1 for d in self.dims):
            raise LinalgError(f"dimensions must be positive: {self.dims}")

    @classmethod
    def of(cls, **factors: int) -> Space:
        """``Space.of(Q=2, W=2)``; keyword order is the canonical order."""
        return cls(tuple(factors), tuple(factors.values()))

    @classmethod
    def qubits(cls, *names: str) -> Space:
        return cls(tuple(names), tuple(2 for _ in names))

    # -- basic queries -----------------------------------------------------
    @property
    def dim(self) -> int:
        return int(np.prod(self.dims))

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise LinalgError(f"no factor named {name!r} in {self.names}") from exc

    def dim_of(self, names: Iterable[str]) -> int:
        return int(np.prod([self.dims[self.index(n)] for n in names]))

    def sub(self, names: Sequence[str]) -> Space:
        """The subspace on ``names``, kept in canonical order."""
        idx = sorted(self.index(n) for n in names)
        return Space(tuple(self.names[i] for i in idx), tuple(self.dims[i] for i in idx))

    def __add__(self, other: Space) -> Space:
        return Space(self.names + other.names, self.dims + other.dims)

    # -- operators -----------------------------------------------------------
    def embed(self, op: np.ndarray, on: str | Sequence[str]) -> np.ndarray:
        """Full-space operator acting as ``op`` on factors ``on`` (in the order given)."""
        on = (on,) if isinstance(on, str) else tuple(on)
        if len(set(on)) != len(on):
            raise LinalgError(f"repeated factor in {on}")
        op = as_matrix(op)
        d_on = self.dim_of(on)
        if op.shape != (d_on, d_on):
            raise LinalgError(f"operator shape {op.shape} does not match factors {on} (dim {d_on})")
        rest = [n for n in self.names if n not in on]
        order = list(on) + rest
        full = kron(op, np.eye(self.dim_of(rest), dtype=complex))
        order_dims = [self.dims[self.index(n)] for n in order]
        # position i of the canonical order is factor order.index(name_i)
        perm = [order.index(n) for n in self.names]
        return permute_operator(full, order_dims, perm)

    def product(self, local: Mapping[str, np.ndarray]) -> np.ndarray:
        """Product of local operators, identity on unlisted factors."""
        mats = []
        for n, d in zip(self.names, self.dims, strict=True):
            mats.append(as_matrix(local[n]) if n in local else np.eye(d, dtype=complex))
        return kron(*mats)

    # -- states -------------------------------------------------------------
    def basis_ket(self, **values: int) -> np.ndarray:
        """Computational basis vector; unlisted factors are in ``|0>``."""
        unknown = set(values) - set(self.names)
        if unknown:
            raise LinalgError(f"unknown factors {unknown}")
        return kron(*[ket(values.get(n, 0), d)[:, None] for n, d in zip(self.names, self.dims, strict=True)])[:, 0]

    def product_density(self, local: Mapping[str, np.ndarray]) -> np.ndarray:
        """Tensor product of the given local densities (all factors required)."""
        missing = set(self.names) - set(local)
        if missing:
            raise LinalgError(f"missing local states for {missing}")
        return kron(*[as_matrix(local[n]) for n in self.names])

    def reduce(self, rho: np.ndarray, keep: str | Sequence[str]) -> np.ndarray:
        """Reduced density on ``keep`` (returned in canonical order of ``keep``)."""
        keep = (keep,) if isinstance(keep, str) else tuple(keep)
        return partial_trace(rho, self.dims, [self.index(n) for n in keep])

    def reorder_vector(self, vec: np.ndarray, order: Sequence[str]) -> np.ndarray:
        """State vector in factor order ``order`` instead of the canonical order."""
        return permute_vector(vec, self.dims, [self.index(n) for n in order])

    def reorder_operator(self, op: np.ndarray, order: Sequence[str]) -> np.ndarray:
        return permute_operator(op, self.dims, [self.index(n) for n in order])
