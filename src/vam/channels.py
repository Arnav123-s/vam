"""Completely positive maps, instruments and environment channels.

Source and derived content
--------------------------
* T3.2: a bit-derived operator acts by congruence ``X R X^dagger``; products
  of densities are not a valid mixing rule.
* T6.2: a system-environment unitary with a fresh environment unit gives the
  Kraus operators ``K_sr = sqrt(q_r) <s|U|r>`` (collision model).
* T10.1: an instrument is a family of CP maps whose sum preserves trace.

Representations
---------------
A :class:`KrausMap` stores Kraus operators ``K_a`` of shape ``(d_out, d_in)``
and acts as ``R -> sum_a K_a R K_a^dagger``. Its Liouville (superoperator)
matrix uses column stacking, ``vec(K R K^dagger) = (conj(K) kron K) vec(R)``,
and its Choi matrix is ``sum_ij |i><j| kron Phi(|i><j|)``.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .linalg import ATOL, LinalgError, as_matrix, dag, hermitian_part, is_psd, unvec, vec


class ChannelError(LinalgError):
    """A map violates a required structural property."""


class KrausMap:
    """The completely positive map ``R -> sum_a K_a R K_a^dagger``."""

    __slots__ = ("d_in", "d_out", "kraus", "label")

    def __init__(self, kraus: Iterable[np.ndarray], label: str = "") -> None:
        ks = [as_matrix(k) for k in kraus]
        if not ks:
            raise ChannelError("a Kraus map needs at least one operator")
        shape = ks[0].shape
        if any(k.shape != shape for k in ks):
            raise ChannelError("Kraus operators have inconsistent shapes")
        self.kraus: tuple[np.ndarray, ...] = tuple(ks)
        self.d_out, self.d_in = shape
        self.label = label

    # -- action ---------------------------------------------------------------
    def __call__(self, R: np.ndarray) -> np.ndarray:
        R = as_matrix(R)
        if R.shape != (self.d_in, self.d_in):
            raise ChannelError(f"{self.label or 'map'}: input shape {R.shape}, expected {(self.d_in, self.d_in)}")
        out = np.zeros((self.d_out, self.d_out), dtype=complex)
        for K in self.kraus:
            out += K @ R @ dag(K)
        return out

    def dual(self, A: np.ndarray) -> np.ndarray:
        """Heisenberg-picture map ``A -> sum_a K_a^dagger A K_a``."""
        A = as_matrix(A)
        out = np.zeros((self.d_in, self.d_in), dtype=complex)
        for K in self.kraus:
            out += dag(K) @ A @ K
        return out

    def effect(self) -> np.ndarray:
        """``sum_a K_a^dagger K_a``; the identity exactly when the map preserves trace."""
        return self.dual(np.eye(self.d_out, dtype=complex))

    def probability(self, R: np.ndarray) -> float:
        """``Tr Phi(R) = Tr(effect R)``."""
        return float(np.real(np.trace(self.effect() @ as_matrix(R))))

    # -- structure ------------------------------------------------------------
    def compose(self, first: KrausMap | FunctionMap) -> KrausMap | FunctionMap:
        """``self o first`` (apply ``first``, then ``self``)."""
        if first.d_out != self.d_in:
            raise ChannelError("dimension mismatch in composition")
        if not isinstance(first, KrausMap):
            return FunctionMap.from_kraus(self).compose(first)
        ks = [K @ L for K in self.kraus for L in first.kraus]
        out = KrausMap(ks, label=f"{self.label}∘{first.label}")
        return out.compressed() if len(ks) > out.d_in * out.d_out else out

    def __matmul__(self, first: KrausMap) -> KrausMap:
        return self.compose(first)

    def tensor(self, other: KrausMap) -> KrausMap:
        """``self kron other`` acting on a two-factor input."""
        return KrausMap([np.kron(K, L) for K in self.kraus for L in other.kraus],
                        label=f"{self.label}⊗{other.label}")

    def scaled(self, c: float) -> KrausMap:
        """``c Phi`` for ``c >= 0`` (keeps complete positivity)."""
        if c < 0:
            raise ChannelError("a negative multiple is not completely positive")
        s = np.sqrt(c)
        return KrausMap([s * K for K in self.kraus], label=f"{c}·{self.label}")

    def superop(self) -> np.ndarray:
        """Liouville matrix ``sum_a conj(K_a) kron K_a`` (column-stacking convention)."""
        return sum(np.kron(K.conj(), K) for K in self.kraus)

    def choi(self) -> np.ndarray:
        """Choi matrix ``sum_ij |i><j| kron Phi(|i><j|)`` of shape ``(d_in d_out, d_in d_out)``."""
        C = np.zeros((self.d_in * self.d_out, self.d_in * self.d_out), dtype=complex)
        for K in self.kraus:
            # sum_ij |i><j| kron K|i><j|K^dagger = |K>><<K| with |K>> = sum_i |i> kron K|i>
            v = np.concatenate([K[:, i] for i in range(self.d_in)])
            C += np.outer(v, v.conj())
        return C

    def compressed(self, atol: float = 1e-12) -> KrausMap:
        """Equivalent map with the minimal number of Kraus operators (from the Choi spectrum)."""
        return choi_to_kraus(self.choi(), self.d_in, self.d_out, atol=atol, label=self.label)

    def is_trace_preserving(self, atol: float = ATOL) -> bool:
        return np.allclose(self.effect(), np.eye(self.d_in), atol=atol, rtol=0)

    def is_trace_nonincreasing(self, atol: float = ATOL) -> bool:
        return is_psd(np.eye(self.d_in) - self.effect(), atol)

    def require_trace_preserving(self, atol: float = ATOL) -> KrausMap:
        dev = float(np.max(np.abs(self.effect() - np.eye(self.d_in))))
        if dev > atol:
            raise ChannelError(f"{self.label or 'map'} is not trace preserving (deviation {dev:.3e})")
        return self

    def __repr__(self) -> str:
        return f"KrausMap({self.label!r}, {len(self.kraus)} ops, {self.d_in}->{self.d_out})"


# ----------------------------------------------------------------------------
# Constructors
# ----------------------------------------------------------------------------

def identity_channel(dim: int) -> KrausMap:
    return KrausMap([np.eye(dim, dtype=complex)], label="id")


def unitary_channel(U: np.ndarray, label: str = "U") -> KrausMap:
    return KrausMap([as_matrix(U)], label=label)


def congruence(Xu: np.ndarray, label: str = "X") -> KrausMap:
    """Single-operator CP map ``R -> X R X^dagger`` (T3.2)."""
    return KrausMap([as_matrix(Xu)], label=label)


def projective_dephasing(projectors: Sequence[np.ndarray], label: str = "Pi_B") -> KrausMap:
    """``R -> sum_r P_r R P_r`` (T7.3); requires a complete orthogonal family."""
    ps = [as_matrix(P) for P in projectors]
    total = sum(ps)
    if not np.allclose(total, np.eye(total.shape[0]), atol=ATOL):
        raise ChannelError("projectors do not sum to the identity")
    return KrausMap(ps, label=label)


def collision_channel(U: np.ndarray, sigma_E: np.ndarray, d_sys: int, label: str = "collision") -> KrausMap:
    """Reduced channel ``R -> Tr_E[U (R kron sigma_E) U^dagger]`` (T6.2).

    ``U`` acts on system kron environment (system first). With
    ``sigma_E = sum_r q_r |r><r|`` the Kraus operators are
    ``K_sr = sqrt(q_r) <s|_E U |r>_E``.
    """
    U = as_matrix(U)
    sigma_E = as_matrix(sigma_E)
    d_env = sigma_E.shape[0]
    if U.shape != (d_sys * d_env, d_sys * d_env):
        raise ChannelError("unitary does not act on system kron environment")
    q, vecs = np.linalg.eigh(hermitian_part(sigma_E))
    Ut = U.reshape(d_sys, d_env, d_sys, d_env)
    ks = []
    for r in range(d_env):
        if q[r] <= 1e-15:
            continue
        # column vector of U on |r> in the eigenbasis of sigma_E
        Ur = np.einsum("aebf,f->aeb", Ut, vecs[:, r])
        for s in range(d_env):
            ks.append(np.sqrt(q[r]) * Ur[:, s, :])
    return KrausMap(ks, label=label)


def replacement_channel(sigma: np.ndarray, d_in: int, label: str = "replace") -> KrausMap:
    """``R -> Tr(R) sigma``."""
    sigma = as_matrix(sigma)
    q, vecs = np.linalg.eigh(hermitian_part(sigma))
    ks = []
    for r, qr in enumerate(q):
        if qr <= 1e-15:
            continue
        for i in range(d_in):
            K = np.zeros((sigma.shape[0], d_in), dtype=complex)
            K[:, i] = np.sqrt(qr) * vecs[:, r]
            ks.append(K)
    return KrausMap(ks, label=label)


def choi_to_kraus(C: np.ndarray, d_in: int, d_out: int, atol: float = 1e-12, label: str = "") -> KrausMap:
    """Kraus operators from a positive Choi matrix (eigen-decomposition)."""
    C = as_matrix(C)
    if not is_psd(C, atol=1e-9):
        raise ChannelError("Choi matrix is not positive: map is not completely positive")
    w, V = np.linalg.eigh(hermitian_part(C))
    ks = []
    for k in range(len(w)):
        if w[k] <= atol * max(1.0, w[-1]):
            continue
        v = np.sqrt(w[k]) * V[:, k]
        ks.append(np.stack([v[i * d_out:(i + 1) * d_out] for i in range(d_in)], axis=1))
    if not ks:
        ks = [np.zeros((d_out, d_in), dtype=complex)]
    return KrausMap(ks, label=label)


def superop_to_choi(S: np.ndarray, d_in: int, d_out: int) -> np.ndarray:
    """Choi matrix of a linear map given by its Liouville matrix (column stacking)."""
    S = as_matrix(S)
    C = np.zeros((d_in * d_out, d_in * d_out), dtype=complex)
    for i in range(d_in):
        for j in range(d_in):
            Eij = np.zeros((d_in, d_in), dtype=complex)
            Eij[i, j] = 1.0
            out = unvec(S @ vec(Eij), d_out)
            C[i * d_out:(i + 1) * d_out, j * d_out:(j + 1) * d_out] = out
    return C


def superop_to_kraus(S: np.ndarray, d_in: int, d_out: int | None = None, label: str = "") -> KrausMap:
    d_out = d_in if d_out is None else d_out
    return choi_to_kraus(superop_to_choi(S, d_in, d_out), d_in, d_out, label=label)


def is_completely_positive_superop(S: np.ndarray, d_in: int, d_out: int | None = None, atol: float = 1e-9) -> bool:
    d_out = d_in if d_out is None else d_out
    return is_psd(superop_to_choi(S, d_in, d_out), atol)


def apply_superop(S: np.ndarray, R: np.ndarray) -> np.ndarray:
    R, S = as_matrix(R), as_matrix(S)
    if S.shape[1] != R.size:
        raise ChannelError(f"superoperator of shape {S.shape} cannot act on a {R.shape} matrix")
    d_out = round(np.sqrt(S.shape[0]))
    if d_out * d_out != S.shape[0]:
        raise ChannelError("superoperator output size is not a square")
    return unvec(S @ vec(R), d_out)


def is_trace_preserving_superop(S: np.ndarray, d: int, atol: float = ATOL) -> bool:
    """``vec(I)^dagger S = vec(I)^dagger`` for a square map on ``d x d`` matrices."""
    vI = vec(np.eye(d))
    return np.allclose(vI.conj() @ as_matrix(S), vI.conj(), atol=atol, rtol=0)


# ----------------------------------------------------------------------------
# Instruments (T10.1)
# ----------------------------------------------------------------------------

@dataclass
class Instrument:
    """Outcome-labelled completely positive maps whose sum preserves trace."""

    maps: Mapping[Hashable, KrausMap]
    label: str = ""
    complete: bool = field(default=True)

    def __post_init__(self) -> None:
        if not self.maps:
            raise ChannelError("an instrument needs at least one outcome")
        dims = {(m.d_in, m.d_out) for m in self.maps.values()}
        if len(dims) != 1:
            raise ChannelError("instrument maps have different dimensions")
        (self.d_in, self.d_out), = dims
        if self.complete:
            dev = float(np.max(np.abs(self.total_effect() - np.eye(self.d_in))))
            if dev > ATOL:
                raise ChannelError(f"instrument {self.label!r} does not sum to a trace-preserving map ({dev:.3e})")

    @property
    def outcomes(self) -> list[Hashable]:
        return list(self.maps)

    def total_effect(self) -> np.ndarray:
        return sum(m.effect() for m in self.maps.values())

    def total(self) -> KrausMap:
        """The unconditioned (trace-preserving) map ``sum_tau J_tau``."""
        return KrausMap([K for m in self.maps.values() for K in m.kraus], label=f"sum {self.label}")

    def probabilities(self, R: np.ndarray) -> dict[Hashable, float]:
        return {t: m.probability(R) for t, m in self.maps.items()}

    def __getitem__(self, outcome: Hashable) -> KrausMap:
        return self.maps[outcome]

    def then(self, first: KrausMap) -> Instrument:
        """Instrument ``tau -> J_tau o first`` (``first`` must preserve trace for completeness)."""
        return Instrument({t: m.compose(first) for t, m in self.maps.items()},
                          label=f"{self.label}∘{first.label}", complete=self.complete)


def projective_instrument(projectors: Mapping[Hashable, np.ndarray], label: str = "readout") -> Instrument:
    """Ideal projective readout: outcome ``tau`` has the single Kraus operator ``P_tau``."""
    return Instrument({t: KrausMap([as_matrix(P)], label=f"P{t}") for t, P in projectors.items()}, label=label)


def map_from_function(fn: Callable[[np.ndarray], np.ndarray], d_in: int, d_out: int | None = None) -> np.ndarray:
    """Liouville matrix of an arbitrary linear map given as a function (for verification)."""
    d_out = d_in if d_out is None else d_out
    S = np.zeros((d_out * d_out, d_in * d_in), dtype=complex)
    for i in range(d_in):
        for j in range(d_in):
            Eij = np.zeros((d_in, d_in), dtype=complex)
            Eij[i, j] = 1.0
            S[:, i + d_in * j] = vec(fn(Eij))
    return S


# ----------------------------------------------------------------------------
# Random maps for property tests (seeded generators only)
# ----------------------------------------------------------------------------

def _random_isometry(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    G = (rng.standard_normal((rows, cols)) + 1j * rng.standard_normal((rows, cols))) / np.sqrt(2)
    Q, R = np.linalg.qr(G)
    d = np.diag(R)
    return Q * (d / np.abs(d))


def random_instrument(d_in: int, outcomes: Sequence[Hashable], rng: np.random.Generator,
                      kraus_per_outcome: int = 2, d_out: int | None = None, label: str = "random") -> Instrument:
    """Random instrument: one isometry ``V: C^d_in -> C^(n k d_out)`` cut into ``n k`` Kraus blocks.

    ``sum_a K_a^dagger K_a = V^dagger V = I``, so the outcome maps sum to a
    trace-preserving map by construction.
    """
    d_out = d_in if d_out is None else d_out
    n, k = len(outcomes), kraus_per_outcome
    V = _random_isometry(n * k * d_out, d_in, rng)
    blocks = [V[i * d_out:(i + 1) * d_out] for i in range(n * k)]
    return Instrument({t: KrausMap(blocks[i * k:(i + 1) * k], label=f"{label}[{t}]") for i, t in enumerate(outcomes)},
                      label=label)


def random_channel(d_in: int, rng: np.random.Generator, n_kraus: int = 3, d_out: int | None = None,
                   label: str = "random") -> KrausMap:
    """Random completely positive trace-preserving map with ``n_kraus`` Kraus operators."""
    return random_instrument(d_in, [None], rng, kraus_per_outcome=n_kraus, d_out=d_out, label=label)[None]


# ----------------------------------------------------------------------------
# Structured maps given by their action (for registers too large for Kraus matrices)
# ----------------------------------------------------------------------------

class FunctionMap:
    """A completely positive map given by its Schrodinger action and its Heisenberg dual.

    Used for large structured maps (index permutations, diagonal congruences,
    Schur-product dephasing) whose Kraus matrices would be wasteful. Complete
    positivity is a property of the construction: every realization that
    builds one states its Kraus form and tests the action against an explicit
    :class:`KrausMap` on small registers. Trace preservation is checked here
    through the dual, exactly as for a :class:`KrausMap`.
    """

    __slots__ = ("d_in", "d_out", "dual_fn", "fn", "label")

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], dual_fn: Callable[[np.ndarray], np.ndarray],
                 d_in: int, d_out: int | None = None, label: str = "") -> None:
        self.fn, self.dual_fn = fn, dual_fn
        self.d_in = int(d_in)
        self.d_out = self.d_in if d_out is None else int(d_out)
        self.label = label

    @classmethod
    def from_kraus(cls, K: KrausMap) -> FunctionMap:
        return cls(K.__call__, K.dual, K.d_in, K.d_out, label=K.label)

    @classmethod
    def identity(cls, dim: int) -> FunctionMap:
        return cls(lambda R: R, lambda A: A, dim, label="id")

    def __call__(self, R: np.ndarray) -> np.ndarray:
        R = as_matrix(R)
        if R.shape != (self.d_in, self.d_in):
            raise ChannelError(f"{self.label or 'map'}: input shape {R.shape}, expected {(self.d_in, self.d_in)}")
        return self.fn(R)

    def dual(self, A: np.ndarray) -> np.ndarray:
        return self.dual_fn(as_matrix(A))

    def effect(self) -> np.ndarray:
        return self.dual(np.eye(self.d_out, dtype=complex))

    def probability(self, R: np.ndarray) -> float:
        return float(np.real(np.trace(self(R))))

    def compose(self, first: KrausMap | FunctionMap) -> FunctionMap:
        """``self o first`` (apply ``first``, then ``self``)."""
        if first.d_out != self.d_in:
            raise ChannelError("dimension mismatch in composition")
        return FunctionMap(lambda R: self.fn(first(R)), lambda A: first.dual(self.dual_fn(A)),
                           first.d_in, self.d_out, label=f"{self.label}∘{first.label}")

    def __matmul__(self, first: KrausMap | FunctionMap) -> FunctionMap:
        return self.compose(first)

    def superop(self) -> np.ndarray:
        """Liouville matrix (small registers only; used to verify the construction)."""
        return map_from_function(self.fn, self.d_in, self.d_out)

    def kraus_map(self) -> KrausMap:
        """Explicit Kraus form from the Choi matrix (small registers only)."""
        return superop_to_kraus(self.superop(), self.d_in, self.d_out, label=self.label)

    def is_trace_preserving(self, atol: float = ATOL) -> bool:
        return np.allclose(self.effect(), np.eye(self.d_in), atol=atol, rtol=0)

    def is_trace_nonincreasing(self, atol: float = ATOL) -> bool:
        return is_psd(np.eye(self.d_in) - self.effect(), atol)

    def require_trace_preserving(self, atol: float = ATOL) -> FunctionMap:
        dev = float(np.max(np.abs(self.effect() - np.eye(self.d_in))))
        if dev > atol:
            raise ChannelError(f"{self.label or 'map'} is not trace preserving (deviation {dev:.3e})")
        return self

    def __repr__(self) -> str:
        return f"FunctionMap({self.label!r}, {self.d_in}->{self.d_out})"
