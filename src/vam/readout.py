"""Boundary readout: projective, nested coarse/fine, and evolution-then-readout instruments.

* T6.4: a measured output ``tau`` has outcome map ``J_tau`` and probability
  ``Tr J_tau(R)``; einselection explains stable alternatives, the instrument
  fixes their probabilities and conditioned states.
* T8.5 (derived): for a macro partition ``{P_j}`` and fine projectors with
  ``sum_a Q_{a|j} = P_j``, ``p(j) p(a|j) = Tr(Q_{a|j} R)``.
* T10.1: the output instrument includes the coherent evolution, the
  coarse/fine interactions and the boundary readout.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping

import numpy as np

from .channels import ChannelError, Instrument, KrausMap, projective_instrument
from .density import condition
from .linalg import ATOL, as_matrix, is_projector


def validate_partition(projectors: Mapping[Hashable, np.ndarray], atol: float = ATOL) -> None:
    ps = [as_matrix(P) for P in projectors.values()]
    for P in ps:
        if not is_projector(P, atol):
            raise ChannelError("readout operator is not an orthogonal projector")
    if not np.allclose(sum(ps), np.eye(ps[0].shape[0]), atol=atol):
        raise ChannelError("projectors do not sum to the identity")
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            if not np.allclose(ps[i] @ ps[j], 0, atol=atol):
                raise ChannelError("projectors are not mutually orthogonal")


def nested_instrument(coarse: Mapping[Hashable, np.ndarray],
                      fine: Mapping[Hashable, Mapping[Hashable, np.ndarray]],
                      label: str = "nested") -> Instrument:
    """Fine readout with outcomes ``(j, a)``, Kraus operator ``Q_{a|j}`` (T8.5)."""
    validate_partition(coarse)
    maps = {}
    for j, Pj in coarse.items():
        Qs = fine[j]
        total = sum(as_matrix(Q) for Q in Qs.values())
        if not np.allclose(total, as_matrix(Pj), atol=ATOL):
            raise ChannelError(f"fine projectors of sector {j!r} do not sum to P_j")
        for a, Q in Qs.items():
            if not is_projector(Q):
                raise ChannelError("fine readout operator is not a projector")
            if not np.allclose(as_matrix(Pj) @ as_matrix(Q), as_matrix(Q), atol=ATOL):
                raise ChannelError("fine projector not inside its coarse sector")
            maps[(j, a)] = KrausMap([as_matrix(Q)], label=f"Q{a}|{j}")
    return Instrument(maps, label=label)


def coarse_then_fine(R: np.ndarray, coarse: Mapping[Hashable, np.ndarray],
                     fine: Mapping[Hashable, Mapping[Hashable, np.ndarray]]) -> dict[tuple, tuple[float, float]]:
    """Two-stage readout of one state: ``{(j, a): (p(j), p(a|j))}``.

    The coarse measurement's back-action (conditioning on ``P_j``) is applied
    before the fine stage.
    """
    R = as_matrix(R)
    out = {}
    for j, Pj in coarse.items():
        Pj = as_matrix(Pj)
        pj = float(np.real(np.trace(Pj @ R)))
        if pj <= ATOL:
            for a in fine[j]:
                out[(j, a)] = (pj, float("nan"))
            continue
        Rj, _ = condition(Pj @ R @ Pj)
        for a, Q in fine[j].items():
            out[(j, a)] = (pj, float(np.real(np.trace(as_matrix(Q) @ Rj))))
    return out


def evolve_then_readout(evolution: KrausMap, projectors: Mapping[Hashable, np.ndarray],
                        label: str = "J") -> Instrument:
    """``J_tau = P_tau o evolution`` for a trace-preserving evolution."""
    evolution.require_trace_preserving()
    validate_partition(projectors)
    return projective_instrument(projectors, label=label).then(evolution)
