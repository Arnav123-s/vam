"""The present-state event loop (T10, T14, F master loop and Parts 1-3).

One external bit ``b`` (T10.1):

    q_b       = Tr B_b(R)                       conditioning mode; 1 in injection mode
    R_b       = B_b(R) / q_b
    p(tau)    = Tr J_tau(R_b)                   output instrument (evolution + boundary readout)
    R_new     = Pi E_tau ( J_tau(R_b) / p(tau) )  re-entry, then canalization

followed by the slow structural update (plasticity) declared by the protocol.

Event-loop theorem (T10.2, derived): with ``B_b`` and every ``J_tau``
completely positive, ``sum_tau J_tau``, ``E_tau`` and ``Pi`` trace preserving,
and nonzero conditioning probabilities, every admitted event yields a
positive normalized state, ``sum_tau p(tau) = 1``, and repeated events form a
controlled Markov process on the complete present configuration.

Present-state rule (T4.1, F-Part 1): a :class:`Machine` holds exactly one
:class:`MachineState` and the static protocol. Nothing it stores grows with
the number of events. Event records are returned to the caller, never kept.

Feedback is not evidence (T10.3): the output is conditioned on once, by
``J_tau``; the re-entry ``E_tau`` is a trace-preserving interaction, never a
second conditioning.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .channels import ChannelError, Instrument, KrausMap
from .density import ZeroProbabilityError, check_density, condition
from .linalg import ATOL, hermitian_part

Structural = dict[str, np.ndarray]


@dataclass
class MachineState:
    """The complete present configuration: retained density plus structural coordinates."""

    R: np.ndarray
    structural: Structural = field(default_factory=dict)

    def copy(self) -> MachineState:
        return MachineState(self.R.copy(), {k: np.array(v, copy=True) for k, v in self.structural.items()})

    def nbytes(self) -> int:
        return int(self.R.nbytes + sum(np.asarray(v).nbytes for v in self.structural.values()))

    def shapes(self) -> tuple:
        return (self.R.shape, *tuple((k, np.shape(v)) for k, v in sorted(self.structural.items())))

    def distance(self, other: MachineState) -> float:
        """Largest entrywise difference (used by the Markov-property audit)."""
        d = float(np.max(np.abs(self.R - other.R)))
        for k in self.structural:
            d = max(d, float(np.max(np.abs(np.asarray(self.structural[k]) - np.asarray(other.structural[k])))))
        return d


@dataclass(frozen=True)
class EventContext:
    """Quantities of one event handed to the plasticity law (not stored by the machine)."""

    bit: int | None
    q_bit: float
    outcome: Hashable
    p_outcome: float
    R_prior: np.ndarray       # the present state before this event
    R_input: np.ndarray       # R_b: after the input interaction/observation
    R_new: np.ndarray         # after readout, re-entry and canalization


@dataclass(frozen=True)
class EventRecord:
    bit: int | None
    q_bit: float
    outcome: Hashable
    p_outcome: float


@dataclass(frozen=True)
class EventProtocol:
    """Declared event maps of one realization (T10.1, T14).

    Every callable receives the current structural coordinates; all numerical
    choices live in the realization that builds the protocol.

    ``input_map(b, s)``: the map ``B_b``. In ``"condition"`` mode the two maps
    must sum to a trace-preserving map (a complete prediction of the next bit);
    in ``"inject"`` mode each must preserve trace.
    ``settle(s)``: the declared closed step used when no external bit is
    supplied (generation after the stream ends, T14 step 15).
    """

    input_map: Callable[[int, Structural], KrausMap]
    output_instrument: Callable[[Structural], Instrument]
    feedback: Callable[[Hashable, Structural], KrausMap]
    canalization: Callable[[Structural], KrausMap] | None = None
    plasticity: Callable[[EventContext, Structural], Structural] | None = None
    settle: Callable[[Structural], KrausMap] | None = None
    fallback_injection: Callable[[int, Structural], KrausMap] | None = None
    input_mode: str = "condition"
    zero_policy: str = "raise"
    label: str = ""

    def __post_init__(self) -> None:
        if self.input_mode not in ("condition", "inject"):
            raise ValueError("input_mode must be 'condition' or 'inject'")
        if self.zero_policy not in ("raise", "inject"):
            raise ValueError("zero_policy must be 'raise' or 'inject'")
        if self.zero_policy == "inject" and self.fallback_injection is None:
            raise ValueError("zero_policy 'inject' needs a fallback_injection map")


class Machine:
    """The Variational Action Machine event loop for one declared protocol."""

    __slots__ = ("protocol", "state", "validate")

    def __init__(self, protocol: EventProtocol, state: MachineState, validate: bool = True) -> None:
        self.protocol = protocol
        self.state = state
        self.validate = validate
        if validate:
            check_density(state.R)

    # -- single-stage helpers -------------------------------------------------
    def input_stage(self, state: MachineState, b: int | None) -> tuple[np.ndarray, float]:
        """``(R_b, q_b)``; ``b = None`` applies the declared settling map."""
        p, s = self.protocol, state.structural
        if b is None:
            if p.settle is None:
                raise ValueError("no external bit and no declared settling step")
            S = p.settle(s)
            if self.validate:
                S.require_trace_preserving()
            return S(state.R), 1.0
        if b not in (0, 1):
            raise ValueError("external input must be a bit")
        B = p.input_map(b, s)
        if p.input_mode == "inject":
            if self.validate:
                B.require_trace_preserving()
            return B(state.R), 1.0
        if self.validate:
            other = p.input_map(1 - b, s)
            tot = B.effect() + other.effect()
            if not np.allclose(tot, np.eye(B.d_in), atol=1e-9):
                raise ChannelError("conditioning maps of the two bit values do not sum to a trace-preserving map")
        try:
            R_b, q = condition(B(state.R))
        except ZeroProbabilityError:
            if p.zero_policy == "raise":
                raise
            # T10.2: the bit is still injected physically; its interpretation stays unresolved.
            F = p.fallback_injection(b, s)
            F.require_trace_preserving()
            return F(state.R), 0.0
        return R_b, q

    def outcome_probabilities(self, state: MachineState, R_b: np.ndarray) -> dict[Hashable, float]:
        J = self.protocol.output_instrument(state.structural)
        return J.probabilities(R_b)

    def complete_event(self, state: MachineState, b: int | None, q_b: float, R_b: np.ndarray,
                       tau: Hashable) -> tuple[MachineState, float]:
        """Readout outcome ``tau``, re-entry, canalization, plasticity."""
        p, s = self.protocol, state.structural
        J = p.output_instrument(s)
        R_tau, p_tau = condition(J[tau](R_b))
        E = p.feedback(tau, s)
        if self.validate:
            E.require_trace_preserving()
        R_new = E(R_tau)
        if p.canalization is not None:
            Pi = p.canalization(s)
            if self.validate:
                Pi.require_trace_preserving()
            R_new = Pi(R_new)
        R_new = hermitian_part(R_new)
        new_struct = s
        if p.plasticity is not None:
            ctx = EventContext(b, q_b, tau, p_tau, state.R, R_b, R_new)
            new_struct = p.plasticity(ctx, s)
            for k, v in s.items():
                if np.shape(new_struct.get(k)) != np.shape(v):
                    raise ValueError(f"plasticity changed the shape of structural coordinate {k!r}")
        new_state = MachineState(R_new, new_struct)
        if self.validate:
            check_density(new_state.R, atol=1e-8)
        return new_state, p_tau

    # -- the loop -------------------------------------------------------------
    def predict(self, b: int) -> float:
        """Probability the current state assigns to external bit ``b`` (conditioning mode)."""
        p = self.protocol
        if p.input_mode != "condition":
            raise ValueError("predictions exist only in conditioning mode")
        return p.input_map(b, self.state.structural).probability(self.state.R)

    def step(self, b: int | None, rng: np.random.Generator) -> EventRecord:
        """Process one external bit (or one settling step when ``b`` is None); sample the output."""
        R_b, q_b = self.input_stage(self.state, b)
        probs = self.outcome_probabilities(self.state, R_b)
        outcomes = list(probs)
        pv = np.array([max(0.0, probs[t]) for t in outcomes])
        total = pv.sum()
        if self.validate and abs(total - 1.0) > 1e-8:
            raise ChannelError(f"output probabilities sum to {total}")
        idx = int(np.searchsorted(np.cumsum(pv / total), rng.random(), side="right"))
        tau = outcomes[min(idx, len(outcomes) - 1)]
        self.state, p_tau = self.complete_event(self.state, b, q_b, R_b, tau)
        return EventRecord(b, q_b, tau, p_tau)

    def force(self, b: int | None, tau: Hashable) -> EventRecord:
        """Process one event with a prescribed output outcome (for exact branch tracking)."""
        R_b, q_b = self.input_stage(self.state, b)
        self.state, p_tau = self.complete_event(self.state, b, q_b, R_b, tau)
        return EventRecord(b, q_b, tau, p_tau)

    def run(self, bits: Iterable[int], rng: np.random.Generator,
            on_event: Callable[[EventRecord], None] | None = None) -> None:
        """Process a stream; per-event records go to ``on_event`` and are not retained."""
        for b in bits:
            rec = self.step(int(b), rng)
            if on_event is not None:
                on_event(rec)

    # -- exact enumeration ----------------------------------------------------
    def branches(self, state: MachineState, b: int | None,
                 min_prob: float = 0.0) -> list[tuple[Hashable, float, float, MachineState]]:
        """All outcomes of one event: ``(tau, q_b, p_tau, next_state)``."""
        R_b, q_b = self.input_stage(state, b)
        out = []
        for tau, pt in self.outcome_probabilities(state, R_b).items():
            if pt <= max(min_prob, ATOL):
                continue
            nxt, p_tau = self.complete_event(state, b, q_b, R_b, tau)
            out.append((tau, q_b, p_tau, nxt))
        return out

    def output_distribution(self, bits: Sequence[int | None], state: MachineState | None = None,
                            min_prob: float = 0.0) -> dict[tuple, float]:
        """Exact probabilities of every output string for a given input sequence.

        Probabilities are conditional on the external bits (the input factors
        ``q_b`` are not multiplied in).
        """
        start = self.state if state is None else state
        frontier: list[tuple[tuple, float, MachineState]] = [((), 1.0, start)]
        for b in bits:
            nxt = []
            for outs, prob, st in frontier:
                for tau, _q, pt, st2 in self.branches(st, b, min_prob):
                    nxt.append(((*outs, tau), prob * pt, st2))
            frontier = nxt
        dist: dict[tuple, float] = {}
        for outs, prob, _ in frontier:
            dist[outs] = dist.get(outs, 0.0) + prob
        return dist

    def sequence_probability(self, bits: Sequence[int], state: MachineState | None = None) -> float:
        """Probability the machine assigns to an external bit string (conditioning mode),
        summed over its own output branches."""
        start = self.state if state is None else state
        frontier = [(1.0, start)]
        for b in bits:
            nxt = []
            for prob, st in frontier:
                try:
                    brs = self.branches(st, b)
                except ZeroProbabilityError:
                    continue
                for _tau, q, pt, st2 in brs:
                    nxt.append((prob * q * pt, st2))
            frontier = nxt
        return float(sum(p for p, _ in frontier))


# ----------------------------------------------------------------------------
# Macro/micro answer generation (F-Part 2, T8.4-8.5, T14 steps 7-12)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    macro: tuple            # the selected continuation (macro sector)
    macro_probability: float
    first: Hashable         # the atomic output actually emitted
    distribution_size: int


def macro_micro_step(machine: Machine, horizon: int, select: str = "max",
                     rng: np.random.Generator | None = None, min_prob: float = 1e-12) -> Plan:
    """Plan over every continuation of ``horizon`` closed steps, then emit only the first output.

    Macro (Feynman scope): the exact distribution over all output strings of
    length ``horizon``, computed by branching the density through the
    machine's own re-entering dynamics. Selection: ``"max"`` takes the most
    probable continuation (the dominant contribution), ``"sample"`` draws one.
    Micro (Newton step): only the first output of the selected continuation is
    emitted, by applying that outcome's instrument map once; the machine then
    re-plans from its new state. The nested-readout identity of T8.5 makes the
    emitted first output consistent with the macro distribution.
    """
    dist = machine.output_distribution([None] * horizon, min_prob=min_prob)
    if not dist:
        raise ValueError("no continuation has nonzero probability")
    if select == "max":
        macro = max(dist, key=lambda k: (dist[k], tuple(map(str, k))))
    elif select == "sample":
        if rng is None:
            raise ValueError("sampling needs a random generator")
        keys = sorted(dist, key=lambda k: tuple(map(str, k)))
        pv = np.array([dist[k] for k in keys])
        macro = keys[int(rng.choice(len(keys), p=pv / pv.sum()))]
    else:
        raise ValueError("select must be 'max' or 'sample'")
    first = macro[0]
    machine.force(None, first)
    return Plan(macro, dist[macro], first, len(dist))
