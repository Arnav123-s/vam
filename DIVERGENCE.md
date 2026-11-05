# Did I Understand the Idea Right? — Divergence Analysis

**Question (Arnav, 2026-11-04):** maybe the idea wasn't understood right?
Maybe some mistake I made?

**Short answer:** Yes. There are specific, checkable divergences between
VAM-as-specified (DESIGN.md) and what the track built. The track validated
structural principles, but it did NOT implement VAM's machine.

## What VAM actually says (DESIGN.md)

1. **Present-state memory.** "The machine is a box whose whole memory is
   its present state: one big table of numbers (a density matrix)."
   "No transcript, no replay buffer, no context window." (T4.1, T10)
   State size is CONSTANT — it does not grow with the number of events.

2. **Event loop (T10.1).** Input bit → nudge state → state evolves by
   itself (Dirac equation) → measure boundary → output bit → feed back in
   → canalize (settle: drop what can never matter, keep what can) →
   plasticity from the external bit only. Nothing is ever written to a
   history log.

3. **Plasticity (T12.1).** Correlation-to-weight law, activity-gated, from
   the external bit only. Re-entered outputs transform the state but are
   NOT evidence (T10.3).

4. **Canalization (T11).** The invariant-algebra lock: what the present
   state must keep. Retention identity ΠK = ΠKΠ.

## What the track built

| VAM says | Track built | Divergence |
|---|---|---|
| Fixed-size density state | Count tables that GROW with data (vocab, bigram entries) | **History log, not present state** |
| Dirac evolution of state | Static tables, no evolution step | **No dynamics** |
| Canalization (settle/drop) | Governor purge (crude analogue) | Partial — purge is a shadow |
| T12.1 plasticity law | Count increments (frequency) | **Different learning rule** |
| Re-entry of outputs | Navigation (search procedure) | Different mechanism |
| No context window | Bigram uses previous word as context | **Explicit history** |

## The key mistake

I reduced "population" to "mixture of n-gram tables." The real VAM idea is
a density-state machine that LEARNS ONLINE through an event loop. My models
are batch count tables. The 2MB failure (Phase 16/17) is a failure of my
n-gram implementation — it does not test VAM's actual machine.

## The test (Phase 18)

Build a minimal FAITHFUL present-state learner and compare it against the
count-based population on the same text:
- State: a FIXED-SIZE weight vector (does not grow with data).
- Update: online plasticity (correlation law) from each observed char only.
- Prediction: from current state only. No stored history, no n-gram tables.
- If the present-state learner is competitive, the idea was misunderstood
  and the scale limit was my artifact. If it is much worse, the count
  tables were doing the heavy lifting and the present-state constraint is
  costly — and the scale limit may be real.

## Result: the idea WAS misunderstood (Phase 18/18b)

- 150KB: present-state **2.58** beats population 2.75 and flat5 2.81.
- 2MB: present-state **2.07** ≈ flat5 2.03 in-distribution; **2.12 BEATS
  flat5 2.26** out-of-distribution (unseen book).
- The scale limit (Phases 13/16/17) was an artifact of the count-based
  implementation, not of VAM's idea. The present-state principle holds at
  scale — and generalizes better, because it learns online and adapts
  rather than memorizing counts.
