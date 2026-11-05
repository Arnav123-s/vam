# End-to-End Plan: The Unified Machine

**Goal:** Go from validated fragments (Phases 1–13) to ONE machine that uses
every piece, tested on synthetic + text, with a final report. Work straight
through, no stopping between phases.

## Phases

### Phase 14: Unified synthetic machine
Combine: population + soft routing (Phase 10) + governor purge (Phase 4).
Source: K=10 regimes (Phase 9 setup).
Question: does purge help when regimes go stale?
Compare: soft-routing alone vs soft-routing + purge vs oracle.
Success: purge ≥ no-purge, both ≈ oracle.

### Phase 15: Unified text machine
Take Phase 11 word population, add governor purge: drop words seen < N
times from the vocab (prune the tail).
Question: does purging rare words help bpb (less noise) or hurt (less coverage)?
Compare: purge vs no-purge at 150KB, vs flat5.
Success: a clear verdict either way, with the threshold mapped.

### Phase 16: Final synthesis
- Update SYNTHESIS.md with the complete arc (Phases 1–15).
- Write a clean pop/README.md presenting the track as a research artifact.
- Push everything to the branch.
- Final report to Arnav.

## Rules (from the track)
- Pre-register each protocol before running.
- Log failures with causes, never tune until they pass.
- Push after each phase.
