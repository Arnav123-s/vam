# The Population Interpretation Track

**Thesis:** Test Arnav's original VAM ideas (from sera-research-raw F/T/N), not
the mis-extracted implementation that produced the old verdict.

**Method:** Every phase pre-registered. Failures logged with causes. Never
tuned until they pass. Classical shadows first; quantum parts flagged as such.

## Results at a glance

| Phase | Claim | Result |
|---|---|---|
| 1–1b | Population beats flat on multi-scale structure | ✓ 0.75 vs 1.00 |
| 2 | Latent regime inference beats noisy key | ✓ 0.72 vs 0.75 |
| 3a–3b | Shared factors + hierarchical disentanglers | ✓ 0.87 / 0.69 |
| 4–4d | Governor purge + disentangled structural radiation | ✓ 0.92 vs 0.99 (hard test) |
| 5a–5b | Navigation beats sampling; latent transform load-bearing | ✓ 1.00 / 0.86 |
| 6 | Integrated machine composes | ✓ 0.71 |
| 7a–7b | Text-like + real text | ✓ 0.58 / 3.29 vs 3.82 |
| 8 | Quantum density-matrix tracker | ~ ties classical (0.70 vs 0.71) |
| 9–10 | Scale breaks on hard ID, fixed by soft routing | ✓ 0.958 ≈ oracle at K=500 |
| 11–13 | Text scaling: hierarchy wins scarce, loses abundant | 2.75 → 2.71 vs flat 2.27 at 500KB |
| 14–15 | Unified machine: soft + purge | ✓ 0.657 / 2.727 (sweet spot) |

## Standing principles

1. **Bootstrap rule.** Latent machinery needs grounded initialization. Never train likelihoods on own outputs.
2. **Disentangle before radiating.** Invariant ⊥ variant, or transfer fails.
3. **Scale reveals.** Toy nulls hide real effects; test where relearning is hard.
4. **Don't identify, weight.** Soft routing (attention) beats hard identification at scale.
5. **Structure must be earned.** Hierarchy doesn't come free from symmetric starts.

## Honest limits

- Quantum: machinery works, no advantage on classical data (Bayes is optimal).
- Scale: soft routing works at K=500; millions untested.
- Text: hierarchy wins when data is scarce, loses when abundant (Phase 13).
- All quantum parts are classical shadows; true implementations unbuilt.

## Layout

- `INTERPRETATION.md` — the five hypotheses, owner's verbatim formulations, testable claims, track rules.
- `proto/` — page, population, sources, slow (tracker), factors, governor, disentangle.
- `runs/` — one script + results JSON per phase. Protocol in each docstring.
- `tests/` — smoke tests.
- `../TRACK.md` — full status, failures, push log.
- `../SYNTHESIS.md` — the complete verdict.
- `../PLAN.md` — the end-to-end plan (Phases 14–16).

## Loop

Build → run → re-evaluate → gap-check → iterate. Failed interpretations are
logged with *why* (extraction? implementation? idea?) — never tuned until
they pass.
