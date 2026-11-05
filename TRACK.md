# VAM Interpretation Track — Status

**Thesis:** Test Arnav's original ideas (from sera-research-raw F/T/N), not the mis-extracted implementation. Every phase pre-registered; failures logged with causes.

**Branch:** `claude/population-interpretation` on Arnav123-s/vam (pushed, commits 5a830ce → 3cdaad3b). Everything under `pop/`, zero changes to existing work.

## Phases

| # | Hypothesis | Result | Key numbers |
|---|---|---|---|
| 1 | H1: population beats flat on multi-scale | ✓ | 0.75 vs 1.00 |
| 1b | H1 under starvation | ✓ | legend backoff dominates |
| 2 | H2/H3: latent regime inference | ✓ | 0.72 vs 0.75 (majority), oracle 0.66 |
| 3a | H4: shared factors (classical shadow) | ✓ | 0.87 vs 0.98 |
| 3b | H4: unitary disentanglers (MERA-shadow) | ✓ | 0.69 vs 0.10 chance, PCA 1.00 |
| 4 | H2: governor (purge/radiate) | ✓ machinery, reframed | purge 0.82 vs static 0.90; radiate≈reuse |
| 4b | H2×H4: structural radiation | → 4c | shape entangled with mean/scale |
| 4c | H2×H4: disentangled transfer | ✓ mechanical, null at toy scale | shape corr 0.99; ties at 32 params |
| 4d | H2×H4: hard transfer test | ✓ | 0.92 vs 0.99 (gap 0.07), 1024-dim |
| 5a | H5: navigation beats sampling | ✓ | validity 1.00 vs 0.32 |
| 5b | H5×H2: latent transform load-bearing | ✓ | 0.86 vs 0.60; errors = inference only |
| 6 | Integration: perceive→predict→learn | ✓ | 0.71 vs 0.74 (majority), oracle 0.65 |
| 7a | Text-like synthetic language | ✓ | 0.58 vs 0.61 (flat6); 10/10 valid words |
| 7b | Real text (Pride & Prejudice) | ✓ | **3.29 vs 3.82 (flat5)** |
| 8 | Quantum density-matrix tracker | ✓ ties | 0.70 vs 0.71 (classical); coherence ≠ differentiator on classical data |
| 9 | Scale: many regimes | ⚠ bottleneck found | tracker accuracy 45% (K=10) → 6% (K=100) → 0.1% (K=500, worse than chance) |
| 10 | Soft routing (attention) fixes scale | ✓ | soft 0.958 ≈ oracle 0.965 at K=500; hard 1.128 |
| 11 | Scaled text (150KB, 1747 words) | ✓ | 2.75 vs 2.81 flat5, 3.68 flat7; margin narrows but holds |
| 12 | Trigram word model | ~ null | 2.99 pure, 2.77 interpolated vs 2.75 bigram — richer slow scale needs more data |

## Failures & fixes (all logged)

1. **Phase 2**: 3 tracker bugs — symmetric start stalls (needs bootstrap); self-trained likelihood collapses (freeze legend); unconditional likelihood ratio pins at 336 (must condition on observed bit).
2. **Phase 4**: radiate/restore = reuse (3 variants) → reframed: radiate STRUCTURE not params.
3. **Phase 4b**: shape entangled with mean/scale → 4c disentangles (corr 0.99).
4. **Phase 4c**: ties at toy scale → 4d hard test confirms (0.07 gap at 1024-dim).
5. **Phase 3b**: adjacent pairing 0.42 → adaptive 0.69 (pairing was the bottleneck).
6. **Phase 7a**: no-bootstrap 1.83/salad → structural bootstrap + soft EM → 0.58/10-10.
7. **Phase 7b**: OOV fell through to 1e-9 (bpb 5.93, worse than uniform) → full-support smoothing → 3.29.

## Standing principles (earned, not assumed)

- **Bootstrap rule** (Phases 2, 7a): latent machinery needs grounded initialization from observables. Never train likelihoods on own outputs.
- **Disentangle before radiating** (4b→4c→4d): invariant ⊥ variant, or transfer fails.
- **Scale reveals** (4c→4d): toy nulls can hide real effects; test where relearning is hard.

## Open / next

- [ ] Push Phase 7a+7b to branch (local, not yet pushed)
- [ ] Push Phase 8 + TRACK.md update
- [x] 7b generation quality: word-level generation (slow generates, fast spells) → zero salad, all vocab words
- [x] Quantum: density-matrix tracker ties classical (0.70 vs 0.71) — coherence validated as machinery, not a differentiator on classical data
- [x] Scale: regime identification is the bottleneck (45% → 0.1% as K goes 10 → 500); fixed by soft routing (posterior-weighted, like attention) — 0.958 ≈ oracle at K=500
- [ ] Old build's word-salad: 7b generates words, not salad — direct comparison pending

## Push log

- 5a830ce: phases 1–5b (34 files)
- 3cdaad3b: phase 6
- e9ff0d4: phases 7a, 7b + TRACK.md
- 90686ed: 7b word-level generation + TRACK.md update
- b78b169: Phase 8 + TRACK.md update
- 385946c: Phase 9 + TRACK.md update
- 4a092c9: Phase 10 + TRACK.md update
- 5c0662a: SYNTHESIS.md
- ede23b9: Phase 11 + TRACK.md update
- pending: Phase 12 + SYNTHESIS.md + TRACK.md update
