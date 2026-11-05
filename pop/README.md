# pop/ — interpretation track

Testing the owner's **original** VAM ideas against the extracted implementation.
Premise: where the published build fails, the failure may sit in the *interpretation*
or the *implementation* — not necessarily the idea.

See **INTERPRETATION.md** for the five hypotheses (H1–H5) as testable claims.

## Status

**Phase 1 (done, 2026-11-04):** H1 machinery — population of 8 small density matrices,
one per (slow × micro) node, blanket-insulated, each Page-sized per node, with an
overlying "legend" layer — on the regime-flip source (slow regime reverses the fast
order-2 pattern; provably ambiguous to flat single-scale models).

| model | bits/bit (3 seeds) | burn-in (first 1000) |
|---|---|---|
| flat order-2 | 0.997 | ~1.0 |
| flat 8-context (slow×micro) | 0.753 | 0.86 |
| **population (8 nodes + legend)** | **0.753** | **0.77** |

- Nodes converge to the true source entropy with an oracle key (0.656 vs 0.64) —
  the machinery is correct; the 0.10 gap is the slow key's 6% noise, not the nodes.
- Prediction drift falls across quartiles (0.0013 → 0.0007): the organization
  **stabilizes** as data grows (H3's criterion, not transfer).
- Honest: steady-state ties the flat 8-context table — the win over flat-2 is the
  slow key. The population's distinctive value (legend sharing under data starvation)
  is **Phase 1b** (done — see below).

**Phase 1b (done, 2026-11-04):** starvation probe — micro orders 2/4/6 (8/32/128
pairs), flat monolithic table vs population, same key. The legend is a backoff:
young/rare nodes trust the slow-marginal instead of their own thin counts.

| pairs | flat bpb | pop bpb | rare-zone flat | rare-zone pop | burn-in flat | burn-in pop |
|---|---|---|---|---|---|---|
| 8 | 0.753 | 0.753 | 0.739 | 0.739 | 0.893 | **0.766** |
| 32 | 0.700 | 0.699 | 0.690 | 0.690 | 1.168 | **0.751** |
| 128 | 0.699 | 0.692 | 0.691 | **0.685** | 1.484 | **0.783** |

- Under starvation the legend dominates: at 128 pairs flat flails at 1.48 bits in
  burn-in while the population holds 0.78. Small steady-state win in the rare zone
  at order 6 (0.685 vs 0.691).
- Observation: deeper micro keys partially compensate slow-key noise (overall bpb
  improves 0.753 → 0.699 with order) — the micro pattern carries regime information
  the noisy slow key misses.
- Remaining gap is still the slow key (oracle: 0.656). Next levers: latent regime
  inference (H2/H5 territory) or tensor cross-node structure (H4).

**Phase 2 (done, 2026-11-04):** latent slow-regime inference. The legend doubles as
the regime estimator: exact binary Bayes filter (`proto/slow.py`), prequential
predict-then-update, likelihood ratio of the *observed* bit.

| key | bits/bit (3 seeds) | regime agreement |
|---|---|---|
| majority-32 | 0.753 | 94.0% |
| **latent Bayes** | **0.723** | **96.2%** |

- Three failure modes found and fixed, all logged: (1) symmetric start never moves
  (needs bootstrap); (2) tracker training its own likelihood model collapses to a
  degenerate fixed point — the likelihood model (legend) must be **frozen** as an
  immutable bedrock while nodes stay plastic (H3's split is what makes latent
  inference possible at all); (3) likelihood ratio must be for the observed bit,
  not p(1|slow) unconditionally (pins odds at a fixed point).
- Remaining gap to oracle (0.656): transition lag after flips + imperfect legend
  likelihoods. Next: richer tracker evidence (node likelihoods) or H4/H2.

**Phase 3a (done, 2026-11-04):** H4 in its minimal classical shadow — the population's
conditional table p(bit|slow,micro) factorized as logit = A_s + B_m (rank-1; the
micro factors B_m "entangled" across slow states). Source with exactly rank-1
structure; slow=1 regime rare (5%) = the transfer zone.

| bits | independent (64 params) | factorized (18) | population+legend |
|---|---|---|---|
| overall | 0.875 | 0.873 | 0.870 |
| **rare-zone (slow=1)** | 0.977 | **0.871** | 0.882 |

- Ordering F < P < I on the transfer zone separates the sharing levels cleanly:
  shared *micro* structure (0.871) > marginal-only backoff (0.882) > no sharing
  (0.977). The H4 principle holds: cross-scale shared structure beats independent
  nodes where data is thin.
- Honest: this is classical rank-1 factorization, not unitary disentanglers on
  density matrices (Phase 3b). SGD lr schedule is ad hoc. Overall ties because the
  rare regime is only 5% — by design, the effect lives in the transfer zone.

**Phase 4 (done, 2026-11-04):** H2 governor as criticality (`proto/governor.py`).
6 sequential regimes (biases 0.15–0.85), 4 slots, regimes return. Governor:
entropy monitor (S/C), purge stalest, radiate converged tables as frozen
invariants, restore on return. Baselines: B1 (slot reuse, no radiate/restore),
B2 (static).

| model | total bpb |
|---|---|
| governor (purge+radiate+restore) | 0.8200 |
| B1 (purge only) | 0.8202 |
| B2 (static) | 0.8975 |

- Purge >> static (0.82 vs 0.90): slot management matters. Entropy stays under
  capacity (S/C 11.9→10.3/12.0). Restores fire correctly (returns match first
  visits: regime 1: 0.608→0.608).
- **Null (robust, 3 variants):** radiate/restore ~= reuse (gaps 0.0002–0.0013).
  Why: rebuilding *counts* is cheap (~200–1000 bits) when the slot *structure*
  is already right. The radiated invariant must be STRUCTURAL to pay.
- Reframing (H2×H4): the governor should radiate learned structure (partitions,
  factors, disentanglers), not parameters. See `runs/phase4_notes.md`.

**Phase 4b (done, 2026-11-04):** H2×H4 structural radiation. Radiate R0's micro-pattern
SHAPE; R1 tests shape vs flipped-shape in a 200-bit shootout. Returns restore directly.

| segment | transfer | baseline |
|---|---|---|
| R1 (shape transfer) | 0.662 | 0.657 |
| R0/R1 returns (restore) | 0.656/0.650 | 0.657/0.650 |

Boundary variant (32-micros, 3k-bit R1): transfer 0.914 vs baseline 0.905 —
shootout picks the RIGHT shape (flipped, as true), yet still ties.

**Why (the real finding):** the transferred shape is entangled with mean and scale
in my encoding — pseudo-counts around base 0.5 pull toward the wrong mean (R1's
true mean 0.4). Shape transfer without *disentangling* shape ⊥ mean ⊥ scale
doesn't beat relearning. This is exactly why H4 needs real disentanglers, not
shape vectors: the disentangler's job is to separate invariant from variant so
the invariant transfers cleanly. Phase 4c (true disentangling factorization per
slot) is the next loop.

**Phase 4c (done, 2026-11-04):** true disentangling (`proto/disentangle.py`). Slot table
factored as p(1|m) = σ(α + β·s[m]); radiate s only; transfer = SGD on (α,β) with s
fixed (2 params) vs 32-param baseline relearning.

| R1 data | transfer (SGD) | baseline | oracle (true α,β) |
|---|---|---|---|
| 5k bits | 0.932 | 0.925 | 0.912 |
| 1.5k bits | 0.949 | 0.951 | ~0.91 |

- Disentangling itself is near-perfect (shape corr 0.99+). The machinery works.
- **Robust null (4th variant):** transfer ≈ relearning. Oracle shows the win is real
  (~0.04 when starved) but SGD noise eats it, and it shrinks as data grows.
- **Why (the deep lesson):** at toy scale, exact counting is an excellent learner and
  structures are simple — there is no transfer problem to solve. Transfer pays when
  the hypothesis space is vast, data is truly scarce, or structure is combinatorial.
  None hold at 32 params / 1.5k+ bits.
- **Honest status:** mechanical validation ✓ (disentangle → radiate → transfer is
  coherent and correct). Performance win requires scale beyond the toy. The H2×H4
  fusion is built and waiting for a harder test, not more toy variants.

**Phase 4d (done, 2026-11-04):** the hard transfer test. 1024 micros (order 10),
shared 1024-dim shape; R0 abundant (100k bits) → disentangle → radiate; R1/R2/R3
scarce (3k bits each, ~3 visits/micro), different calibrations.

| regime | transfer (2-param) | baseline (1024-param) |
|---|---|---|
| R1/R2/R3 avg | **0.917** | 0.990 |

Gap ~0.07, all seeds. Baseline hopeless at 3 visits/micro; radiated shape +
2-param calibration holds. **The principle is confirmed where it matters:**
disentangled structural radiation beats relearning when relearning is hard.
Shape recovery corr 0.95.

**Phase 5b (done, 2026-11-04):** H5×H2 latent navigation (`runs/phase5b_latent_nav.py`).
Validity depends on a hidden regime (R=0: sum%3==0, p=0.7; R=1: sum%4==0,
p=0.3). Transform = exact Bayes re-inference of the regime from the prefix,
per emission.

| method | validity |
|---|---|
| greedy | 0.298 |
| nav, no transform (assume R=0) | 0.596 |
| nav + transform (re-infer R) | **0.864** |

Regime agreement 0.862 ≈ validity 0.864: whenever the regime is inferred
right, navigation is perfect; errors are inference errors only. **The
transform is load-bearing with latent state**, as predicted. The H5×H2
fusion works.

**Phase 5a (done, 2026-11-04):** H5 collapse as navigation (`runs/phase5a_navigation.py`).

**Phase 3b (done, 2026-11-04):** unitary disentanglers, the real H4
(`runs/phase3b_disentangle.py`). 32-dim data: 4 slow dims (var 4) rotated by a
random orthogonal (dense entanglement) + 28 fast dims (var 1). Metric: fraction
of true slow variance in 4 kept dims.

| method | slow variance captured |
|---|---|
| global SVD/PCA (optimal) | 0.999 |
| hierarchical learned, adaptive pairing | **0.69** |
| hierarchical learned, adjacent pairing | 0.42 |
| hierarchical fixed (Haar) | 0.13 |
| none | 0.10 |

- The MERA-shadow is a real disentangler: 7x chance, and learning matters
  (0.69 vs 0.13 fixed). Reversible, O(n), scale-respecting.
- Gap-check: adjacent→adaptive pairing closed half the PCA gap (0.42→0.69);
  the rest is greedy locality vs global optimum on *dense random*
  entanglement. On data with true local multi-scale structure it should do
  better — that's the next test, not this one.
- Honest: classical shadow only. Real unitary disentanglers on density
  matrices (coherence, not just variance) remain untested.
12-bit sequences, constraint sum%3==0. Prefix 6 → complete 6 (64 candidates).
Scope (all completions) → thread (max-prob valid) → emit → transform (re-condition).

| method | validity | mean logprob |
|---|---|---|
| greedy (sampling) | 0.322 | -8.85 |
| greedy+repair | 0.656 | -8.79 |
| **navigation** | **1.000** | **-7.83** |

- Scope+thread beat sampling on validity AND quality (~1 nat). V7 bounds sampling;
  navigation doesn't sample. Honest: transform implemented (per-emission
  re-conditioning) but not decisive HERE — the choice is made at scope time.
  Transform becomes load-bearing with latent state (re-inference per emission):
  H5×H2 fusion, next loop.

## Layout

- `INTERPRETATION.md` — the five hypotheses, owner's verbatim formulations, what was
  extracted instead, testable claims, track rules.
- `proto/` — `page.py` (Page-limit per-node basis sizing via Marchenko–Pastur edge),
  `population.py` (nodes, blankets, legend), `sources.py` (synthetic sources).
- `tests/` — smoke tests (`PYTHONPATH=. python3 tests/test_population.py`).
- `runs/` — `phase1_regime_flip.py` (protocol in docstring) + `results_phase1.json`.

## Loop

Build → run → re-evaluate → gap-check → iterate. Failed interpretations are logged
with *why* (extraction? implementation? idea?) — never tuned until they pass.
