# Audit 2: the nested fractal (N) work, `45d1209..6fbfdc7`

- Task: T1.4. Author: the engineer session (`[private session reference removed]`), with the Opus advisor on the
  mathematical calls (every conclusion it informed is written below, because nobody else can read the advice).
- Date: 2026-10-20. Repo state read: `cac24c0` (the helper's smoke run included). Code was **not changed**.
- Environment of the evidence: Python 3.13.16, numpy 2.5.3, scipy 1.18.1, numba 0.68.0 (E2's pins), 4 cores.
  Scripts are inline; their output is quoted. Corpus-dependent claims are marked, because this container has no
  PG-19 (the helper restored it in its own container; the helper's `e3_smoke.json` is read for fields only).
- Judged against: spec N, T (sections 7.1, 10.1, 10.3, 11.4, 13.1, 14 step 7, 15.3), DESIGN section 10, and
  PROTOCOL E3. Labels: S = source statement, D = derived result, C = chosen realization (T1.1).

## Summary

| ID | Severity | Blocks | Finding | Fix |
|---|---|---|---|---|
| A1 | High | T2.1 | Of Part 4's settings only (hbar 1, euclidean, sample, unverified) is the specification's own output instrument; the rest are chosen realizations of N-L7, 6 of 32 are duplicates | PROTOCOL amendment (manager); no code |
| A2 | Medium | T2.1 | The area-law reading rule reads truncation and dissipation as physics | PROTOCOL amendment (manager); small `e3_nfa.py` change in T1.5 |
| A3 | Medium | G1 | "Cache bond" and the density cap depend on the host; PROTOCOL says 90, this host gives 64 | T1.5: pin and record |
| A4 | Medium | T2.1 | E2's reproduction check (note 2) is not exercised by the smoke run; the helper's "met" is unsupported | process: run it first, before Part 1 |
| A5 | Low | none | The text Page edge has no power beyond depth 2 to 3; the turnaround is always 1 byte, so Part 2 stage 1 spends a full read on a candidate at 5.27 bpb | PROTOCOL amendment (manager) |
| A6 | Low | none | No test that frozen scoring leaves storage untouched under `verify` or `governor` (it does) | T1.5: one test |
| A7 | Low | none | `page.py` says the Page edge is N-L5's growth rule; the code uses a sighting count (text) and a cache or entropy bond (chain) | T1.5: docstring |
| A8 | Info | none | `coherent_mix` is valid: an explicit enlarged state, a chosen realization; the docstring's wording should say so | T1.5: docstring |
| A9 | Info | none | Sinks on the text machine, doorkeeper memory, T12.3 drift, measured run times | none |

"Blocks" names the first row of the ledger that should not start before the item is closed or consciously
accepted. Items marked "none" can ride along with T1.5.

## A1. N-L7: which Part 4 setting is the machine's own instrument (High)

**Judged against.**
- T7.1 (S): one propagator, "not an unrelated answer-scoring system".
- T10.1 (S): the output instrument `J_tau`; Feynman path sums "are not an additional stochastic answer generator".
- T14 step 7 (S): "do not convert the complex phases into unrelated positive path costs".
- T15.3 (S): "Replace complex amplitudes by softmax or shortest-path costs: Not done".

**Evidence.** `src/vam/nfa/collapse.py:131-155` (`first_step`) and `:156-186` (`generate`); `tests/test_nfa_collapse.py:42,49`.

| Setting (PROTOCOL Part 4) | What it computes | Judged |
|---|---|---|
| hbar 1, euclidean, `sample`, unverified | `Z_b = sum over paths with first bit b of p(path)`, which marginalizes to the machine's own `q_b`; then a draw | **The instrument** (D: sum of products of its own predictions; `test_hbar_one_is_the_machines_own_prediction`; `part1_l7` checks `|Z - p0| < 1e-10`). |
| same, `collapse` | the larger of `q_0`, `q_1` | Greedy decoding: a decision rule outside the machine. Not the instrument. |
| hbar 0 (any form, mode) | the first bit of the minimum-`S` path over 8 bits (`first_step`, `hbar == 0.0` branch, line 143) | Shortest path over `-ln p`: the "shortest-path costs" T15.3 declines. C, as N-L7's own stationary-phase limit. |
| euclidean, 0 < hbar < 1 | weights `p^(1/hbar)` over paths | Tempering of probabilities: a positive-cost reweighting (T14 step 7, T15.3). C. |
| feynman, hbar > 0 | amplitudes `sqrt(p) e^{i S/hbar}`, `S = -ln p`, `Z = |A|^2` (line 148) | The phase is built from the probability. `S = -ln p` is the integral of the observation generator along the *chain's* event flow (`nfa/flow.py`, tested); on the nested text machine that is an analogy, and `e^{iS/hbar}` is not the phase of any propagator in the machine. At hbar = 1 `Z_b` is not `q_b` (the phases between paths do not cancel to the marginal). C, and not T7.1's path sum. |
| `verified=True` | only sectors with `ev > protect` drive the readout (`collapse.py:102`) | An evidence threshold on the readout: C, N-L6's "radiate completed invariants"; it is a selection rule and must be labelled so. |

Duplicates: at hbar 0, `Z` is one-hot, so `form` is ignored and `mode` has no effect (the `sample` draw of a
one-hot `Z` is deterministic). The 4 combinations of (form, mode) give identical output for each `verified`:
6 of the 32 settings are duplicates. Reporting them as 32 independent settings would overstate the grid.

**Why High.** The handoff's rule is that no scoring, path cost, beam or selection rule is added to the machine,
and Part 4's grid is where all of them appear, under the name of N-L7. Nothing there is a bug, but a result
table that does not label which row is the machine would let the writing track be judged on rows that are not
VAM.

**Required (manager, before the run, dated):** a PROTOCOL amendment to Part 4 saying
1. the specification's own instrument is (hbar 1, euclidean, sample, unverified); it is the row to read for
   "what the machine writes";
2. every other row is N-L7's realization (C), reported as observation, and none is called better by the table;
3. hbar-0 rows are reported once (the 6 duplicates are marked, not counted);
4. `RESULTS.md` (via `report.py`) carries the label per row. This label can be added by the helper after the
   run, from `hbar`, `form`, `mode` and `verified`, without a rerun.

No code change is needed. The sentence on the `S = -ln p` analogy belongs in the amendment.

## A2. The area-law reading rule (Medium)

**Judged against.** N-L5 (S): the bond is governed by `S(rho_A) <= |dA|/4`. PROTOCOL: "the largest operator
entanglement of any cut after 40 bits for 4, 6, 8 and 10 cells ... bounded in the chain length means the area
law holds".

**Evidence.** `experiments/e3_nfa.py:281-304` (`part1_area_law`) run in full here (about 2 min):

```
cells engine               max_cut_entropy_bits  largest_bond  discarded (Hilbert-Schmidt weight)
4  plain_bond64            1.549                 32            2.5e-28
4  governed_cache_bond     0.832                 16            6.0e-13
6  plain_bond64            1.646                 64            0.112
6  governed_cache_bond     0.662                 64            2.0e-08
8  plain_bond64            2.817                 64            0.959
8  governed_cache_bond     0.409                 64            2.4e-06
10 plain_bond64            3.473                 64            2.349
10 governed_cache_bond     0.301                 64            8.7e-06
```

1. The plain engine at bond 64 discards 0.11, 0.96 and 2.35 of the Hilbert-Schmidt weight at 6, 8 and 10 cells
   (a value above 1 means the engine has thrown away more than the state it holds). Its entanglement numbers are
   a property of the truncation, not of the state. The rule "bounded in chain length" cannot be read from
   them in either direction.
2. A truncated engine is bounded by construction: a bond of 64 caps the entropy at `log2 64 = 6` bits at every
   length. Only rows whose discarded weight is negligible say anything.
3. The governed engine's small, falling entropies are produced by its sinks (they dephase everything outside
   the context). That shows the governor works, not that the plain dynamics obey an area law. Its
   discarded weight is small (at most 9e-6), so those rows are readable as what they are.
4. The measured quantity is the operator entanglement of `vec rho` (bond entropy of the MPO), not N-L5's
   `S(rho_A)`, the von Neumann entropy of a block. They are different quantities; the first is not bounded
   by the second.
5. Four sizes cannot tell a bound from slow logarithmic growth.

**Required (manager, dated before the run), reading rule:**
- Only rows with `discarded <= 1e-6` are read for the area law; rows above are reported and marked "truncation-bound, not read".
- The plain engine's rows at 6 to 10 cells are therefore marked as not readable, as they stand.
- The governed result is stated as "entanglement kept bounded by the sinks", a result about N-L6 acting on
  N-L5, not a measurement of the plain chain.
- The quantity is named "operator entanglement of the stored density", with S(rho_A) not claimed.
- A cheap addition (T1.5, optional): record `discarded` next to each row's entropy in the report.

## A3. Host-dependent cache constants (Medium, blocks G1)

**Evidence.** `src/vam/nfa/bounds.py:47,100-117,173` reads the caches from `/sys/devices/system/cpu/cpu0/cache`;
`governor.py:96` takes `max_bond = bounds.max_mpo_bond()` when none is given; `part1_chain_blankets`,
`part1_area_law` and `part1_chain_fidelity` call it that way ("governed_cache_bond", `always=True`).

```
$ python -c "from vam.nfa import bounds; print(bounds.max_mpo_bond(), bounds.max_density_dim())"
64 45          # this host: lscpu L2 = 4 MiB over 4 instances
```

PROTOCOL and the commit messages name "cache bond 90" (a 2 MiB L2). This host gives 64, and the helper's smoke
run shows a largest bond of 64 for the governed cache-bond engines, so its host is likely 64 too. The
provenance block of `e3.json` does not record the value. Part 1 rows named "cache bond" therefore change with
the machine that runs them, and PROTOCOL's "90" is not what is run.

**Required (T1.5):** pass the PROTOCOL value explicitly (`max_bond=90` where PROTOCOL says 90), and write
`bounds.max_mpo_bond()`, `bounds.max_density_dim()` and the detected cache sizes into `provenance` as
information. The manager decides whether 64 or 90 is the PROTOCOL value (it must be one number, stated).

## A4. E2's reproduction check was not exercised (Medium, blocks T2.1)

**Evidence.**
- `experiments/e3_nfa.py:424`: `reproduced = smoke or abs(start_bpb - e2_bpb) < 1e-9`. In smoke mode the check is
  skipped by construction, and the smoke start value is a 256 KiB reading, not E2's.
- The helper's RUNLOG says "`reproduced` = True (engineering note 2 is met in this environment)". The statement
  is unsupported: `True` comes from the `smoke or` clause.
- The check runs in the full run only at the top of Part 2, after the whole of Part 1.

**What was verified instead (the code side of the risk).** Default options of the current `NestedMachine` are
bit-identical to `45d1209` (the E2 code), including after evictions:

```
$ python cmp.py          # old nested.py from 45d1209 vs current, E2 depths, counting law, prior 4, 118 KB of text
table_bits=10  identical  max|dq| 0.0  evictions 3955686 vs 3955686  tables equal True   frozen reading equal True
table_bits=22  identical  max|dq| 0.0  evictions 83750 vs 83750      tables equal True   frozen reading equal True
```

`e2_curriculum.make` changed only by adding optional keys (`git diff 45d1209..6fbfdc7 -- experiments/e2_curriculum.py`);
for an E2 spec the machine is the same object. The numba loop has no `fastmath`. So a mismatch could come only
from the environment, which the pins address. What remains unverified is the value on full A-train: this container has no corpus.

**Is forcing `reproduced = True` in smoke acceptable?** Yes as a smoke-mode convenience (a 256 KiB reading cannot equal E2's value), no as evidence: it must never be reported as note 2 being met.

**Required (process, no code):** before the full run, the helper runs one command that scores the start
specification (full `A_train`, frozen `A_validation`) and compares it with `2.0316126330797104` to 1e-9. The
manager decides (and amends PROTOCOL) what happens on a mismatch before Part 1 starts. The cost is one reading
(Part 2's first row, which is then reused through `--resume`).

## A5. The text Page edge and the turnaround (Low)

**Judged against.** N-L2 (S): the Page limit gives the number of vectors "before hitting the Page turnaround".
PROTOCOL N-L2(b): "the depth with the largest rank is the turnaround ... used in Part 2".

**Evidence (4 MiB of real English prose, here Newton's *Opticks* and the vim manual, as no PG-19 is in this container;
`page.depth_profile`, depths 1 to 7 bytes; rank and cutoff):**

```
real English         d1 3 (0.833)  d2 2 (0.995)  d3 0 (1.02)  d4..d7 0 (1.10)    turnaround 1
byte-shuffled null   d1 1          d2 1          d3 1         d4..d7 0          turnaround 1
iid, same unigram    d1 1          d2 1          d3 1         d4 1  d5..d7 0     turnaround 1
```

1. Validity (the check asked for): on text with no structure the rank never exceeds 1 (no false directions), so
   the split-half edge is not too low. On the spec text (118 KB, 29 blocks) the profile degenerates to rank 0 and
   is unusable at that size.
2. Power: from depth 3 on, the cutoff exceeds 1, the value of the trivial direction, so even "structure
   exists" is not detected (rank 0). The cutoff grows with the number of observed contexts `m` (the
   Marchenko-Pastur edge `(sqrt m + sqrt n)/sqrt N`), so the ranks at depth 3 to 7 measure how rare the contexts are,
   not how much structure there is. Rank 0 is outside the method's meaning.
3. The turnaround is "the depth of the largest rank". With ranks 3, 2, 0, ... it is 1 byte. The spec text's
   "Page turnaround" is a property of an entropy curve (half the system), not of a sample-size-dependent noise edge.
4. The consequence in Part 2 stage 1: `cut = depths <= 8 x turnaround` is `(8,)`, one level. Measured here on 512 KiB of the same
   English: depths `(8,)` give 5.27 bpb, `(8,16)` 3.04, `(8,16,24)` 2.30, the default 2.00. Stage 1 will reject
   the candidate on A-validation, as the protocol intends, but only after one full A-train read (about 40 MiB
   of bits) spent on a configuration that cannot win. The comparison is not wrong; it is wasted.

Corpus dependence: the numbers above are from this container's English text; PG-19 A-train may differ in detail, but the
mechanism (cutoff above 1 from depth 3) depends on the stream length and the number of contexts, not on the
topic. The E3 run records the real profile; this finding predicts turnaround = 1 and candidate `(8,)`.

**Required (manager, dated before the run):** either (a) drop the cut candidate when the turnaround is below 3
bytes (state the rule, and report the profile as a measurement that did not constrain the choice), or (b) keep it and
accept the cost. Not blocking. N-L2 also gives a result for text (the profile) that is reported either way.

Related: in the helper's smoke run `N_L2.criteria.a_known_sources` is **False**: `S4_echo4`, window 8 gives rank 17 for 16
states at 40,000 bits (weakest direction 0.24, cutoff 0.189). At the full 400,000 bits the criterion passes
(`part1_l2(..., smoke=False)`: `{'a_known_sources': True}`, run here). The smoke value reflects the small sample, not a defect; the smoke run's pass
condition is "completes". Verdict: at the full size the criterion resolves (True, run here on the 400,000-bit sources).

## A6. Frozen scoring and storage: verified, one test missing (Low)

**Brief check:** with `verify` or the governor on, frozen scoring must not admit, evict or drain sectors.

**Evidence.** Compiled loop: every doorkeeper update, admission, drain and occupancy change is inside the
`if learning:` block (`nested.py`, `_make_loop`); `NestedReference.step` is gated the same way.
Measured on 400,000 bits then 200,000 frozen bits, tables of 2^11 slots (heavy eviction pressure), for default,
`verify=2`, `verify=3 + coherence=0.5`, `governor`, and `governor + verify 2 + protect 0.5 + sink 0.25`:

```
_wn, _tags, _door, _occ identical before and after frozen reading: True (all five configurations)
_stats: only the "bits read" counter (index 2) changes (400000 -> 600000); evictions, refusals, protected knocks unchanged
```

The existing `tests/test_nested.py:172` (`test_frozen_reading_changes_no_table`) checks only the default
configuration and only `_wn` and `_tags`.

**Required (T1.5):** extend that test over the five option sets above, comparing `_wn`, `_tags`, `_door`, `_occ`
and the three counters. No machine change.

## A7. Docstring: the Page edge as N-L5's growth rule (Low)

`src/vam/nfa/page.py:36` says "The same edge is the growth rule of N-L5: a bond grows only for a singular value above
it". `src/vam/nfa/governor.py:40-47` states the opposite for the chain ("Why the Page edge is not used to cut the
chain's bonds"), and the nested text machine's growth rule is a sighting count (`verify`). N-L5's "rank expansion
only upon statistically verified, non-noise invariants" is thus realized by (C) a count of sightings on text and by
(C) a cache or entropy bond on the chain; neither is a statistical test against the Page edge. **Required
(T1.5):** correct `page.py:36`, and add the label to DESIGN section 10 (manager).

## A8. `coherent_mix` (N-L3): verdict (Info, no defect)

**Judged against.** T11.4 (S): isometric nesting preserves `Tr(rho A)`. T13.1 (S): a coherent superposition of
density descriptions "requires an explicit enlarged state, such as chosen purifications plus a layer label. The
mixture `p rho_0 + (1-p) rho_1` is not itself evidence of coherent cross-layer interference."

**Evidence** (`nested.py:191-198`, compiled twin in `_make_loop`):

1. Bounds (200,000 random draws including `c = 1`, `share` in `{0, 1}`, `p0` and `own` in `{0, 1}`): output in
   `[3.8e-5, 0.99999]`, never NaN. Stronger, the output lies in the hull `[min(p0, own), max(p0, own)]` on 20,000
   draws, which gives N-L3's PROTOCOL criterion `[eps, 1 - eps]` directly whenever `p0` and `own` are (the law clamps
   both).
2. Isometric nesting: `own = p0` leaves `p0` unchanged for every `c` and `share`: max deviation 2.2e-16 over
   100,000 draws. This is the exact statement of T11.4 for this step.
3. The enlarged state is explicit. For `c = 1`, `T_b = 2 |<+|<b| Psi>|^2` with
   `|Psi> = sqrt(1-s)|L0>|h_p> + sqrt(s)|L1>|h_o>`, `|h_x> = sqrt(x_0)|0> + sqrt(x_1)|1>`; for general `c` the label
   state is `(1-c) diag + c |a><a|`. Checked numerically against `coherent_mix` on 20,000 draws: max deviation
   1.1e-15 (`T_b` unnormalized, as in the code, then divided by `T_0 + T_1`).
4. `P_0 + P_1 = [[1, 0.6], [0.6, 1]]` for `(p, o) = (0.9, 0.1)`: not the identity, which is why the code
   normalizes. That is not a defect: the readout is the trace-nonincreasing element `|+><+| (x) |b><b|`
   ("conditioned on the label being erased"), the conditioning case of T10.1, followed by normalization.
5. The amplitudes are real and the relative phase is zero by choice: C, not derived. With `c = 1` and hypotheses
   that disagree (`p0 = 0.9`, `own = 0.1`, `s = 0.5`) the readout is 0.5 for every `c`; with agreement
   (0.9 and 0.8) it moves 0.85 to 0.8536 as `c` goes 0 to 1. The effect is small by construction.

**Verdict:** valid, an explicit enlarged state, a chosen realization (C). T13.1's warning does not apply to
`c > 0`; it would apply if the label were presented as interference without the purification. **Required
(T1.5, docstring only):** the `nested.py` module docstring says "read with `P_b = |phi_b><phi_b|` ... normalized
over `b`"; add "(a conditioning step on the erased label, T10.1)", the purification, and the label C.

## A9. Remaining checks and measurements (Info)

**Sinks on the text machine.** The text sink `ev <- ev e^{-2 rate}` per knock (`nested.py:326`, `sink_factor`) is a
discrete analogue of `Re lambda = -2 rate`: it acts on a log-evidence scalar once per colliding newcomer, not on a
density per unit time. It is a chosen realization (C) and should be so labelled in DESIGN. On the chain the
sink is the real thing: `governor.py` `sink_liouvillian` has eigenvalues `0, -2 rate, -2 rate, 0` (tested), and it
is a completely positive trace-preserving map, so it fits T10.1's `Pi` slot: the event-loop form is intact.
No change to the event loop is required by any finding.

**Doorkeeper and governor semantics** (compiled equals reference, all options, counters included; 60,000 bits,
512 slots, `verify 2` and `governor + verify 2 + protect 0.5`): `max|dq| = 2.2e-16`; evictions 11621/11621, 10064/10064;
refusals 145872/145872, 145303/145303; protected knocks 0/0 and 1273/1273. The `part1_text_survival` criterion
already passes in the helper's smoke run (plain 0.0 kept; verify 2: 0.591 of 0.600; governor: 0.651 of 0.658; the function ignores `smoke`).

**Memory.** `verify > 1` or `governor` add 128 MiB (the doorkeepers, 32 MiB per hashed level; hashed levels 2 to 5):
449.5 MiB to 577.5 MiB for the E2 start configuration. Information for the run host.

**T12.3 part 2.** `read(..., drift=...)` measures `||w_after - w_before||` over all rows of `bm.w`, which includes
sectors created or replaced in the block, so "does not settle" mixes growth with change of established sectors
(the earlier handoff's note 7). Not re-examined further here; it is the manager's call whether T4 adds the
established-sector drift as a clearly labelled addition.

**Part 4 and the end-of-A state.** `writing_grid` copies the machine for every prompt and setting
(`machine.copy()`), and `copy` copies `_wn`, `_tags`, `_stats`, `_door`, `_occ`; `collapse.generate` changes only the copy's
register. `generations` copies too. The end-of-A machine is untouched; this is the reading of the code, and A6's
script shows storage is unchanged under frozen reading.

**Run times (this host, one job at a time; two parallel jobs on 4 cores made single events 30 times slower, so
the helper should not run chain parts in parallel with anything).**

| Part | Measured | Estimate for the full part |
|---|---|---|
| 5-cell untruncated sink cost (`SINK_GRID` 5 cells, 512 events here) | 0.23 to 0.29 s per event, bond 128, steady; 124 s for 512 events | 2,048 events x 2 engines: about 17 min |
| 12-cell governed chain fidelity (bond 128, 120 bits) | about 3.9 s per event steady from event 12 (52 events in 203 s); bond 32: 120 events in 41 s | about 8 min per bond-128 engine; bonds 32, 64, 90 and the entropy bond are cheaper (cap 90); the 7 engines: 30 min or less |
| 8-cell untruncated engines of `chain_blankets` (cap 4096, cutoff 0; plain with every qubit einselected, and governed at sink rate 40) | 40 events in 10.6 s and 11.6 s; bond 128, discarded weight 1.9e-27 (so "untruncated" holds) | under 1 min each; the part's four truncated engines are cheaper or equal (the smoke run's 7.6 s at 5 cells scales to minutes) |
| `part1_area_law` (4, 6, 8, 10 cells, plain and governed) | 2 min in total here | 2 min |
| Part 4, 16 writing settings in smoke (2 prompts x 8 bytes) | 50.9 s (helper) | 32 settings x 8 prompts x 48 bytes: about 40 min (scaling by bits written; not measured) |
| Part 2 | 5 rows in 16.7 s on 256 KiB (helper) | full A-train rows: not measurable here (no corpus); the helper's smoke has no basis for it |

The note-3 question for 5 cells is answered: no amendment is needed (the bond stays at 128, far below 1024, on
the test bytes. The bond growth on A-train's first bytes was not measured; I used non-English bytes for speed).

## Checks of the brief, in one place

| Check in the brief | Verdict | Where |
|---|---|---|
| `coherent_mix` bounds | holds, hull property | A8 |
| isometric nesting for every `c` (T11.4) | holds to 2.2e-16 | A8 |
| explicit enlarged state, not a mixture presented as interference (T13.1) | holds, chosen realization; docstring to say so | A8 |
| doorkeeper and governor: admission, protection, sink | compiled equals reference; sink is analogue on text, real on chain | A9 |
| frozen scoring leaves storage untouched | holds; test to extend | A6 |
| N-L7 against T7.1, T14 step 7, T15.3; `S = -ln p` as the observation-generator integral | proven on the chain (`flow.py`), analogy on text; only (hbar 1, euclidean, sample) is the instrument | A1 |
| the Page edge on text (split-half) | valid (no false directions), no power from depth 3 | A5 |
| area-law reading rule with four sizes | not readable as written | A2 |
| E3 runtime and reproduction (notes 2 and 3) | note 2 unexercised, note 3 acceptable at 5 cells, a new host-dependence found | A3, A4, A9 |
| Part 4 copies leave the end-of-A state untouched | holds | A9 |

## What the manager must write before T2.1 (dated before the run)

1. Part 4 amendment (A1).
2. Area-law reading rule (A2).
3. The cache bond: one stated number (A3).
4. What happens if E2's start value does not reproduce (A4).
5. Optional: Part 2 stage 1 rule for the depth-cut candidate (A5).

T1.5 (engineer): A3 (pin and record), A6 (test), A7 and A8 (docstrings), and optionally the A2 `discarded` column. Each has its
module's tests and ruff; `e3_nfa.py` is touched only for A3 and A2, before T2.1 starts.
