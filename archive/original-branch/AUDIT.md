# Independent audits

Each audit is a read-only review of the mathematics and the code against the specification documents
(T and F; from Audit 2 on, also N). The reviewer runs in a separate session, writes nothing in the repository, and
reports findings ranked by severity. The reviewer is a separate session of the most capable model, the owner's
choice (2026-10-18). Every finding gets a response here: fixed (with the test that now covers it), corrected
(a claim reworded to match the evidence), or noted.

## Audit 1 (2026-10-18, at commit de3fd8e)

The reviewer re-ran the suite (251 tests passed) and the verifier (26/26 headline results), and
checked by its own scripts the core mathematics, every exact number of T, the engine agreement, the
window floors, the oracles and the KT baseline; all held. It found no critical problem, three major,
three minor and three notes. All were answered in commit aa59cc9 and the reruns that followed it.

| # | Severity | Finding | Response |
|---|---|---|---|
| 1 | major | The "consolidating" law is not the consolidation of T12.2; it is a count-dependent friction that makes the weight a running average (frequency counting), and every headline criterion was judged under it | Corrected and logged as decision D10. The literal T12.1 law is tested to give T12.2's geometric transient exactly; reports now show both laws, and E2 runs its arms for the best literal-law machine |
| 2 | major | While reading, nothing was emitted or re-entered, so T14 steps 11-12 never ran on data, and the README said they did | Fixed: an active-port qubit; after every external bit the machine emits with its own prediction, conditions once and re-enters the bit. At zero mass this provably changes no prediction; the mass sweep now runs the full loop |
| 3 | major | The canalization algebra was closed over the maps at one weight, so it did not hold for a learning machine (it collapses to the scalars at `w = 0`) | Fixed: `DenseMaps.canalization()` seeds the algebra with the sector projectors, so it holds at every weight and keeps what plasticity reads; the canalized learning machine is tested to predict and learn exactly as the plain one |
| 4 | minor | A new nested sector copied its parent's raw relation, which moves the prediction at creation (against T11.4) | Fixed: its own prediction now equals the coarser readout, so creation changes nothing (tested, with the old rule as a negative control) |
| 5 | minor | DESIGN claimed mutation checks, 10^5-10^6-event storage tests, long-run drift checks, an AUDIT.md, and a tensor network carrying VAM-1, none of which existed | Corrected or added: negative controls listed as they exist; storage over 10^6 events and positivity over 20,000 events tested; this file; the tensor-network claim reworded. The engine now exists (`vam1_tn.py`): exact when untruncated, and with mass its truncation is shown not to be faithful. A second engine stores the density as an MPO (`vam1_mpo.py`); its truncation is faithful for the fully einselected chain and approximate (about 1e-3 at 12 cells) with context einselection, measured (DESIGN section 5) |
| 6 | minor | The committed receipt and the README status were out of date | Fixed: receipt regenerated at the current commit; README status rewritten |
| 7 | note | Every massless number tests a classical context model | Stated next to the E1 tables and in DESIGN |
| 8 | note | E1 criterion 1 is two-sided as worded | Noted in the protocol; no conclusion depends on it |
| 9 | note | E2 risks: other-language textbooks, repeated editions reaching the unseen books, a weak control for T12.3 part 1 | Repeated books are now left out before books are held out; the catalog's language field must be exactly `en`; the weakness of part 1 is stated in the protocol |

### Details

**1. The learning law.** The reviewer tracked the transient `|w_n − target| / |w_0 − target|` under one
repeated lesson. The literal law with `κ dt/γ = ln 2` gives 1/2, 1/4, 1/32, 1/1024 after 1, 2, 5, 10
repetitions, exactly T12.2's `r^n`. The count-dependent law gives `exp(−H_n)` (0.368, 0.223, 0.102,
0.053), a harmonic decay. Both facts are now tests (`tests/test_vam1.py`). The E1 criteria were
registered for the count-dependent law and stay as written; `RESULTS.md` adds, without a criterion,
that the literal law would miss criterion 1 (it settles up to about 0.025 above the floor, and beats the
floor by up to about 0.10 on the delayed echo, because its weights carry recent history).

**2. Emission and re-entry.** T14 step 11 emits the first output and step 12 re-enters it "before moving
to the next external bit"; T15.3 lists "update the machine only after a complete answer" as a
reinterpretation that was not done. The new active port is one qubit at the right boundary; its content
enters the chain as the left-mover `L_{c−1}` at the next transport. The two ports are distinct, as
T10.3 requires. Tests: the emission probabilities equal the machine's own prediction of the next bit;
at zero mass two runs with different emissions give identical predictions and weights; with mass the
emitted bit changes later predictions; the weights never depend on the emitted bit; the pointer engine
equals the dense engine emission for emission; the bit kernel equals the dense engine with the full loop.

**3. Canalization while learning.** Every weight-dependent factor is `O_b(w) = Σ_s sqrt(π_s(b)) P_s`.
Once the `P_s` are in the algebra, its dual maps the algebra into itself for every `w`, so seeding with
the `P_s` and closing over the weight-independent maps (injection, evolution, re-entry) gives one lock
valid at all weights. Tests: dimension `2^k` at zero mass for `k = 1, 2`; retention defect below 1e-10 at
`w = 0`, equal weights and random weights; the canalized learning machine matches the plain one on the
alternating stream; a lock closed at `w = 0` alone has dimension 1 and a retention defect above 1e-3 at
learned weights (the reviewer measured 0.0296).

**4. Nested creation.** With the reviewer's example the old rule moved the prediction from 0.769262
to 0.757061 at creation. The new sector's relation is `w = (2 p − 1)/(1 − 2ε)` with `p` the coarser
readout, and evidence 0, so its mixture share is 1/2 of a value equal to the readout: the prediction is
unchanged to rounding.

### Results before and after the fixes

E1 and E1b were rerun from clean commits after the fixes (E1 at aa59cc9, E1b at 5520ea0). Numbers are
cross-entropy in bits per bit over the last quarter of each stream (lower is better).

**E1 context sweep.** Every row is identical to the last digit, as the zero-mass equivalence requires:
at zero mass the emitted and re-entered bits cannot change a prediction.

**E1 mass sweep** (delayed echo; the oracle is about 0.29 and the window floor 1.0). Before the fixes the
machine read without emitting; now every external bit is followed by an emission and a re-entry.

| Source | θ | before | after |
|---|---|---|---|
| S4 d=6 | 0 | 1.0008 | 1.0008 |
| S4 d=6 | 0.2 | 1.0008 | 1.0004 |
| S4 d=6 | 0.4 | 0.9937 | 0.9925 |
| S4 d=6 | 0.8 | 0.9010 | 0.9318 |
| S4 d=9 | 0 | 1.0001 | 1.0001 |
| S4 d=9 | 0.2 | 1.0003 | 1.0003 |
| S4 d=9 | 0.4 | 1.0018 | 1.0003 |
| S4 d=9 | 0.8 | 0.9789 | 0.9843 |

The zero-mass rows agree exactly. With mass, the re-entered bits now travel through the chain with the
external ones, which dilutes the delayed signal: at θ = 0.8 the machine still beats the window floor on
both delays, by a little less than before. The conclusion of E1 does not change.

**E1b nested machine** (last quarter; N16 nests contexts up to 16 bits deep, N64 up to 64).

| Source | oracle | N16 before | N16 after | N64 before | N64 after |
|---|---|---|---|---|---|
| S1 order-2 Markov bits | 0.6444 | 0.6465 | 0.6464 | 0.6765 | 0.6755 |
| S2 T2.2 incomplete codes | 0.7961 | 0.7965 | 0.7965 | 0.8365 | 0.8357 |
| S3 order-1 bytes over 16 letters | 0.3712 | 0.3805 | 0.3805 | 0.3920 | 0.3914 |
| S4 delayed echo d=6 | 0.2864 | 0.2868 | 0.2868 | 0.3059 | 0.3049 |
| S4 delayed echo d=9 | 0.2865 | 0.2879 | 0.2879 | 0.3121 | 0.3108 |

A new distinction now starts as the coarser readout instead of a copy of its parent's raw relation, so
creating it no longer moves the prediction. Every row improves or stays equal; the gain is largest for N64,
whose deep levels keep creating new distinctions and fill their tables (about 0.8 to 1.3 million
evictions per run on the delayed echo), while N16 never evicts.

## Audit 2 (2026-10-22, the nested fractal architecture, commits bfc1e1b to 859974c)

The reviewer read every N layer (`src/vam/nfa/`, the N options of `nested.py`), the E3 script and
report generator against N and the E3 protocol, while E3 was running. It re-ran the layer tests (all
passed), checked the compiled text machine against `NestedReference` line by line, checked E3's grids,
the tie rule and the absence of leakage against the protocol, and ran the pre-N machine (bfc1e1b's own
package) against the current one. One bug changed a result, and it had already been found and fixed in
859974c; the rest are a vacuous check, a false bound in DESIGN, labels and notes. The responses are in
cef6d39 (code and tests) and the documentation commit that follows it.

| # | Severity | Finding | Response |
|---|---|---|---|
| 1 | major (changed a result; already fixed) | N-L2 on text: one-off byte pairs whiten to entries of 1 in the split-half difference, the edge rises to 1 and every rank is 0 | Fixed in 859974c (verified support, min count 8), confirmed by the reviewer. Its two gaps are answered: the support is now trimmed to a fixed point (rows and columns are re-counted inside the support until neither changes; test `test_verified_support_is_a_fixed_point`, which fails on the old code). At min count 8 the profile is unchanged (ranks 20, 40, 54, 58, 58, 61, 60, turnaround 6); at min count 1 it becomes 7, 19, 57, 61, 62, 62, 55 instead of 7, 0, 0, 57, ... The fragility of the turnaround (the support leaves out up to 49% of positions at 7 bytes) is stated in the report, with the profiles at min counts 1 and 4 beside it |
| 2 | minor (vacuous check) | The loss-integral check started from the basis state, where the filter does not move, so it was true by construction | Fixed: E3 and the test now start after three reading events (a mixed state), and the test asserts the state moves along the flow and that the integral differs from Tr(K R(0)). Measured: 2.9e-11 (criterion 1e-10). Criterion (b) already exercised the identity on non-trivial states |
| 3 | minor (false claim) | DESIGN's bound "T_b ≥ ε(√r + √s)²" is false for c < 1 (c = 0, r = s = ½, p = o = ε gives ε) | Corrected in DESIGN 10.1: T_b ≥ εK with K = (1 − c) + c(√r + √s)², and T_0 + T_1 ≤ K by Cauchy-Schwarz. The result (readout in [ε, 1 − ε]) and E3's grid check stand |
| 4 | minor (labels) | The report printed one commit for a resumed run, the 6fbfdc7 parts were stamped afterwards, and smoke runs printed "pass" for the reproduction check | Fixed: the report prints each part's commit and time, marks stamps added after the run, and lists the code files changed between those commits (from git). Part 2 now records its own commit. A smoke run records the reproduction as not checked |
| 5 | note | At hbar = 0 the form and the mode cannot change the step, so 8 of the 32 writing settings give 2 distinct outputs | Fixed in the report: identical settings are grouped and said to be one setting, and their outputs are printed once |
| 6 | note | `test_defaults_leave_the_machine_unchanged` compared the new code with itself | Fixed: the test now checks the current machine against numbers produced by the pre-N commit's own package (both laws, table bits 8 and 22, 60,000 bits, a frozen copy; learning and frozen bits per bit, the last prediction, evictions and the first eviction). They agree bit for bit; the reviewer separately found E2's 2.0316126330797104 reproduced exactly by E3 |
| 7 | note | The default doorkeeper takes 8 GiB per level at table bits 30; `test_measured_throughput` fails under background load; the smoke partial predates `seconds_by_engine` | The doorkeeper is two 4-byte buckets per slot, about a quarter of the table's own memory at any size, and `nbytes()` declares it, so it is kept. The throughput test measures the machine it runs on and is run alone for the receipt. The report reads old partial files without that key |

The reviewer also proved the sink finding of E3 (DESIGN 10.2): predictions, emissions and learning
read only the diagonal of ρ, and a coherence can reach the diagonal only through the coin in one cell
m, from differences that arrived at R_{m−1} and L_{m+1}. Context R sites are dephased exactly
(g = ∞), R_0 and L_{c−1} are refilled with classical bits, so this needs k + 1 ≤ m ≤ c − 2: impossible
when at most two cells lie outside the context (c − k ≤ 2), where interior sinks of any rate change no
prediction, emission or weight. With three or more such cells, or finite g, they can, as measured. This
holds for the untruncated engine; with truncation the sinks change the bonds and so the truncation error.

