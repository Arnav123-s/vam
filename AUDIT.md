# Independent audits

Each audit is a read-only review of the mathematics and the code against the two specification
documents (T and F). The reviewer runs in a separate session, writes nothing in the repository, and
reports findings ranked by severity. The reviewer is a separate Claude Opus 5.5 session, the owner's
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

## Audit 2 (2026-10-20, of `45d1209..6fbfdc7`; report `audits/audit2.md`, commit `a5118ed`)

Answers by the manager. Protocol changes are PROTOCOL E3 Amendment 1, dated before any full run.

| Finding | Severity | Answer |
|---|---|---|
| A1 Part 4 instrument | High | Accepted. Amendment 1 (A1) labels every row; duplicates at `hbar = 0` reported once; `report.py` label (T1.5). |
| A2 area-law reading rule | Medium | Accepted. Only rows with discarded weight at most 1e-6 are read; named operator entanglement; `discarded` column (T1.5). |
| A3 host-dependent cache bond | Medium | Accepted. The cache bond is 90, passed explicitly; detected values recorded in provenance (T1.5). |
| A4 reproduction check not exercised | Medium | Accepted. Recomputed before the run (T1.6); a mismatch stops the run until its cause is stated. |
| A5 text Page edge, turnaround | Low | Noted. Stage 1 kept as written; the depth-cut row is reported as a negative if it loses. |
| A6 frozen scoring test | Low | Accepted. Test over the option sets (T1.5). |
| A7 page.py docstring | Low | Accepted (T1.5). |
| A8 coherent_mix | Info | No defect; docstring clarified (T1.5). |
| A9 remaining checks | Info | Noted. |

## Checkpoint C2 (2026-10-20, `audits/c2.md`, commit `ef2f633`)

Answers by the manager (PROTOCOL E3 Amendment 4). R1 and R3 are reported as measured. The four gaps:
N-L2 on text and the chain's late sinks were measured again as T4.2 and T4.3 (added after the result,
`results/e3_t4.json`); the inert halt is replaced in E4's design by a raised admission bar (DESIGN section 11), but E4's Part 2 chose the
per-level governor, so the chosen machine does not use it; the cost of sinks is reported for E3, and E4's chosen
machine runs without sinks (the sink-rate-0 row won on A-validation). The Part 2 grid extension was not approved (tuning after the
test split was read) and remains an open question.

