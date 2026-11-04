# VAM-1 experiment protocol (written before any run)

This protocol was fixed on 2026-10-18, before any VAM-1 experiment ran. Every run uses fixed seeds and
saves its outputs as JSON under `experiments/results/`. `RESULTS.md` is generated from those files.
Nothing is tuned on a test split.

All losses are prequential: each bit is scored by the probability the machine assigned to it **before**
it learned from that bit. Bits per byte (bpb) is the sum of `-log2 q_b` over a byte's 8 bits.

## E1. Known sources (the test sources)

Purpose: check that VAM-1 learns what it should, on sources whose best possible loss is known exactly.
Code: `src/vam/sources.py` (sources), `src/vam/baselines.py` (reference learner),
`experiments/e1_known_sources.py` (the run). The sources and their exact references are tested in
`tests/test_sources.py`.

| Source | Exact references |
|---|---|
| S1: order-2 Markov bits, `P(0 \| s) = (0.9, 0.2, 0.6, 0.35)` | entropy rate; window floor |
| S2: the T2.2 incomplete codes (A=0, B=01, C=10, probability 1/3 each, no delimiters) | forward filter, which equals the T2.2 density machine; window floor |
| S3: order-1 *byte* source over 16 letters (`a`..`p`), random transition table (Dirichlet 0.5, seed 1234) | entropy rate; window floor (exact enumeration over the 8 byte phases) |
| S4: delayed echo `x_t = x_{t-d} XOR noise(0.05)`, d = 6 and d = 9 | `h(0.05)` per bit; window floor is 1 bit when `k < d` |

Three references are recorded for every source, all on the same stream:
- **Oracle loss.** The exact source model's prequential loss.
- **Window floor.** `H(next bit | last k bits)` of the stationary process. No fixed rule from the last
  k bits to a probability can beat it on average. A learner whose weights keep moving can, because its
  weights then carry memory of recent history.
- **KT.** The sequential Bayesian count predictor (Krichevsky-Trofimov) with the same k-bit context. It
  sees exactly what a massless VAM-1 sees, so the gap to it measures the learning law alone.

Run:
- Every source gives 2^22 bits (seeded).
- Massless VAM-1 (bit kernel) is run with k ∈ {2, 4, 8, 12, 16} under both laws: constant friction
  (dt = 0.05) and consolidating friction (dt = 1).
- Loss is reported over the whole stream and over its last quarter.
- **Mass sweep on S4.** The pointer engine with c = 6 cells and k = 4, θ ∈ {0, 0.2, 0.4, 0.8},
  consolidating law, over 2^18 bits. With mass, a bit can turn into a left-mover, travel back, and
  re-enter the context about `2j + 1` events later, so mass is the one mechanism here that could see
  past the window.

Pass criteria, fixed now:
1. Consolidating law, every source, k ≤ 8: the last-quarter loss is within 0.01 bits per bit of the
   window floor. This shows the law converges to the right conditional probabilities.
2. Consolidating law, every source and every k: the last-quarter loss is within 0.01 bits per bit of
   KT's last-quarter loss, or below it. This shows the law learns about as efficiently as Bayesian counting.
3. The constant law and the mass sweep are reported as measured, with no criterion.

Revision note: this section was revised on 2026-10-18 before any experiment ran.
- S3 changed from order-3 to order-1 bytes. An order-3 byte source needs a 31-bit window, beyond
  every k in the sweep, so it could only have tested the representational limit.
- The window floor and the KT reference were added, so that representation and learning are measured
  separately.
- A smoke run at 2^16 bits (`results/e1_small.json`) checked the code before the full run. No criterion
  changed after it. One sentence did: it showed the constant law beating the window floor on S4, so
  the floor is now described as a bound for fixed rules, not for learners whose weights keep moving.

Post-audit revision (2026-10-18, after E1 and E1b ran and before any E2 run). The independent audit
(`AUDIT.md`) found three things that bear on E1. The criteria above are unchanged.
- The reader did not emit and re-enter an output after each bit, which T14 steps 11-12 require. VAM-1
  now does. At zero mass this changes no prediction and no weight (tested), so every window row is
  unchanged by construction. The mass sweep runs the full loop now, so its numbers change. E1 is rerun
  in full from the fixed commit; the first run's numbers are kept in `AUDIT.md`.
- The "consolidating" law is not the consolidation of T12.2. It is a count-dependent friction that T
  does not state (decision D10), and it makes the weight a running average, which is frequency
  counting. Criteria 1 and 2 were registered for it and stay as written. Read with that in mind,
  criterion 2 is close to true by construction. `RESULTS.md` also reports, without a criterion, how the
  literal T12.1 law (constant friction, in every table from the start) compares with the floor.
- Criterion 1 is two-sided as worded ("within 0.01"). A learner whose weights keep moving can beat the
  floor; no reported conclusion depends on this.

## E1b. Nested levels on the known sources (added 2026-10-18, before running it or E2)

Step 14 of the master loop (`src/vam/nested.py`, decision D9) adds nested context levels:
- New fine sectors start as copies of their coarse parents.
- The readout mixes the levels by the Bayesian evidence that each finer distinction predicts better.

Run on the E1 streams (same seeds, 2^22 bits), consolidating law, prior 0:
- **N16.** Levels of 2, 4, 8, 12 and 16 bits, all stored directly.
- **N64.** Levels of 8, 16, 24, 32, 48 and 64 bits, with 2^22-slot tables.

Criterion:
- N16's last-quarter loss is within 0.01 bits per bit of the best fixed window of E1 on that source.
  This means the nesting costs little where no deep structure exists.
- N64 is reported as measured. It is the configuration meant for text, and on short-memory sources its
  deep levels are expected to cost something.

Disclosure: development runs with another seed (S1 to S3 at 2^18 and 2^21 bits, seed 1, and 3 MiB of
Python source text) shaped the readout rule, which moved from a hard switch to the Bayesian mixture.
They also showed that a new-sector prior helps stationary sources but hurts text. The criterion above
was written after those runs and before this run.

Post-audit revision (2026-10-18, before any E2 run). The audit found that a new fine sector copied its
parent's raw relation, which moves the prediction at the moment the sector is created, against T11.4.
A new sector's own prediction now equals the coarser readout, so creating it changes no prediction
(tested). E1b is rerun from the fixed commit with the criterion unchanged; the first run's numbers are
kept in `AUDIT.md`.

## E2. Textbook curriculum (owner's instruction of 2026-10-18)

The owner's instruction: real data, structured textbooks first, then unstructured data after a good
amount of structured data, plus every test source above.

**Data** (selection fixed by rule in `experiments/fetch_gutenberg.py`; nothing is picked by hand)
- **Stage A, structured.** English Project Gutenberg texts whose Library of Congress subjects include
  `Textbooks`, which is the cataloguers' own label for a textbook. Books for learning a foreign language
  are excluded, because they mix languages.
  - Reading order: by Library of Congress class, then by ebook number.
  - The stage stops at 64 MiB of downloaded text.
- **Stage B, unstructured.** English fiction from English and American literature (classes PR and PS)
  that is not a textbook. Reading order is seeded random; the stage stops at 64 MiB.
- **Data source, revised 2026-10-18 before any E2 run.** This cloud machine cannot reach gutenberg.org,
  and the owner, away from their laptop, said to continue without it. The rules above need the Gutenberg
  catalog's subject labels, so E2 runs on PG-19 instead (`experiments/fetch_pg19.py`): DeepMind's fixed
  snapshot of 28,752 Project Gutenberg books published before 1919, from its public bucket (dataset
  Apache-2.0, books in the public domain). PG-19 has titles but no subject labels, so the stages are:
  - **Stage A, structured:** the title names an instructional form (text-book, elements of, elementary,
    primer, introduction to, manual, handbook, lessons, treatise, principles of, ...) or a school subject
    (arithmetic, algebra, geometry, physics, chemistry, botany, grammar, logic, ...), and the text is
    didactic: at least 10 words per 10,000 among exercise, lesson, fig., example, problem, definition,
    theorem, proposition and experiment. Titles alone mislead (*Cobb's Anatomy* is a humour book), so
    the text decides. The threshold was set before any E2 run from a survey of the 394 title matches and
    150 random other books: it sits above the 95th percentile of the random books (8.2 per 10,000), and
    no random book exceeded 15. Books for learning another language are left out. Reading order: titles
    naming a subject first, then the rest, each by ebook number. The stage stops at 64 MiB or when the
    candidates run out (about 50 MiB qualified in the survey).
  - **Stage B, unstructured:** every book whose title matches neither rule, in a seeded random order,
    up to 64 MiB.
  - **English:** at least 12% of a book's words must be among twenty common English function words.
  - The repeat, held-out, cleaning and split rules are unchanged. The Gutenberg-catalog selection above
    stays available (`fetch_gutenberg.py`) as a replication once the owner can run it.
- **Cleaning** (`vam/corpus.py`):
  - Text is UTF-8, line endings become `\n`, and Gutenberg's licence header and footer are removed.
  - Nothing else changes; mathematics stays as the source writes it.
  - There is no tokenization or segmentation: the machine reads raw bytes as bits (F-Part 10).
  - Every cleaned file's SHA-256 is recorded in `MANIFEST.json`.
- **Repeats.** A book that repeats an earlier one, in either stage, is left out, so no edition of a
  training book can reach the unseen books. A repeat has the same cleaned-text SHA-256 as an earlier
  book, or the same title and authors once lowercased with punctuation and spacing removed
  (`experiments/build_corpus.py`; added after the audit, before any E2 run).
- **Held out.** Every tenth remaining book of each stage is held out and never trained on.
- **Reference yardstick (not a criterion).** lzma with a 64 MiB dictionary, measured as the extra
  compressed size of a test split after its training split. This is a familiar compressor's number
  for the same text, so the bpb values have a scale.

**Splits**
- Each book is split by position: the first 80% is train, the next 10% is validation and the last 10% is test.
- Held-out books are never trained on at all (unseen books).

**Arms** (one machine state each; no resets inside an arm)
1. **A then B.** The machine reads all of A-train, then B-train. This is the owner's curriculum.
2. **B only.** The machine reads B-train with no prior experience.
3. **Shuffled A then B.** Same as arm 1, but A-train is sentence-shuffled within each book. This control
   keeps the local statistics and destroys the textbook order.
4. **A only.** The machine reads A-train only, for reference.

**Configuration**
- Configurations are chosen on A-validation only, from this grid (`grid()` in
  `experiments/e2_curriculum.py`):
  - machines: fixed windows with k ∈ {16, 24} bits, and nested levels (8, 16, 24, 32, 48, 64 bits;
    2^22-slot tables);
  - laws: the literal T12.1 law (constant friction) at three rates, κ dt / γ ∈ {0.02, 0.05, 0.2}, and
    the count-dependent friction of decision D10 (dt = 1);
  - the new-sector prior ∈ {0, 4} only acts through the activity count, which the literal law never
    reads, so it is varied for the D10 law only.
- Two configurations are then fixed for every arm and every test: the best under the literal law (the
  specification's machine, reported first) and, if different, the best overall.
- Added on 2026-10-18, before any E2 run: the nested machines in the grid, and the visitation statistic
  computed from the data rather than from one machine's counts, so that it applies to every grid member.
- Revised on 2026-10-18 after the audit, before any E2 run: the literal law at three rates instead of
  one, the arms for the best literal-law configuration, and the repeat rule under **Data**. Every E2
  machine is massless, so the emission and re-entry added after the audit change none of its numbers.

**Measurements**
- bpb on A-test, B-test and the unseen books. These are evaluated with learning frozen, from the
  state at the end of each stage.
- The learning curve: bpb per 1 MiB block during training.
- **Transfer.** bpb on B-test after "A then B" minus bpb after "B only". A negative value means the
  structured stage helped.

**Textbook hypothesis (T12.3), in its three parts**
1. **Ordered information creates recurring correlations.**
   - Measure: over the 24-bit contexts of the stream as read, the entropy of context visitation and the
     share of visits beyond the 8th to each context (visits to familiar contexts).
   - Comparison: A-train against shuffled A-train.
   - Limit (noted after the audit): sentence shuffling keeps every context inside a sentence, so these
     statistics can differ only near sentence boundaries. This part is a weak test and is reported as
     measured.
2. **The correlations change and then stabilize the weights.**
   - Measure: weight drift ‖Δw‖ per MiB across Stage A, and the consolidation counts n_s.
3. **Unfamiliar arrangements of the same relations engage the organization.**
   - Measure: bpb on the unseen books and on A-test, against the same text with its words shuffled
     inside each sentence. That control has the same relations and destroyed arrangement.
   - Reading: a learned structure should make the true text much cheaper than its shuffle. The gap is
     reported, not assumed.

**Generation.** From the end-of-A state, VAM-1 answers short prompts taken from A-test by macro/micro
generation, with horizon 8 bits (one byte). The prompts and outputs are reported verbatim, as observed
behaviour. They carry no claim beyond what the losses show.

## E3. The nested fractal architecture, layer by layer (owner's design N, 2026-10-18)

The owner's seven-layer design (`spec/VAM_Nested_Fractal_Architecture.md`) was built into the machine
(DESIGN section 10, `src/vam/nfa/`, and three options of the nested text machine). E3 measures, for every
layer, whether it does what the design says it does, and then runs the whole machine on E2's data and arms
against E2. Written on 2026-10-18 before any E3 run.

**Pilot, declared.** While building, a pilot read the first 8 MiB of A-train and scored the first 1 MiB of
A-validation (frozen) for single options (nested, counting law, prior 4): plain 1.980 bpb; coherence
0.75: 1.979; verify 2: 1.935, verify 3: 1.928; governor: 1.915 (protect 0.5: 1.911; sink rate 0.25: 1.905).
No test split, held-out book or B data was read. The grid below was fixed after the pilot and includes
every value it tried.

**Part 1. Designed behaviours** (one pass/fail criterion each, stated here)
- N-L1 (bits to equations). (a) The register cap is `min(Bekenstein, word) = 64` bits and the nested
  machine's deepest level fits it. (b) Every reading and generation event integrated as a differential
  equation by an ODE solver equals the machine: largest difference in `R`, `q`, `p_tau`, `w`, `n`
  below 1e-8 over 10 events, for the 2-cell grid of `tests/test_nfa_flow.py`. (c) `-ln q_b` equals the
  integral of `Tr(K_b R)` along the observation flow to 1e-10.
- N-L2 (the Page limit). (a) On S1, S2, S4 (d = 3, 4) and i.i.d. bits (400,000 bits, windows of 6 and
  8 bits), the Page rank equals the number of hidden states (4, 3, 8, 16, 1) whenever the weakest
  direction is above the measured edge, and never exceeds it. (b) Text: the Page rank of the byte
  Hankel matrix (previous `d` bytes, next byte) of the first 4 MiB of A-train for `d = 1..7`; the
  depth with the largest rank is the turnaround. Reported, and used in Part 2.
- N-L3 (density matrix). Every coherent readout probability lies in `[eps, 1 - eps]` and isometric
  nesting holds for every `c` (tests). Its effect is measured in Part 2.
- N-L4 (blankets). On the chain (8 cells, context 2, theta 0.5, 40 bits of English, then 5 forced
  events): the conditional mutual information `I(cell j-1 : cell j+1 | cell j)` for every `j`, and the
  largest change of any cell outside a local disturbance's event cone. Criterion: the cone leak is below
  1e-12 for an untruncated engine; for the truncated engines it is reported. (Neither the plain nor the
  unit-rate governed engine can be held untruncated at 8 cells: the governed one passes bond 1,300
  within 10 events. The untruncated engines, with no bond cap below 4096 and cutoff 0, are the plain
  engine with every qubit einselected and the governed engine with sinks at rate 40; the criterion
  holds if both leak less than 1e-12 and discard less than 1e-20. The truncated engines reported are
  the plain engine at bond 64, the governed engine at the cache bond, the governed engine started at
  the Page threshold, and the governed engine with the entropy-governed bond of margin 4.)
- N-L5 and N-L6 on the chain. Same setting: the plain MPO engine at bond 64 against the governed engine
  (unit-rate interior sinks, cache bond 90): discarded Hilbert-Schmidt weight, cone leak, CMI profile,
  Renyi-2 entropy over qubits. At 12 cells (context 4, theta 0.3, 120 bits): governed engines at bonds 32,
  64 and 90 against the governed engine at bond 128, largest prediction difference, beside the ungoverned
  numbers of `experiments/diagnostics/context_convergence.txt` (5.6e-4 at bond 64); in the same run the
  entropy-governed bond (`bond="entropy"`, each cut keeps `ceil(margin 2^S)` directions under the cache
  cap) at margins 2, 4 and 8, with its bond sizes. The area law: the largest operator entanglement of
  any cut after 40 bits for 4, 6, 8 and 10 cells (context 2, theta 0.5), plain at bond 64 and governed
  at the cache bond (bounded in the chain length means the area law holds). And the cost of the
  sinks: prequential bits per bit of the exact (untruncated) machine with and without sinks on the same
  stream (the first 256 bytes of A-train) and the same emissions, with the largest change of a
  prediction, for 3, 4 and 5 cells (the sinks change the machine; the size of that change is reported,
  not assumed). Found while building, before the run: with the context einselected exactly (`g = inf`)
  and at most two cells outside the context, the sinks change no prediction (to rounding;
  `tests/test_nfa_chain.py`), because no coherence the coin makes can return to a population; with
  three or more such cells, or a finite `g`, they do. The grid includes both cases.
- N-L5 and N-L6 in the text machine. A sentence learned in a 512-slot table, then 60,000 random bits: the
  fraction of the sentence's deepest-level sectors still stored, for plain, verify 2 and governor.
  Criterion: plain keeps under 2%, the other two keep at least 97% of what they held.
- N-L7 (path integral). `hbar = 0` equals the macro/micro "max" planner bit for bit; `hbar = 1`
  (euclidean) gives the machine's own prediction (tests).

**Part 2. Configuration, chosen on A-validation only** (full A-train, frozen A-validation, as E2)
- Start: E2's best overall configuration (nested levels 8-64 bits, 2^22-slot tables, counting law, prior
  4). Its A-validation bpb is recomputed first; it must equal E2's 2.0316 (reproducibility check).
- Stages, each keeping the previous stage's choice:
  1. N-L2 depths: the default depths, against the default depths cut at the Part 1 turnaround
     (levels deeper than `8 x turnaround` bits removed; kept only if the turnaround is below 8 bytes).
  2. N-L3 coherence `c` in {0, 0.25, 0.5, 0.75, 1}.
  3. N-L5 verify in {1, 2, 3}.
  4. N-L6 governor: off, or on with protect in {0.5, 1, 2} bits and sink rate in {0.1, 0.25, 0.5, 1}, each
     with verify 1 and with the stage-3 choice.
- Ties within 0.0005 bpb go to the simpler setting (fewer options on).

**Part 3. The whole machine on E2's arms.** The Part 2 configuration runs E2's four arms (A then B, B
only, shuffled A then B, A only) on the same streams, with every E2 measurement (A-test, B-test, unseen
books, transfer, T12.3 parts 2 and 3, learning curves). The table puts it beside E2's best overall
configuration and lzma. New counters: evictions, doorkeeper refusals, protected knocks, occupancy.

**Part 4. Writing (N-L7).** From the end-of-A state of the Part 3 machine, the 8 E2 prompts from A-test,
48 bytes each, for every setting of: `hbar` in {0, 0.25, 0.5, 1}; form euclidean or feynman; collapse
or sample (seeded); verified emission off or on. Measured per setting: the share of repeated 8-byte
windows (loops), the share of generated words found in A-train's vocabulary, the share of printable bytes,
the share of valid UTF-8, and the machine's own bits per byte of what it wrote. Outputs are reported
verbatim. These are observations; no setting is called better than another beyond what the numbers show.

**Amendment 1 (2026-10-20, after Audit 2, before any full E3 run; manager).**
- (A1) Part 4: the setting `hbar = 1`, euclidean, sample, verified off is the machine's own output
  instrument (T10.1, T7.1) and is labelled so. Every other setting is a chosen realization and is
  labelled so: `hbar = 0` is a shortest path over `-ln p` (T15.3), euclidean with `hbar < 1` is a
  tempering (T14 step 7), the feynman form builds its phase from the probability, and verified emission
  is an evidence threshold. At `hbar = 0`, form and mode have no effect: those rows are reported once.
  `report.py` writes the label on every row. The grid itself is unchanged.
- (A2) The area-law reading rule: only rows whose discarded Hilbert-Schmidt weight is at most 1e-6 are
  read for the area law; the others are reported as truncation-bound. The quantity is named operator
  entanglement (of `vec(rho)` across the cut), not `S(rho_A)`. Governed rows are read with the note that
  their boundedness comes from the sinks.
- (A3) The cache bond is 90, as written above, passed explicitly and not detected from the host. The
  detected `max_mpo_bond()`, `max_density_dim()` and cache sizes are recorded in provenance as
  information only.
- (A4) Before the full run, E2's start value is recomputed on full A-train, frozen A-validation (HANDOFF
  T1.6). If it differs from `2.0316126330797104` by more than 1e-9, the full run does not start; the cause
  is found and stated first, and any tolerance is a further dated amendment. No tolerance is set in
  advance.
- (A5) Part 2 stage 1 stays as written: the depth-cut candidate is run and reported, though Audit 2
  shows the text Page edge has no power from depth 3 on (so the turnaround is a lower bound).
- Runtime: a part that runs past about 1 h without finishing is stopped; a smaller setting for that part
  only is written as a further dated amendment, and the run resumes. Chain jobs run one at a time.

**Amendment 2 (2026-10-20, Part 2 stopped at the 1 h limit with no chosen configuration; manager).** Part 2
is a fixed list of independent configurations, each saved on completion (4.6-5 min each, 12 of about 36
done). It is progressing, not stalled, so the grid is kept unchanged and the 1 h limit of Amendment 1 is
replaced for Part 2 by 6 min per configuration (about 3.6 h in total). Parts 3 and 4 keep the 1 h rule.
No Part 2 result was read to choose this.

**Amendment 3 (2026-10-20, Parts 3-4 stopped by an out-of-memory kill with nothing of Part 3 saved;
manager).** Parts 1 and 2 stay as run at `1029e49`. Parts 3 and 4 run with the machine code (`src/`,
`e2_curriculum.py`) of `1029e49` and the `e3_nfa.py` of `f87184d`, which only releases each prompt's
machine copies and saves Part 4 as it is written (bit-identical outputs, `tests/test_e3_memory.py`).
Measured peak about 3.3 GiB, against 13.9 GiB at the kill. `e3.json` records both commits. No Part 3 or
Part 4 result existed when this was written.

**Amendment 4 (2026-10-20, after checkpoint C2, `audits/c2.md`; manager). Added after the E3 results, and
labelled so wherever reported.** E3's numbers stand as run; nothing here changes them.
- T4.2, N-L2 on text: the Page rank of the byte Hankel matrix with a count floor (fixed in the script
  before it reads PG-19; the value and its reason written in its docstring). Reported as a measurement
  only, never used to cut depths.
- T4.3, N-L6 on the chain: the governed engine started so that its sinks act before the cache cap binds
  (the cap-limited capacity as the trigger), at 8 cells, same setting as Part 1's blankets. Measured:
  discarded weight, cone leak, CMI, bonds, beside the Part 1 rows.
- Not approved: extending the Part 2 grid (C2's T4.1). It tunes the configuration after the test split
  was read; it is listed as an open question, not run.
- Results go to `results/e3_t4.json`, from `experiments/e3_t4.py`; `report.py` adds them under E3 as
  "added after the result".

## E4. N above one equation, on text (DESIGN section 11, decision D11)

Written on 2026-10-20, before any E4 code and before any E4 run. E3's A-test numbers (A-test 2.1309 bpb, lzma 2.0031) informed
the design; E4 therefore holds E3's chosen base fixed and selects only E4's new factors, on A-validation. A-test is read once, at the
end, for the chosen configuration and the controls. E4 changes only by amendments dated before the run they affect.

**Purpose.** E3 realized one equation (a 64-bit register, 8 bytes of context). E4 builds the levels above it (units of bytes, sentences,
registers of registers, one global governor) as present-state registers read as more context levels. Questions:
Q1. Does the hierarchy lower the bits per byte on A-test and on unseen books, against the E3 machine, on the same splits and arms?
Q2. If so, is it the added span, and not the added capacity or the added levels?
Q3. Do the upper levels use order?
Q4. Does the gain sit where the long-match hypothesis puts it (windows of 16 and 32 bytes found in A-train)?
Each is reported whatever the answer. A null result is a statement about this realization of the hierarchy, not about N.

**Base (held fixed, from E3).** Counting law, prior 4, byte levels 8 to 64 (tables `2^22`), coherence 1.0, verify 3, governor with protect 0.5
and sink rate 0.1 (E3's chosen row, `results/e3.json` `part2.chosen`). The cache bond 90 is not used (text only).

**Added levels (DESIGN 11.4).** `W1, W2, W4` (ring of 4 unit codes), `S0, S1` (sentence), each `2^21` slots with a doorkeeper of `2^22`
entries, in the order byte levels, W, S. Units close at `mix(last 3 bytes) mod 6 == 0`; sentences at `mix(code) mod 16 == 0`. Not tuned.

### Part 0. Gate: identity and reproduction (before any E4 row)

- (a) With the upper levels off, the E4 machine's prediction arrays on 1 MiB of A-train (frozen, then learning) equal the E3 machine's
  bit for bit, and the E3 chosen configuration's A-validation bpb reproduces the saved value of `e3.json` to 1e-9 (same pins).
- (b) Paired, interleaved timing of the default path against the E3 head (T1.7's method): the gap must stay within the noise it showed
  (3%).
- If either fails, no E4 row runs; the cause is found and stated first.

### Part 1. Designed behaviours (one pass/fail criterion each, stated here)

- **N-L1.** (a) Every register is at most `register_cap` = 64 bits and every arity at most `line_words` = 8 (asserted); the state is under 128 bytes and its size does not change
  after reading 1 MiB and 8 MiB. (b) The share of distinct units that share a code (collisions), on the first 4 MiB of A-train: reported.
- **N-L2 (measurement, never a selector).** For each upper level, the Page rank of the whitened Hankel matrix (level context against next
  byte) with a count floor of 256 on the first 4 MiB of A-train, and the same on the byte-shuffled copy. Criterion: the null never exceeds
  rank 1. The real ranks are reported and gate nothing.
- **N-L3.** (a) Over 20,000 random states of the full machine, the readout lies in `[min, max]` of the component readouts. (b) Adding an
  upper sector whose prediction equals the coarser readout leaves the readout unchanged to 1e-12 for `c` in {0, 0.5, 1}. (c) No NaN or Inf over every run.
- **N-L4 (containment).** On a frozen, trained machine, flip one byte at a random position of a 256 KiB stream; the predictions
  beyond the span are bit-identical to the unflipped run, for 20 flips: after 8 bytes for the byte levels alone; after the ring has turned over (4 closed units after resynchronization) for the W levels; after the sentence closes (S0) and one
  sentence more (S1).
- **N-L5.** (a) Verified growth: on 8 MiB of i.i.d. random bytes, every upper level's occupancy stays below 1% with verify 3, and above 50% with verify 1. (b) The storage in bytes does not change from the start of reading to the end.
  (c) The mean entropy of the label density per cut, reported; the area law `S_cut <= 1 bit` is vacuous at bond 2 and is stated so.
- **N-L6.** (a) Survival: a sentence learned in tables of 512 slots, then 60,000 random bits: plain keeps less than 2% of the deepest upper level's sectors, verify 2 and the global governor keep at least 97% of what they held. (b) Global occupancy never exceeds 1. (c) The raised bar changes the doorkeeper's refusal count
  at verify 2, 3 and 4 on 2 MiB of A-train; if it does not at all three, the raise is removed and the governor is E3's scope with
  the global occupancy (decided now, and stated in the report).
- **N-L7.** The instrument (hbar 1, euclidean) equals the machine's prediction to 1e-10 on the full state; `hbar = 0` equals the macro/micro planner.

### Part 2. Configuration, chosen on A-validation only

Each row reads full A-train and scores frozen A-validation. Ties within 0.0005 bpb go to the simpler setting (fewer options on). Each row is saved and
resumable; a row that runs past 15 minutes is stopped and its setting is replaced by one amendment (dated before the resume), as in E3's Amendment 2.
- Stage 1, segmentation (with every upper level on): content-defined units (default), and the text rule (a byte outside letters, digits and the apostrophe closes a
  unit; `. ? !` followed by a space or newline closes a sentence), labelled as a departure from F Part 10.
- Stage 2, level set: all, W only, S only (E3's six levels are the baseline from Part 0).
- Stage 3, governor: per-level (E3's), global without the raised bar, global with the raised bar, each at sink rate 0.1; then sink rate 0 for the best of them (the pre-registered ablation, reported, and selectable like any other row).
- Stage 4, verify of the upper levels only (the byte levels keep 3): 2, 3, 4. A passage repeated once is seen once before, so under verify 3 an upper level
  gives it no slot until its third sighting; this stage measures that interaction (verify 2 and 4 are the new rows).
Rows: 2 + 2 + 3 (+1 ablation) + 2 = 9 new rows.

### Part 3. The whole machine on E2's arms

The Part 2 configuration runs E2's arms on the same streams, with every E2 and E3 measurement (A-test, B-test, unseen books, transfer,
forgetting, T12.3 parts 2 and 3, learning curves, counters). Saved in two units, each under the 1 hour rule: 3a (A only, A then B), 3b (B only, shuffled A then B, seed 1).
Optional 3c: the shuffled arm with a second seed, run last and reported if it finishes. Sentence shuffling now changes E4's contexts, so the
transfer comparison with the shuffled control means something here; it is read with the spread of the seeds when 3c exists.

### Part 4. Controls (A only arm, for the chosen configuration)

- **C0, upper levels off:** E3's saved numbers (no rerun).
- **C1, equal storage and compute, no extra span:** the same five upper levels with keys computed from the byte register at depths 24, 32, 48, 64, 64 with new salts (duplicates of byte levels): the same slots, doorkeepers and time, no
  span beyond 8 bytes.
- **C2, order destroyed, content kept:** the W and S contexts built from commutative combinations (sum of the unit codes) instead of ordered mixes.

### Part 5. Decomposition by window class (the long-match hypothesis)

For each byte of A-test, the class is the longest of 32, 16 and 8 for which the window of that many bytes immediately before it occurs in A-train; class 0 if
none. Reported: the share of bytes and the bpb per class, for the E3 machine (retrained, same code, same split) and the E4 machine, and the difference. lzma's per-byte cost is not available, so this table does not compare with lzma.
A-train windows are held as sorted 64-bit rolling hashes (one size at a time). Present-state is not touched: this is a measurement taken outside the machine.

### Part 6. Writing

From the end-of-A state of the Part 3 machine, the 8 E2 prompts of E3 (48 bytes): the instrument row (hbar 1, euclidean, sample, unverified), and `hbar = 0` collapse, labelled as a chosen realization.
Verbatim, with the E3 outputs beside them. No ranking.

### Reading rules (fixed now, not tuned)

- **R1** (headline either way): E4's A-test bpb against E3's 2.1309 and lzma's 2.0031 (conditional on the A split).
- **R2** (the hierarchy helps): E4 beats E3 by at least 0.01 bpb on A-test and on A-heldout (unseen books).
- **R3** (span, not capacity): E4 beats C1 by at least 0.005 bpb on A-test and A-heldout.
- **R4** (order): E4 beats C2 by at least 0.005 bpb on A-test and A-heldout.
- **R5** (long matches): more than half of E4's total bit gain over E3 on A-test comes from the classes 16 and 32.
- **R6** transfer and forgetting, reported beside E3's (-0.0143 against -0.0167; forgetting 0.055).
- Every number comes from a saved JSON file that records its commit and seeds; E3's numbers are read from `e3.json`, never typed.

### Budget (a 16 GB, 4-core host; derived from saved E3 data, not from new runs)

- Inputs: an E3 Part 2 row took 276 to 304 s (6 levels; 33.1 MB A-train plus 4.1 MB validation: about 1.0e6 bits/s); E3's Parts 3 and 4 took 46 min
  (RUNLOG, 2778 s), of which writing was 32 settings at 10 to 16 s, leaving about 40 min for the arms (288 MB read in total);
  the E3 machine is 577.5 MiB (449.5 MiB without doorkeepers), 112 MiB per hashed `2^22` level; the sampled peak of E3's arms was 2.6 GiB (4.5 machines).
- Scaling: time in proportion to levels (11 against 6: x1.83); memory as the machine size times the copies alive, as in T2.3 (copies released and collected after each prompt).
- The E4 machine: 937.5 MiB. Peak memory: about 4.5 x 0.92 = 4.1 GiB, plus the streams. The run is stopped above 10 GiB.

| Unit | Estimate | Basis |
|---|---|---|
| Part 0 | 10 min | one 6-level row at 5 min, and timing |
| Part 1 | 20 min | small streams (256 KiB to 8 MiB) |
| Part 2 | 9 rows x 9 to 10 min, 85 min | 276 to 304 s x 1.83 = 505 to 555 s; cap 15 min per row |
| Part 3a | 31 min | 121 MB x 15.3 s/MB (8.3 s/MB in E3, x1.83) |
| Part 3b | 43 min | 167 MB |
| Part 3c (optional) | 26 min | 100 MB |
| Part 4 | 28 min | two A-only arms, 54 MB each |
| Part 5 | 20 min | E3 retrain 5 min, E4 8 min, classes |
| Part 6 | 2 min | two settings |
| Total | about 4.0 h (4.4 h with 3c) | |

Rules: one job at a time (parallel jobs slowed each other 10 to 30x on 4 cores); a unit that runs past 1 hour is stopped and a smaller setting
for that unit is written as a dated amendment; every unit is saved as it ends; the per-row estimate of Part 2 is re-measured at build time, and if a row exceeds 15 minutes the grid
is reduced by amendment before the run.

**E4 Amendment 1 (2026-10-20, after the build `4a27a0e`, before any E4 run; manager).**
- Part 0 (b): the paired timing of the upper-off path of E4's own loop against the nested machine is
  reported, not a gate (E3's loop is untouched, so the 3% rule has nothing to guard). Bit-identity and the
  reproduction of E3's chosen A-validation value stay gates.
- Part 1, N-L6 (c): the raised bar is checked on tables of 2^14 and 2^12 slots, where the Page point is
  reached; at full size it cannot be reached in 2 MiB, so the check would fail by construction.
- Budget: about 4.4 h (4.9 h with `--seed2`), from the measured 2.2x rate ratio; peak about 4.2 GiB, stop
  above 10 GiB; a Part 2 row above 900 s stops the run.
- If the global governor never reaches the Page point at full size, its rows are reported as such.

**E4 Amendment 2 (2026-10-20, after the E4 results and the claims check `audits/claims.md`; manager).
Added after the result, and labelled so.** One control: E3's chosen configuration with sink rate 0, read
and scored on the same splits as E4, so that R2's comparison with E3 separates the hierarchy from the
removal of sinks. E4's registered reading rules and numbers stand as run.

## E5. Does the taught machine use what it was taught? (approved by the owner 2026-10-21, before any E5 run)

Moved from `E5_DRAFT.md` at `eacf59e` (G3 passed; owner approval T7.2 via the planner, 2026-10-21). The question
set is `data/e5/questions.json`, SHA-256 `2706322fb9c371ff0c12cfc302bb5f8a193bf06bb650c091076e9467892d5e62`,
unchanged. Decisions D1-D11 as recommended; stage 5b confirmed; the reproduction scripts stay in the draft's
appendix.

### E5.1. Question and claim (S: manager's task, owner's instruction)

After one fixed teaching stream (dictionary, general prose, mathematics, physics, then named primary works and the authors' own accounts of their reasoning), does E4's chosen machine, unchanged, give lower cost to the correct answer of held-out questions that follow from the taught material than the same machine that was not taught stages 3 to 5?

Bound on the claim (S: T15.4): the state is a 64-bit register, its context statistics and E4's upper levels. A fact reaches an answer only if question and answer appear close together in a form like the teaching text. E5 therefore bounds "uses taught knowledge" from one side. It cannot show understanding. Either outcome is reported.

### E5.2. Rules kept (S: HANDOFF, T7.1, T10.1, T14, T15.3)

- The machine is E4's chosen configuration, read from `results/e4.json` `part2.chosen` (S): `{"machine": "hier", "law": "counting", "prior": 4.0, "coherence": 1.0, "governor": true, "protect": 0.5, "sink_rate": 0.0, "verify": 3, "hier_levels": "WS", "hier_segment": "text", "hier_verify": 4}`. Code at E4's head; no change.
- No scoring, path costs, beams, softmax or thresholds are added to the machine. The event loop is unchanged. Learning is forward only: no transcript, no replay, no second pass over any stage.
- Everything, the dictionary included, is read as raw bytes, as bits. Cleaning is E2's (`vam.corpus.clean_text`: UTF-8, `\n`, and the text between the `*** START OF` and `*** END OF` Project Gutenberg markers when both are present) and nothing else (S). Files with an older header format keep it: a scan (`data/e5/header_scan.json`, script A.e5_headerscan.py) finds the old "SMALL PRINT" boilerplate in 4 S2 files, 2 S4, 2 S5 and 2 C files, and Project Gutenberg text in all stages. Those bytes are taught like any other and are not in a question source or a stem (checks, section 5).
- The cost of an option is a measurement taken outside the machine: each bit is charged the probability the machine gave it before seeing it (PROTOCOL line 7; `m.run(bits, learning=False)` returns those probabilities).
- Numbers in any report come only from saved JSON that records the commit and seeds.

### E5.3. Curriculum, one continuous stream, no reset (C: order from the task; D: sizes)

Stages in order: **1** dictionary, **2** general prose, **3** mathematics, **4** physics, **5a** primary works, **5b** the authors' own accounts of their reasoning (5a and 5b are read as the single stage 5, 5a first). Stage 5 comes last, so recency favours it; this is a risk (R2), not a hidden choice.

#### 3.1 Source table

Bytes are bytes of cleaned text. "Trained" excludes the held-out parts; "held out" is never read by the machine except in the frozen held-out measurement (section 8).

| Stage | Source | Licence | Rule | Bytes | SHA-256 | Flags |
|---|---|---|---|---|---|---|
| 1 dictionary | Webster's Unabridged Dictionary (text marked 1913 Webster, with a 1913 supplement), Project Gutenberg #29765 (PG-19 train-split copy; fixed by name, not by rule) | public domain (1913); PG-19 dataset Apache-2.0 | whole book, cut into 64 KiB blocks at line ends, every 20th block held out | 26585810 trained + 1376846 held out | cleaned file `7139d6bba7079f521662846364be0e72da8c30dfe1b02f481c3a1528bb5c0100` | licence not independently verified (PG header only); Webster has three PG-19 copies, #29765 chosen, the others excluded by the repeat rule |
| 2 general prose | PG-19 stage-B rule of E2 (`fetch_pg19.py`): title matches neither FORM nor SUBJECT; seeded shuffle, seed 20261018; English check; repeat rule; whole books until >= 32 MiB | public domain (US) per PG header; PG-19 dataset Apache-2.0 where fetched from PG-19 | 69 books trained, every 10th book held out (7 held out) | 30206436 trained + 3443380 held out | list hash `9cf9de102fed5dccaf1d811d327396557548a433f90d8c9366dd45478e094aba` (per-file hashes in `sources.json`) | 76 of 76 books are also in E2's MANIFEST.json (same rule and seed, so the same first books) |
| 3 mathematics | Project Gutenberg catalog, English texts with LoCC class QA, ascending ebook number, excluding stage-5 ids, foreign-language teaching, repeats, books under 60% letters (digit dumps); cap 24 MiB | public domain (US) per PG header; PG-19 dataset Apache-2.0 where fetched from PG-19 | 54 books trained, every 10th book held out (6 held out) | 16716924 trained + 2438489 held out | list hash `a77e8f2719f9e9b4e5d6b8c3c90ae5f062f97e9cb7afe2607feb4e24165e1bf6` (per-file hashes in `sources.json`) | supply-limited: 19155413 bytes available, under the cap; 53 QA books had no fetchable text; 6 books also in E2's MANIFEST |
| 4 physics | Project Gutenberg catalog, English texts with LoCC class QC, ascending ebook number, same exclusions, cap 24 MiB | public domain (US) per PG header; PG-19 dataset Apache-2.0 where fetched from PG-19 | 74 books trained, every 10th book held out (8 held out) | 22128698 trained + 3113447 held out | list hash `d3263ecb324094821f428cb0553a386c8ae3f85673ba5c4042ecbc368368409b` (per-file hashes in `sources.json`) | cap reached (last book overshoots); 10 QC books had no fetchable text; 7 books also in E2's MANIFEST |
| 5a primary | Newton, Principia (Motte translation, with the Life of Newton), PG #76404 | public domain (PG header) | named work; last 3% held out | cleaned file 1457485 B | `459ecf96ae4b8405d11bc79ac2ac9c71d77e9df609edcb0a96ecc14560e39572` | html (converted); licence from PG header only |
| 5a primary | Newton, Opticks (4th ed., 1730), PG #33504 | public domain (PG header) | named work; last 3% held out | cleaned file 567127 B | `38fc7c50ef7e802da45d8016cd32ad0281db6836f16a98ff38c8c8a85393aa52` | txt; licence from PG header only |
| 5a primary | Einstein, Relativity: The Special and General Theory (Lawson transl.), PG #30155 | public domain (PG header) | named work; last 3% held out | cleaned file 186921 B | `824ee60bad8ed05c7e876cc614fe734fdbeb78afd05d0befc3a4068f96f70c5d` | txt; licence from PG header only |
| 5a primary | Leonardo da Vinci, The Notebooks (Richter transl.; PG "Complete" = vols 1+2), PG #5000 | public domain (PG header) | named work; last 3% held out | cleaned file 1381410 B | `d9414526fd5181425aa27466ffe534f0ba4e5df10787726aa5933129b847ebb3` | txt; licence from PG header only |
| 5a primary | Euclid, The First Six Books of the Elements, ed. Casey, PG #21076 | public domain (PG header) | named work; last 3% held out | cleaned file 476650 B | `338bf7fa1cbcb6cbc4ed617b486fc5024b994d4f45a2bb0bbeb953effc68b547` | html (converted); licence from PG header only |
| 5a primary | Plato, Republic (Jowett), PG #1497 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 1195161 B | `816fa05bbb4baadfcfc1e03da2bb332ed0fae30c16e8393b74639da129883392` | PG-19 copy |
| 5a primary | Plato, Charmides (Jowett), PG #1580 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 121657 B | `9d8388d0007de3408f7c42557b8625f0bf547280e3ba442b9733b0d5434508ab` | PG-19 copy |
| 5a primary | Plato, Laches (Jowett), PG #1584 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 66688 B | `ac409cfd670cf7279f81d2b5f95ea4d1305d27ed5afbc5988e8717ba6ad3cbdc` | PG-19 copy |
| 5a primary | Plato, Euthydemus (Jowett), PG #1598 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 114304 B | `aecb2daa57a2c61b5b1aa650cb1de47770e4b245c8570ff3775bcdb8e50acfdf` | PG-19 copy |
| 5a primary | Plato, Cratylus (Jowett), PG #1616 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 299964 B | `2f29b47091fc37a48d5470b62d117a84c023042f1649f46a96385e6522e59f11` | PG-19 copy |
| 5a primary | Plato, Ion (Jowett), PG #1635 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 35490 B | `6a78795ff4c083762baea1618ff6849b2d4800785caba3054124e81bdcdd87b4` | PG-19 copy |
| 5a primary | Plato, Meno (Jowett), PG #1643 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 123020 B | `59c8ace60d24b61bf0453deb131371ae5ad9858cea46365d603829e0383c117e` | PG-19 copy |
| 5a primary | Plato, Apology (Jowett), PG #1656 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 85851 B | `8052b747de1fe25715ea96f48937df69521754a8ca83aacf92034100b8943b90` | PG-19 copy |
| 5a primary | Plato, Crito (Jowett), PG #1657 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 47257 B | `f0edc62362a155eb3fb0c8137d9b19e38bed9de7d6ede00b1d1a3a17b3898d85` | PG-19 copy |
| 5a primary | Plato, Phaedo (Jowett), PG #1658 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 232976 B | `3d31d86bbc5a1f6bed45850357ef843442b7b8b53d381e6692d5c37dd08d2b2c` | PG-19 copy |
| 5a primary | Plato, Gorgias (Jowett), PG #1672 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 328352 B | `ae83d821db959ae4c8b33419e7b754b37d04da42c3dacb4d35d2f318832551f5` | PG-19 copy |
| 5a primary | Plato, Parmenides (Jowett), PG #1687 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 197261 B | `66aeb8eeccd854420abcf39d7769fbd7281684f23087f6d06be9ffc8e505428e` | PG-19 copy |
| 5a primary | Plato, Philebus (Jowett), PG #1744 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 261522 B | `b18fd1cf01d2726ae6ba52b17ca9ef767f8081472177b3a5200815b617a6cb39` | PG-19 copy |
| 5b own accounts | Einstein, Sidelights on Relativity (Ether and Relativity; Geometry and Experience), PG #7333 | public domain (US) per PG header | added at the owner's request; last 3% held out | cleaned file 51862 B | `343b983ad9e86ed635dd4a6bfa63f51fb570a844bef53379dff566d582128af8` | gutenberg direct; licence from PG header only |
| 5b own accounts | Einstein, The Meaning of Relativity (four lectures, Princeton, 1921), PG #36276 | public domain (US) per PG header | added at the owner's request; last 3% held out | cleaned file 145229 B | `d62efc44dd0b7608bb051ee32cec626ff271fd81e6faf5572b4e32d99dcd3e6b` | gutenberg direct; licence from PG header only |
| 5b own accounts | Einstein, Fundamental Ideas and Problems of the Theory of Relativity (Nobel lecture), PG #69572 | public domain (US) per PG header | added at the owner's request; last 3% held out | cleaned file 23268 B | `c567dee8ff660929de929baef1b788fcfc3139ab0c5c13a9e53a34b64fbd95a1` | gutenberg direct; licence from PG header only |
| 5b own accounts | The Principle of Relativity (papers by Lorentz, Einstein, Minkowski, Weyl; several authors, not Einstein alone), PG #66944 | public domain (US) per PG header | added at the owner's request; last 3% held out | cleaned file 408764 B | `bc9d5acc5e2da31b165899a28ba7968f9d7d71fae11b2b0a1fd3bdcaa9359f70` | gutenberg direct; licence from PG header only |
| 5b own accounts | Leonardo da Vinci, Thoughts on Art and Life (extracts from the Notebooks; transl. Maurice Baring; the "Einstein" in the catalogue entry is Lewis Einstein, the editor, not Albert), PG #29904 | public domain (US) per PG header | added at the owner's request; last 3% held out | cleaned file 267795 B | `4e0d47bc68bb4aab78845487d607b613c4e5a41c1dac5f195a8f51f0202f7d69` | gutenberg direct; licence from PG header only |
| 5b own accounts | Xenophon, The Memorabilia (Dakyns): Xenophon's account of Socrates, not Socrates' or Plato's own words, PG #1177 | public domain (US) per PG header | added at the owner's request; last 3% held out | cleaned file 412124 B | `422d8edfb90b4e8e2e4d5880b8fe94df594584738cbaf8e04cedc196a5294634` | gutenberg direct; licence from PG header only |
| 5b own accounts | Xenophon, The Apology (Dakyns): Xenophon's account of Socrates' defence, PG #1171 | public domain (US) per PG header | added at the owner's request; last 3% held out | cleaned file 47107 B | `27c3595a9b455c4f5518920d04251245283520decabfdd443d1cf33b65449ba9` | gutenberg direct; licence from PG header only |

| C extra prose (replaces 3-5 in arm C) | PG-19 stage-B order continued after stage 2; same exclusions; last book cut at a line end so that bytes equal the trained bytes of stages 3-5 | 105 books, 47125859 bytes (= 47125859) | list hash `488313e65a1839ef49cd67cd9bc56f38a209bac9c44bc75729d86613b75c164e` | 69 of 105 books also in E2's MANIFEST |

Totals (D, from `sources.json`): trained bytes by stage `{"S1": 26585810, "S2": 30206436, "S3": 16716924, "S4": 22128698, "S5": 8280237, "C_extra": 47125859, "need": 47125859}`; held-out bytes by stage `{"S1": 1376846, "S2": 3443380, "S3": 2438489, "S4": 3113447, "S5": 255008}`. Trained bytes of one T arm: 103918105.

Notes on the table:
- Stage 5 is called by what it is. No work is called a diary. Newton and Leonardo left no separate diaries on Gutenberg; Leonardo's Notebooks are his own notes, and Newton's accounts of his method are in the Opticks Queries and the Principia's rules of reasoning, which are inside the works above.
- **Stage 5b was added at the owner's direct request during this task, not by the manager's specification.** The owner asked for the authors' own descriptions of their ideas and how they reached them. What exists in public domain on Gutenberg is the set above. Entries that are not what the label might suggest: #66944 is a collection by several authors; #29904 is extracts from the Notebooks, whose catalogue "Einstein" is Lewis Einstein, the editor; the Xenophon works are accounts of Socrates by Xenophon. No question is drawn from stage 5b (section 5), so 5b changes only what the machine has read.
- **Heath's Euclid is not on Project Gutenberg** (the catalogue has Casey's edition and Heath's Archimedes). Casey is used (owner decision D3).
- **Licences.** Every row's licence is the Project Gutenberg header and catalogue entry (public domain in the USA) and, for PG-19 copies, the dataset's Apache-2.0. None is independently verified outside the USA. Flagged unverified for owner review: all Einstein items (translations), #29904, and the converted-HTML works.
- **Frozen supply.** `sources.json` records the exact files and hashes. The run reads those files and never re-selects. This matters because the fetch depended on network state: Gutenberg returns 406 to clients that do not accept gzip, so the first fetch (`fetchlist.py`, urllib) marked 99 QA/QC books unavailable; `curl --compressed` (`fetch_html2.py`) recovered 36 of them, and 53 QA and 10 QC books still have no text at either URL. The fetch of every stage-5 and 5b file and of the dictionary is `fetch_s5.py`; re-running it on 2026-10-21 reproduced every cleaned-file hash in the table.
- **Digit filter (C, added after seeing the supply).** The first QA selection included files such as "Catalan's Constant to 1,500,000 places" and "The Value of Zeta(3)" (megabytes of digits). A rule was added before the final build: a book needs at least 60% letters among non-space characters. It dropped 13 QA books. It was chosen after seeing the data, which is why it is labelled C.
- **S3 is supply-limited** (all eligible QA books are used and the stage is under its 24 MiB cap); **S4 reached the cap** (the last book overshoots it).
- **Overlap with E2's `MANIFEST.json`** (D, `data/e5/overlap.json`, by id and normalised title; sources in MANIFEST / sources): S1: 0/1; S2: 76/76; S3: 6/60; S4: 7/82; C: 69/105; S5: 0/25. Stage 2 uses E2's stage-B rule and seed, so its books are E2's stage-B books. No stage-5 source and not the dictionary is in the MANIFEST. Overlap with the question sources is tested in section 5. Whether overlap with E2 matters is owner decision D6.

#### 3.2 Rules (D unless marked)

- **S1** fixed by name (C).
- **S2** (D): PG-19 books whose title matches neither the instructional-form nor the school-subject pattern of `fetch_pg19.py`, in the seeded shuffle of seed 20261018 (S: E2), skipping ids excluded below, non-English books (E2's 12% function-word test) and repeats (same cleaned SHA-256 or same normalised title); whole books until at least 32 MiB.
- **S3, S4** (D): Project Gutenberg catalogue (`pg_catalog.csv`, fetched 2026-10-20), `Language=en`, `Type=Text`, LoCC class QA (mathematics) or QC (physics); ascending ebook number; skip stage-5 ids, other editions of stage-5 works, books for learning a foreign language (E2's FOREIGN pattern), repeats, non-English, digit-heavy; stop at 24 MiB. Book of a class in both goes to QA.
- **Exclusion list** (C): the ids of every stage-5 work and every other edition of one (`excl_ids.json`: 29765, 33504, the Plato dialogues, 30155, 5001, 36114, 28801, 76404, 5000, 4998, 4999, 21076 and the 5b ids), applied to S2, S3, S4 and C.
- **Held out** (C): S1 every 20th 64 KiB block; S2, S3, S4 every 10th book of the stage order (E2's rule); S5 the last 3% of each work, cut at a line end. A question whose source quote lies in a held-out part is dropped (this removed 3 items).
- **Why 32 MiB for stage 2 and not E2's 64 MiB per stage (C):** E4's A-train was 33.1 MB, a size at which E4's tables were at 49% global occupancy (`results/e4.json`, A-only arm). The whole T stream is 99 MiB. A 64 MiB stage 2 would put the total near 130 MiB and over an hour per arm for a stage that carries no question. Owner decision D1.

#### 3.3 Arms (S: manager)

- **T (taught):** stages 1, 2, 3, 4, 5a, 5b.
- **C (control):** stages 1 and 2 as in T, then the "C extra prose" stream in place of stages 3 to 5. It has the same bytes as T's trained stages 3 to 5 (cut at a line end), is chosen by the same rule, uses the same seed, and is disjoint from T's prose by construction (it continues the order after stage 2). For the held-out measurement the C stream is paused at the byte offsets where T's stage 3, 4 and 5 end, with no reset.
- **C0 (optional, reference):** E4's end-of-A machine. `results/` holds only JSON, so the state would be rebuilt from A-train (about 10 minutes at E4's rate). Recommendation: skip (D8).

### E5.4. Machine facts the run needs, from the code (S: `vam/src/vam/nfa/hier.py`, `experiments/e4_hier.py`)

- Capacity counters: `counters(m)` in `e4_hier.py` returns `evictions`, `refused`, `protected`, per-level `occupancy` and `global_occupancy` (from `m.evictions`, `m.refused`, `m.protected`, `m.occupancy()`, `m.global_occupancy()`). They are recorded at the end of every stage, and again as deltas per stage.
- Register snapshot and restore without copying the 937 MiB tables: `m.state` returns the register list and ring state; `m.set_state(list)` restores it. `m.copy()` duplicates `_wn`, `_tags`, `_door`, `_occ`, `_stats` and `_st`. That frozen reads leave the tables alone, and that `state`/`set_state` is a complete snapshot, are **read from the code and not yet tested**; they are pre-run gates in section 6.
- `m.run(bits, learning=False)` returns per-bit probabilities of the bits read. Cost of a byte string = `-log2` sum over its bits.

### E5.5. Questions (S for quotes, C for questions and distractors, D for checks)

File: `data/e5/questions.json`; SHA-256 `2706322fb9c371ff0c12cfc302bb5f8a193bf06bb650c091076e9467892d5e62` (also in `questions.sha256`). Authored 2026-10-20 and frozen 2026-10-21, before any run. The file records, per item: id, kind, source work, stem, four options, answer index, option byte length, the source quotes with locators, the written-out reasoning, the key terms, the largest overlap of the stem with any teaching text, the option counts in T and in C, and whether the answer is the most frequent option in T.

**Kinds.** Recall (a): the stem is a lead sentence plus the words just before a fact in the source, the options are four strings of equal byte length (numbers or words from the same work or same shape), one correct. Inference (b): the stem gives a premise of the taught material and asks for a quantity that follows by a one-step computation from two taught facts (or one fact and one definition); the options are equal-length numbers or phrases. The written-out chain of each is in `reasoning`. After the checks: **79 recall and 61 inference questions** from Opticks, the Notebooks, the Principia and the Life of Newton, Relativity, Euclid (Casey) and Plato's dialogues. No question comes from stage 5b, from the dictionary, mathematics or physics.

**Common knowledge.** Facts are chosen for being specific to these editions (Newton's measured numbers, Leonardo's rules in braccia, the Life of Newton's dates, Casey's wording of terms, Jowett's wording). That a fact is not common knowledge is a judgement, not a check; the stem-free baseline (section 6) measures how far the options alone give it away.

**Mechanical checks (D, `e5_questions.py`, log in `data/e5/checks.json` and `dropped.json`; thresholds fixed before they were applied):**
1. The four options have equal byte length and are distinct (all 166 authored items pass).
2. The exact stem and every stem plus option, whitespace-normalised, occur in no stage text of T and not in the C extra prose (all pass).
3. Largest common substring (bytes) between the stem and any stage text of T or C: at most 80 for recall and 48 for inference. This bounds how much of a question can be a verbatim continuation of a passage; items above it are dropped.
4. Every source quote of an item is found in the trained part of its stage-5 text (not held out, not in S1 to S4, not in C).
5. For inference: the answer string is searched, as a whole word, in all T text and in C within 100 bytes of any of the item's key terms. For answers of at least 4 bytes, **any hit in T, including one in the item's own source quote, drops the item** (this removed I-PLA-02, I-PLA-03, I-PLA-04, I-PLA-13 and I-EUC-08). Answers under 4 bytes ("two", "ten", "six", short numerals) are exempt, because they co-occur with any key by chance; counts for all items are logged. **14 of the 61 kept inference items are exempt** (I-NOT-01, I-NOT-03, I-PLA-08, I-PLA-10, I-PLA-11, I-PLA-12, I-PLA-14, I-EUC-03, I-EUC-09, I-EUC-10, I-EUC-11, I-EUC-12, I-OPT-07, I-PRI-12); I-PLA-08 and I-EUC-10 have hundreds of key co-occurrences in T and pass only through the exemption.
6. Option counts in T and in C are logged. They are not a criterion; they define the familiarity subsets in section 7.

Authored 166 (92 recall, 74 inference); dropped 26: R-OPT-05 (lcs>80); R-OPT-09 (lcs>80); R-OPT-10 (lcs>80); R-NOT-12 (lcs>80); R-PRI-03 (lcs>80); R-PRI-09 (lcs>80); R-PRI-12 (lcs>80); R-PRI-13 (lcs>80); I-PRI-04 (lcs>48); I-PLA-02 (cooccurrence); I-PLA-03 (cooccurrence); I-PLA-04 (cooccurrence); R-REL-05 (quote-not-in-trained-S5); R-EUC-11 (lcs>80); R-EUC-14 (lcs>80); R-EUC-19 (lcs>80); R-EUC-20 (lcs>80); I-EUC-01 (lcs>48); I-EUC-04 (lcs>48); I-EUC-08 (cooccurrence); I-NOT-14 (lcs>48); I-NOT-16 (lcs>48); I-PRI-09 (lcs>48); I-REL-08 (quote-not-in-trained-S5); I-PLA-13 (cooccurrence); I-REL-10 (quote-not-in-trained-S5). Kept **79 recall, 61 inference.** The answer is the most frequent option in T for 23 of the recall and 11 of the inference items.

**Positions** are shuffled once with `random.Random(f"{seed}/{id}")`, seed 20261020, and fixed in the file.

### E5.6. Measurement (D, specified, not run)

For each arm, at the end of the stream (after all stages, learning frozen from then on):
0. **Pre-run gate (both must pass before any question is scored):** (a) on a few questions, the cost of each option through `set_state(S_end)` equals, bit for bit, its cost on a `m.copy()` of the end-of-stream machine; (b) a SHA-256 of each of `_wn`, `_tags`, `_door` and `_occ` is equal before the first frozen read and after the last (the doorkeeper and occupancy arrays are the ones that could move). If either fails the measurement is redesigned before the run.
1. Take `S_end = m.state`. Define the fixed neutral preamble `PRE = b"\n\nQuestion and answer.\n\n"` (C): the W and S levels do not flush at `\n\n`, so the last bytes of the previous stage (T: the 5b text; C: prose) otherwise leak into every question. PRE is identical in both arms.
2. For each question and each option `o`: `m.set_state(S_end)`; read `PRE + stem` with `learning=False` and discard the probabilities; read `o`'s bytes with `learning=False`; the **cost** is the sum of `-log2 p` over the bits of `o` only. Option lengths are equal by construction, so bits and bits per byte rank options identically; both are saved.
3. The chosen option is the one with the lowest cost. A tie counts as wrong. Nothing is tuned: no scaling, no priors, no calibration.
4. **Stem-free baseline (pre-registered):** the same with `PRE + o` and no stem. The share of items the baseline already gets right, per arm and kind, is reported; the stem's contribution is `cost(o | stem) - cost(o | no stem)`.
5. Save per item and per arm: the four costs, the baseline costs, the choice. All reported figures are computed from that JSON.

Held-out and capacity records (**measurements, not criteria**): at the end of each stage (and, for C, at the matching byte offsets), frozen bits per byte on the held-out passages of every stage up to and including it (total 34.5 MB of frozen reads); `counters(m)` and its per-stage deltas; trained bytes; seconds. This gives, per stage, transfer to unseen passages, forgetting of earlier stages, and how much capacity the stream used.

### E5.7. Criteria (D from the numbers below; C for the choices)

Unit: one question = one trial. Chance is 1/4 (four options, ties wrong, so chance is at most 1/4). All tests are exact; one-sided; alpha = 0.025 per kind (Bonferroni over the two kinds, so the family is 0.05). The three criteria are reported separately and are not combined.

- **P1 (recall, taught above chance):** T is correct on at least X of the N = 79 recall questions, X = 28: the smallest count whose binomial tail under p = 0.25 is at most 0.025 (`stats.json`). It is the same test for inference with N = 61, X = 23, reported as P1-inference (descriptive; the manager's P1 is on recall).
- **P2 (recall, T beats C):** among questions where exactly one arm is correct, T is the correct one at least as often as an exact sign test at alpha = 0.025 requires, **and** T's accuracy minus C's is at least D = 0.10 (8 of 79 items).
- **P3 (inference, T beats C):** the same on the 61 inference questions, D = 0.10 (7 of 61 items).

**Why these N, X, D.** N is what survived the checks; the recommended 60 per kind is met, inference only just (61). X comes from the exact binomial: 28 of 79 is 35.4%, 23 of 61 is 37.7%. D: with about 20 discordant items the smallest gap the sign test can reject is 10 items (12.7% of 79), so the sign test is the binding criterion and D = 0.10 only forbids a "significant" but trivial gap. **Power** (exact, T and C independent, C at chance 0.25; `stats.json`): recall `{"T=0.3,C=0.25": 0.073, "T=0.35,C=0.25": 0.217, "T=0.4,C=0.25": 0.449, "T=0.45,C=0.25": 0.696, "T=0.5,C=0.25": 0.875, "T=0.6,C=0.25": 0.993}`; inference `{"T=0.3,C=0.25": 0.059, "T=0.35,C=0.25": 0.163, "T=0.4,C=0.25": 0.339, "T=0.45,C=0.25": 0.557, "T=0.5,C=0.25": 0.757, "T=0.6,C=0.25": 0.966}`. A true T accuracy of 0.40 is detected with probability 0.45 (recall) and 0.34 (inference); 0.50 with 0.88 and 0.76. A null here is a statement about gaps larger than about 15 points.

**Pre-registered secondary analyses** (reported with the criteria, never replacing them): (i) P2 and P3 on the subset where the answer is not the most frequent option in T (familiarity-matched subset: the numbers in section 5 minus the most-frequent ones); (ii) the stem-free baseline accuracy per arm; (iii) per source work, descriptive only; (iv) accuracy and the T minus C gap with the 28 items that have a distractor absent from all of T excluded (the `option_counts_T` field); (v) P3 on the 47 inference items whose answers are at least 4 bytes (the items that went through the co-occurrence check).

### E5.8. Expectation (S: T15.4; the manager's wording)

Recall above C is possible: the stem is close in form to the teaching text, and E4's byte and word levels can carry a fact over a few dozen bytes. Inference is expected near chance: the premises are in the stem, the answer is a computed quantity that usually appears nowhere in the teaching text, and nothing in the machine adds or compares numbers. Both results are reported. If T beats C on inference, the first suspicion is exposure of the answer's digits and words (section 7 secondary analyses), not reasoning.

### E5.9. Budget (D from saved E4 data)

- Rate: E4 Part 3a read 112.3 MB of training (A-train once, then A-train and B-train) and took 2105.4 s with its evaluations (`results/e4.json` `part3a.seconds`): 18.8 s/MB all-in. Part 2 rows took 531 to 690 s.
- One arm: 103918105 trained bytes plus 34.5 MB of frozen held-out reads at the same rate: about 2600 s, 45 minutes. The questions add 140 x 5 short reads (about 0.9 MB of bits): seconds. Two arms about 1.5 hours; C0 would add about 10 minutes.
- Memory: E4's recorded peak is about 5.0 GiB (HANDOFF T6.4); the register snapshots add kilobytes. Stop above 10 GiB.
- One job at a time. A stage that runs past 1 hour is stopped and a dated amendment written.

### E5.10. Owner decisions, each with a recommendation

- **D1. Stage sizes.** 32 MiB general prose, 24 MiB caps for mathematics and physics (E2 used 64 MiB per stage). Recommend as drafted (about 99 MiB total); raising stage 2 to 64 MiB adds about 10 minutes per arm and moves no question.
- **D2. Mathematics supply.** Only 19.2 MB of mathematics qualifies by the QA rule. Recommend accepting it (a smaller stage 3 is a fact about the supply) rather than adding books by hand.
- **D3. Casey instead of Heath for Euclid.** Heath's *Elements* is not on Project Gutenberg. Recommend Casey (public domain, on Gutenberg, quotes verified). If the owner wants Heath, a file with a verified licence must be supplied and new questions authored.
- **D4. Plato.** The 13 Jowett dialogues in PG-19. Recommend these; the other Jowett dialogues (Timaeus, Symposium, Theaetetus, Laws and others) are on Gutenberg and can be added at the cost of the 5a size, with no question drawn from them.
- **D5. Stage 5b (the authors' own accounts).** Included at the owner's request. Recommend keeping it as drafted: it carries the authors' descriptions of method (Einstein's lectures, Xenophon's Socrates, Leonardo's thoughts) but no question depends on it. If the owner wants questions about stated reasoning, a new batch must be written and frozen before the run, and this draft's hash changes.
- **D6. Overlap with E2.** Stage 2 is E2's stage B; some stage 3 and 4 books are in E2's MANIFEST. Recommend accepting: E5 never uses E2's held-out books as a test, and the overlap is logged (`overlap.json`). Alternative: exclude MANIFEST ids (cost: stage 2 starts later in the same order).
- **D7. Digit filter.** Recommend keeping it, labelled C.
- **D8. C0.** Recommend skipping (no saved E4 end-of-A machine in `results/`; the comparison that answers the question is T against C).
- **D9. Where the fetched texts live.** About 270 raw files and the built streams are not in git. Recommend committing hashes only (`sources.json`) and keeping one tarball outside git; the run verifies every hash before reading.
- **D10. Alpha and multiplicity.** 0.025 per kind, one-sided. Recommend as drafted.
- **D11. Licences.** Recommend the owner confirm the items flagged in section 3.1 (Einstein translations, #29904, #66944) or drop stage 5b.

### E5.11. Risks, each with the cheapest check

| # | Risk | Cheapest check |
|---|---|---|
| R1 | Table saturation: 99 MiB into E4's tables (49% global occupancy after 33 MB, 1.2e9 refusals in E4's A-only arm); late facts may be refused or evicted | The per-stage counters in section 6, read before any question is scored; a run on the first 40 MiB of T shows the trend |
| R2 | Recency: stage 5 is last, so T's facts are the freshest; T versus C does not separate "taught" from "recent" | Report accuracy by source position in stage 5; a later T-reordered arm (stage 5 first) is the clean control and is not in this draft |
| R3 | Familiarity or frequency: answers may be cheap because of their string frequency, not their link to the stem | Stem-free baseline and the familiarity-matched subset (section 7) |
| R4 | Leak through state: last bytes of the previous stage condition every question | The fixed preamble and the restore from `S_end`; check that the cost of a fixed neutral string is the same after each restore |
| R5 | Frozen reads change tables or the snapshot is incomplete | The two pre-run gates of section 6 |
| R6 | Question text taught elsewhere (another edition, an excerpt, a quotation inside a stage 3 or 4 book) | The exact-substring and largest-overlap checks already ran against every stage text and C; rerun on any change of sources |
| R7 | Distractors unseen in T make the answer cheap by familiarity | Option counts logged; the 28-item exclusion analysis |
| R8 | Inference items test arithmetic the machine cannot do, so a gap reflects digit exposure | Section 7 secondary analyses; read P3 with them |
| R9 | Power: inference N is 61; a 40% true accuracy is detected 34% of the time | State the detectable gap with the result; add questions before the run only by a dated amendment |
| R10 | Licences unverified; converted HTML (Principia, Euclid, three Einstein items) may carry conversion artefacts | Owner confirmation; the converter is in the appendix and the converted text is hashed |
| R11 | Network dependence of the fetch | `sources.json` freezes files; verify hashes before the run |
| R12 | Time or memory over budget | One arm first; stop rules in section 9 |
| R13 | The Life of Newton is a nineteenth-century biography inside the Principia file, not Newton's words | Report per source work |

### E5.12. Reproduction

All scripts are inline in the appendix; none is stored under `experiments/`. Order: `fetchlist.py` (catalogue, `qaqc.json` and the first QA/QC fetch), `fetch_html2.py` (recovers books `fetchlist.py` missed), `fetch_s5.py` (dictionary, stage-5 and 5b files), `prose_fetch.py` (prose; needs `excl_ids.json`, listed below), `conv.py` (HTML to text, imported by the fetchers), `e5_texts.py`, `e5_assemble.py`, `e5_overlap.py` (streams), the `items_*.py` files (questions, authored), `e5_questions.py` (checks), `e5_freeze.py` (file and hash), `e5_stats.py`, `e5_headerscan.py`. `excl_ids.json` is the list in section 3.2. Paths in the scripts are those of the drafting machine (`/path/to/e5-work`). Scratch outputs not committed: the fetched texts and built streams (see D9). Committed: this file and `experiments/data/e5/` (`questions.json`, `questions.sha256`, `checks.json`, `dropped.json`, `overlap.json`, `stats.json`, `header_scan.json`, `sources.json`).


## E6. The shared operator level O (DESIGN section 12, V2 and V6; written 2026-10-21, before any E6 code and before any E6 run)

Labels as in DESIGN section 12: **S** source (spec, code, saved result), **D** derived, **C** chosen realization (T15.4). Owner decisions 1 to 6 of DESIGN 12.5 are approved at their defaults (G5, 2026-10-21): O is a chosen realization, its unit is the byte, the machine reads text, and re-entry acts only on emitted symbols. E6 holds E4's chosen row, whose segmentation is the text rule (`hier_segment: text`), labelled as E4 labelled it; decision 4 (modality-free by default, the text rule as an arm) governs the later phases where segmentation is chosen, and E6 does not implement it. Proposal P3 (a mixing node per partial-byte state and a top-mixture fallback) is accepted. Everything below is fixed now; E6 changes only by amendments dated before the run they affect.

### E6.1. Question and claim (S: DESIGN 12.2, V2)

On the bytes that the exact-context tables have not seen, does a shared linear operator level O lower the cost, against E4 and against the same machine with O at rank 1? V1 says the tables cannot generalize to such bytes; V2 says O can, at a bounded risk (a stage costs at most 2 bits per node against its input). E6 measures the size. A null result is a statement about this realization of O on this text, not about shared operators.

### E6.2. Rules kept (S: HANDOFF, T7.1, T10.1, T10.3, T14, T15.3; E5.2)

- E4's machine is E4's chosen configuration (`results/e4.json` part2.chosen), unchanged. With O off, the new code path is bit-identical to E4 (Part 0). No scoring, path cost, beam, softmax, threshold or learned mixer is added; the O stage is the machine's own two-hypothesis coherent mixture (DESIGN 12.1.2), with E3's coherence held at 1.0.
- O is learned forward only. It reads counts of the stream as read; no transcript is stored. Its operators are compiled from counts that end before the block they predict, so no byte is predicted by an operator that has seen it (Part 0 tests this). Frozen reads write nothing, in the tables or in O's counts.
- The cost of a byte is the probability the machine gave it before seeing it, measured outside the machine (PROTOCOL line 7).
- There is no configuration search. A-test is read once per arm. Every constant is listed here (E6.3) and is not tuned.

### E6.3. O (C realization of the observable-operator form of T2.2; D from Hsu, Kakade and Zhang 2009 and Boots, Siddiqi and Gordon 2011)

O has a state `b ∈ ℝᵏ`, operators `B_x ∈ ℝ^{k×k}` for each byte x, a start state `b_1` and a normalizer `b_∞`. The next-byte distribution is `p(x) = b_∞ᵀ B_x b / Σ_{x'} b_∞ᵀ B_{x'} b`; after reading byte x the state is `b ← B_x b / (b_∞ᵀ B_x b)`.

- **Moment store (fixed size, T11.5).** Pasts are the hash of the last `L = 4` bytes into `P = 2¹²` buckets (**C**). L is the byte depth with the largest Page rank in `results/e3_t4.json` T4_2_text.A_train (depth 4, rank 51; the N-L2 turnaround; **D**). Counts: `C2[f, p]` = number of positions with past p followed by byte f; `C3_x[f, p]` = number of positions with past p, byte x, then byte f. `C3` is a dense array of `2²⁸` 32-bit counters (1 GiB, declared; the largest cell count is logged and a cell that would overflow is a build error); `C2` and a split-half copy of `C2` (blocks of 4096 positions, 3 random splits, seed 0, as `e4_hier.level_hankel`) for the Page rank. The store is a fixed-size sum of counts, not a transcript.
- **Compile.** After every `2²⁰` bytes read with learning on (`2¹⁵` bytes in `--smoke`, so that the smoke run exercises the compile, the mixing nodes and gate (f); the code is the same), the operators are rebuilt from the counts alone: the whitened SVD of `C2` (`page.whiten`), the rank `k = min(256, max(2, r))` where r is the Page rank of `C2` with the count-floor rule of `e4_hier.floor_rank` (floor 256, split-half edge, margin 0.1, `page.page_spectrum`); `Ũ = D_r^{1/2} U_k`; `B_x = (Ũᵀ C3_x)(Ũᵀ C2)⁺`; `b_1 = Ũᵀ π` with π the byte marginal; `b_∞ = (C2ᵀ Ũ)⁺ m` with m the past marginal. The state is carried over a recompile by re-projection, `b ← Ũ_newᵀ (Ũ_oldᵀ)⁺ b`. The first block is read with O off (no operators yet). **Bad update rule (C):** if the raw mass `b_∞ᵀ B_x b` of the byte actually read is at most `ε_O · Σ_{x'} |b_∞ᵀ B_{x'} b|`, the state is reset to `b_1` instead of being divided by it; the number of resets is logged per arm. The pseudo-inverse cutoff and the count scale are module constants recorded in the result file.
- **Prediction.** The 256 byte probabilities `b_∞ᵀ B_x b` are floored at `ε_O = 2⁻¹²` and renormalized; the negative-mass share is logged before flooring (per byte: the sum of negative parts over the sum of absolute values; and the share of bytes with any negative part). The probability of bit 0 given the bits of the byte read so far is the mass of the bytes with that prefix and bit 0 over the mass of the bytes with that prefix, clipped to `[ε, 1 − ε]` with the machine's ε.
- **Rank-1 arm (C, flagged: it fixes the comparator of C1).** O's prediction is the empirical byte unigram `π` of the counts at the last compile (the best constant predictor), independent of the state; the stage, the nodes, the floor and the compile schedule are those of the main arm. This is the conservative comparator. The k = 1 spectral compile is not used because on text it gives a past-frequency-weighted distribution, equal to π only for an i.i.d. stream (gate (d)).
- **State size.** `b` (k ≤ 256 floats) and the operators (`256·k²` floats, at most 134 MB) are fixed by k and independent of the stream length.

### E6.4. How O enters the machine (C; P3)

O is a stage of the chain right after level 0 (`depth 8`): after level 0's own prediction `p_0`, `p_0' = coherent_mix(p_0, p_O, share, 1.0)` with `share = 2^EV/(1 + 2^EV)`, and the finer levels then see `p_0'` as the coarser readout, so new sectors are born at O's prediction (isometric nesting, T11.4). There are 255 mixing nodes, one per partial-byte state `u` (the bits of the byte read so far with a leading 1), each with its own evidence `EV_O[u]`, starting at 0 and updated at every bit read with learning on: `EV_O[u] += log2(q_O) − log2(q_{p_0})`, never drained. The proof of DESIGN V2(a) applies to this stage: its cost against its input is at most 2 bits per node, 510 bits over the whole stream. The loop is a copy of `hier._make_loop` with one extra input (the per-bit O probability, absent for the first block); E4's loop is not edited.

### E6.5. Part 0: gates (before any E6 row; each is also a test)

- (a) With O off, the E6 code path equals E4's: the per-bit probabilities on the first 4 MiB of A-train (learning, then frozen) are bit-identical to `hier.HierMachine`, and the E4 arm's A-test equals `results/e4.json` part3a.A_only.A_test to 1e-9.
- (b) O reads counts only: `compile` takes the count store and nothing else, and two stores with equal counts and equal split-half copies give identical operators (the counts are sums, so the order of the blocks inside one compile window does not matter); a compile never reads the stream.
- (c) Convergence to a known HMM: the same compile, run with `L = 1` and one past bucket per symbol, on a stream from a 3-state HMM with distinct emission rows, gives next-symbol probabilities within a mean absolute error of 0.02 of the true ones at 10⁶ symbols with the bad-update rule in place, and a Page rank of at most 3. (The threshold is fixed now; if the build cannot meet it, that is reported before any arm runs.)
- (d) The rank-1 arm's prediction equals the empirical unigram of the counts and does not depend on the state; and the k = 1 compile returns π on an i.i.d. stream.
- (e) Frozen reads leave the count store, the tables and the operators unchanged (hashes before and after, as the E5 gates), and restore O's state exactly.
- (f) The stage regret: on the A-train reading of the E4+O arm, for every node the loss of the stage output exceeds the smaller of its input's loss and O's loss by at most 2 bits; if it does, the build has an error and nothing is reported.
- (g) Re-entry and causality (V6). In generation mode, re-entering an emitted symbol changes O's state unless it is an eigenvector of that symbol's operator, and leaves the counts, the tables and the operators untouched; while reading, external bytes are not re-entered. Perturbing one byte in block j leaves the operators used for block j unchanged (they come from counts that end before block j).
If any gate fails, no E6 arm runs; the cause is found and stated first.

### E6.6. Arms and measures

Arms (one machine each; A-train read once with learning, then frozen reads on copies that share the count store): **E4** (O off); **E4+O** (rank by the Page rule); **E4+O at rank 1**. Measures per arm: A-test and the unseen books of stage A (`A_heldout`) in bits per byte, frozen; the decomposition of A-test by window class (E4 Part 5: the longest of 32, 16 and 8 for which the preceding window occurs in A-train, class 0 if none), share and bits per byte per class; for the O arms the negative-mass share on A-test and A-train, the compiled rank at each compile, the per-node evidences and the stage regret, and the capacity counters (evictions, refused, protected, occupancy) at the end. The E4 arm also reports the E4-reproduction check of Part 0 (a). The new-sector birth point is the only way O touches the tables. Also reported: the number of distinct 4-byte contexts per past bucket, the largest cell of `C3`, the number of bad-update resets, and a by-book bootstrap of the class-0 difference of C1 (resampling the books of A-test, 1000 draws, seed 0), as an uncertainty for the margin, not a criterion.

### E6.7. Criterion, reading rules, fallback (fixed now, not tuned)

- **C1 (the criterion, as in DESIGN 12.4).** On A-test, the bits per byte of the class-0 bytes (class 0 as defined in E6.6) of E4+O are at least 0.02 lower than E4's and at least 0.02 lower than the rank-1 arm's. Both differences are computed in the report from the saved JSON. Class 0 is 23.4% of A-test bytes (`results/e4.json` part5.shares), E4's class-0 cost 3.0868.
- **Reported, not criteria:** the same differences on all bytes, on A-test, and on the unseen books; the gain by window class; O's negative-mass share; the rank-1 comparison on all bytes; the stage regret against its bound.
- **Reading.** If C1 passes: O carries shared structure on text that the tables do not, by at least the stated margin; the size is the measure. If C1 fails and E4+O is not worse than E4 on A-test: O adds less than 0.02 bpb on class 0 here. If E4+O is worse than E4 on A-test by more than 0.0005 bpb (E4's tie band, `TIE`): E6 **loses**.
- **Fallback (P3), only if E6 loses.** The top mixture of DESIGN V2(a): E4 and E4+O are read side by side; their final readouts are mixed by the same two-hypothesis coherent mixture with one node per class of the deepest stored level of the E4+O machine (at most nine classes: at most 18 bits against E4). It is built and run as an amendment dated before it runs, with its own budget; it is not built if E6 passes or merely fails C1 without losing.
- **What E6 does not test.** V3 to V5 (sums, families), the loop and the feedback port. A pass here says nothing about them.

### E6.8. Budget (D, from saved E4 data; to be re-measured by the smoke run)

- E4's A-only arm took about 10 minutes (`results/e4.json` part2.rows 531 to 690 s for A-train plus A-validation; part3a 2105 s with its other units). Each O arm adds the O pre-pass (about `256·k` multiply-adds per byte: `k ≈ 50` gives about 4 × 10¹¹ over A-train) and 33 compiles (each a Gram SVD of a 256 × 4096 matrix and `256` products of size `k × 4096`): estimated 5 to 10 minutes per arm at `k ≈ 50`. **The Page rank on 256 × 4096 counts from 33 million positions may be well above 50, up to the cap of 256**; at `k = 256` the per-byte cost is about `1.3 × 10⁵` multiply-adds (the 256 × k prediction and the k × k update), about `4 × 10¹²` over A-train, which could exceed the 45-minute stop. The smoke run (256 KiB, which cannot reveal the full-size rank) is therefore accompanied by a timing check with k forced to 256 on 1 MiB, extrapolated to A-train. If the extrapolated arm exceeds 45 minutes, the cap on k is lowered **before the run**, by an amendment dated before it and chosen from that timing alone (the largest cap meeting the stop), never from any result. Three arms plus the decomposition and the frozen reads: about 1 hour at `k ≈ 50`, the worst case stated by the timing check.
- Memory: E4's recorded peak is 5.0 GiB; O adds the 1 GiB count store, the operators (at most 134 MB) and the per-block O probabilities (8 MiB per block of `2²⁰` bits, i.e. per 128 KiB of bytes); estimated peak 6.2 GiB. The run is stopped above 10 GiB.
- The E2 corpus must be restored first (`fetch_selection.py` and `build_corpus.py`, T1.2) because E6 reads E4's A-stage streams. Rules: one job at a time; each arm is a unit saved as it ends (`results/e6.json`, partial `e6_partial.json`, `--resume`); an arm that runs past 45 minutes is stopped and the cause (O's cost) written as a dated amendment before it resumes; a failed gate stops the run.

### E6.9. Expectation (S: T15.4)

O is a linear low-rank predictor of the next byte from a state that summarizes the whole stream through learned operators. On bytes whose last 8 to 32 bytes the tables have not seen, the finest levels are absent and the readout is that of short contexts; O can add information about the longer past. How much is the measurement. The spectral estimate is consistent only for finite-rank models (Hsu et al.), and text is not one, so a small gain, none, or a loss are all possible; each is reported. E6 bounds "O helps on unseen contexts" from one side; it does not show that O generalizes a relation (V3) or does arithmetic.


**E6 Amendment 1 (2026-10-22, after the E6 result and `audits/e6_diag.md`; manager). Added after the result, and
labelled so: C1 failed as registered and that verdict stands.** Two diagnostic reruns, each of the E4+O arm only,
against the saved E4 and rank-1 arms, on the same splits, with the same seeds and no other change:
- R-a (diagnosis fix 2): past_bits 14 at L = 4, with C3 over the observed rows. It tests hash mixing against C3
  sparsity.
- R-b (diagnosis fix 1): k forced to 24 at the E6 settings. It excludes rank instability.
Each is read with C1's arithmetic and reported beside E6, never in place of it. Stopping rule: if both leave O at or
above rank 1 on class 0, the reading is "spectral O at L <= 4 learns nothing beyond the byte unigram on this
corpus", and tuning stops. The density form of O is not triggered by this amendment. A stop on time or memory
follows E6's own rules.

## E7. Rotation families on a synthetic relation (DESIGN section 12, V3 to V5, proposals P1 and P2; written 2026-10-22, before any E7 code and before any E7 run)

Labels as in DESIGN section 12: **S** source, **D** derived, **C** chosen realization (T15.4). Phase 2 is recorded as the owner's decision in HANDOFF at `6cacad2` (relayed by the manager), with these changes to DESIGN 12.4: the E4+O arm is dropped, P1 and P2 apply, and the linear-operator control of V3 is kept because the E6 code makes it cheap. The synthetic data are labelled synthetic wherever reported (owner decision 5). Everything below is fixed now; E7 changes only by amendments dated before the run they affect.

### E7.1. Question and claim (S: DESIGN 12.2, V3, V4, V5)

Can a machine that holds E4's tables plus a family stage generalize an additive relation to pairs it has not seen, where E4 alone cannot (V1) and a linear operator cannot (V3)? V4 says a held-out triple is determined by the seen ones if and only if its row is in the integer lattice L of the seen rows, and that the readout of a determined triple is exact (T8.3). V5 says the angles can be learned forward-only from counts, by a joint decomposition of the maps `T_a`, approximately when pairs are missing. E7 measures the V5 route, inside the machine, on the determined held-out triples and separately on the others, and sets it beside the exact V4 route. A null result is a statement about this realization, not about T. **What E7 does not show:** that the machine infers an unknown theorem or learns a textbook (T8.3: the input explicitly supplies the structure). Both family stages carry the additive hypothesis; E7 measures whether a machine with that structural prior uses it, which E4 cannot.

### E7.2. Rules kept (S: HANDOFF, T7.1, T10.1, T14, T15.3; E5.2; E6.2)

- E4's machine is E4's chosen configuration (`results/e4.json` part2.chosen), unchanged, read from scratch on the synthetic stream only (no PG-19 first: the question is the relation, and a taught machine is E9's). With every family stage off the E7 loop is bit-identical to E4's (gate (a)). No scoring, search, softmax, threshold-on-output or learned mixer is added in the machine; a family stage is the machine's own two-hypothesis coherent mixture with coherence 1.0, as O's stage in E6.4.
- A family stage is learned forward only from the counts of the equations read so far (the set of distinct equations and nothing else is stored; fixed size for a fixed token inventory, T11.5). Frozen reads write nothing.
- The cost of a byte and the choice among candidates are measured outside the machine from probabilities the machine gave before seeing the byte (PROTOCOL line 7); no answer is selected by an outside scorer: top-1 among candidates is a reading of the machine's own probabilities, as E5 read its options.
- No configuration search: every constant is listed here; the constants of the V5 route were fixed from a design-time run on stage 1 (E7.7) and are not changed after.

### E7.3. Synthetic data (C, labelled synthetic; P1, P2)

- **Lines.** `a+b=c\n` with decimal numerals and no padding, c = a + b. A **numeral** is a maximal run of ASCII digits: the unit above the byte of P2. It coincides with E4's text-rule word unit on these lines (the other bytes `+ = \n` are not word bytes; gate (h)). Tokens are numerals; the relation is on tokens, never on bytes.
- **Stage 1: single digits.** Operands 0 to 9, results 0 to 18; 55 unordered pairs a ≤ b. **Stage 2: two-digit operands.** Operands 0 to 99 (numerals of one or two digits), results 0 to 198; 5,050 unordered pairs.
- **Splits (P1).** A fraction h of the unordered pairs is held out, h = 0.2 and h = 0.4, drawn with `numpy.random.default_rng(seed)` over the sorted pair list. **Evaluation seeds** stage 1: 10 to 19 for each h; stage 2: 1 to 3 at h = 0.2 and 1 at h = 0.4. Seeds 0 to 9 at stage 1 and seed 0 at stage 2 are **development draws**: the design-time runs of E7.7 and the choice of the V5 constants used them, and no number from them is a result. Nothing is run on an evaluation seed before the run. Both orders of an unordered pair are seen or both held out (commutativity is not what is being learned). The held-out pairs of every draw are saved with their hash.
- **Lattice classification (P1).** For each draw the seen rows `e_a + e_b − e_c` generate the integer lattice L (integer row reduction to a canonical form, `audits/e7_lattice.py`). A held-out triple is **determined** if `e_a + e_b − e_c ∈ L`, **undetermined** otherwise. The held-out set is not chosen to be determined: the draw is random, the classification is computed from the seen rows after the draw, and everything is reported for determined and undetermined triples separately, with the fraction determined per draw. The reduction's cost varies with the draw: a design-time run on three draws of another generator at stage 2, h = 0.4, gave no result in 28 minutes (coefficient growth), while seed 0 here finished in seconds. So at stage 2 the classification has a 15-minute bound per draw; if it does not finish, it is reported as not computed and the items are reported unclassified.
- **Training stream.** The seen ordered equations, `E = 20` epochs, each epoch the seen equations in a fresh random order (the draw's seed), concatenated. The held-out equations never appear. A **relation control** has the same pairs and lines with a random commutative table in place of the sum (c uniform over the stage's results, drawn once per seed), so no additive structure exists.
- **Candidates.** The candidates for an item are all results of the stage (19 at stage 1, 199 at stage 2). An item is a held-out ordered equation (both orders of each held-out pair at stage 1; one order, chosen by the seed, for a sample of 100 held-out pairs per draw at stage 2, to bound the frozen reads).

### E7.4. The family stages (C realization of V4 and V5; D for the algebra)

Both stages work on the set of distinct equations read (a, b, c). Both emit, at an answer byte, a distribution over the stage's result tokens and enter the machine in the same way (below). They differ in how the distribution is made.

- **F5, the V5 route (the arm that E7's criterion reads).** Let `C[a, b, c] = 1` if the equation `a+b=c` was read. V5's joint decomposition of the maps `T_a` is the rank-R canonical (CP) decomposition of this tensor: `C[a,b,c] ≈ Σ_j A[a,j] B[b,j] G[c,j]`, complex factors. For complete data of the closed case (cyclic, every token both operand and result) the columns are the characters and the decomposition is Jennrich's simultaneous diagonalization of the `T_a` (V5, proof in the closed case). With held-out pairs the `T̂_a` have missing columns (V5, item 2), and the standard method of the field for a decomposition with missing entries is the masked fit: the cells `(a, b)` that were read are fitted and the others are not scored (never as zeros). **Fit (C, fixed):** alternating ridge least squares over A, B, G (each update a closed-form solve; the symmetric start `B = A`), ridge `λ = 10⁻³`, initial scale 0.3 from a seeded complex normal (`default_rng(seed + 1000)`), 300 sweeps, **rank R = the stage's result count** (19 at stage 1, 199 at stage 2: T8.3's period of at least the result count; fixed, not read from the data). **Readout (T8.3, Fourier):** for an item (a, b) the amplitude over results is `G (A[a] ∘ B[b])`, and `p_F(c) = (1 − η)·|amplitude_c|² / Σ|amplitude|² + η/R` with `η = 2⁻⁶` (**C**, fixed). **Fit-residual grouping (C, in place of V5's commutator threshold, which needs complete columns):** an operand token whose read cells are reproduced with root-mean-square error above 0.05 (relative to the unit cells) is dropped from the family and the stage abstains on items that contain it. On development draws this separates the additive relation from a random table at h = 0.2 and only weakly at h = 0.4 (E7.7). The stage does not abstain otherwise, so on undetermined items it answers confidently (this is the difference from F4).
- **F4, the V4 route (reported beside F5, and the classifier of P1).** For an item (a, b) the determined candidate is the unique token c with `e_a + e_b − e_c ∈ L`; with none or several the stage abstains. When it fires, `p_F` puts `1 − η` on that candidate and η spread over the other results. F4 is an oracle-like upper bound (V4: determined triples are consequences of the seen ones for every assignment of angles) and carries no learning claim beyond V4.
- **Compile (both).** The stage is rebuilt when the set of distinct equations changes: after epoch 1 (epoch 1 is read with the stages off, as O's first block in E6), and never again, because every seen equation appears in epoch 1. The frozen reads use that final fit. A stage never uses a line before it has been read (gate (e)).
- **Entry into the machine (both).** The same slot as O (E6.4): after level 0, `p_0' = coherent_mix(p_0, p_F, share, 1.0)`, `share = 2^EV/(1 + 2^EV)`, 255 nodes keyed by the partial-byte state u, `EV_F[u]` updated only at the bits of answer bytes read with learning on and never drained; new sectors are born at `p_0'`. **Answer bytes** are the bytes after `=` up to and including the terminating newline, found by a deterministic parser of the byte stream (`numeral + numeral =`). At an answer byte where the stage is active, `p_F` for the next byte is the exact marginal of its distribution over results spelled as `digits\n`, given the answer digits read so far. At every other byte the stage is inactive. The cost of the stage against its input is at most 2 bits per node (V2a, with `own = p_F`), 510 bits over a run.
- **Linear-operator control (V3's claim; cheap because E6 built it).** The E6 operator machine (E4+O, spectral, E6 constants) on the same stream, with its compile every `2¹²` bytes (the streams are far shorter than 2²⁰) and otherwise unchanged. It is a control for V3's prediction that a linear O does not generalize to held-out sums, not an arm of the criterion. Stage 1, h = 0.2 only.

### E7.5. Gates (before any E7 row; each is also a test)

- (a) With every family stage off, the E7 loop equals E4's: per-bit probabilities bit-identical on a random stream of 2¹⁷ bytes, learning then frozen.
- (b) Lattice code: for 200 random small systems, canonical-form equality equals membership by the Hermite normal form of `sympy` (the script of DESIGN 12.9); on a complete stage-1 stream F4's determined candidate is always the right sum.
- (c) Closed case of V5: on addition modulo 10 with every ordered pair read (nothing held out), F5's top-1 is 1.0 on all 100 pairs; on stage 1 with nothing held out, F5's top-1 is at least 0.99.
- (d) Soundness of F4: on every draw, every determined held-out triple's candidate is the true sum (a theorem; a failure is a code error and nothing is reported).
- (e) Causality and frozen reads: frozen reads leave the tables, the equation set and the stage's evidences unchanged and restore the state exactly (hashes, as the E5 gates); changing one line of epoch e leaves the stage used in epoch e unchanged.
- (f) The stage regret of a family stage against its input and against `p_F` is at most 2 bits per node (the E6 gate (f)).
- (g) E4 memorizes: on its own seen equations E4's top-1 is at least 0.99 at stage 1 (otherwise E7 does not interpret the held-out result and stops, reporting E4's training failure).
- (h) The parser agrees with the text rule: on the training stream every answer byte found by the parser lies in the unit that E4's text rule closes at `\n`.
If any gate fails, no E7 arm runs; the cause is found and stated first.

### E7.6. Arms and measures

Arms (one machine per draw, h and arm): **E4**; **E4 + F5** (V5 route); **E4 + F4** (V4 route); at stage 1, h = 0.2, the **E4+O control**; and the relation control for E4, E4 + F5 and E4 + F4 (stage 1, h = 0.2, seeds 0 to 4). Per item: the machine's bits for each candidate (the prefix `a+b=` read once, then each candidate and its newline from the state after the prefix, learning off, as `e5_taught.costs_after`), top-1 (lowest cost), the bits of the correct candidate, and whether the item was determined. Per draw and arm: top-1 on determined held-out items, on undetermined held-out items, on all held-out items, on seen equations (a sanity check of gate (g)), the mean bits of the correct candidate, the fraction of held-out items determined, the number of tokens and components, F5's fit residual and rank, and the fraction of items on which F4 abstained. Reported with every draw (min, median, mean, max across draws).

### E7.7. Criterion, reading rules (fixed now, not tuned)

- **C-E7 (the criterion, as in DESIGN 12.4 with P1).** At h = 0.2, the mean across draws of **E4 + F5**'s top-1 on **determined** held-out items is at least 0.90 at stage 1 and at least 0.90 at stage 2, and exceeds E4's mean on the same items by at least 0.5. The mean is computed in the report from the saved JSON; the minimum across draws and every draw are shown beside it.
- **Reported, not criteria:** the same for F4; h = 0.4; the undetermined items (F5 answers them, F4 abstains, so F4's top-1 there is what E4 leaves); all held-out items together (the unfiltered number P1 warns about); the fraction determined per draw; the relation control (a stage that fires where no additive structure exists is a false fire: F5's firing rate and top-1 on the control against chance, F4's firing rate); the E4+O control (V3's prediction: near E4's).
- **Reading.** If C-E7 passes: a machine whose family stage learns the angles forward-only from counts by the V5 route generalizes this additive relation to the held-out pairs that the seen ones determine, and E4 and the linear O do not. If it fails while F4 passes: the V4 algebra holds, and the learned route is the limit (the masked decomposition does not recover the characters from this many cells); the F5 versus F4 gap on determined items, and F5's accuracy as a function of how many pairs are held out, measure it. If both fail with gates passed, the exactness is lost in the mixture or the readout; the evidences and per-item bits locate it. The undetermined items cannot be answered by any method that uses only the seen triples (V4); they are not a failure of a stage.
- **Design-time numbers (D; not results; `audits/e7_lattice.py` for the lattice, and the V5 route run once on stage 1 with the constants of E7.4 before this section was written).**
  - Fraction of held-out triples determined at stage 1 over 60 draws: 0.942 at h = 0.2 and 0.902 at h = 0.4 (minimum 0.636 and 0.727); DESIGN V4 reports 0.933 and 0.930 over a different draw. At stage 2, h = 0.2, three draws: 1.000.
  - F5 at stage 1 (ten draws): top-1 on determined held-out items 0.92 at h = 0.2 (minimum 0.75) and 0.50 at h = 0.4; on undetermined items 0.00. With 1,000 sweeps 0.93; with R = 12 0.76 and with R = 30 0.86 at h = 0.2, so the bar of 0.90 is at the edge of what this route gives at stage 1. Operands 0 to 29 (R = 59, one draw): 1.00 at h = 0.2 and 0.99 at h = 0.4. Operands 0 to 99 (R = 199, one draw, 60 sweeps): 1.00 at h = 0.2 and 0.998 at h = 0.4, with every held-out item determined at h = 0.2; the fit takes seconds.
  - Relation control, stage 1, ten draws: F4 fires on 0 of the held-out items at both h (rank of L about 18; no unique candidate), so it never fires on a random table; F5's top-1 on the random table is 0.01 (h = 0.2) and 0.05 (h = 0.4), chance being 1/19 = 0.05. The per-token relative residual of the F5 fit (`audits/e7_residual.py`) is 0.0025 (maximum 0.008) on the sum and 0.29 on a random table at h = 0.2, so the 0.05 rule abstains on every token of the random table there; at h = 0.4 it is 0.019 (3% of tokens above 0.05) on the sum and 0.066 (76% above 0.05) on the random table, a weak separation. The threshold 0.05 was written before these were measured and is not tuned.
  - A naive zero-filled Jennrich diagonalization of one random combination of the `T̂_a` (no masking) gives top-1 of 0.24 (modulus 10, h = 0.2) and 0.03 (modulus 100): the V5 closed-form route does not survive missing pairs, which is why F5 is the masked decomposition.

### E7.8. Budget (D; to be re-measured by the smoke run)

E4's rate is 28.6 s/MiB (`results/e6_timing.json`). Stage 1: the stream is about 10 KB, so a unit is dominated by building the machine and the frozen candidate reads (about 110 items × 19 candidates), under 30 s; F5's fit is about 1 s. Units: 20 draws × (E4, F5, F4) + 10 O-control + 15 controls = 85 units ≈ 45 minutes. Stage 2: about 1.5 MB per draw (≈ 45 s) plus 100 items × 199 candidates of frozen reads (an estimated 2 to 3 minutes) and F5's fit (R = 199: seconds at 60 sweeps, to be measured at 300 by the smoke run) plus the lattice (about 3 minutes per draw at h = 0.2 in the design-time run): 4 draws × 3 arms = 12 units ≈ 45 minutes. Total about 1.5 hours; stop a unit at 15 minutes (cause written as a dated amendment before it resumes). Memory: E4's tables and a stage ≈ 3 GiB (E6's peak was 2.7 GiB); stop above 10 GiB. Each unit is saved as it ends (`results/e7.json`, partial `e7_partial.json`, `--resume`). One job at a time.

### E7.9. Expectation (S: T15.4)

E4 reads each seen equation from its tables and has nothing on a held-out one (V1); I expect its top-1 on held-out items near the frequency of the most common result. F4 is exact on the determined items by V4. F5 is the learned route: at stage 1 and h = 0.2 the design-time run puts it around 0.92 on determined items, so a pass is not guaranteed and a fail at h = 0.4 is expected. The linear control is expected near E4 (V3). The relation control should show no gain; at h = 0.2 F5 should abstain by the residual rule, at h = 0.4 it may fire, and a loss bounded by the stage regret (2 bits per node) is then possible. E7 does not test arithmetic on unseen operands, carry beyond the numeral token, or learning without the additive hypothesis; those are V8, E8 and the later phases.

**E7 Amendment 1 (2026-10-22, before any evaluation-seed run; manager's ruling on the engineer's draft, below as
written).** Rulings:
- The stage sits at the top slot, after the deepest level. The slot was chosen on development seeds only.
- The after-level-0 ablation is run and reported as a measurement of dilution, not a criterion.
- Stage-1 evaluation seeds are 20-29 and control seeds 20-24, because 10-11 were exposed by a smoke run. Stage-2
  seeds are unchanged.
- Gate (c) reads the fitted readout. That the 0.05 residual rule does not fire on the modular closed case is
  reported as a finding.


Written after the E7 build and a smoke run, before any evaluation draw was run. Evidence below is from development
seeds only (stage 1 seeds 0 to 2, stage 2 seed 0, h = 0.2); no number from them is a result.

## 1. Slot of the family stage (replaces the "Entry into the machine" slot of E7.4)

**Proposal:** the stage enters after the top level, mixing the final readout of the chain with `p_F` (nodes keyed by
the partial-byte state, evidence updated at answer bits only, never drained), instead of after level 0.

**Evidence (stage 1, h = 0.2, seeds 0 to 2, determined held-out items):**

| slot | arm | top-1 | bits of the correct result |
|---|---|---|---|
| after level 0 (E7.4 as pushed) | E4 | 0.000 | 7.99 |
| after level 0 | E4 + F5 | 0.102 | 3.43 |
| after level 0 | E4 + F4 | 0.102 | 3.36 |
| top | E4 + F5 | 1.000 | 0.55 |
| top | E4 + F4 | 1.000 | 0.46 |

Stage 2 (seed 0, h = 0.2, slot after level 0): E4 + F5 and E4 + F4 both 0.48 top-1 (E4 0.00).

**Reason (D, a reading of the mechanism, not separately tested):** sectors below the stage see its prediction as
their coarser readout, but the finer levels' evidence accumulated in epoch 1 (the stage is off then) is never drained
at sink rate 0, so on a held-out equation the finer levels' memorized answers override a stage placed early. At the
top the stage is mixed with the final readout, where on held-out equations the readout is unconfident.

**Consequences.** The proof of V2a (cost at most 2 bits per node against the stage's input, 510 bits over a run) holds
at any slot, since it is relative to the stage's input; the gate (f) check is unchanged (max excess 1.02 bits in the
runs above). New sectors are no longer born at the stage's prediction. The script has `--slot top`; the default stays
`after0` until this is ruled.

**Optional ablation (cost about 3 minutes):** also report the E7.4 slot (`--slot after0 --stages 1`, writing
`results/e7_after0.json`), since "does the exactness survive the machine's mixture" is itself a finding.

## 2. Seeds (replaces the seed lists of E7.3 and E7.6)

The first smoke run used stage-1 seeds 10 and 11, and the slot decision above followed from it. Stage-1 evaluation
seeds move to **20 to 29** (both h), the relation control to seeds **20 to 24** (E7.6 said 0 to 4, which E7.3 lists
as development draws), development draws are stage 1 seeds 0 to 19 and stage 2 seed 0. Stage 2 evaluation seeds
(1 to 3 at h = 0.2, 1 at h = 0.4) are unaffected: no stage-2 evaluation seed has been run.

## 3. Gate (c) (replaces its wording in E7.5)

Gate (c) was written as "F5's top-1 is 1.0 on all 100 pairs of addition modulo 10". F5 as a stage scores 0.1 there:
after 300 sweeps the rank-10 fit of the closed case leaves a per-token residual of 0.32 to 0.45, the 0.05 rule drops
every token and the stage abstains. The gate is therefore measured on the fitted readout (`fourier_readout`), which
gives 1.0; on stage 1 with nothing held out the stage itself (rule included) gives 1.0 with mean residual 0.0016.
**Finding:** the 0.05 rule would not fire on V5's own closed case in the modular form at 300 sweeps; it separates the
integer relation from a random table at stage 1 (E7.7), and nothing more is claimed for it.

## 4. Time (replaces E7.8)

Measured on development draws (hier at 28.6 s/MiB): stage 1, 6 s per unit (the smoke run: 18 units in 107 s), 85
units, about 9 minutes. Stage 2 (seed 0, h = 0.2): E4 32 s, F4 46 s, F5 155 s (the F5 fit takes 115 s at R = 199, 300
sweeps; the lattice takes 1.8 s on this draw), about 4 minutes per draw for three arms; four draws about 16 minutes.
The lattice is built once per draw and shared by the arms. Total about 30 minutes; the 15-minute bounds stand.
Memory: the smoke run's peak was not measured separately; E4's tables dominate (about 3 GiB expected, stop above 10).


## E6b. Word-class levels on text (DESIGN section 12, V2′; written 2026-10-22, before any E6b code and before any E6b run)

Labels as in DESIGN section 12: **S** source, **D** derived, **C** chosen realization (T15.4). E6b is the owner's decision relayed in HANDOFF (`6cacad2`) and the manager's brief of 2026-10-22, after E6 (G6: O added nothing beyond the byte unigram) and E7 (G7 passed). E6b holds E4's chosen row (`results/e4.json` part2.chosen), whose segmentation is the text rule (`hier_segment: text`, labelled as E4 labelled it). Everything below is fixed now; E6b changes only by **at most one** amendment, dated before the run it affects (E6b.7 reserves it for the top-mixture fallback).

### E6b.1. Question and claim (S: DESIGN 12.2, V2′)

On text bytes whose exact last-2-unit context the tables have not seen, do levels keyed by the **classes** of the last 1, 2 and 4 units lower the cost, against E4 and against the same levels with a **random** class map of the same size and class sizes? V2′ says the coarsening loss of a class map is exactly its within-class mutual information (D) and that pooling helps where counts are scarce; E6b measures the size. A null result is a statement about this clustering of this text, not about canalization (T11.2). The random control matters: E6 showed that any added stage has a state-independent effect (the byte unigram alone: 0.037 bpb on class 0).

### E6b.2. Rules kept (S: HANDOFF, T7.1, T10.1, T14, T15.3; E5.2; E6.2)

- E4's machine is E4's chosen configuration, unchanged. With the K levels off the E6b code path is bit-identical to E4 (gate (a)). No scoring, search, softmax, threshold-on-output or learned mixer is added; K levels are ordinary upper levels of the chain.
- The class map is learned forward only. It reads counts of the unit stream as read; no transcript is stored. Each map is built from counts that end before the block it is used on (gate (d)). The map is fitted to counts, never to costs or answers, and there is no backpropagation through the machine. Frozen reads write nothing (tables, counts, map).
- The cost of a byte is the probability the machine gave it before seeing it, measured outside the machine (PROTOCOL line 7). There is no configuration search; A-test is read once per arm; every constant is listed in E6b.3.

### E6b.3. The class map (C realization of T11.2; D from Brown et al. 1992 and Levy and Goldberg 2014)

- **Units (S).** E4's text-rule units: a unit closes at the first non-word byte and includes it; its code is the 64-bit hash E4 already holds (the W ring). The unit stream is the sequence of these codes.
- **Count store (fixed size, T11.5).** 2¹⁷ rows indexed by the top bits of the unit code; per row a majority-vote tag (one 64-bit code and a count, the Boyer–Moore rule: a matching code increments, another decrements, a zero count takes the new code), and for each side (previous and next unit) 512 counters (32-bit) indexed by `code mod 512` of the neighbor's code: 2¹⁷ × 1,024 × 4 B = 512 MiB. After each block read with learning on, every unit of the block updates its row's tag and its left and right counters (units of the block boundary are carried over); the store is a fixed-size sum of counts, not a transcript.
- **Row size (decided at design time).** With 2¹⁷ rows, 3,944 of the 32,768 most frequent units lose their row, and 5.8% of A-train unit tokens (6.9% of A-validation's) end in class C against 4.7% for the ideal top-32,768 map: kept, because the loss is about one point of tokens.
- **Compile.** After every `2²⁰` bytes read with learning on (`2¹⁵` bytes in `--smoke`), from the counts alone: the `V = 2¹⁵` rows with the largest tag counts (at least 4); the matrix of positive pointwise mutual information of row against feature over those rows (`max(0, log(n_rf · T / (r_r · c_f)))`, no smoothing); its leading `d = 64` right singular vectors (the eigenvectors of `PᵀP`, 1,024 × 1,024, float64), the rows projected on them and normalized to unit length; then **spherical k-means with C = 256 classes**, each row weighted by its tag count, 15 Lloyd iterations. The first compile starts from 256 rows drawn without replacement with `default_rng(0)` and weights; each later compile starts from the weighted class means, in the new coordinates, of the previous map's assignment (rows that were in the previous map), and a class with no member is reseeded from a random row, so class ids persist from compile to compile. The map is the arrays `tag[row]` and `class[row]` for the selected rows. **A unit's class** is `class[row]` if the row is selected and `tag[row]` equals the unit's code; otherwise the reserved class **C (256, unknown)**. Before the first compile every unit is class C.
- **Why this method, and C = 256 (C).** Brown clustering (greedy merging by average mutual information, Brown et al. 1992) is the field's standard but is quadratic in the vocabulary per merge pass; the exchange algorithm (Martin, Liermann and Ney 1998) is incremental but also a greedy search per word. PPMI plus a low-rank factorization plus k-means (Levy and Goldberg 2014; Schütze 1995) is the cheaper standard, reuses the Gram/eigen machinery of E6's compile, is deterministic given the seed, and its coarsening loss is the quantity of V2′(a) up to the quality of the fit. 256 classes is the manager's suggestion, fixed in advance, not tuned.
- **K levels (C).** K1, K2 and K4 are keyed by the classes (ids 0 to 256) of the last 1, 2 and 4 closed units, the partial unit's running hash and the partial byte, with their own salts: the same key shape as W1, W2 and W4 with the unit codes replaced by classes. Each has 2²¹ slots, verify 4 (E4's upper levels' value, `hier_verify`), the same doorkeeper and the same governor. **Declared order in the mixture: byte levels, K1, W1, K2, W2, K4, W4, S0, S1** (each K level is the class counterpart of the next W level: a W context determines its K context, so a new W sector is born at its class counterpart's readout, T11.4; the order is never selected).
- **Random control (C).** The same machine with a **random class map**. "Class sizes" means **rows per class** (the number of units in each class), matched; the token mass per class is reported for both maps and is not matched (design time: the class with the most mass holds about 1.67 million tag counts in the real map and about 1.72 million in a rows-matched random map, the median class 12 thousand against 7 thousand, so the mass distributions are close at the top; the manager may rule to match token mass instead). At each compile the rows selected get control classes drawn so that the row counts match the real map's: at the first compile a random permutation (`default_rng(0)`) of the real classes over the selected rows (sizes exactly equal); at later compiles a row that already had a control class keeps it (persistent, so the control's contexts are as stable as the real map's), and a newly selected row takes the real class of a uniformly random other newly selected row. The control never uses the factorization or the k-means. Its class sizes at the end are reported beside the real ones.

### E6b.4. Sets and arms

- **Class 0** (E4 Part 5, as E6.6): A-test bytes whose preceding window of 32, 16 or 8 bytes is absent from A-train.
- **U2, the bytes with an unseen last-2-unit context (D).** An A-test byte is in U2 if E4's W2 context of that byte, the pair (last two closed unit codes, running hash of the partial unit) at the byte's first bit, occurs at no byte position of A-train. It is computed once, offline, from the stream alone by the pre-pass keys (the same key function the machine uses), and saved with its hash. Design-time sizes on A-validation (`audits/e6b_coverage.py`; an upper bound, E6b.9): U2 is 41.8% of the bytes and holds 72% of class 0. **Disclosure:** the first version of that script read A-test unit keys once, at design time; no number from it is used here and the design was not changed by it (the numbers cited are from A-validation, the declared store). U1 (the same for the last unit) is reported for reference.
- **Arms.** **E4** (K off; the E6b code path); **E4+K** (the clustered map); **E4+K random** (the control). Each arm reads A-train once with learning, then A-test and A-heldout frozen on copies that share the count store and the map.

### E6b.5. Gates (before any E6b arm; each is also a test)

- (a) With the K levels off, the E6b pre-pass and machine equal E4's: per-bit probabilities bit-identical on 4 MiB of A-train (learning, then frozen), and the E4 arm's A-test equals `results/e4.json` part3a.A_only.A_test to 1e-9.
- (b) The pre-pass keys of the K levels equal a pure Python writing of the same state machine on random text; the keys of W1 to S1 equal `hier`'s.
- (c) The compile reads the count store and the previous map and nothing else; two stores with equal counts and an equal previous map give equal maps; the control has the real map's class sizes at the first compile (equal multisets) and keeps assignments across compiles.
- (d) Causality: perturbing a byte in block j leaves the map used for block j unchanged; frozen reads leave the tables, the count store and the map unchanged (hashes, as the E5 gates).
- (e) A synthetic check of the clustering: on a generated stream whose words come from 8 known classes (same-class words share the successor distribution), the compile recovers the classes with purity at least 0.9.
If any gate fails, no E6b arm runs; the cause is found and stated first.

### E6b.6. Measures

Per arm: A-test and A-heldout in bits per byte, frozen; the decomposition of A-test by window class (E4 Part 5) and by U2 against its complement and by class 0 ∩ U2; the occupancy and capacity counters of E4 at the end of A-train; the per-book class-0 and U2 bits, for a by-book bootstrap of each difference of the criterion (1,000 draws, seed 0), as an uncertainty, not a criterion. For the K arms: compile seconds, rows selected, the share of A-test unit tokens in class C, the class-size quantiles, the share of unit tokens whose class changed between consecutive compiles (stability), and for the real map the within-class purity against the control's.

### E6b.7. Criterion, reading rules (fixed now, not tuned)

- **C-E6b.** On A-test, the bits per byte of the class-0 bytes **and** of the U2 bytes of E4+K are each at least **0.02** lower than E4's and at least 0.02 lower than E4+K random's: four differences, all computed in the report from the saved JSON.
- **Why 0.02 (D, from E6).** It is E6's C1 margin. E6's by-book bootstrap of a class-0 difference had a 95% half-width of about 0.002 over 83 books, so 0.02 is about nine times the sampling noise; it is 0.65% of E4's class-0 cost (3.087) and, if the gain were confined to U2 (42% of the bytes), at least 0.008 bpb on all bytes. E6's byte unigram alone was worth 0.037 on class 0, which is why the comparator that matters is the random control.
- **Reported, not criteria:** the same differences on all bytes of A-test and on the unseen books (A-heldout); the gain by window class and on class 0 ∩ U2; U1; E4+K against the control on all bytes; the stability and purity measures.
- **Reading.** If C-E6b passes: classes of units carry shared structure that the exact contexts and a size-matched arbitrary partition do not, by the stated margin. If it fails and E4+K beats the control by less than 0.02: the classes help less than the margin here. If E4+K is not better than the control (within 0.0005 on A-test, E4's tie band): the effect of the K levels is capacity, not classes. If E4+K is worse than E4 on A-test by more than 0.0005: E6b **loses**, and the one amendment may build V2a's top mixture (E4 and E4+K, one node per class of the deepest stored level), dated before it runs.
- **What E6b does not show.** That classes are semantic; anything about reasoning, relations or the loop; that VAM beats lzma.

### E6b.8. Budget and stop rules (D; to be re-measured by the smoke run and a timing check)

- E6's E4 arm took about 15 minutes; the K arms read three more levels (14 against 11) and add 31 compiles. The design-time clustering step (Gram eigenvectors of 32,768 × 1,024, k-means, 15 iterations) took about 16 seconds including the counting, so 31 compiles are under 10 minutes. Estimate: E4 about 15 minutes, each K arm about 20 to 25, total about 1 hour. A **timing check** before the run reads 3 MiB of A-train with K on, reports seconds per MiB and per compile and extrapolates to A-train. **Pre-declared rule (not an amendment):** if a K arm extrapolates over 45 minutes, V (the rows clustered) is halved, repeatedly, until the extrapolation is under 45 minutes, from that timing alone and before the run; the result file records V. The one allowed amendment is reserved for the top-mixture fallback of E6b.7 and takes precedence for it; a timing problem the V rule cannot fix stops the run and goes to the manager.
- Memory: E6's peak was 2.7 GiB; the count store adds 0.5 GiB, three K levels 0.45 GiB of tables and doorkeepers, the class arrays and the block buffers a little: estimated 4 GiB; stop above 10 GiB.
- The E2 corpus must be restored first (T1.2). One job at a time; each arm is a unit saved as it ends (`results/e6b.json`, partial `e6b_partial.json`, `--resume`); an arm past 45 minutes is stopped and the cause written as a dated amendment before it resumes; a failed gate stops the run.

### E6b.9. Expectation (S: T15.4)

The class levels have counts where the exact levels have none (design-time, A-validation, an upper bound: a seen K2 context for 57% of U2 and a seen K1 context for 91%), but a class context carries less than a word context (V2′(a)), and the first bytes of the next word, where a class helps most, are a minority of the bytes of U2. The field expects a moderate gain at best (Brown et al. 1992; Goodman 2001). A small gain, none, or a loss are all possible; each is reported. E6b bounds "classes of units generalize on text"; it does not show reasoning and does not promise to beat lzma.


**E6b note (2026-10-22, before the full run; manager).** Gate (e) is fixed at 16 clusters, seed 0, on 8 known classes
(purity 0.996). Recovery depends on the k-means start: at C = 8 it ranged 0.75-0.996 across seeds, and at 16 clusters
0.87-0.996 over seeds 0-7. This is reported as a finding. The run's clustering is unchanged, and the one amendment is
still reserved for the fallback.

**E6b Amendment 1 (2026-10-22, after the E6b result and `audits/e6b_diag.md`; manager). Added after the result, and labelled so: C-E6b failed as registered, E4+K lost to E4 on A-test by more than the tie band, and both verdicts stand.** This is the amendment E6b.7 reserved, the top mixture of DESIGN V2a, and the only one E6b allows.
- **What it is.** E4 and E4+K read side by side from the same bits. Their final per-bit readouts are mixed by the machine's own two-hypothesis coherent mixture (coherence 1.0, `share = 2^EV/(1 + 2^EV)`), with **one node per class of the deepest contributing level of the E4+K machine** (the deepest level whose sector is present and mature at that bit, as the loop's own test; level 0 always contributes, so there are at most 14 classes for its 14 levels). Each node's evidence starts at 0, is updated at every bit read with learning on (`EV += log2(q_K) - log2(q_E4)`) and is never drained; frozen reads update nothing. By V2a the mixture costs at most 2 bits per node against the better of its two inputs: **at most 28 bits** over a stream. (E6b.7 gave no count; V2a's "nine classes" was for E4's chain, not E4+K's.) The mixture reads the finished readouts of two machines, so it can recover E4's exactness where the deepest level is deep and keep K where it is shallow; it does not change what happens inside E4+K.
- **Arms and splits.** Two arms, each reading A-train once with learning and then A-test and A-heldout frozen, on the same streams and with the same seeds and constants as E6b: **M** (E4 with E4+K) and **M random** (E4 with E4+K random). The component machines are the E4 and K arms of E6b and must reproduce their A-test bits per byte in `results/e6b.json` to 1e-9 (gate).
- **Criterion and reading.** C-E6b's arithmetic is applied to the mixtures: the class-0 and U2 bits per byte of M at least 0.02 below E4's and at least 0.02 below M random's (four differences, computed in the report). It is reported **beside the original** C-E6b, never in place of it, with A-test, A-heldout, the by-book bootstrap, and the share of bits at which each node prefers K. M against E4 on A-test is read with E4's tie band (0.0005).
- **Gates (each also a test).** (a) the loop copy that also outputs the deepest contributing level equals hier's loop with the added lines removed, and the machines built on it equal the E6b machines bit for bit; (b) the mixture's regret against the better input is at most 2 bits per node on A-train; (c) frozen reads leave the tables, the count store, the map and the node evidences unchanged; (d) the components reproduce E6b's A-test to 1e-9.
- **Development check first (manager's ruling, 2026-10-22).** A check on A-validation only, no A-test read (`--dev-check`: a 3 MiB prefix of whole A books, frozen reads on their validation parts; variants E4, E4+K as built, the map fixed after its first compile, the K levels after S1; `results/e6b_dev.json`, reading in `audits/e6b_dev.md`) precedes the run of this amendment and decides its content, the fallback above (A) or the K-after-S1 order (B), before any A-test read; the manager rules on it. **Dev check decided A, 2026-10-22** (manager's ruling after `audits/e6b_dev.md`: option B, the K-after-S1 order, is rejected by the check and the staleness candidate is not supported); the amendment's content is the fallback above (A), run by `experiments/e6b_mix.py` (`--smoke`, `--resume`, results in `results/e6b_mix.json`, read by `report.py` beside the original C-E6b). No A-test read preceded this decision. No further amendment to E6b follows; the result is reported as it stands.
- **Budget.** Two machines side by side: about 5 to 6 GiB, stop above 10 GiB. Reading A-train takes about 20 minutes per arm (E4 426 s and E4+K 731 s in E6b), 40 to 60 minutes for both with the frozen reads; an arm past 45 minutes is stopped. One job at a time; each arm saved as it ends (`results/e6b_mix.json`, partial `e6b_mix_partial.json`, `--resume`). No further amendment.

## Reporting rules

- Every number in `RESULTS.md` comes from a saved JSON file, which records its commit and seeds.
- Negative or null results are reported in the same table as positive ones.
- The capability claims follow T15.4. VAM-1 measures online prediction and generation from raw bits
  with no stored history and no backpropagation. Whether that is textbook-level understanding is the
  open question, and these measurements bound it from both sides.
