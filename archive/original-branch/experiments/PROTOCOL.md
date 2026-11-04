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
  *Amendment of 2026-10-22, during the run, before Part 2 started:* the first measurement (run of
  2026-10-18 at commit 6fbfdc7) gave rank 0 at every depth. Cause: the analysis kept every observed
  context and continuation, and rare ones whose counts fall in alternate halves whiten to a block of
  singular value 1 in the split-half difference, so the edge rose to 1 above every direction
  (`tests/test_nfa_page.py::test_rare_cells_cannot_lift_the_edge` reproduces it). The text profile now
  uses the verified support: contexts and continuations seen at least 8 times in each half of every
  split (8 is E2's threshold for a familiar context). Both measurements are reported. The known
  sources keep the original analysis, where every context is frequent.
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

*Amendment of 2026-10-22, after the independent review (AUDIT.md, Audit 2), during Part 2:* (1) the
check N-L1 (c) started from the basis state, where the observation flow does not move, so it held by
construction; it now starts after three reading events (a mixed state), with the same criterion.
(2) The verified support of N-L2 is trimmed to a fixed point (rows and columns re-counted inside the
support until neither changes); at min count 8 this changes no number of the text profile (checked on
the same 4 MiB), so the turnaround that Part 2 uses is unchanged. N-L2 also reports the text profile at
min counts 1 and 4, to show how the turnaround depends on the threshold. (3) When Parts 2 to 4 finish,
Part 1 is run again in full at the final code, so that every Part 1 number comes from one commit; the
first N-L2 text measurement stays as recorded. Parts 2 to 4 are not re-run: they use only the
turnaround, which these changes leave as it was, and the text machine, which they do not touch.

*Amendment of 2026-10-22, after Part 2 ended (post hoc, exploratory):* Part 2 chose the smallest sink
rate (0.1) and the smallest protection (0.5 bits) of its grid, and the A-validation bpb fell
monotonically toward both edges. `experiments/e3_explore.py` scores, on A-validation only, the chosen
configuration with sink rates 0.05, 0.025 and 0 (no sink: a protected sector is never drained) and with
protection 0.25 at sink rates 0.1 and 0.05; its first row recomputes the chosen configuration, which must
give Part 2's value exactly. It is reported apart from E3, as an observation chosen after seeing the
data; it changes neither Part 2's choice nor Parts 3 and 4, and no test split is read.

## Reporting rules

- Every number in `RESULTS.md` comes from a saved JSON file, which records its commit and seeds.
- Negative or null results are reported in the same table as positive ones.
- The capability claims follow T15.4. VAM-1 measures online prediction and generation from raw bits
  with no stored history and no backpropagation. Whether that is textbook-level understanding is the
  open question, and these measurements bound it from both sides.
