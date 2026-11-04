# Variational Action Machine (VAM): build design

Status: build plan, written before implementation (2026-10-18). Updated in place as decisions are made.

Specification: the owner's two documents, stored unchanged in `spec/`:

- `spec/VAM_Coupled_Theory_and_Proofs.md` (cited below as **T**, with its section numbers, e.g. T10.2)
- `spec/VAM_Framework_Foundational_Principles.md` (text of the owner's DOCX, cited as **F**, e.g. F-Part 7)
- `spec/VAM_Nested_Fractal_Architecture.md` (the owner's seven-layer design of 2026-10-18, cited as **N**, e.g. N-L5; section 10 below)

This build is standalone. Nothing from earlier SERA or SERA-U work is used.

---

## 1. In simple words

The machine is a box whose whole memory is its present state: one big table of numbers (a *density matrix*) that says how likely each internal arrangement is and how the arrangements relate to each other. Bits come in one at a time. Each bit nudges the state. The state then moves by itself according to physics-style rules (the Dirac equation, which makes changes spread at a limited speed). The outside world "looks at" only a few boundary parts of the state, and that look turns fuzzy possibilities into one definite output bit. That output bit is immediately fed back in, so the machine changes because of what it just said. Then the state "settles" (canalization): details that can never matter again are dropped, everything that can matter is kept. Nothing is ever written to a history log.

Learning happens forward only: when a relation keeps showing up in the state, a slow "weight" drifts toward it (like a riverbed slowly shaped by water). No backpropagation, no replay of old data.

What we will be able to show for certain: every rule is implemented exactly as the documents define it, every worked example in the documents is reproduced exactly, every proved property is checked by tests, and the complete machine runs on any file's raw bits. What is still an open scientific question (the documents say this themselves, T15.4) is whether these rules, left to run on real data such as textbooks, grow the "right" structure for understanding. We build the testbed for that question and report the measured answer, whatever it is.

---

## 2. Ground rules taken from the documents

1. **Three labels on everything** (T1.1): *source statement* (in F or the author's clarifications), *derived result* (proved from stated equations), *chosen realization* (a specific coupling, encoding or number picked to make a calculation determinate). Every module docstring and every realization record carries these labels. A chosen realization is never presented as the author's unique law.
2. **Present-state execution** (T4.1, T10): the machine object holds the complete current state and nothing that grows with the number of events. No transcript, no replay buffer, no context window.
3. **No reverse-time learning** (F-Part 3, T3.3, T12.1): weight changes come from the same interaction that drives the vectors, evaluated forward on the current state.
4. **Positivity is structural** (T3.2, T10.2): every state update is a completely positive map; densities are never multiplied directly; normalization happens only where the mathematics conditions on an observed outcome.
5. **Complex amplitudes stay complex** (T7, T15.3): path sums keep phases; nothing is replaced by softmax or path costs.
6. **Numerical choices are declared before a run** (T14): each run reads a frozen realization record; results are written to files; tables are generated from files, never typed by hand.
7. **Zero-probability observations are not normalized** (T10.2): they raise, or the bit is injected without interpretation, or the represented space is extended by an isometry (T11.4), by declared policy.
8. **Honest scope** (T15): proved, calculated and open claims are kept apart in every report.

---

## 3. Architecture

Python 3.13 reference implementation. NumPy and SciPy for numerics (complex128 throughout), SymPy for exact rational and symbolic checks of every exact number in T, pytest and Hypothesis for tests. Speed comes after correctness (T13.4 says no throughput figure is established without a complete kernel and measured memory traffic, so we measure instead of assuming).

```
vam/
  DESIGN.md  README.md  RESULTS.md (generated)  pyproject.toml
  spec/                         the two source documents, unchanged
  src/vam/
    linalg.py        Pauli/Clifford algebra, partial trace, entropies, purity, trace distance, matrix functions
    space.py         named tensor registers (V, W, B, E, ...), embedding local operators, reduced states
    density.py       validated density operators and constructors (pure, mixture, R = AA†/Tr)
    channels.py      Kraus maps, instruments, Stinespring/collision channels, Liouville and Choi forms, CP/TP checks
    lindblad.py      Lindblad generators, semigroups, dephasing, sector dissipator, limit projector Π = lim e^{tL}
    dirac.py         Dirac/Clifford matrices in any dimension, lattice Dirac Hamiltonians (Wilson, staggered),
                     split-step Dirac walk with exact light cone, Feynman checkerboard path sum
    hamiltonian.py   joint vector-weight-boundary Hamiltonians, activity-register interactions (T4.5), term supports
    boundary.py      active interaction graph, generator-level boundary test, conditional mutual information I(I:E|B)
    gaussian_blanket.py   the exact spiral Markov blanket (T5)
    paths.py         exact finite path sums, influence functional, two-path density weights, dark-port interference (T7)
    semiclassical.py relativistic characteristics and Newtonian limit, stationary phase, macro-micro action matching,
                     wave-packet propagation (T8)
    readout.py       projective and nested coarse/fine instruments, einselection (T6.4, T8.5)
    canalization.py  idempotent locks, relation-preserving maps, the retention identity ΠK = ΠKΠ, exact
                     canalization discovery, isometric nesting, coarse channel with remainder flag, consolidation
    constraints.py   support-invariant hard constraints (T13.2)
    plasticity.py    correlation-to-weight law (T12.1), consolidation (T12.2), generalization bound (T12.3)
    geometry.py      density-derived metric g(R) and position-dependent Dirac transport (T3.4, F-Part 7)
    bits.py          universal bit substrate: any file or byte stream to bits, sensory and active ports (F-Part 10)
    machine.py       the present-state event loop (T10) and the 15-step algorithm (T14), branch enumeration,
                     storage accounting, the Markov-property audit
    tensornet.py     scalable joint states: MPS and locally purified density operators, local gates and
                     channels, truncation accounting (T13.1, T13.4, F-Part 9)
    bitkernel.py     bit-sliced execution of the einselected (pointer) dynamics, with measured throughput (F-Part 8)
    realizations/    every chosen realization, each with a frozen record:
                     incomplete_codes (T2.2), noncommuting_pulses (T2.3), two_mode (T9), fourier25 (T8.3),
                     vam1 (the reference machine for arbitrary bit streams, section 5 below),
                     vam1_tn (VAM-1 on a locally purified density), vam1_mpo (VAM-1 on an MPO density)
  tests/             one test file per section of T and part of F
  experiments/       protocols written before running, scripts, saved outputs (JSON)
  verify/            run_all.py writes RECEIPT.json: every check, its value, its pass/fail, counts
```

### The event equation the machine runs (T10.1)

For the current state `R`, next external bit `b`, output instrument `{J_τ}`, feedback channel `E_τ`, canalization `Π`:

```
q_b         = Tr B_b(R)                       (prediction of the bit, conditioning mode)
R_b         = B_b(R) / q_b
p(τ | R, b) = Tr J_τ(R_b)                     (J contains Dirac-Zurek evolution + boundary readout)
R_new       = Π E_τ J_τ B_b (R) / Tr(J_τ B_b(R))
```

Structural classical coordinates (slow weights) are part of the complete state and evolve by the T12.1 law after each event. `B_b`, `J_τ`, `E_τ`, `Π` may depend on them (T10.1 allows this; the overall map is then not one linear channel, and contraction claims are only made where T says they hold).

---

## 4. Specification-to-code map

Every row becomes tests that reproduce the exact numbers in T and check each proved property on random instances.

| Spec | What is built | Module | Exact results to reproduce |
|---|---|---|---|
| T1.2, F-Part 4 | density operators, purity gauge, partial trace | density, linalg | Tr ρ² bounds 1/d to 1 |
| T2.1-2.2 | incomplete bit configurations as density sectors; Kraus instrument for A=0,B=01,C=10 | realizations/incomplete_codes | populations (1/2,1/2,0),(3/4,0,1/4),(2/3,1/3,0); bit probs 2/3,2/3,3/4; P(010)=1/3; parses 1/9 each; Σ K†K = I |
| T2.3 | order-sensitive composition, BCH | realizations/noncommuting_pulses | Bloch (0,-1,0) vs (1,0,0); P(X=+1)=1/2 vs 1; BCH second order |
| T3.1-3.2 | joint vector-weight densities, congruence update | density, channels | marginals, positivity, R1R2 non-Hermitian counterexample |
| T3.3 | reciprocal Heisenberg equations from one Hamiltonian | hamiltonian | both derivative identities |
| T3.4 | density-derived metric | geometry | g ≥ g0 I |
| T4.1 | present-state sufficiency | machine | equal states give equal futures |
| T4.2 | generator-level boundary | boundary | d⟨O_I⟩/dt independent of E |
| T4.3 | statistical blanket, quantum CMI | boundary | pointer-label blanket has I(I:E|B)=0 |
| T4.4 | relay counterexample | boundary | I(I:E|B)=1 bit |
| T4.5 | activity registers N_e | hamiltonian, boundary | d⟨Z1⟩/dt = 2⟨N12 Y1 X2⟩ |
| T5 | spiral with exact Markov blanket | gaussian_blanket | minors 2,4,10,16; A; Lyapunov; mean (b/2,0), cov I/2; eigenvalues -2±2i; vorticity 4; radius 1/4 at log 2 |
| T6.1-6.3 | Dirac Hamiltonian in open-system setting, collision channel, Lindblad | dirac, channels, lindblad | Clifford relations; Σ K†K = I for K_sr |
| T6.4 | einselection by dephasing | lindblad | coherence factor exp(-γ(a-a')²t/2); halving at γt = (log 2)/2 |
| T7.1 | exact finite path sum | paths | equals operator product |
| T7.2 | influence functional | paths | reduced density from two-path sum equals partial trace |
| T7.3 | dark port; sector dissipator | paths, lindblad | p_dark = (1-Re d)/2 (0, 1/4, 1/2); e^{tD_B} closed form |
| T8.1 | Newtonian limit of the Dirac band | semiclassical | dispersion; characteristics; error O(p⁴/m³c²) |
| T8.2 | stationary phase, needle separation | semiclassical | leading term and error order; nonstationary decay |
| T8.3 | 25-configuration interference | realizations/fourier25 | a_r = 1, others 0 for all r; k = 5a+b nesting |
| T8.4 | macro-micro action matching | semiclassical | z = 1/2; actions 1/2, 3/2, 2; Hessian 16/3; packet centre 2t, variance 0.01+0.0025t² |
| T8.5 | nested coarse/fine readout | readout | p(j)p(a|j) = Tr(Q_{a|j} R) |
| T9 | vector-weight-output-feedback calculation | realizations/two_mode | R_Q = R_W; purity 7/8; -√2 vs -1/√2; p0 = 3/4; 4√2/3 and 0; 5/6, -1/2; 11/12, 1/4; 11/16,1/16,1/16,3/16; 32 branches |
| T10 | event-loop theorem, feedback is not evidence | machine | positivity, normalization, Σ_τ p = 1, Markov property by induction |
| T11.1 | lock from dynamics | lindblad, canalization | Π² = Π, CPTP, spectral condition |
| T11.2 | relation-preserving many-to-one map | canalization | relational expectations preserved |
| T11.3 | retention identity ΠK = ΠKΠ | canalization | sequence probabilities identical |
| T11.4 | incompatible locks; isometric nesting | canalization | Z-then-X loses Z; isometry preserves expectations |
| T11.5 | storage independent of event count | machine | measured constant state size over long runs; 2^B ≥ 2^n bound |
| T12.1 | correlation-to-weight law | plasticity | mean 0.4, variance 0.015 at log 2; limits 0.8, 0.02 |
| T12.2 | consolidation | plasticity | multipliers 1/2,1/4,1/32,1/1024 |
| T12.3 | generalization bound | plasticity | |p-p'| ≤ D(R,R'); postselection can break it |
| T13.1 | hierarchy, coarse channel with remainder | canalization, tensornet | CPTP; trace accounting |
| T13.2 | persistent hard constraints | constraints | invariance; zero eigenvalue alone is not enough |
| T13.3 | locality, exact event cone, commuting disjoint updates | dirac, machine | split-step walk has zero amplitude outside cone; lattice Hamiltonian has tails |
| T13.4 | storage and precision accounting | tensornet, paths | |Ã-A| ≤ (δ/ħ)Σ|a_j|; split-step error order |
| T14 | the 15-step algorithm | machine, realizations/vam1 | each step is a named stage |
| F-Part 8 | bit-level execution | bitkernel | measured updates per second on this hardware, not quoted ones |
| F-Part 9 | hierarchical density tensor network | tensornet | MPS/LPDO correctness against dense |
| F-Part 10 | universal bit substrate | bits | same interaction algebra for text, image, audio bytes |

---

## 5. VAM-1: the reference machine for arbitrary bit streams

T15.4 says the principles alone do not fix the couplings, so a complete run needs one declared realization. VAM-1 is that realization (`src/vam/realizations/vam1.py`, decision D8). Every element is a chosen realization built only from mechanisms in T and F.

**Registers.**
- `V`, a lattice Dirac field of `c` cells. Cell `j` holds a right-moving mode `R_j` and a left-moving mode `L_j` (one qubit each). This is the two-sublattice (staggered) form of the split-step Dirac walk: its one-excitation sector is exactly `split_step_walk(c, θ)` (tested), and a disturbance moves at most one cell per event (exact event cone, T13.3).
- `Bs`, the sensory port at the left boundary, and `Ba`, the active port at the right boundary. Both use the same weight-controlled port algebra; the code uses their Stinespring-reduced Kraus forms, tested against the explicit port qubit.
- The active port holds one qubit `A`: the machine's last emitted bit, waiting for the next transport to carry it into the chain as the entering left-mover `L_{c−1}`. The two ports are distinct, as T10.3 requires: the sensory port takes the environment's bits, the active port takes the machine's own.
- Structural classical coordinates: one slow weight `w_s` and one activity accumulator `n_s` per context sector `s`, where `s` is the right-mover content of the first `k` cells (`R_0` is the newest bit).

**One external bit (reading, conditioning mode; T14 steps 2-15 in order).**
1. Observe at `Bs`: Kraus `O_b = Σ_s sqrt(π_s(b)) P_s` with `π_s(0) = (1 + (1−2ε) w_s)/2`.
   - The machine's prediction of the bit is `q_b = Σ_s π_s(b) Tr(P_s R)`, and coherence inside a sector survives (T7.3).
   - The floor `ε > 0` keeps every possible bit possible (T13.2: a lack of evidence is not a hard constraint), so `q_b > 0` always.
2. Inject, using the collision model of T6.2.
   - Right-movers advance one cell and left-movers retreat one cell.
   - `R_{c−1}` and `L_0` leave with fresh environment units, which are traced out.
   - The bit enters as `R_0`, and the content of the active port enters as `L_{c−1}` (the port is left empty).
3. Dirac-Zurek evolution.
   - The mass coin `exp(−iθ(XX+YY)/2)` acts in every cell.
   - Then the sector dissipator `exp(D_B)` acts on the context sectors, or on every pointer configuration in the fully einselected limit.
4. Emit (step 11). The output instrument is the active-port readout `J_τ = O_τ`, so the machine emits a bit with exactly its own prediction for the next bit, and the state is conditioned on that output once.
5. Re-enter (step 12). `E_τ` writes `τ` into the active port by a collision. It is trace preserving and conditions on nothing (T10.3). The emitted bit enters the chain at the next transport and moves leftward; with mass, the coin mixes it into the right-movers and so into the context.
6. Plasticity follows the activity-gated T12.1 law, from the external bit only.
   - Inputs: the posterior sector activity `a_s = Tr(P_s O_b R O_b)/q_b` and the correlation `C_s = z_b a_s`.
   - Fixed point: `w_s = (g/κ) E[z | s]`, so `π_s` learns the source's conditional bit statistics forward, with nothing replayed.
   - The literal T12.1 law uses a constant friction `γ`. Under one repeated lesson its transient shrinks by the fixed factor `r = exp(−κ a dt/γ)` per repetition, which is exactly the geometric consolidation `r^n` of T12.2 (tested with `r = 1/2`: 1/2, 1/4, 1/32, 1/1024).
   - With the option `consolidating=True`, the friction is `γ_s = γ + κ dt n_s`. T does not state this law; it is a chosen addition (decision D10). With `g = κ = γ = dt = 1` its step is `1 − exp(−1/(n+1)) ≈ 1/(n+1)`, so the weight becomes a running average of the sector's `z`: frequency counting, close to the KT estimator. Its transient decays harmonically, `exp(−H_n)`, not geometrically, so it is not the consolidation of T12.2 (tested).
   - Results are reported for both laws; the literal law is the specification's.

In the massless limit the context sector is always definite, so conditioning on the emitted bit changes nothing, and left-moving outputs never reach the right-movers. The predictions and weights are then exactly those of a reader that emits nothing, whatever it emits (tested against the bit kernel). The bit kernel and the nested machine therefore compute the full loop exactly at zero mass. With mass, emission and re-entry change the dynamics, and the pointer engine runs them.

**Generation (answering, T14 steps 11-15).**
- There is no external bit; the declared settling step is the identity.
- The output instrument is the active-port readout `J_τ = O_τ`. The machine therefore emits with its own predictive probabilities, and the emitted bit is conditioned on once.
- Re-entry `E_τ` writes `τ` into the active port, exactly as in reading. Because no external stream remains, the sensory port receives the emitted bit too: `τ` is injected (step 2, which also carries the port's content into the chain) and the chain evolves (step 3) before the next emission.
- Plasticity is off during generation: re-entered outputs transform the state but are not evidence (T10.3).
- The macro/micro rule of F-Part 2 is `macro_micro_step`:
  - Macro: the exact distribution over every continuation of a declared horizon, computed by branching the density through the machine's own re-entering dynamics.
  - Micro: emit only the first bit, re-enter it, then plan again.
- A stopping condition is declared per experiment (T14 step 15).

**Canalization.** The invariant-algebra lock (D7) gives what the present state must keep. The event maps depend on the weights and the weights move, so the algebra must serve every weight and must contain what plasticity reads. `DenseMaps.canalization()` therefore seeds it with the sector projectors `P_s` and closes it under the weight-independent maps (injection, evolution, re-entry); every weight-dependent factor `O_b(w) = Σ_s sqrt(π_s(b)) P_s` then maps it into itself. Tested: with zero mass the algebra is the context populations, of dimension `2^k` for `k = 1` and `k = 2` (two cells); the retention identity holds at `w = 0`, at equal weights and at random weights; and the canalized learning machine predicts, emits and learns exactly as the plain one. A lock closed over the maps at one weight only collapses to the scalars at `w = 0` and erases what learning needs (a tested negative control). With mass, left-movers bring information back into the context, and the retained algebra grows.

**What VAM-1 can and cannot show.**
- With zero mass the chain is an exact sliding window over the last `c` bits, and the state is a product of basis states of those bits. Massless VAM-1 is therefore exactly a classical `k`-bit context model with one moving weight per context (under the count-dependent friction, a counting estimator close to KT). The learning law converges to the source's conditional bit statistics given the last `k` bits. That is a real, measurable capability: online prediction and generation from raw bits, with no history and no backpropagation. It does not exercise the coherent field.
- With nonzero mass the contexts become mixtures of shifted and reflected bit patterns, weighted by how well each predicted. Whether that helps is measured, not assumed.
- Whether this grows textbook-level understanding is the open claim of T15.4; section 6.3 tests it.

**Nested levels: step 14 of the master loop (`nested.py`, decision D9).**
- The massless machine keeps several context partitions at once, nested from coarse to fine (by default the last 8, 16, 24, 32, 48 and 64 bits). Each level is a coarse or fine readout of the same pointer configuration (T8.5, T13.1), and each learns by the same T12.1 law.
- A fine sector is created the first time its context occurs. Its relation is set so that its own prediction equals the coarser readout at that moment (the mixture of the levels above it), and its evidence starts at zero, so adding it leaves the machine's prediction exactly unchanged (tested, with a negative control: copying the parent's raw relation instead moves the prediction). This is isometric nesting (T11.4): creating a distinction changes nothing already known.
- Readout: each fine sector keeps `ev`, its log2-likelihood ratio against the coarser readout over its own visits. The levels are mixed from coarse to fine with weight `2^ev / (1 + 2^ev)`, the Bayesian posterior from even odds. A distinction that does not change the predicted future fades (step 13), while several supported levels contribute together (step 8).
- Storage is declared and fixed: deep levels live in hash tables with check tags, and a new context that lands on an occupied slot replaces it. Every replacement is counted.
- Tests: with one level it equals the bit kernel bit for bit; the compiled tables equal an independent dictionary implementation, with and without forgetting.

**Engines (identical semantics, tested against each other).**
- Dense (`DenseMaps` with the generic `Machine`): the full coherent density, up to 4 cells.
- Pointer (`PointerEngine`): the exact fully einselected limit on configuration probabilities, up to about 10 cells, emission for emission equal to the dense engine.
- Bit kernel (`bitkernel.py`): the massless einselected limit on packed bits for long contexts and long streams, with measured throughput.
- Nested machine (`nested.py`): the massless limit with nested context levels (step 14).
- Tensor network (`realizations/vam1_tn.py`, `TensorEngine`): the same reading and generation events on a locally purified density (`tensornet.py`, T13.1, T13.4, F-Part 9), with dephasing 0 or infinite only (the finite sector dissipator is a correlated channel). Untruncated it equals the dense engine on the whole test grid, the pointer engine emission for emission, and the bit kernel at zero mass for 2000 bits; the event cone is exact at 20 cells. With mass and einselection the purification grows with the history, and truncating it is not a faithful approximation: at 6 cells the truncated engine misses the pointer engine by up to 0.019 per prediction, and at 24 cells about 0.6 of weight is discarded per event. The engine is correct machinery and runs at 24 cells, but no VAM-1 result uses a truncated run. The next engine is that fix.
- Density MPO (`realizations/vam1_mpo.py`, `OperatorEngine`): the same events on the density itself, stored as a matrix-product operator (`vec(rho)` as an MPS with local index `2s + s'`), every map applied as its superoperator. It supports every dephasing value, since the finite sector dissipator is an MPO of bond 2. The density keeps no history: measured before the engine was written, in the fully einselected chain the bond for a relative Hilbert-Schmidt error of 1e-6 is 6-25 up to 10 cells, where the register allows about a million at 10 cells. Untruncated it equals the dense engine on the whole grid (finite dephasing included), the pointer engine for 300 bits with an exact bond of at most 12, and the bit kernel; the event cone is exact at 20 cells. Truncated to bond 64 it follows the exact engines: within 1.6e-6 per prediction at 8 cells and 1.1e-5 at 10 cells fully einselected (600 and 300 bits), and within 3e-6 at 6 cells with context einselection, every emission equal. Positivity is not structural for an MPO, so it is measured: the smallest reduced eigenvalue at 24 cells is -3e-10. The limit is coherence. With context einselection the L movers and the R movers past the context are never dephased, so the chain is a coherent Dirac walk there and its operator entanglement grows with the length: at 12 cells (theta 0.3, no exact engine reaches it) bond 64 differs from bond 128 by up to 5.6e-4 per prediction, and bonds 16 and 32 by 3.3e-3 and 1.5e-3, so the error roughly halves per doubling of the bond. Full einselection at 20+ cells is within reach at modest bonds; context einselection at 20+ cells is approximate at about the 1e-3 level with bond 64 (at 24 cells it discards about 0.3 of the Hilbert-Schmidt weight per event), and the bond it needs is the open cost.

---

## 6. Verification

### 6.1 Exact and property tests
- Every exact number in T is reproduced symbolically (SymPy) and numerically.
- Every proved property is checked on seeded random instances (random CPTP maps, random states, random Hamiltonians), plus targeted counterexamples T itself gives (Z-then-X locks, relay blanket, postselection).
- Negative controls: where a broken variant is plausible, the test also shows the check catching it. Examples: a feedback map that conditions a second time is rejected (T10.3); the independent-marginal derivative misses the exact one by a factor of two (T9.2); a canalization closed over the maps at one weight erases what learning needs (D7); a nested sector created from its parent's raw relation moves the prediction (D9).

### 6.2 Machine audits
- Markov property: two different histories reaching the same state produce identical future distributions.
- Storage: the bit kernel and the nested machine keep a constant state size over 10^6 events (`tests/test_long_runs.py`); the dense event loop over 20,000 events; the experiments' registers and tables are fixed by construction.
- Positivity and normalization: the dense event loop validates every event and, after 20,000 events on a massive, partly coherent chain, has trace error below 1e-12 and no negative eigenvalue below −1e-12.
- Retention identity checked for every event map of each realization.

### 6.3 Developmental experiments (protocols written before running, seeds fixed, outputs saved)
1. Known sources (the T2.2 code source, k-th order Markov sources): log-loss converges to the source entropy rate; controls with plasticity off and with shuffled streams.
2. **Real-data curriculum (owner's instruction, 2026-10-18): structured data first, unstructured after.**
   - Stage A, structured: textbooks streamed as raw bits in reading order (default: freely licensed OpenStax and public-domain mathematics and science texts, unless the owner names others).
   - Stage B, unstructured: after a declared amount of structured data (set in the protocol before the run), switch to unstructured text and other raw bytes (web-style prose, images, audio) through the same machine, with no reset: the state carries over, since the state is the memory.
   - Measured throughout: bits per byte on held-out structured and unstructured text, weight stabilisation, and whether Stage A changes how fast Stage B is learned (compared with a machine fed Stage B alone).
   - Plus every test source the build supports (section 6.3.1, the known sources).
3. The three-part textbook hypothesis of T12.3, measured on the Stage A data: (i) structured order creates recurring correlations, (ii) they change and stabilise the weights, (iii) unfamiliar arrangements of the same relations engage the learned organization. Each part is reported as shown or not shown.

### 6.4 Independent audit
After each phase an independent reviewer checks the mathematics and code against T and F, read-only. The reviewer is a separate Claude Opus 5.5 session, the owner's choice (2026-10-18). Findings and responses are recorded in `AUDIT.md`.

---

## 7. Work split

| Work | Who | Why |
|---|---|---|
| Theory-to-code design, core mathematics, machine, verification, final calls | this thread (Claude Opus 5.5, high effort) | precision-critical |
| Independent mathematical and code audit after each phase | separate Claude Opus 5.5 reviewer, read-only | an adversarial second reader catches what the author misses |
| Well-specified modules with written acceptance tests (bit kernel and benchmark, tensor-network layer) | helper on Claude Sonnet 5.5, medium effort, in its own files only | routine coding against a precise spec |
| Mechanical chores (full test runs, formatting, receipt regeneration) | helper on Claude Haiku 5.5, low effort | cheapest model that can do it |

Helpers never post to the thread and never commit; this thread reviews and integrates everything. One thread is used rather than several, because the modules share one package and one test suite, and a single owner of the integration avoids conflicting edits.

---

## 8. Phases

1. **Core mathematics and every exact result in T** (sections 2-13 of T, F-Parts 1-7).
2. **Scalable state and execution**: tensor networks, Dirac light cone at scale, bit kernel with measured throughput (T13, F-Parts 8-9).
3. **The complete machine**: VAM-1, generation by re-entry, macro/micro decoding, developmental experiments (T10, T12, T14, F-Parts 2-3, 10).
4. **Audit, fixes, results report** (`RESULTS.md` generated from saved outputs).

Each phase ends with a commit and push to the branch, a test receipt, and a short report.

---

## 9. Decision log (where the documents leave a choice open)

| # | Question | Choice | Reason |
|---|---|---|---|
| D1 | Location | branch `claude/project-thread-20ve0y` of `sera-research-raw-20261005`, folder `vam/` | the branch this session may push to; a draft PR keeps it reviewable |
| D2 | Units | ħ = 1 unless T gives a value (T8.4 uses ħ = 0.01) | T's own convention in T9 |
| D3 | Vectorization | column stacking, vec(AXB) = (Bᵀ⊗A)vec(X) | standard; documented in channels.py |
| D4 | Lattice Dirac discretization | three provided: Wilson-term Hamiltonian, staggered (Jordan-Wigner chain), split-step walk | T13.3 contrasts continuous-time tails with exact event cones; all three are needed to test that |
| D5 | Weight-law noise | zero-noise idealization by default, thermal noise optional | T12.1 allows it if stated |
| D6 | Zero-probability observation | default: raise; options: inject, extend by isometry | T10.2, T11.4 |
| D7 | Canalization construction | besides the limit lock and relation-preserving map, an exact construction: the smallest unital *-algebra containing the readout effects and closed under the Heisenberg duals of all event maps; Π is the trace-preserving conditional expectation onto it. For a learning machine the algebra must serve every reachable value of the structural coordinates and contain what plasticity reads; for VAM-1 it is seeded with the sector projectors (audit finding 3) | it satisfies ΠK = ΠKΠ by proof (section 11.3 of T asks for a Π with this property but gives no general construction) |
| D8 | VAM-1 field and ports | cells of right/left-moving modes (staggered split-step walk); sensory port at the left boundary, active port (one qubit feeding the entering left-mover) at the right; after every external bit the machine emits `J_τ = O_τ` and re-enters `τ` at the active port (T14 steps 11-12; until the audit the reader was silent, audit finding 2); while generating, the emitted bit also enters the sensory port; plasticity only from external bits; probability floor ε | exact Dirac walk in the one-excitation sector with an exact event cone; T10.3 keeps outputs from counting as evidence and puts them through a different port; T13.2 forbids treating unseen bits as impossible |
| D9 | How step 14 extends the hierarchy | nested context levels with isometric creation (a new sector's own prediction equals the coarser readout; audit finding 4), Bayesian level mixture by per-sector evidence, fixed hashed storage with counted replacement | T14 steps 8, 13 and 14 ask for retained multiple contributions, canalization that removes only distinctions irrelevant to the future, and nesting for new distinctions; a fixed window cannot extend itself |
| D10 | Friction of the weight law | the literal T12.1 law (constant friction γ) is the specification's; an optional count-dependent friction `γ_s = γ + κ dt n_s` is a chosen addition that T does not state. It turns the law into frequency counting (step ≈ 1/(n+1)), whose transient decays harmonically, not as the r^n of T12.2. Every experiment reports both laws (audit finding 1) | the literal law is the specification; the counting variant is kept because it was in the pre-registered E1 criteria and the E2 grid, and is labelled as a choice |
| D11 | How N's layers above one equation are realized on text (E4) | Upper levels are fixed-size registers of completed lower units (unit boundaries computed from the bits; unit codes of 64 bits; arity 4), read as more context levels of the existing evidence mixture; one global governor with a raised admission bar past the Page point; output through the machine's own instrument. Section 11. | The text machine realizes only N-L1 (one 64-bit register); the 2.131 vs 2.003 bpb gap to a dictionary coder is a hypothesis about structure beyond 8 bytes. A match model that looks back into stored text is excluded by T4.1/T11.5; a registers-of-registers hierarchy is what N and F Part 10 describe. A chosen realization (T15.4), not the unique one |

---

## 10. The nested fractal architecture (N, the owner's design of 2026-10-18)

The owner's instruction: integrate every layer, leave nothing out, make each part work where it does not
work as written, and measure where the design does what it was designed to do. The goal is unchanged:
one machine that reads raw bits (textbooks first), predicts and generates. Every layer below is a running
mechanism of the machine, with the test that it does what N says and the measurement that shows its
effect. New code lives in `src/vam/nfa/`, except where a layer extends an existing kernel.

| Layer (N) | What N asks | Realization in VAM | Measured by |
|---|---|---|---|
| L1 bits | every modality is configurations of bits | `bits.py` (existing): any file's bytes, most significant bit first | runs on text and on non-text bytes |
| L1 equation | a sentence is a flow dx/dt = f(x, t) | `nfa/flow.py`: each bit's event is the solution over one tick of a master equation dR/dt = -i[H, R] + D(R) driven by the bit (coin Hamiltonian, sector dephasing, injection as a collision); the recorder keeps the trajectory x(t) of local observables | integrating the equation reproduces the event map (1e-10) |
| L1 register bound | the Bekenstein bound with the CPU register sets the information per equation | `nfa/bounds.py`: the exact Bekenstein bound 2πRE/(ħc ln 2) (CODATA constants) and the hardware's word, cache-line and cache sizes read from the machine; the cap per equation is the smaller of the two, 64 bits: one context word per level (the nested levels already stop at 64) | every configuration checked against the cap; bpb as depth approaches 64 |
| L2 Page limit | the basis of vectors is sized by the Page limit; beyond it no new mutual information | `nfa/page.py`: Page's exact mean entropy S(m, n) = Σ_{k=n+1}^{mn} 1/k − (m−1)/(2n), the Page curve, and the edge of a Page-random spectrum (Marchenko–Pastur, the spectrum of a random state); a direction is kept only above that edge. Applied to the stream's predictive (Hankel) matrix, giving the number of state vectors the stream's equation needs, and to choosing the nested levels | recovered dimension against the known dimension of E1 sources; pruned levels against the fixed six |
| L3 density matrix | lift the vectors to quantum vectors; ρ = Σ p_i \|ψ_i⟩⟨ψ_i\|, probabilities in [0, 1], no Inf/NaN | coherent readout of the nested levels: each level is a hypothesis with amplitude √(w_i p_i(b)); the readout is p(b) ∝ Tr(P_b ρ) with ρ = (1−c) diag(w) + c \|√w⟩⟨√w\|. Coherence c = 0 is today's classical mixture, c = 1 a pure superposition (Born-rule pooling). Probabilities stay in [ε, 1−ε] by construction | bpb against c on textbook validation; finiteness over every run |
| L3 cache-bounded size | the size and rank of ρ are bounded by the cache | `nfa/bounds.py`: the largest density dimension and MPO bond whose working set (16 bytes per complex entry) fits L1 or L2, used as the MPO engine's default cap | throughput across the cache boundary |
| L4 blankets | every node has sensory, internal and active parts; nodes interact only through them | nodes are the chain's cells (sensory: incoming movers; active: outgoing movers) and the nested levels (sensory: the coarser readout; active: the refined prediction). The event map is local (tested: the event cone and the neighbour test). `nfa/blankets.py` measures the conditional mutual information I(μ:η\|b) of node triples and the containment of a perturbation | CMI per node; containment exact within one event |
| L5 tensors | density matrices assemble into a multi-scale MPO | `vam1_mpo.py` (the chain's density as an MPO) and the nested levels (the scales of the learned part) | (existing tests) |
| L5 area law | the bond is governed by the entanglement across the cut | per-cut operator entanglement entropy S_cut, and an entropy-governed bond χ_cut = ⌈2^{S_cut}⌉ with margin, capped by the cache bound | S_cut against chain length (area law: bounded) |
| L5 verified growth | the rank grows only for statistically verified, non-noise invariants | in the MPO: a new singular direction is kept only above the Page edge of its cut; in the nested machine: a new fine sector is admitted only once its context is verified (seen before, in a small doorkeeper filter), so one-off contexts never take storage | bond on noise against structure; bpb and evictions at fixed storage |
| L6 global blanket | sensory port, internal core, active port around the whole machine | VAM-1's two ports (existing) | (existing tests) |
| L6 Page governor | track total information against capacity; at the Page threshold halt expansion, purge noise through dissipative sinks with Re λ = −2, radiate invariants | `nfa/governor.py`. MPO: Rényi-2 entropy S₂ = −log₂ Tr ρ² (the squared norm of vec ρ) against the qubit count; above half of it (the Page time) no bond grows, and a unit-rate dephasing sink (its coherence eigenvalue is exactly −2) acts on the cut with the excess. Nested machine: stored information against table capacity; at half capacity new sectors are admitted only if verified, unverified evidence decays, and verified sectors are kept against replacement | entropy pinned at the threshold; cost at 24 cells; predictions against the ungoverned engine |
| L7 collapse | Feynman path integral over the field, destructive interference, stationary phase to least action, minimal output | the field: the chain's evolution is the exact path sum (T7.1, `paths.py`), einselection the interference loss, stationary phase gives Newton (T8, `semiclassical.py`), existing. Generation: `nfa/collapse.py` sums over the machine's h-bit continuations γ with action S[γ] = −ln p(γ) and weight e^{−S/ħ} (the Euclidean form of e^{iS/ħ}); ħ = 1 is the full sum, ħ → 0 the least-action path; the first bit of the collapsed path is emitted, then the machine re-plans | generation quality against ħ (loops, valid text) |

The engine that composes all of it is `nfa/engine.py`; experiment E3 (protocol in `experiments/PROTOCOL.md`,
written before it runs) measures each layer's designed behaviour and the capability of the whole on the E1
sources and the E2 corpus. Where a layer cannot be made to do what N says, the report says what it does
instead, with the measurement.

---

Measured (2026-10-20): E3 (`RESULTS.md`, `audits/c2.md`) passes every Part 1 criterion; C2 found four gaps no
criterion caught (N-L2 on text, the inert halt, the cost of sinks, the Page trigger's late sinks), two of which
T4.2 and T4.3 measured as added after the result.

## 11. N above one equation, on text (E4; decision D11, a chosen realization; written 2026-10-20, before any E4 code)

Labels: **S** source statement, **D** derived result, **C** chosen realization (T1.1, T15.4: the coupling law between layers is a
declared choice, not the unique one). Nothing here is built until the manager's gate G2 passes.

### 11.1 Why, and what E3 left open

On text the machine of E3 realizes only N-L1: every level is a window of one 64-bit register, 8 bytes of context at most
(DESIGN section 10, L1 row). N and F Part 10 put structure above that: bits, then words, then sentences as equations "of at most
one register each", then vectors, densities, tensors and blankets above them. E4 builds the levels above one equation.

What E3 measured (audits/c2.md), and what E4 takes from it:

| E3 finding | What E4 does with it |
|---|---|
| A-test 2.131 bpb against lzma 2.003 (conditional on the A split); the machine's contexts end at 8 bytes. Why lzma wins is a **hypothesis**: long matches (repeated names, boilerplate, passages across books) | Pre-registered decomposition of A-test by whether the preceding 8-, 16- or 32-byte window occurs in A-train (protocol, Part 5). If the hypothesis holds, the gain of E4 sits in the long-window classes |
| N-L2 on text measured nothing (rank 0 at every depth, cutoff 1.1) | N-L2 stays a measurement with a shuffled null and a count floor; it never selects a level (11.3). It does not depend on the result of T4.2, which is still pending when this is written |
| The text governor's halt is inert: admission needs `max(verify, 2)`, which is `verify` for verify >= 2; occupancy ran on to 0.87 to 0.98 | One global governor, and a halt that changes behaviour at every verify setting (11.3, N-L6), with the E3 behaviour as a pre-registered control; if it changes nothing at any verify setting in Part 1, it is removed |
| Sinks cost accuracy at every rate tried (0.1 best, the lowest); rate 0 untested | Rate 0 is a pre-registered ablation in the protocol, reported, not a tuning of E3 |
| Deepest tables ended at 0.87 occupancy after A and 0.98 after B: the machine is capacity-bound | More levels add capacity as well as span; the protocol fixes a control with equal storage and compute and no extra span |

The result of E3's A-test informed this design. E4 therefore holds E3's chosen base fixed (coherence 1.0, verify 3, governor with
protect 0.5 and sink rate 0.1, counting law, prior 4, depths 8 to 64) and selects only E4's new factors, on A-validation. A-test is
read once, at the end.

### 11.2 Constraints, each satisfied explicitly

1. **Present state only (T4.1, T11.5; S).** Long-range structure enters only through registers of fixed size that the stream itself
   updates. A match or dictionary model that looks back into stored text breaks this rule, and lzma-conditional is exactly that
   (it uses A-train as a dictionary). "Beat lzma" is therefore not a like-for-like target: E4 measures how much of that gain a
   present-state machine can recover. The state of the machine is declared in bytes and does not grow with the stream (a test).
2. **N-L1's 64-bit equation (S).** Each register is at most 64 bits. An upper equation holds a fixed number of lower equations (its
   arity). N-L1's own bound on arity is the cache line: 64 bytes, 8 registers. The value inside that bound is **C**: arity 4.
   A single wider hash over a longer byte span would be a bigger equation, not a layer above one, and is not used.
3. **No tokenizer (F Part 10; S).** Unit boundaries are a fixed function of the bits (11.4), not a text rule. A whitespace and
   punctuation rule is a tokenizer and departs from F; it appears as one labelled arm, and the choice between them is an owner
   question (11.8).
4. **Event loop and output instrument unchanged (T7.1, T10.1, T14, T15.3; S).** Upper levels join the existing evidence mixture
   as more context levels. No scoring, no path cost, no selection is added. Writing defaults to the instrument (hbar 1,
   euclidean, sample, unverified). `collapse.Readout` advances the full state (every register) and memoizes on it.
5. **Forward learning only (T12; S);** new distinctions by isometric nesting (T11.4; S); storage fixed and declared (T11.5; S).
6. **Defaults bit-identical and free when off (engineering).** With no upper level, E4's machine equals E3's machine exactly
   (tested, and the E3 chosen A-validation value is reproduced from the saved row), and the compiled loop pays nothing for the
   option (paired timing, as T1.7).

### 11.3 The seven layers above one equation: one mechanism each

| Layer (N) | Mechanism on text | Label | Where the layer is trivial or only measured |
|---|---|---|---|
| L1 bits to equations | Level B (as E3): the 64-bit byte register. Level W: units ("chunks") of bytes close at content-defined boundaries; a closed unit is an equation of at most 64 bits (11.4); the W register is a ring of the last 4 unit codes. Level S: closed units fold into a sentence register; a sentence closes at a content-defined boundary; the S state is the running sentence code and the last 2 sentence codes. Contexts of an upper level are functions of its register and the byte being read | C (arity, boundaries, codes); S (register bound) | none |
| L1 Bekenstein bound | Every register is at most `register_cap` = 64 bits; arity at most `line_words` = 8; asserted at construction; the state size in bytes is a declared number | S, D | by construction |
| L2 Page limit | **A measurement, not a selector.** For each upper level, the Page rank of the whitened Hankel matrix (level context against next byte) with a count floor of 256 (T4.2's rule, fixed there before PG-19) on the first 4 MiB of A-train, with a byte-shuffled null of the same bytes. Criterion: the null never exceeds rank 1. Ranks on a filtered row set fall with depth because rows are removed, so they never cut a level | D (method), C (floor) | measured only; C2 showed the unfiltered method has no power on text |
| L3 density matrix | The existing coherent mixture over levels (`coherent_mix`), with E3's coherence 1.0 held fixed. Each added level adds one two-hypothesis density to the chain. Output stays inside the hull of the component readouts (so inside `[eps, 1 - eps]`). Isometric nesting (T11.4) holds for any context partition, because a new sector whose prediction equals the coarser readout leaves the readout unchanged for every `c`: upper sectors need not refine the byte contexts | D (A8 of audit 2; derived from `coherent_mix`) | the label state is an explicit enlarged state; relative phase zero is C |
| L4 blankets | A level's sensory port is the readout of the levels before it plus the completion events of the units below; its active port is its refined readout. A level reads only its own register tuple. Containment holds by construction; tested by perturbing one byte of a frozen run: predictions beyond the registers' span are bit-identical (11.5) | D | exact by construction |
| L5 tensors | The chain of levels is a matrix-product chain of two-hypothesis densities: bond dimension 2 per cut. The area law `S_cut <= 1 bit` is therefore **vacuous** here; the mean label entropy per cut is reported as a measurement, and the claim is no more than that. Verified growth: a context takes a slot of an upper level only at its `verify`-th sighting (doorkeeper, as E3). The MPO of the chain engine (E3 Part 1) is the tensor network at the density level; E4 does not rebuild it on text | D; vacuous area law stated | bond 2 is trivial; verified growth is the substantive mechanism |
| L6 global governor | One governor over the total slot capacity of every hashed level (E3's was per level). Below half of the total occupancy: growth is free. From half on: an unverified context needs `verify + 1` sightings (the halt: the bar rises, so the halt changes behaviour at every verify setting, where E3's `max(verify, 2)` changed it only at verify 1); proven sectors (evidence above `protect`) are protected from replacement; knocks drain a protected sector by the sink factor. Sink rate 0 is a pre-registered ablation | C (the rule), S (Page point, sinks) | the halt is kept only if Part 1 shows it changes the refusals at verify 2, 3 and 4; else it is removed (decided in advance) |
| L7 collapse | The machine's own output instrument, as E3, on the full state. Settings of N-L7 other than the instrument remain chosen realizations and are labelled (Amendment 1). Writing is trimmed to the instrument and one chosen realization | S (instrument), C (rest) | none |

### 11.4 Units, codes and registers (the definitions the build must follow)

All hashes are fixed 64-bit mixers (the SplitMix64 finalizer already used by `nested.context_hash`) with fixed salts.

- **Unit (word-level equation).** A byte closes the current unit when `mix(last 3 bytes) mod 6 == 0` (**C**: 3 bytes of lookahead-free
  context; modulus 6 gives units near the length of a word in English text; the rule depends only on the last 3 bytes, so it
  resynchronizes within 3 bytes after any change). The unit's running code `c` is `mix(c, byte)` from a fixed seed; at closure the
  code is final and `c` restarts. A unit is a configuration of at most one register (64 bits).
- **W register (ring).** The last 4 closed unit codes. Contexts: `W1 = mix(c1, c, u)`, `W2 = mix(c1, c2, c, u)`, `W4 = mix(c1..c4, c, u)`
  where `c1` is the latest closed code, `c` the code of the unit in progress and `u` the bits of the current byte read so far
  (with the implicit leading 1). Levels `W1`, `W2`, `W4` (**C**: three depths of the ring).
- **Sentence (sentence-level equation).** The sentence code `s` folds each closed unit code: `s = mix(s, code)`. A sentence closes when
  `mix(code) mod 16 == 0` at a unit closure (**C**: modulus 16 gives about 100 bytes), then `s` restarts. The S state keeps the
  last 2 closed sentence codes. Contexts: `S0 = mix(s, c, u)` (the whole sentence so far), `S1 = mix(p1, s, c, u)` with `p1` the
  last closed sentence code. Levels `S0`, `S1`.
- **Order.** The mixture runs coarse to fine as in E3: byte levels 8 to 64, then `W1, W2, W4`, then `S0, S1`. The order decides which level earns
  evidence credit; it is **C**, declared here and never selected.
- **Many-to-one compaction (D).** A unit code is a many-to-one image of the unit's bytes: the coarse-graining channel of T13.1, with a remainder
  (the collision of two different units to one code). At 64 bits the rate is negligible; the protocol measures it on the text
  (the share of distinct units sharing a code).
- **Present state (D).** The registers depend only on the stream, not on learning, so they can be computed in a pre-pass while reading;
  state sufficiency (T4.1) holds: the state is `R_b`, `c`, the ring of 4 codes, `s`, the last 2 sentence codes and the bit position,
  a fixed state of under 128 bytes. The upper-level keys are functions of that state.
- **Storage (T11.5, declared).** Each upper level has `2^21` slots (3 doubles and a 32-bit tag, 28 bytes: 56 MiB) and a doorkeeper of `2^22`
  32-bit entries (16 MiB): 5 levels = 360 MiB, on top of E3's 577.5 MiB (449.5 MiB without doorkeepers): about 0.92 GiB of tables. The
  size is fixed and does not grow with the stream.

### 11.5 What each layer must be shown to do (the criteria are in PROTOCOL E4, fixed there)

Containment is tested on a frozen machine (learning off), because with learning on a perturbation also changes the tables. The span after
which a perturbation of one byte must have no effect: for `W`, until the ring has turned over (4 closed units after resynchronization);
for `S0`, until the sentence closes; for `S1`, until one sentence closes after that. The byte levels: 8 bytes.

### 11.6 What would not be new

- Sector storage, evidence, doorkeeper, coherent mixture and sink maths are `nested.py`'s; the compiled loop receives a per-level key
  stream (from a pre-pass) in place of `reg & mask` for upper levels, and the default path is untouched and timed.
- The governor's scope option and the raised bar are options off by default.
- `collapse.Readout` is extended to the full state; the E3 settings are unchanged.

### 11.7 Reading rules fixed before any run

The questions E4 answers, and how the answers are read, are written in PROTOCOL E4 (criteria, controls, budgets, the decomposition by
window class). None is tuned on the test split.

### 11.8 Owner questions (not decided here)

1. Content-defined units (modality-free, F Part 10) against a text rule for words and sentences (a tokenizer): E4 runs both, labelled, and
   reports the cost of the principled choice.
2. Whether a unit above the word should be a sentence by content-defined boundary (what E4 builds) or by another principle.
3. Arity and moduli were chosen (4, 6, 16). E4 does not tune them; a later experiment may.

### 11.9 The advisor's conclusions (Opus, recorded because no other session can read them)

1. Long-range structure may enter only through fixed-size registers that the stream updates; a match or dictionary model over stored text breaks T4.1/T11.5, and
   lzma-conditional is such a model, which makes "beat lzma" a different kind of target (11.2, point 1).
2. An upper level is a configuration of lower registers of at most 64 bits each; its arity needs a bound from N-L1 itself (the cache line, 8 registers), and the value chosen inside the bound is C (arity 4).
   A single wider hash over a longer byte span is a bigger equation and is not a layer above one.
3. Unit boundaries computed from the bits are modality-free; a whitespace or punctuation rule is a tokenizer (F Part 10), so it is one labelled arm and an owner question.
4. Blanket containment holds by construction and is tested by perturbing a frozen run; on text the level chain has bond 2 per cut, so the area law is vacuous and is reported as such.
5. N-L2 is a measurement with a shuffled null, never a selector. The halt is either redefined so that it changes behaviour at every verify setting, or removed; the protocol states the test that decides it.
6. Sinks: rate 0 is a pre-registered ablation, not a tuning of E3.
7. The long-match hypothesis is tested by a pre-registered decomposition of A-test by window class; the controls fix equal storage and compute with no extra span (E3's machine is capacity-bound) and destroyed order.
8. A-test is read once, at the end; E3's chosen base is held fixed.
9. Isometric nesting holds for any context partition (so upper sectors can join the chain), level order is C and never selected, unit-to-code compaction is T13.1's many-to-one coarse-graining with a remainder, and registers that depend only on the stream may be computed in a pre-pass.
10. Budgets are taken from saved E3 data and scaled by level count; one job at a time.

Added by the engineer after the advisor: with `verify` 3 a passage seen once before has no slot at an upper level until its third sighting, so the long-match gain may be small for once-repeated text; Stage 4 of Part 2 therefore varies the upper levels' verify separately from the byte levels'.

---

## 12. Shared operators, rotation families, the reasoning loop, the feedback port and the teaching order

Written 2026-10-21 (task T8.1, Phase 0 of the owner-approved plan "VAM: from memorizing to learning"). Design only: no `src/` code, no `experiments/` code, no change to `PROTOCOL.md`. Nothing here is built or run before the owner's gate G5.

Labels: **S** source statement (a spec section, the code as it stands, or a saved result); **D** derived (a proof or computation from S, written out); **C** chosen realization (T1.1, T15.4: a choice, not the unique one). Spec sections are cited as T, F and N; code as `file:line`; saved numbers by file and key.

### 12.0 Why, and the verdicts at a glance

E5 (`results/e5.json`, `RESULTS.md`) bounded what the machine does with taught text: recall above the control only where the answer is the most frequent option (9 against 4 of the questions that remain, p 0.11), inference at chance (P3, 11 against 11). §12 asks why, and what the machine would need in order to carry a relation to a case it has not seen, to do a multi-step problem, to be corrected, and to be taught in a sensible order. The plan proposes thirteen verdicts. Each is proved or refuted below against the spec and the code. Where a verdict fails as stated, it says so and gives the fix. The fixes are engineer proposals; none changes the plan until the manager rules (12.6).

| # | Claim in the plan | Result | Where it fails or is narrowed |
|---|---|---|---|
| V1 | Exact-context tables do not generalize to unseen contexts or sums | **holds**, by construction; also for contexts seen fewer than `verify` times | none |
| V2 | A shared linear operator level O helps on unseen text contexts | **holds as a mechanism with bounded risk**; magnitude is E6's | (a) the bound is per stage input, 1 bit (c = 0) or 2 bits (c = 1) per sector incarnation, not E4+O against E4; (b) the measured ranks are 19 to 51, not "35 to 48", on frequent rows only; (c) the cited theorem is for finite-rank models |
| V3 | A linear O fails on held-out arithmetic pairs | **holds** | none (proved by indistinguishability, on the byte-level Hankel too) |
| V4 | Tokens as rotations on one generator, read by interference, do held-out sums | **holds for held-out triples in an integer lattice L**; the condition "connected pair graph" is false as stated | two connected counterexamples (12.2, V4); carry is not additive |
| V5 | The angles are learned forward-only by a closed form | **holds in the complete, closed case**; for missing pairs the determination is V4's lattice test, not the eigendecomposition | closure and completeness needed; T8.3 itself says it does not show learning |
| V6 | Re-entry acts | **holds for O** (the state changes) | the next prediction changes only under observability; re-entry off while reading is C and departs from the literal T14 step 12 |
| V7 | A loop with sampled traces does not improve answers on average | **holds** in expected probability and in log-loss of the correct answer | not a statement about hard-decoded accuracy; the condition on re-entry is explicit |
| V8 | A loop with least-action traces and taught worked steps helps multi-step problems | **half holds** | the "rank grows with n" half is true only when the number of distinct problems grows; the "each step needs only its own low rank" half is false in general (copying); the mechanism that survives is composition of steps that were each seen |
| V9 | Reading the word "wrong" does not teach correction | **holds** for the tables (proved at c = 0, checked numerically at c > 0) | not for O after a truncated recompile |
| V10 | A feedback port with signed coupling works | **holds with the update specified**; two hazards | the plan's wording "subtract the moments" is not defined for a sector; "wrong" carries almost no information about the right answer |
| V11 | Correction by reading: counting law no, capped law yes | **holds**; formula made exact and checked | the cap costs accuracy on saturated sectors (derived) |
| V12 | Question recognition from cues, given data | **holds for format recognition from surface cues only** | not content recognition |
| V13 | Teaching order matters, by mechanism | **holds in part**; the last sentence **fails** | capacity locking is independent of the plasticity law: "lateness costs time, not capability" is false under the governor |
| V2′ | Word-class levels, built by canalization from forward-only neighbor counts, help text bytes whose exact last-2-word context is unseen | **holds as a mechanism with a stated loss**; magnitude is E6b's | (a) the coarsening loss is exactly the within-class mutual information; (b) a table level has no useful aggregate cost bound; (c) the class map is a compile-time fit, the ids are aligned across compiles, not stable by proof |

### 12.1 The machine as it stands (S: code, spec)

Everything below is read from `src/vam/nested.py` (the nested machine) and `src/vam/nfa/hier.py` (E4's machine, whose compiled loop repeats the same arithmetic), configured as E4's chosen row (`results/e4.json` part2.chosen: counting law, prior 4, coherence 1.0, verify 3, upper-level verify 4, governor on with protect 0.5 and **sink rate 0.0**, levels W and S, text segmentation).

1. **Sectors.** A sector is a hashed table slot of a level ℓ, keyed by a function of the register at that level together with the bits of the current byte read so far. It holds a relation `w ∈ [-1, 1]`, a count `n` and an evidence `EV` (bits). Its own prediction of bit 0 is `p0 = ½(1 + (1 − 2ε) clip(w))` (`nested.py:213`), ε = 2⁻¹².
2. **Chain readout.** `p_0 = own_0`. For ℓ ≥ 1, if the sector of the present context is stored (and `n ≥ mature`, with `mature = 0`), `p_ℓ = coherent_mix(p_{ℓ-1}, own_ℓ, share, c)` with `share = 2^EV / (1 + 2^EV)`; otherwise `p_ℓ = p_{ℓ-1}` (`nested.py:196-204`, `655-672`). The prediction is the last `p_ℓ`. At `c = 0` this is the ordinary two-hypothesis mixture; at `c = 1` it is the Born pool `(√((1−s)q_c) + √(s q_o))²`, normalized over the two bit values. E3's chosen coherence, held fixed by E4, is 1.0.
3. **Learning.** At every visit to a stored sector with ℓ ≥ 1, `EV += log2(own_b) − log2(p_{ℓ-1}(b))` (own computed before the update; `nested.py:358`, `hier.py:440`). Then `w ← T_b + (w − T_b) e^{−κ dt/γ_n}`, `n ← n + 1`, with `T_b = ±g/κ` and `γ_n = γ + κ dt n` for the counting law (D10), `γ_n = γ` for the literal T12.1 law (`nested.py:207-210`; `Config` defaults g = κ = γ = dt = 1). Reading a bit therefore moves the relation toward the observed bit's target at every touched sector.
4. **Birth.** A context takes a slot only at its `verify`-th sighting (byte levels 3, upper levels 4 in the chosen row; doorkeeper), and a slot is created with `w = w_of(p_{ℓ-1})`, `n = min(parent_n, prior)`, `EV = 0`: its own prediction equals the coarser readout (isometric nesting, T11.4; `hier.py:424`). Replacing a stored sector starts a new **incarnation** with `EV = 0`.
5. **Governor.** Past half occupancy, a sector with `EV > protect` is not replaced; the knock does `EV *= e^{−2 sink_rate}` (`nested.py:331`, `hier.py:410`). At sink rate 0.0 this is the identity.
6. **State.** The machine's state is the register, the upper-level rings and counters (`HierMachine.state`); the tables are structure. Frozen reads (learning off) write no table (the E5 gates: `results/e5.json` arms.T.gates).

### 12.2 The verdicts

#### V1. Exact-context tables on unseen contexts or sums: WILL NOT generalize

**Claim (S, plan).** An unseen context only inherits from coarser levels.

**Proof (D, by construction).** Fix a history h. By 12.1.2 the prediction is a function of the parameters of the sectors `(ℓ, k_ℓ(h))` for those ℓ whose sector is stored; an absent sector contributes nothing (`p_ℓ = p_{ℓ-1}`, `nested.py:655-672`). By 12.1.3 a reading changes only the sectors of the contexts it touches, and by 12.1.4 a sector is born equal to its coarser readout. Write ℓ*(h) for the deepest stored level. Then `p(·|h)` is a function of `(k_ℓ(h))_{ℓ ≤ ℓ*(h)}` and of the parameters those sectors hold; it depends on the history only through those keys. For arithmetic, take the held-out string "3+4=". Every level whose key contains both operands (a byte level of depth ≥ 4, a W or S level whose unit or sentence contains both) has an absent sector, because the string occurs nowhere in the taught text; a hash collision is rejected by the 32-bit tag, so it is not a source of generalization (probability about 2⁻³² per probe). The levels that remain keyed by "+4=" (depth 3) hold the statistics of "x+4=" over every operand x seen with 4, so their prediction is the mixture over x. It equals the prediction for any other first operand. So the prediction cannot depend on the first operand: among sums a + 4 with a uniform over ten values, the best a-independent guess names a sum with probability at most 1/10. ∎

**Extends to rarely seen contexts.** A context seen fewer than `verify` times is also absent (doorkeeper): in the chosen row an upper-level context needs four sightings and a byte-level context three. A fact stated once in a text is not stored at the upper levels at all. This is consistent with E5, where recall followed frequency (`results/e5.json`; HANDOFF "E5").

**Spec check.** T4.1 (the state is the tables and registers, no transcript); T11.4 (isometric nesting: this is exactly why an unseen sector inherits); T13.2 (an unseen configuration is not impossible: the ε floor, no hard zero, creates no generalization); F-Part 9 section 1 asks for relational probabilities rather than static weights: V1 shows E4's realization does not provide the relational generalization that T12.3's third part needs ("unfamiliar arrangements expressing the same relations"); T12.3 itself says that guarantee is conditional on a structural equivalence being established. **Result:** holds; none of it needs weakening.

#### V2. A shared linear operator level O on unseen text contexts: WILL (mechanism proved, magnitude measured)

**What O is (C, T2.2 as the model).** T2.2's exact example is a linear map `M_u` on a population vector z with `p(u) = 1ᵀ M_u z` and `z' = M_u z / p(u)`. O is that form with learned real matrices: a state `b_t ∈ ℝ^k` (k ≤ 256), operators `B_x` for each byte x, a start state `b_1` and a normalizer `b_∞`, with `p(x) = b_∞ᵀ B_x b_t` and `b_{t+1} = B_x b_t / (b_∞ᵀ B_x b_t)`. This is an observable operator model (Jaeger 2000). T2.2's matrices are nonnegative and T3.2's congruence update is the quantum version; the learned `B_x` may have negative entries, so predictions can be negative: they are floored at ε (T13.2) and the share of negative mass is logged. The Kraus form (T2.2's density-operator realization) is the positive realization and is not attempted until linear O works. State size is fixed (`b_t` and the operators: 256·k² floats at most), independent of the stream (T11.5); the moment counts are aggregate statistics of a fixed-size store, not a transcript (F-Part 1, T4.1).

**(a) The cost of adding a stage is bounded (D).**

*Setting.* A stage of the chain (12.1.2) mixes its input readout with one more predictor, for each of its sectors. Take one sector incarnation with visits `i = 1..m` after birth. Write `q^c_i, q^o_i` for the probabilities that the coarser readout and the own prediction gave to the bit then observed (both in [ε, 1−ε]), `L^c_i = Π_{j≤i} q^c_j`, `L^o_i` likewise (`L_0 = 1`), `B_i = L^c_i + L^o_i`. Then `EV_{i−1} = log2(L^o_{i−1}/L^c_{i−1})` (the evidence starts at 0 at birth and nothing else changes it at sink rate 0; the first use is at the second visit, so the share at first use is ½), `r = 1 − share = L^c/B` and `share = L^o/B` at the previous step. The sector's own prediction need not be a learned relation: any predictor with probabilities in [ε, 1−ε] will do.

*Claim.* `Σ_i −log2 p_i ≤ min(Σ_i −log2 q^c_i, Σ_i −log2 q^o_i) + R`, with `R = 1` at c = 0 and `R = 2` at c = 1.

*Proof at c = 0.* `p_i = r q^c_i + s q^o_i = (L^c_i + L^o_i)/(L^c_{i−1} + L^o_{i−1}) = B_i/B_{i−1}`. The product telescopes to `B_m/B_0 = (L^c_m + L^o_m)/2 ≥ max(L^c_m, L^o_m)/2`. ∎

*Proof at c = 1.* With bit probabilities `q_0, q_1` under each hypothesis, `N_b = (√(r q^c_b) + √(s q^o_b))²` and `p_i = N_{b_i}/Z_i`, `Z_i = N_0 + N_1 = 1 + 2√(rs)·BC`, where `BC = Σ_b √(q^c_b q^o_b) ≤ 1` (Cauchy–Schwarz). Put `ρ_i = 2√(L^c_i L^o_i)/B_i ∈ (0, 1]`. Then `2√(rs) = ρ_{i−1}`, so `Z_i ≤ 1 + ρ_{i−1}`, and `N_{b_i} = (√L^c_i + √L^o_i)²/B_{i−1} = B_i(1 + ρ_i)/B_{i−1}`. Hence `p_i ≥ B_i(1 + ρ_i)/(B_{i−1}(1 + ρ_{i−1}))`, the product telescopes, and with `B_0 = 2, ρ_0 = 1`: `Π p_i ≥ B_m(1 + ρ_m)/4 ≥ max(L^c_m, L^o_m)/4`. ∎

*Check (D, computed).* 20,000 random sequences of up to 60 visits with arbitrary predictors: the largest observed excess over the best of the two was 1.0000 bit at c = 0 and 1.8848 bits at c = 1 (script in 12.9; an earlier variant with extreme-probability draws reached 1.9856, still under 2).

*Conditions the code must meet (S).* The proof needs `mature = 0` (so the share at first use is ½; with `mature > 0` the bound is `loss ≤ loss_c + 1 + log2(1/(1−s_0))`, and `≤ loss_o + 1 + log2(1/s_0)`, s_0 the share at first use), `sink_rate = 0` (a nonzero rate changes `EV` between visits and breaks the telescoping; E4's chosen row has 0.0), and `EV` updated only by the rule in 12.1.3. A replaced sector is a new incarnation, so the sum is over incarnations, not slots. O's own mixing node (C: one node per partial-byte state u, 255 nodes, never evicted) has the same form with `own = p_O`, so the whole O stage costs at most 2 × 255 = 510 bits over the entire stream.

*Scope, stated plainly.* The bound is relative to the stage's input. It is **not** a bound on E4+O against E4: the sectors below O are born at the readout that includes O (12.1.4), so their own predictions and evidences differ from E4's. The plan's claim "adding a level costs at most a bounded number of bits per sector" is true of the added stage and false as a statement about the whole machine. Fix (C, proposal, only if E6 shows a loss): add one top mixture of the E4 readout and the E4+O readout per class of "deepest stored level" (at most nine classes, so at most 18 bits in all), which does bound E4+O against E4.

**(b) Low rank is measured, with the numbers corrected (S).** E4's N-L2 (`results/e4.json` part1.N_L2 rows): W1 rank 33, W2 23, S0 17 (W4 and S1 kept no rows), each against a byte-shuffled null of rank 1. The byte-level numbers (`results/e3_t4.json` T4_2_text.A_train, depths 1 to 7 bytes) are **19, 40, 44, 51, 50, 48, 44** with 81 columns kept; the nulls are rank 1 (none at depths 6 and 7). The plan's "35 to 48" does not match that file. All of these are Page ranks of the **frequent** rows only (count floor 256: 81 to 3,242 rows kept of up to 858,000), so they say how many directions the frequent contexts carry above the random edge; they neither bound nor equal the rank of the whole Hankel matrix, and 40 to 51 of 81 columns is not "very low" in absolute terms. It is far above the null and well inside k ≤ 256.

**(c) Consistency (S, external).** Hsu, Kakade and Zhang (2009; JCSS 2012) prove that a spectral algorithm learns a hidden Markov model of rank m consistently, with sample complexity polynomial in m, 1/σ_m(P_{2,1}), 1/ε and log(1/δ). The theorem needs a finite-rank model with σ_m > 0. Text is not of finite rank, so the theorem gives the mechanism (the compiled operators converge to the best rank-k description as counts grow), not a promise on text. The negative predictions the spectral estimate can produce are handled by the ε floor, and (a) holds for any valid distribution, including the floored O.

**Why O can differ from the tables on an unseen context (D).** O's state after a string is `B_{x_n}···B_{x_1} b_1`, a function of the whole string through learned operators that were estimated from shorter shared pieces, not a key that must match exactly. Whether that function predicts better than the coarser readout on the bytes whose 8-byte window is absent from the teaching text is an empirical question; E6's criterion is the measurement.

**Spec check.** T2.2 (the operator form); T3.2 (positivity of the congruence update; O departs from it by allowing signed entries, flagged); T4.1 and T11.5 (fixed-size present state); T13.2 (floor, no hard zero); N-L2 (the Page rank sizes the basis: k is capped at 256 and compiled from the measured rank, not tuned); N-L3 (O joins the existing coherent mixture). **Result:** the mechanism holds; (a) narrowed, (b) corrected, (c) scoped.

#### V3. A linear O on held-out arithmetic pairs: WILL NOT

**Claim (S, plan).** The state after "3+" never meets "4", so the operator is unconstrained there.

**Proof (D), on the byte-level Hankel.** Let `P` be a process over byte strings, and `Ĥ` the empirical Hankel matrix of its training strings (counts of past-then-future). O is a function of `Ĥ` alone (compile reads only counts). Take the training set T = all strings "a+b=c" with `a + b = c` for pairs (a,b) in a seen set S, with (3,4) ∉ S, and let `P'` agree with `P` on every seen string and assign "3+4=9" instead of "3+4=7". Both processes have the same observed counts, so the same `Ĥ`, hence the same O and the same predictions on the held-out string; but they disagree on it. Both have Hankel rank at most 10 + c₀ (ten operand classes plus the fixed framing states): the rows "past ends in a+" for the ten values of a, and every completion of the unobserved entries still has at most ten such rows. So a rank bound k ≥ 10, which O has (k up to 256), does not separate `P` from `P'`; and a rank bound k < 10 cannot even represent the training data. The singular value decomposition of the zero-filled count matrix treats unseen entries as observed zeros, not as missing; completing a missing entry needs structure the matrix does not have (the (a)×(b,c) block has ten linearly independent rows). ∎

**On T11.2.** The plan says T11.2 merging cannot help because there is no observed future. The exact statement: T11.2 (compaction) and T11.3 (retention) merge two configurations only if their future behaviour is the same (`ΠK = ΠKΠ`), and the future of the held-out state was never observed. T11.2 itself is the nuisance-subsystem map `Π(ρ) = Tr_N ρ ⊗ σ_N`; it does not concern operators.

**Result:** holds. It is the reason V4 and V5 exist.

#### V4. Tokens as rotations on one shared generator, read by interference: WILL, for held-out triples in the lattice L

**Setting (C, realization).** Each token a carries a unitary `U_a = e^{iθ_a G}` with one Hermitian generator G, so `U_a U_b = U_{θ_a+θ_b}`. T2.3 writes the action of a bit as `e^{δG_u}` and stresses that the `G_u` do not commute (order sensitivity); a **commuting** subfamily is the exception, not what T2.3 states, so this is a chosen realization. T8.3 supplies the readout: on 25 configurations, `F` the Fourier matrix, `D_r` the phase ramp, `F D_r F† |0⟩ = |r⟩` exactly (the output amplitude is `δ_{kr}`); `F D_r F†` is the cyclic shift by r, so rotations compose as shifts and the readout is exact. T3.1 allows the angle to be an eigenvalue of a signed weight observable.

**What the seen data determine (D).** Let the tokens be 1..n and θ ∈ 𝕋ⁿ, 𝕋 = ℝ/2πℤ. A seen triple (a, b, c) with `a + b = c` is the equation `θ_a + θ_b − θ_c = 0`, i.e. the integer row `r = e_a + e_b − e_c`. Let L be the ℤ-span of the seen rows. **Claim.** For an integer vector r, every θ satisfying all seen equations satisfies `⟨r, θ⟩ = 0` **if and only if r ∈ L**.

*Proof.* (⇐) is a sum of equations. (⇒) Suppose r ∉ L. The quotient G = ℤⁿ/L is a finitely generated abelian group, so its characters into 𝕋 separate points (G is a finite direct sum of copies of ℤ and ℤ/m). Take χ with χ(r̄) ≠ 0 and set θ_k = χ(ē_k). Then every seen row gives 0 and r gives χ(r̄) ≠ 0. ∎

So a held-out triple is determined **iff** `e_a + e_b − e_c ∈ L`. The rational span is not enough: torsion matters.

**The plan's condition is false as stated.** "With a connected pair graph the solution is unique up to gauge" fails on connected examples (checked with a Hermite normal form; script 12.9):
- *Span failure.* Seen (1,3,4) and (2,2,4); held out (1,1,2). The graph is connected through 4, but `2e₁ − e₂` is in neither the integer nor the rational span of `e₁+e₃−e₄` and `2e₂−e₄`. Two worlds fit the data: θ = (0, 1, 2, 2) (and 2θ₁ ≠ θ₂) and θ = (½, 1, 1½, 2).
- *Torsion (the integer lattice is not its saturation).* Seen (1,1,3), (2,2,3), (2,4,5); held out (1,4,5). Then `e₁ + e₄ − e₅ = (e₂ + e₄ − e₅) + (e₁ − e₂)`, and `e₁ − e₂` is in the saturation of L but not in L; the Smith normal form of the seen rows has an elementary divisor 2. Over 𝕋, `2(θ₁ − θ₂) = 0`, so `θ₁ − θ₂ ∈ {0, π}`: two worlds fit every seen triple and disagree on 1 + 4. This freedom is a discrete one that no global rotation removes, so it is not a gauge.

For the unit-generated family the test reduces to: every token in the held-out triple satisfies `e_k − k e₁ ∈ L`, i.e. is reachable from the unit by coefficient-1 derivations through seen triples. A token that appears in no other seen triple (the result 18 = 9+9 appears only there) is never determined.

**Design-time computation (D, script 12.9).** One-digit sums: tokens 0 to 18, the 55 unordered pairs a ≤ b, a fraction h held out at random, L from the rest, 60 draws (seed 3). The fraction of held-out triples in L: 0.933 on average at h = 0.2 and 0.930 at h = 0.4 (minimum 0.727 over the draws). With ordered pairs (the commuted pair a + b / b + a seen) it is 0.984 and 0.977, because the commuted pair is the same row. So some held-out sums cannot be determined by any method, and E7's 90% criterion must be read on the determined ones (12.6, proposal P1).

**Carry (D).** Carry is not additive in a single character. With a cyclic period of at least 19, the sum of two digits is read exactly as one token 0..18 (T8.3's readout); a digit and a carry are then functions of that token, supplied by worked steps (V8). Multi-digit operands are multi-byte at O's byte unit, so the maps `T_a: b → c` are not defined between single tokens there (12.6, proposal P2).

**Evidence outside the machine (S, external).** Power et al. (2022) and Nanda et al. (2023) found that networks trained on modular addition learn Fourier (rotation) features. Those networks were trained by backpropagation with weight decay; this is evidence that rotations solve additive relations, not that they can be learned forward-only.

**Result:** the readout and the composition law hold exactly (T8.3); the data determine a held-out triple iff it is in L.

#### V5. Learning V4's angles forward-only, without number labels

**Claim (S, plan).** The per-operand maps `T_a: b → c` commute for additive relations; a joint eigendecomposition gives the characters and `e^{iθ_a}`; commuting learned operators are merged into one generator (T11.2).

**Proof in the closed case (D).** Let the tokens be the cyclic group ℤ_m with every residue both an operand and a result, and let `T_a` (estimated from counts of `P(c | a, b)`) be complete. Then `T_a = S^a`, the shift by a, which are circulant, commute (`T_a T_b = T_{a+b} = T_b T_a`) and are simultaneously diagonalized by the characters `χ_j(k) = e^{2πi jk/m}` (the columns of T8.3's `F`) with eigenvalues `e^{2πi ja/m}`. The eigenvalue of `T_a` on the j-th character is `e^{iθ_a j}`, `θ_a = 2πa/m`. A single eigendecomposition of a generic linear combination `Σ x_a T_a` (Jennrich's simultaneous diagonalization) returns the common eigenvectors when the combined eigenvalues are distinct. Forward-only: it reads counts only (compile at 1 MiB, like O). ∎

**What the plan's wording hides (D).**
1. **Closure.** `T_a` must map one space to itself. With bytes, operand and result digits are different byte strings and single tokens only for one-digit results; a result like "13" is two bytes. The maps are defined at a unit above the byte (12.6, P2).
2. **Completeness.** If pairs are held out, `T̂_a` has missing columns and `T̂_a T̂_b ≠ T̂_b T̂_a`; the joint eigenvectors are then approximate (Davis–Kahan: the error in the eigenvectors is of order the commutator over the eigenvalue gap) and the diagonalization is iterative, not closed-form. The exact answer for a missing pair is V4's lattice test; the eigendecomposition is the operator form of the same fact.
3. **Merging (T11.2).** The plan cites T11.2 for merging commuting operators into one generator. T11.2 is `Π(ρ) = Tr_N ρ ⊗ σ_N` (nuisance compaction); T11.3 requires `ΠK = ΠKΠ` for the compaction to keep the future. Merging operators whose commutator is small is linear algebra (a commuting family of diagonalizable operators has a joint eigenbasis) plus a threshold, and the threshold is C, declared before E7. T8.3 itself says: "the input explicitly supplies r; the machine has not inferred an unknown theorem or learned a textbook." It supports the readout (V4), not the learning.

**Result:** holds in the complete closed case (closed form), approximately for missing pairs; coverage (80%, 60%) is E7's to measure, with the lattice caveat.

**Addendum, 2026-10-22 (E7 design-time check, D; manager's ruling, `PROTOCOL.md` E7).**
1. **The closed-form route fails when pairs are missing.** The plan's route is a joint eigendecomposition of one generic combination of the maps `T̂_a` estimated from counts. With the held-out columns zero-filled (the maps then no longer commute), its top-1 on the held-out sums of addition modulo 10 is **0.24 at h = 0.2 and 0.20 at h = 0.4**, and **0.03** modulo 100 (h = 0.2 and 0.4). The approximation V5 item 2 allowed ("the error is of order the commutator over the gap") is not small here. On the integer sums of E7 the route is not even defined: the maps from operands 0 to 9 to results 0 to 18 are truncated shifts, nilpotent on the token space, so the only eigenvalue is 0; V5's closure condition (every token both operand and result, a cyclic group) fails. Evidence: `audits/e7_jennrich.py` (modulus 10 over 20 draws, modulus 100 over 5).
2. **The replacement, F5 (C, a chosen realization).** The joint decomposition of the `T̂_a` is the rank-R canonical (CP) decomposition of the equation tensor `C[a, b, c]`; with held-out pairs it is fitted on the cells that were read only (masked alternating ridge least squares), and read out by T8.3's Fourier readout. It is a compile-time fit to counts, like O's SVD: never fitted to answers or costs, never fed the held-out cells, with no backpropagation through the machine. At design time (development draws, not results) it gives top-1 0.92 on the determined held-out sums of stage 1 at h = 0.2 and 0.50 at h = 0.4 (`audits/e7_cp.py`). F5 does not change the verdict of V4 (the data determine a held-out triple iff it is in L); the exact lattice route F4 is the upper bound and the classifier of E7, not an arm of its criterion.


#### V6. Re-entry acts: WILL, for O

**Proof (D).** In O, re-entering the emitted symbol τ applies `b' = B_τ b / (b_∞ᵀ B_τ b)`. `b' = b` iff `B_τ b = λ b`, an eigenvector. So unless b is an eigenvector of `B_τ` the state changes. The next-symbol distribution `x ↦ b_∞ᵀ B_x b'` changes whenever the difference `b' − b` is not in the kernel of the readout; in a minimal (observable) representation distinct states have distinct futures, so some future differs, but the *next* prediction may coincide. The counts are untouched: re-entry is not learning (T10.3). ∎

**Spec check and the departure from T14 step 12 (C).** T9.3 and T9.4 show re-entry reshaping the joint density and the next probabilities; T10.1 puts `ℰ_τ` after the output instrument; T10.3 says re-entry is not new evidence. T14 step 12 re-enters every emitted bit before the next external bit. If O applied `B_τ` after every external byte while *reading*, the state would read each byte twice. So in O re-entry applies only to symbols the machine itself emits that are not in the external stream (generation); while reading, the external byte is the only update. This departs from the literal rule and is **C**; it is owner decision 6 (12.5). In the massless text machines of E3 and E4 the re-entry changes no number, so the departure changes none of their results.

#### V7. A loop with sampled traces: WILL NOT improve answers on average

**Claim restated exactly (D).** Fix a state S, learning off, and let the post-emission state equal the state after *reading* the emitted symbol (`Φ(S, x) = B_x S` normalized), i.e. re-entry equals the reading update. For any symbol event y at a later position, `E_{x₁ ~ p(·|S)}[ p(y | Φ(S, x₁)) ] = p(y | S)` (the tower property: `Σ_{x₁} p(x₁|S) p(y|x₁, S)` is the marginal). Hence:
1. the expected probability the machine gives to the correct answer after a self-sampled trace equals its marginal probability of that answer before; sampling traces adds no expected probability;
2. the expected log-loss of the correct answer cannot decrease: `E[−log p(y|S₁)] ≥ −log E[p(y|S₁)] = −log p(y|S)` (Jensen).

**What it does not say.** It is not a statement about a hard-decoded answer. If the machine's mode is wrong, sampling randomizes the decoded answer and its accuracy can exceed that of the deterministic wrong mode. E8 reports bits of the correct answer (as E5 did), which 1 and 2 cover. The condition matters: if re-entry used a channel different from the reading channel (as VAM-1's active-port coupling does, DESIGN D8), the chain rule would not hold and the identity would fail. The O re-entry of V6 satisfies it.

**Spec check.** T10.1 (path sums are not an additional stochastic generator: the sampling is the instrument's sample setting, the E4 default writing mode); T10.2 (Markov property, so `p(y|S)` is well defined); T10.3; T14 steps 11 and 12. **Result:** holds in the stated metric.

#### V8. A loop with least-action traces plus taught worked steps, on multi-step problems: HALF HOLDS

**Claim (S, plan).** A one-shot answer to an n-step problem needs a rank that grows with n; the trace commits and re-enters each step, so each step needs only its own low rank. Only with worked steps in the curriculum; otherwise V7 applies.

**First half (D): true when the number of distinct problems grows with n.** The rank of an operator model equals the Hankel rank, which is at least the number of distinct residual futures. In "a+b=c" with n-digit a, the pasts "a+" for the 10ⁿ values of a have pairwise different futures (the distribution of the completions "b=c" differs), so the rank is at least 10ⁿ. With k ≤ 256 the one-shot limit is therefore two digits for the first operand alone. It is **not** true for iterating a function on a fixed finite domain: after reading x₀ the future of `fⁿ(x₀)` has only m distinct values, so the rank is at most m whatever n is.

**Second half (D): false in general.** The trace does not by itself give each step a bounded rank. Each step must read its operands from the problem text, and a finite-state O that must emit a digit located d bytes back needs rank on the order of 10ᵈ when the d intervening bytes are themselves information to be remembered (as in "478+695="), and about 10·d when only one digit at a fixed offset matters and the gap is fixed framing; the exact tables reach 8 bytes (depths 8 to 64 bits) plus the W and S registers, and only for strings already seen (V1). A scratchpad that rewrites the remaining problem each step needs a copy of the remaining digits at every step, which is the same cost. With k ≤ 256 the working state that can be carried is about two digits. So "each step needs only its own low rank" holds only when each step's inputs lie within what the state and the exact window carry, i.e. a **step-local format**. In that format, for column addition, the machine is a finite-state transducer with a carry bit and the loop is not even needed. The loop is needed where the format forces non-local access but the steps are local after rewriting, and for that the rank argument gives no support.

**What survives, and is the real mechanism (D).** The loop composes steps that were each seen in teaching. A chain of n lookups `h(g(f(x)))` has mⁿ possible compositions but only 3m step facts. One-shot, the machine needs the composition itself seen (V1: no generalization) or inferred (V3: a linear O cannot). The loop asks f(x), re-enters the result, then asks g, then h, and each question is a context the teaching text contained. If each step is right with probability ρ, the chain is right with ρⁿ, against a chance level that falls like m⁻¹; with ρ = 0.95 and n = 5 that is 0.77. This is a coverage argument, not a rank argument, and it needs no assumption about O.

**On "least action" (D, C).** The machine's ħ → 0 collapse (DESIGN §10 L7; `collapse.py`) weights an h-bit continuation by `e^{−S/ħ}` with `S = −ln p`, the Euclidean form of `e^{iS/ħ}`. T8.2 is the stationary-phase theorem for the oscillatory `e^{iS/ε}` and says "stationarity is not synonymous with minimum action"; the ħ → 0 limit of the Euclidean weight is Laplace's method, a different theorem. T15.3 rejects replacing complex amplitudes by "shortest-path costs"; the Euclidean form is already a labelled chosen realization of N-L7 (§10), and N-L7 itself says the collapsed path is "the classical trajectory of least action", so V8 uses it as it stands (C). It is exact over the h-bit horizon, not a beam. If the owner reads T15.3 as excluding it, V8's trace becomes the instrument's default sampled trace and V7 applies.

**Stopping (S, C).** T14 step 15: after the external stream ends a continued answer "requires a specified boundary stopping condition; it is not inferred from the absence of bytes alone". T11.1 gives the existence of a limit for a dissipative semigroup (the lock), not a rule. The plan's rule (stop when ‖Δb‖ < δ, at a learned end-of-answer, or at 64 bytes) is C; δ is declared in the E8 protocol.

**Spec check.** T8.2, T14 steps 7 to 9 and 15, T15.3, N-L7, F-Part 2 (macro then micro, each token re-entering). **Result:** the compositional mechanism holds; the rank half is narrowed and the per-step-rank half is withdrawn. Fix (C, proposal): E8(i) uses a step-local format (the operands of each step adjacent to it, or a state small enough to carry), and carries a one-shot arm and a no-trace arm so that the effect of the loop is separated from the effect of the format; "worked steps against none" stays as planned.

#### V9. Learning from the word "wrong" by reading: WILL NOT

**Claim (S, plan).** Positive-only learning raises p(answer | question) at every reading.

**Proof (D).** Reading text changes a sector only by 12.1.3: `w` moves toward the observed bit's target and `n` grows. Reading "Q … A: 7. That is wrong." touches the sectors of the contexts ending in each byte read; the sectors on the path of "7" after "A:" move toward "7" and nothing in the sentence moves them away. The readout of the observed bit is also non-decreasing along the chain. Induction on ℓ: `p_0 = own_0` rises. For ℓ ≥ 1, write `f(c, o, s) = (1 − s)c + s·o` (c = 0). A reading raises the coarser readout (c' ≥ c) and the own prediction (o' ≥ o) and moves the share toward whichever hypothesis gave the observed bit the larger probability (s' ≥ s if o ≥ c, s' ≤ s otherwise), because `EV` changes by `log2(o/c)`. Since f is nondecreasing in c and in o, `f(c', o', s') ≥ f(c, o, s')`; and f is linear in s with slope `o − c`, so `f(c, o, s') ≥ f(c, o, s)` in both cases. ∎ at c = 0.

**At c > 0 (D, computed).** Not proved; checked on 300,000 random draws each at c = 0.5 and c = 1 with the coarser readout also updated: no violation (script 12.9). The verdict does not rest on it: the "wrong" bytes enter only as more text.

**What the machine does learn from "wrong".** It learns to continue "7." with "That is wrong", i.e. to emit the word after the answer, and it raises p(7 | question) by one reading. Nothing lowers it.

**Scope.** For a Phase 3 machine with O, a truncated-SVD recompile is not monotone in the counts, so the per-reading monotone claim covers the tables only. The verdict holds in substance there too: reading "wrong" produces no negative coupling, only counts.

**Spec check.** T12.1 (the force `gC − κw` drives w toward the observed correlation), T10.3, T12.2. **Result:** holds.

#### V10. A feedback port with signed coupling: WILL, with the update specified

**Claim (S, plan).** T12.1's force with signed C, T3.1's signed weights and N-L6's global sensory port give a feedback port: "wrong" subtracts the emitted answer's moments, "right" adds them; the world supplies the sign, the machine computes no score.

**What the spec allows (S).** T12.1: the force on w is `gC − κw` and C is a bounded correlation observable (the sign is free); the weight relaxes toward `gC/κ`. T3.1: a signed interaction weight is an observable `Ŵ` with eigenvalues of either sign. N-L6: the Global Sensory Blanket "ingests external bitstreams from the world"; a feedback signal is such a stream, but a **distinct signed port** is a chosen realization (C). Two cautions from the spec: T1 rejects replacing the mechanism with "an external correctness supervisor", and F-Part 3 rejects backpropagation's "external objective loss function and global supervisor". The port is argued to be neither: the sign arrives as input bits through a port (T14 step 2, T10.3 keeps ports separate), the update is the local T12.1 law at the sectors the emitted answer touched, forward-only, with no loss and no gradient, and the machine computes no score of its own answers. Owner decision 1 covers it.

**What "subtract the moments" means for one sector (C, specified here).** A sector holds only `w`, `n`, `EV`. A feedback event has a weight λ ≥ 1 and acts on the sectors on the emitted path (each node of the bit tree of the emitted bytes) under one **restriction R: only sectors with `n < n_max` are touched** (n_max is the capped-law cap of V11, declared in E8; the same threshold is used under the counting law). Saturated sectors (the top bit node, order-0 and order-1 contexts) carry general statistics, not this question, and are skipped. A path with no stored sector on it is a no-op and is logged (E8 reports the share of emitted paths on which feedback acted).
- **right:** the ordinary reading update, λ times (reading the answer λ times as correct text);
- **wrong, form S (signed push, T12.1 with C → −C):** `w ← −T_b + (w + T_b) ρ_n^λ`, `ρ_n = e^{−κ dt/γ_n}`, `n` unchanged: the sector relaxes toward the *other* bit's target, λ steps at its own step size;
- **wrong, form U (unlearn, milder):** remove `λ_eff = min(λ, (n − prior)·p_emit)` observations of the emitted bit from the running mean, `w ← (n·w − λ_eff·T_b)/(n − λ_eff)`, `n ← n − λ_eff` (so `n` never falls below the prior pseudo-count and nothing divides by zero). U cannot push a sector below the prediction it inherited at birth; S can.
`EV` is not changed by feedback (the sign carries no information about which of own and coarser predicts the world). Which form is used is C; E8(ii) compares them.

**Two-node example (D).** Top bit node of an ASCII byte: n = 10⁶, p0 = ½. Deep sector for the question's context: n = 10 (4 pseudo-counts at the inherited w = 0, six readings of bit 0), so `w = 0.6`, p0 = 0.8. A wrong answer whose first bit is 0, λ = 4. *Top node:* n ≥ n_max, skipped by R: unchanged. *Deep node, form S:* `ρ₁₀ = e^{−1/11} = 0.913`, `ρ⁴ = 0.695`, `w' = −1 + 1.6 × 0.695 = 0.112`, p0 = 0.556. With λ = 16: `ρ¹⁶ = 0.234`, `w' = −0.626`, p0 = 0.187: the sector now rejects the emitted bit. *Deep node, form U, λ = 4:* `λ_eff = min(4, 6 × 0.8) = 4`, `w' = (6 − 4)/6 = 0.333`, p0 = 0.667; with λ = 16 `λ_eff = 4.8`, `w' = 0.231`, p0 = 0.615, and no further feedback takes it below p0 = 0.5, the inherited value.

**The hazard R removes (D).** Without R, the signed push under the capped law moves every node with n ≥ n_max by the same step `1 − e^{−κ dt/γ_cap}`, about 1/257 per unit λ at n_max = 256, including the top node: one wrong answer with λ = 256 would move the top bit most of the way toward 1, the non-ASCII half 0x80 to 0xFF. The count-removal form of the plan ("subtract the moments") is scaled by 1/n only under the counting law; under the capped law, for n ≥ n_max, the correct inverse of one reading is `w ← T + (w − T)/r`, which moves a saturated node by about 1/W per observation whatever its real n. R avoids both. E8 logs the drift of the top node's p0 over the feedback phase under both laws; a drift above a declared tolerance fails the run.

**Effect on the same and on new questions (D).** On the same question, by construction, the deep sector moves at once if it is stored (and mature). On new questions the information is asymmetric. "Right" on "3+4=7" supplies a positive equation `θ₃ + θ₄ − θ₇ = 0` (V4): informative, and it determines every triple in the lattice. "Wrong" on "3+4=8" supplies only the exclusion of one value: `−log2(1 − p(8))` bits, small when p(8) is small. So E8(ii)'s re-test on held-out questions should be expected to move through "right" and through corrections the world supplies by text, much less through "wrong" alone.

**Result:** holds with this update (forms S and U under restriction R); the plan's wording "subtract the moments" is replaced by it; the port is C. Limit: feedback acts only where a sector is stored; correcting an answer inherited from coarser levels needs form S, and an answer with no stored sector on its path needs to be taught by reading (V11).

#### V11. Correction by reading: the counting law WILL NOT, the capped law WILL

**Proof (D).** After n readings of the wrong bit, the sector's w is near `−T`. Reading the right bit m times under `w ← T + (w − T) e^{−κ dt/γ_n}`:
- counting law, `γ_n = γ + κ dt·n_total`: the step is about `1/(n_total + 1)`, so the correction needs about n readings to bring w to 0 (p = ½) and more to win by a margin: the transient decays harmonically, not as `rⁿ` (D10; T12.2's `rⁿ` holds only for the constant-friction channel);
- capped law, `γ_n = γ + κ dt·min(n, n_max)`: for n ≥ n_max the friction is the constant `γ_cap = γ + κ dt n_max` and `r = e^{−κ dt/γ_cap}`, so `w_m = T(1 − 2 r^m)` and the correction wins (w > 0) when `r^m < ½`, i.e. `m > ln 2 · (γ + κ dt n_max)/(κ dt)` readings, independent of how many wrong readings there were. This is T12.2's `F^m(ρ) = Π_L(ρ) + r^m(ρ − Π_L(ρ))`. The plan's `r = 1 − 1/(n_max + 1)` is its linearization; the exact r is the exponential.

**Checked (D, computed, script 12.9).** With g = κ = γ = dt = 1: the counting law flips after exactly n readings (10, 100, 1,000, 10,000 for n = 10, 100, 1,000, 10,000); the capped law with n_max = 256 flips after min(n, 179) readings, and the formula gives `ln 2 · 257 = 178.1`.

**The capped law equals T12.1's literal law at large n (D).** At n ≥ n_max the capped friction is constant, which is T12.1's classical weight equation `dw = (g⟨C⟩ − κw)/γ dt` with `γ = γ_cap`; the counting law is D10's chosen addition and the literal law is T12.1's, so the cap is the point where the chosen law meets the specification's.

**What the cap costs (D).** A saturated sector tracks a frequency with an exponential window of about `W = n_max + 1` visits instead of all visits, so its estimate has variance about `p(1 − p)/(2W)` and an excess of about `1/(4 W ln 2)` bits per saturated binary prediction (0.0014 bits at W = 257, so about 0.011 bits per byte on contexts that are saturated). `n_max` is therefore chosen on validation (E8), not fixed here.

**Result:** holds, with the exact formula and the cost.

#### V12. Recognizing a question from cues: WILL, for the format, given the data

**Proof (D).** The machine's recognition of a context is its next-byte distribution in the state. After a question the shallow cues are recent bytes ("?", a newline, "Q:"), exact short contexts (depth ≤ 3 bytes) that are stored after a few sightings. If in the teaching text a fraction ρ of "?\n" occurrences are followed by the answer marker, the sector's own probability converges to ρ with error of order `1/√n`. So a machine taught question-then-answer turns learns to put its mass on answer-format bytes after a question-shaped state. **What it is not:** content recognition. The cue is a surface form; a rhetorical question with the same cue gets the same distribution, and E5 already showed that content-conditioned answers follow frequency. It needs question-then-answer turns in the data (owner decision 3). Spec check: T4.1 (the state is what recognizes), T2.1 (completeness and interpretation are observables of the present state, not hard decisions), F-Part 1. **Result:** holds for format recognition.

#### V13. Teaching order MATTERS, by mechanism: HOLDS IN PART; THE LAST SENTENCE FAILS

**Measured facts (S: `results/e5.json`, arms T and C, per stage).**
- Early material is forgotten: held-out bits per byte of the dictionary (S1) rise from 2.039 after stage 1 to 2.423 after stage 5 in T and to 2.408 in C; the prose stage's held-out bpb rises from 1.969 to 2.072 in T.
- Capacity: after stage 5 the occupancy of levels 4 to 7 (48-bit, 64-bit, W1, W2) is 0.962, 0.991, 0.987, 0.979; refused sightings 3.85 × 10⁹, protected knocks 5.0 × 10⁷, evictions 8.1 × 10⁷ (arm T, `arms.T.stages.S5.counters`).

**Mechanisms (D).**
1. *Counting-law lock on shared contexts.* Under the counting law a sector with n readings needs about n readings to be overturned (V11). The lock concerns **already visited** contexts; a context never seen has n = 0 and full plasticity, so material in new contexts is learned at full speed at any stage.
2. *Capacity locking, independent of the plasticity law.* Past half occupancy the governor refuses unverified contexts (`verify + 1` sightings) and protects sectors with `EV > protect`; at the occupancies above, new contexts at the saturated levels are mostly refused. This is a loss of capability for new contexts, not a loss of time.
3. *Protection by evidence.* A sector's `EV` measures how much better it predicts than the coarser readout. Early in teaching the coarser readout is weak, so many sectors beat it, earn evidence above `protect` and are protected; after O (or a stronger base) saturates, the same text would not have earned it, and those slots are spent on distinctions that became redundant. This is the content of "too early: the Page rank is not saturated" (D, mechanism; its size is an empirical matter for E9).
4. *O is order-invariant.* O's compile is a function of the counts alone, so if the moment store is a fixed-size sketch of sums (no eviction), the compiled O does not depend on the order of the teaching. Order then acts only through the tables (1 to 3).

**The last sentence fails.** "Under the capped law, lateness costs time, not capability" is true of mechanism 1 only. Mechanism 2 is the same under every law: E5's late material met saturated levels. **Fix (C, proposal):** the readiness rule becomes: the Page rank of O on a fixed held-out probe is saturated **and** the frozen held-out bpb gain on that probe is below δ per MB **and** the refusal rate of the saturating levels is below δ_r per MB (or their occupancy below ρ). The probe is a fixed passage evaluated frozen every 4 MiB; per-MiB *training* bpb is not usable (it varies by ±0.2 from book to book in `results/e5.json` arms.T.stages.*.curve_bpb_per_MiB). δ, δ_r and ρ are **unknown**: no saved measurement sets them, and E9's protocol declares them before the run from a pilot of the probe's noise.

**Spec check.** T12.2 (consolidation `rⁿ`, and its failure under the counting law), T12.3 (a learned equivalence needs to be established first), T11.5 (storage is fixed, so capacity is a real constraint), N-L6 (the Page governor), N-L2. **Result:** the order matters through 1 to 4; the rule is C.

#### V2′. Word-class levels on text bytes with an unseen exact context: WILL (mechanism proved, loss exact, magnitude measured)

Added 2026-10-22, after E6 (G6: C1 failed, stopping rule) and E7 (G7 passed), at the manager's direction (owner decision relayed in HANDOFF `6cacad2`). **Why this and not O again.** E6 found that O's spectral estimate, hashed 76 to 1 and sparse (0.29 counts per cell in C3), added nothing beyond the byte unigram; `audits/e6_diag.md`. The idea that survives is T11.2: merge configurations whose futures are the same, and let the merged configuration pool counts. For text the natural configurations are words, and E4 already holds them (the unit codes of the W ring).

**What it is (C).** A **class map** from unit code to a class id in 0..C−1 (C = 256, fixed in advance), plus one reserved id C for units outside the map. The map is rebuilt at each compile (every 2²⁰ bytes read with learning on) from **forward-only counts** of the unit stream: for each unit, how often each hashed neighbor followed it and preceded it. The rebuild is distributional: positive PMI of unit against neighbor feature, a low-rank factorization (d = 64), then spherical k-means weighted by unit frequency. Three new levels, each an ordinary upper level of E4's chain (the same ext columns, the same tables, doorkeeper, governor and evidence mixture): **K1** = hash(class of the last unit, the partial unit so far, the partial byte), **K2** = the classes of the last two units, **K4** = the classes of the last four units. Declared order in the mixture: byte levels, K1, W1, K2, W2, K4, W4, S0, S1. A new W sector is therefore born at the readout of its class counterpart, as every sector is born at the readout of the level before it (T11.4).

**(a) The coarsening loss is exactly a mutual information (D).** Let U be the previous unit, g the class map and X the next symbol (a byte of the next unit). With exact distributions, the best predictor given g(U) loses against the best predictor given U by `H(X | g(U)) − H(X | U) = I(X; U | g(U))`, because g(U) is a function of U (chain rule: `H(X|g(U)) = H(X|U, g(U)) + I(X; U|g(U))`, and `H(X|U, g(U)) = H(X|U)`). The same holds for the last 2 or 4 units with g applied componentwise, and for the multi-byte future by the chain rule. This is T11.2's `ΠK = ΠKΠ` made quantitative: a merge is lossless iff the merged units have the same future distribution, and the loss of an inexact merge is the average divergence of its members from their pooled distribution (a generalized Jensen–Shannon divergence).

**(b) What pooling buys (an approximation from S: Krichevsky and Trofimov 1981; Rissanen 1984).** A context seen n times costs a sequential estimator about `(A − 1)/2 · log₂ n` bits of learning per context over an effective alphabet of A symbols. Merging m contexts that share a future saves roughly `(m − 1)(A − 1)/2 · log₂ n` bits of learning and costs `n·m·I` bits of coarsening loss. So pooling helps exactly where counts are scarce and the members' futures are close, which is where the exact context is **unseen** (n = 0, the saving is the whole cost of a first sighting). Where the exact context is well estimated it can only lose; the evidence mixture then keeps the exact level (the K level is an input of the chain, not a replacement).

**(c) Coverage (D, computed at design time on A-validation against A-train; `audits/e6b_coverage.py`).** The declared store (2¹⁷ hashed rows with a majority-vote tag, 512 hashed neighbor counters per side, the 2¹⁵ rows with the largest tag counts, positive PMI, 64 Gram eigenvectors, weighted spherical k-means, C = 256), one clustering from the counts of all of A-train: an **upper bound** for the maps in force while A-train is read (the machine reads it under 31 evolving maps), and a count of keys, not costs. 7.87 million unit tokens, 204,677 distinct units; 3,944 of the 32,768 most frequent units lose their row to another unit, and **94.2% of A-train unit tokens (93.1% of A-validation's) are in the map**, the rest in class C. The share of A-validation bytes whose exact context key is seen in A-train is **0.885 (last unit), 0.582 (last two units)**; for the class keys **0.961 (K1) and 0.822 (K2)**. The set U2, the bytes whose exact last-2-unit context is unseen, is **41.8%** of A-validation; **57.5%** of it has a seen K2 context and **90.6%** a seen K1 context. U2 holds 72% of the class-0 bytes (the window class of E4 Part 5), and class 0 is 25% of U2. The first version of the script read A-test keys once (a different store: exact lookup of the top 32,768 units, 1,024 features, full SVD); nothing was tuned on those numbers and they are not cited. So the class levels have counts exactly where the exact levels have none; whether those counts predict is E6b.

**(d) The cost of adding a level (D, scope stated).** Each K level is a stage of the chain in the form V2a proves: at most 1 bit (c = 0) or 2 bits (c = 1) per sector incarnation against its input, because the sector's own prediction is mixed with the coarser readout by the evidence rule. For a table level this is **not** a useful aggregate bound: the number of incarnations is the number of contexts (up to 2²¹ slots per level), and the bound says nothing about E4+K against E4 (the sectors above are born at readouts that include K). E6b measures the net effect; V2a's fallback, a top mixture of E4 and E4+K, is available if E6b loses.

**(e) Limits, stated plainly (S, C).**
1. **A class is distributional, not semantic.** Units are E4's text-rule units (a word with the delimiter byte that closes it), so "the " and "the," are two units; classes may group by function, by position or by punctuation habit. Nothing here claims meaning.
2. **A compile-time fit to counts, like O's SVD.** The map is never fitted to costs or to answers, never fed bytes of the block it will predict (it uses counts that end before the block), and there is no backpropagation through the machine. Class ids are aligned to the previous map by maximum overlap (Hungarian), not stable by proof; a unit that changes class changes the contexts it appears in.
3. **Fixed size (T11.5).** The unit rows are hashed (2¹⁷ rows with a majority-vote tag per row); units whose row is taken by another unit, or outside the 32,768 most frequent, are in class C. About 5% of tokens are outside the top 32,768, more are unknown at read time.
4. **E6's lesson applies.** Any added stage has a state-independent effect (E6's byte unigram alone was worth 0.037 bpb on class 0). The control is therefore a **random class map with the same C and class sizes**, not E4 alone.
5. **The field's expectation is moderate.** Brown et al. (1992) found class-based language models help mostly when interpolated and when data are scarce; Goodman (2001) reports small gains for classes over strong n-gram baselines. Nothing here promises to beat lzma.

**Spec check.** T11.2 and T11.3 (canalization, `ΠK = ΠKΠ`: an approximate merge with exact loss (a)); T11.4 (isometric nesting: a new sector is born at the coarser readout); T11.5 (fixed-size present state: the class map and the neighbor counts are aggregate statistics, not a transcript); T13.2 (floor, no hard zero: the K levels use E4's ε); N-L2 (a rank chosen by the data, here d fixed in advance); N-L3 (the new levels join the existing coherent mixture). **Result:** the mechanism holds, the loss is exact, the size is E6b's.

### 12.3 What changes if the verdicts stand (D)

E5 measured frequency-weighted recall. §12 says why: exact tables cannot generalize (V1), a linear O adds shared structure but cannot invent a held-out sum (V3), rotations with a shared generator can when the relation is additive and in the lattice (V4, V5), and a loop composes only steps it has seen (V8). Correction by reading needs the capped law (V11) or the port (V10); the world's sign is the only source of "wrong" (V9). The order of teaching matters through capacity as much as through plasticity (V13).

### 12.4 Phases (S: the plan; the engineer's proposals are marked and wait for the manager's ruling)

**Phase 1, E6: the shared operator level O (V2, V6).** `src/vam/nfa/operator.py`.
- `MomentStore`: hashed counts; past length from the N-L2 turnaround; the future is the next byte.
- `compile()` every 1 MiB: whitened SVD and Page rank, reusing `page.page_spectrum`, `whiten`, `split_edge` and `e4_hier.floor_rank`; k ≤ 256; `B_x = Uᵀ P_x (Uᵀ P)⁺`; `b_1` and `b_∞`; the state carried over by re-projection.
- `step` and `predict` with an ε floor (T13.2); the negative-mass share is logged.
- Mixture: an `extp` column on a copy of `hier._make_loop`, with O right after level 0, so new contexts are born at O's prediction.
- Tests: convergence to a known HMM; rank 1 gives the unigram; compile reads only counts; O off is bit-identical to E4; re-entry changes O's state and never the counts.
- Arms: E4, E4+O, and E4+O at rank 1. Criterion: class-0 bytes (window absent from A-train) at least 0.02 bpb better than E4 and than the rank-1 arm. About 1.5 hours. Gate G6.
- *Proposals.* P3: O's mixing node keyed by partial-byte state (255 nodes), so the stage cost is at most 510 bits (V2a); the top-mixture fallback if E4+O loses to E4 (V2a); the E6 tests include "the cost of the O stage against its input stays within 2 bits per node" as a checked bound.

**Phase 2, E7: rotation families (V4, V5).** `families()` groups commuting tokens (threshold declared); a joint eigendecomposition gives the generator and angles; `exp(iθG)` with the T8.3 Fourier readout. Synthetic a+b=c, single digits then two, with 20% and 40% held out and the pair graph connected. Arms: E4, E4+O, E4+O+families. Pass: at least 90% at 20% held out; V3's failure reported. Gate G7.
- *Proposals.* P1: choose the held-out set by the lattice test (Hermite or Smith form of the seen rows) and score the 90% criterion on held-out triples that are in L, reporting the others separately (a mean of 0.93 of held-out triples are in L at 20% and 40% held out, minimum 0.73, so an unfiltered 90% depends on the draw); hold out unordered pairs unless commutativity is itself what is being learned. P2: multi-digit operands are multi-byte at O's byte unit, so the `T_a` maps are not defined there; two-digit sums go through the sum token and the worked steps, or the family is defined at a unit above the byte.

**Phase 3, E8: the loop, the feedback port and the capped law (V7 to V11).**
- Ports: sensory, active, and a global feedback port (right +, wrong −, weight λ).
- Capped law, with n_max chosen on validation.
- Least-action loop with re-entry, stopping when ‖Δb‖ < δ (T11.1 for the limit, δ declared), at a learned end-of-answer, or at 64 bytes.
- (i) worked carry steps against none, one-shot against the loop; (ii) exam feedback, then a held-out re-test, against none; (iii) *Two New Sciences*, counting against capped, with a no-marker arm; three seeds where sampling enters. Gate G8.
- *Proposals.* P4: E8(i) in a step-local format with a one-shot arm and a no-trace arm (V8). P5: the feedback update is "unlearn `λ_eff` observations" (V10), with λ chosen on validation, and E8(ii) reports "right" and "wrong" separately. P6: re-entry only for emitted symbols (V6, owner decision 6).

**Phase 4, E9: the teaching order (V12, V13).** Stages PG-19, then Webster's, conversation (dialogues), synthetic mistake-correction and worked steps, the textbooks, and an exam with feedback, each advanced by the readiness rule. Arms: this order, reversed, and fixed sizes. Measures: E5's questions, retention, the feedback re-test. Gate G9.
- *Proposal.* P7: the readiness rule also watches the refusal rate or occupancy (V13), on a fixed frozen probe.

### 12.5 Owner decisions (defaults in brackets)

1. O, families, the feedback port and the capped law as chosen realizations (T15.4) [approve].
2. O's unit [byte].
3. The conversation source [dialogues now].
4. Segmentation [modality-free, with the text rule as an arm].
5. Synthetic data, labelled [yes].
6. *Added by the engineer (V6):* re-entry acts only on symbols the machine itself emits, not on external bytes being read; this departs from the literal "re-enter after every bit" of T14 step 12 [approve; the alternative reads each external byte twice].
7. *Added by the engineer (V8):* whether the Euclidean form of N-L7 is accepted as "least action" under T15.3 [accept, as §10 already does; otherwise the trace is the sampled instrument and V7 applies].

### 12.6 Engineer proposals needing the manager's ruling

P1 and P2 (E7), P3 and the top-mixture fallback (E6), P4 (E8(i)), P5 (E8(ii)), P6 (re-entry), P7 (E9). None changes Phases 1 to 4 as written until ruled; each follows from a verdict above.


**Rulings (manager, 2026-10-21, after G5).** Owner decisions 1-6 approved at their defaults; decision 7 accepted
as a trial (the Euclidean N-L7 at hbar -> 0 is the loop's trace, labelled C; the sampled-instrument trace is
V7's control). P1-P7 are accepted as proposed, each a chosen realization written into its experiment's
protocol before the run: P3 for E6; P1, P2 for E7; P4, P5, P6 for E8; P7 for E9 (its thresholds from the E9
pilot). P3's top-mixture fallback is built only if E6 loses against E4.

**Note to Phase 3 (owner-approved addition, 2026-10-21, via the planner; D, numerical methods with proofs, no new
selection rule: the same sum computed exactly or with a certified bound, T14 step 7).**
1. The loop's trace at hbar -> 0 is found by best-first (Dijkstra) search on S = -ln p. The step costs are
   nonnegative, so the first complete path popped is the least-action path. In the planner's design check it was
   exact in 40/40 cases at h = 8 and 12, expanding 11.4 nodes against 511 and 17.5 against 8,191.
2. A Feynman-form collapse expands prefixes in Euclidean order and bounds each open prefix by
   sum sqrt(p) <= sqrt(2^r P(prefix)) (Cauchy-Schwarz). It stops when the |A_0| and |A_1| intervals separate, so
   the decision is certified equal to the full sum. Check: 40/40 equal, h in {8, 12}, hbar in {0.25, 0.5, 1}, 2-3x
   fewer nodes.
3. Feynman sampling to a tolerance on Z saves little with this bound (465-478 nodes against 511 at tol 1e-3), so
   it is used only where sampling is needed.
4. Euclidean at hbar = 1 is the machine's own prediction, and needs no paths.
E8 uses method 1 for the trace and method 2 for any Feynman step. Each needs a test that its decisions equal full
enumeration on random states. `collapse.paths` currently drops steps with p <= 1e-12 without a bound; method 2's
bound makes that cut certifiable.

### 12.7 Not doing

Match models, beams, answer selection, learned mixers, grid extensions after the test split, the lzma diagnostic, and the Kraus form of O until linear O works. Also not done: an external scorer of answers, replay of a transcript, backpropagation.

### 12.8 The advisor's conclusions (recorded because no other session can read them; the advisor's model is not visible from this session, so it is not named)

1. The citation glosses of the plan did not all match the spec text: T2.2 is the nonnegative linear-map example, not spectral learning; T8.3 shows exact Fourier readout and says it does not show learning; T8.2 is oscillatory stationary phase, not the Euclidean limit; T11.1 is the lock as a semigroup limit, not a stopping rule; T11.2 is nuisance compaction, not operator merging. §12 cites what the text says and marks what is D or C.
2. V4's "connected pair graph" is false as a condition: the exact criterion is membership of `e_a + e_b − e_c` in the integer row lattice L; two connected counterexamples (span failure, torsion) are given, and the test belongs in E7.
3. V10: "subtract the moments" has to be defined for a sector holding only `w`, `n`, `EV`; relaxing toward the other bit's target is stable but moves every node on the path equally and, under the capped law, drifts the top bit toward the non-ASCII half; unlearning scales by `1/n`.
4. V2(a): the bound is per stage input and per incarnation; it needs `mature = 0` and sink rate 0 (the chosen row has both) and says nothing about E4+O against E4.
5. V9: monotonicity along the chain needs the induction on the coarser input; for O after a truncated recompile it does not hold per reading.
6. V6 and V7 depend on re-entry equalling reading and learning being off; "re-entry off while reading" departs from T14 step 12 and is an owner decision.
7. V3 is a statement about any learner from counts, not only about the toy block.
8. The plan's changes are proposals, not silent rewrites.
9. *Second advisor round, after the first commit.* V10's first "unlearn" formula divided by `n − λ_eff` and inverted a sector instead of erasing it; its worked example was inconsistent with the mean semantics (six readings of bit 0 give w = 1, not 0.8, and inherited pseudo-counts are not readings). It is replaced by forms S and U with the pseudo-count bound.
10. The 1/n scaling that protects the top bit node holds only under the counting law; under the capped law, which Phase 3 uses, saturated nodes move by about 1/W per observation. Restriction R (feedback only on sectors with n < n_max) is the fix, and E8 must log top-node drift under both laws.
11. V8's copying cost is 10ᵈ only when the intervening bytes must be remembered; with a single digit at a fixed offset and fixed framing it is about 10·d. The verdict stands.

### 12.9 Design-time computations (D, scripts)

These produced the numbers in V2, V4, V9 and V11. They are design checks, not experiment results; each is rerun as a gate in the phase that uses it. The Python and library versions are the repository's pinned environment.

```python
# (1) V2a: regret of the two-hypothesis chain stage against the best of {coarser, own}
import sys, math, random
sys.path.insert(0, "src")
from vam.nested import coherent_mix, _share
random.seed(1)
def worst(c):
    w = -1e9
    for _ in range(20000):
        EV = Lc = Lo = Lm = 0.0
        for _ in range(random.randint(1, 60)):
            pc, po = (min(max(random.random(), 1e-6), 1 - 1e-6) for _ in range(2))
            b = random.random() < 0.5
            q = (lambda p: 1 - p) if b else (lambda p: p)
            pm = coherent_mix(pc, po, _share(EV), c)
            Lm += -math.log2(q(pm)); Lc += -math.log2(q(pc)); Lo += -math.log2(q(po))
            EV += math.log2(q(po)) - math.log2(q(pc))
        w = max(w, Lm - min(Lc, Lo))
    return w
print(worst(0.0), worst(1.0))        # 1.0000 and 1.8848 in the run reported
```

```python
# (2) V4: held-out triples determined by the seen ones (integer lattice membership)
import random, numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form
def row(a, b, n=19):
    v = [0] * n; v[a] += 1; v[b] += 1; v[a + b] -= 1; return v
def in_lattice(rows, r):
    return hermite_normal_form(Matrix(rows).T) == hermite_normal_form(Matrix(rows + [r]).T)
random.seed(3)
pairs = [(a, b) for a in range(10) for b in range(a, 10)]      # unordered
for h in (0.2, 0.4):
    fr = []
    for _ in range(60):
        held = set(random.sample(pairs, round(h * len(pairs))))
        seen = [row(a, b) for a, b in pairs if (a, b) not in held]
        fr.append(sum(in_lattice(seen, row(a, b)) for a, b in held) / len(held))
    print(h, np.mean(fr), min(fr))   # 0.933 / 0.727 and 0.930 / 0.727 in the run reported
t = lambda a, b, c, n=6: [(i == a) + (i == b) - (i == c) for i in range(n)]
print(in_lattice([t(1, 3, 4), t(2, 2, 4)], t(1, 1, 2)))                       # False
print(in_lattice([t(1, 1, 3), t(2, 2, 3), t(2, 4, 5)], t(1, 4, 5)))           # False (torsion)
```

```python
# (3) V9: the readout of the observed bit does not fall after one reading (coarser input also updated)
import random, math
from vam.nested import coherent_mix, _share
random.seed(5)
for c in (0.0, 0.5, 1.0):
    bad = 0
    for _ in range(300000):
        pc, po = random.uniform(.01, .99), random.uniform(.01, .99)
        EV = random.uniform(-6, 6); b = random.random() < 0.5
        qb = (lambda p: 1 - p) if b else (lambda p: p)
        r0 = qb(coherent_mix(pc, po, _share(EV), c))
        tgt = 0.0 if b else 1.0
        pc2 = pc + random.random() * (tgt - pc); po2 = po + random.random() * (tgt - po)
        EV2 = EV + math.log2(qb(po)) - math.log2(qb(pc))
        bad += qb(coherent_mix(pc2, po2, _share(EV2), c)) < r0 - 1e-12
    print(c, bad)                     # 0 for each c in the run reported
```

```python
# (4) V11: readings needed to overturn n wrong readings (g = kappa = gamma = dt = 1)
import math
def flips(n_wrong, cap=None):
    T, w, n = 1.0, 0.0, 0.0
    fr = lambda n: 1.0 + (n if cap is None else min(n, cap))
    for _ in range(n_wrong):
        w = -T + (w + T) * math.exp(-1 / fr(n)); n += 1
    m = 0
    while w <= 0:
        w = T + (w - T) * math.exp(-1 / fr(n)); n += 1; m += 1
    return m
print([flips(n) for n in (10, 100, 1000, 10000)], [flips(n, 256) for n in (10, 100, 1000, 10000)])
# [10, 100, 1000, 10000] and [10, 100, 179, 179]; ln 2 * 257 = 178.1
```
