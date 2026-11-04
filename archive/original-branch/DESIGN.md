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
After each phase an independent reviewer checks the mathematics and code against T and F, read-only. The reviewer is a separate session of the most capable model, the owner's choice (2026-10-18). Findings and responses are recorded in `AUDIT.md`.

---

## 7. Work split

| Work | Who | Why |
|---|---|---|
| Theory-to-code design, core mathematics, machine, verification, final calls | this thread (the most capable model, high effort) | precision-critical |
| Independent mathematical and code audit after each phase | a separate reviewing session of the same model, read-only | an adversarial second reader catches what the author misses |
| Well-specified modules with written acceptance tests (bit kernel and benchmark, tensor-network layer) | a helper session on a mid-sized model, medium effort, in its own files only | routine coding against a precise spec |
| Mechanical chores (full test runs, formatting, receipt regeneration) | a helper session on the smallest model, low effort | cheapest model that can do it |

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
| D11 | N-L3: which density matrix | a two-hypothesis density per nesting step (the coarser readout against the finer sector, weights (1 − share, share), coherence c), applied level by level | one density over all levels at once would break isometric nesting (a new sector would move the readout); the pairwise form keeps it exactly for every c (tested) |
| D12 | N-L5: what "verified" means for a new sector | a doorkeeper of two-way buckets (16-bit tag and count) per hashed level; a context takes a slot at its v-th sighting; a newcomer replaces the weaker of the pair; an admitted context is cleared from the door | a direct-mapped door let colliding recurrent contexts block each other forever, and an aging door verified noise (both measured while building) |
| D13 | N-L6 on the text machine: capacity and sinks | capacity is the slots of each hashed level; from half full (Page's turning point) a newcomer needs two sightings and a knock on a protected sector (evidence above `protect` bits) multiplies its evidence by e^(−2 rate) instead of evicting it | the table is the text machine's register; Re λ = −2 is N's sink |
| D14 | N-L6 on the chain: measure, capacity, sinks | S2 = −log2 Tr ρ² (free on the MPO), capacity the qubit count; unit-rate σ_z dephasing on every qubit the readout does not einselect; at the threshold the bond is frozen | the sinks drain exactly the coherence the readout never reads; on short chains they change no prediction at all |
| D15 | N-L2 on text: which contexts the Page analysis counts | contexts and continuations seen at least 8 times in each half of every split (E2's "familiar") | with every observed context kept, rare cells lift the split-half edge to 1 and every rank to 0 (E3's first measurement) |
| D16 | N-L7: the form of the path integral | Euclidean e^(−S/ħ) by default (ħ = 1 is the machine's own prediction, ħ → 0 least action); the literal real-time sum |Σ √p e^(iS/ħ)|² as an option | the Euclidean form is the one whose limits are the machine's own quantities; both are measured in E3 |

---

## 10. The nested fractal architecture (N, the owner's design of 2026-10-18)

The owner's instruction: integrate every layer, leave nothing out, make each part work where it does not
work as written, and measure where the design does what it was designed to do. The goal is unchanged:
one machine that reads raw bits (textbooks first), predicts and generates. Every layer below is a running
part of the machine, with the test that it does what N says. Experiment E3 (protocol in
`experiments/PROTOCOL.md`, results in `RESULTS.md`) measures each layer and then runs the whole machine
on E2's books. New code is in `src/vam/nfa/`; the text-machine options are in `src/vam/nested.py`. Every
option is off by default, and with all of them off the machine is bit-identical to the one E2 ran
(tested).

### 10.1 As built

| Layer (N) | What N asks | What the machine does | Code | Tested by |
|---|---|---|---|---|
| L1 bits | every modality is configurations of bits | any file's bytes, most significant bit first | `bits.py` | existing |
| L1 equation | a sentence is a flow dx/dt = f(x, t) | every event is the unit-time solution of seven ODE flows on the whole state: the normalized no-jump filter dR/dt = −½{K, R} + Tr(KR) R with K = −Σ ln π_s(b) P_s (its integral dℓ/dt = Tr(KR) is the bit's code length, ℓ(1) = −ln q); transport and the environment units entering and leaving as exchange Hamiltonians (π/2)(I − SWAP); the coin H = θ/2 Σ(XX + YY); the sector dissipator g(Π − I) (g = ∞ is the exact projection); emission; re-entry; plasticity dw/dt = −(κ dt a/γ_s)(w − target), dn/dt = a. `trajectory()` records x(t) for a sentence | `nfa/flow.py` | an ODE solver reproduces every reading and generation event to 1e-10 (`test_nfa_flow.py`) |
| L1 register | the Bekenstein bound with the CPU register | exact Bekenstein and Bekenstein-Hawking bounds (CODATA); the register, cache line and caches read from the machine; one equation's label holds min(Bekenstein of a 1 µm silicon register, 1.6e22 bits; word, 64) = 64 bits; every nested depth is checked against it | `nfa/bounds.py` | `test_nfa_bounds.py` |
| L2 Page limit | the basis is sized by the Page limit | Page's exact mean entropy, the Page curve, the random-state and Marchenko-Pastur edges with Tracy-Widom width, and an empirical split-half edge; the Page rank of the whitened Hankel matrix is the number of predictive states. On text: the byte depth profile over the verified support (D15) and its turnaround, which proposes where the nested levels stop. The flow's own trajectory is exact, so its basis is cut at the register's resolution instead (`trajectory_basis`) | `nfa/page.py`, `nfa/flow.py` | recovers 3, 4, 8 and 16 hidden states and 1 for i.i.d. bits, never more (`test_nfa_page.py`) |
| L3 density matrix | lift to quantum vectors, ρ = Σ p_i \|ψ_i⟩⟨ψ_i\|, probabilities in [0, 1] | coherent readout of the nested levels (D11): the prediction of bit 0 is T_0 / (T_0 + T_1), T_b = Tr(P_b ρ) with ρ = (1 − c) diag(1 − s, s) + c \|a⟩⟨a\|, a = (√(1 − s), √s), P_b = \|φ_b⟩⟨φ_b\|, φ_b = (√p_b, √o_b). Always in [ε, 1 − ε]: with r = 1 − s and K = (1 − c) + c(√r + √s)², every T_b ≥ εK (both levels predict at least ε), while T_0 + T_1 = (1 − c) + c[1 + 2√(rs)(√(p_0 o_0) + √(p_1 o_1))] ≤ K by Cauchy-Schwarz | `nested.coherent_mix` | the density form, the bound, and isometric nesting for every c (`test_nested.py`) |
| L3 cache size | size and rank bounded by the cache | largest density dimension (55 in L1d, 362 in L2) and MPO bond (90 in a 2 MiB L2: the two-site tensor D·16·D·16 bytes) | `nfa/bounds.py` | the governed engine's default bond |
| L4 blankets | each node has sensory, internal and active parts | nodes are the chain's cells; the conditional mutual information I(cell j−1 : cell j+1 \| cell j) of every triple; the containment of a local disturbance (it moves one cell per event; the only faster path is the observation acting on the whole context region, the machine's sensory blanket) | `nfa/blankets.py` | GHZ 1 bit, Markov chain 0; untruncated containment to 1e-12 (`test_nfa_chain.py`) |
| L5 tensors | density matrices assembled into a multi-scale tensor network | the chain's density as an MPO of vec ρ; the text machine's nested levels as the learned scales | `vam1_mpo.py`, `nested.py` | existing |
| L5 area law | the bond follows the entanglement across the cut | the cache bond with a cutoff, or `bond="entropy"`: each cut keeps ⌈margin · 2^S⌉ directions (S its Schmidt entropy) under the cache cap | `nfa/governor.py` (`EntropyBond`) | bonds follow the entropy, predictions stay those of the fixed bond (`test_nfa_chain.py`) |
| L5 verified growth | rank grows only for verified, non-noise invariants | text: the doorkeeper (D12), one-off contexts never take a slot. Chain: the state is exact, so a weak Schmidt direction is real correlation, not noise; there the unverified part is the coherence the readout never einselects, and the sinks of L6 drain it (the Page edge is used where noise exists: sampled matrices, L2) | `nested.py`, `nfa/governor.py` | one-off 40-bit contexts take no slot; repeated text is admitted at its second sighting (`test_nested.py`) |
| L6 global blanket | sensory port, internal core, active port | VAM-1's two ports; `nfa/flow.py` makes the environment units entering and leaving explicit | `vam1.py`, `nfa/flow.py` | existing |
| L6 Page governor | track information against capacity; at the Page threshold halt expansion, sinks with Re λ = −2 purge noise, invariants are kept | text (D13): from half full, admission needs two sightings and knocks on protected sectors drain them by e^(−2 rate). Chain (D14): S2 against the qubit count; from the threshold (or always) unit-rate σ_z sinks, superoperator diag(1, e^−2, e^−2, 1), on every qubit the readout does not einselect; verified (context) sectors are never drained; the bond stops growing at the threshold | `nested.py`, `nfa/governor.py` | Re λ = −2 against expm of the Liouvillian; the governed engine equals the dense machine with sinks to 1e-10; a learned sentence survives noise (`test_nfa_chain.py`, `test_nested.py`) |
| L7 collapse | path integral over the field, interference, stationary phase, the least-action step emitted | paths of h future bits with action S = −ln p; Z_b = Σ e^(−S/ħ) (Euclidean) or \|Σ √p e^(iS/ħ)\|² (Feynman); collapse emits the larger Z_b, sample draws from Z; ħ = 0 is least action, ħ = 1 the machine's own prediction; `verified=True` lets only sectors above `protect` bits speak (emission through the global blanket) | `nfa/collapse.py` | ħ = 0 equals the macro/micro planner bit for bit; ħ = 1 equals the prediction (`test_nfa_collapse.py`) |

### 10.2 What the measurements say (E3 Part 1; the numbers are in `RESULTS.md`)

- The chain obeys the area law only with the governor. Without it the largest cut entropy grows with the
  chain (1.5 bits at 4 cells to 3.5 at 10) and the bond-64 engine throws away most of the state; with it
  the entropy stays under 1 bit and falls with length, and the engine is near exact (discarded weight
  1e-7 against 0.96 at 8 cells). At 12 cells the governed engine at bond 64 is within 7e-6 of bond 128,
  against 5.6e-4 for the ungoverned engine.
- The blankets become real: with the governor a disturbance outside the event cone leaks 1.6e-6 (plain:
  1.5e-3), and neighbouring cells are nearly independent given the cell between them (0.002 to 0.04
  bit against up to 0.3).
- The sinks are cheap. With the context einselected exactly and at most two cells outside it they change
  no prediction at all (1e-15): a coherence the coin makes splits into a right-mover and a left-mover
  that never meet again, so it never reaches a population (the proof is in `AUDIT.md`, Audit 2). With longer chains or finite g single
  predictions move (up to 0.06) but the bits per bit change by at most 1.4e-4.
- The area-law bond makes the chain cheap: at 12 cells, margin 8 holds bonds of 18 at most and runs 70
  times faster than bond 128, with predictions within 6e-4.
- On text, the Page rank of the byte Hankel matrix rises with context depth and levels off at 6 bytes
  (20, 40, 54, 58, 58, 61, 60 verified directions for 1 to 7 bytes, on 4 MiB of A-train).

### 10.3 Departures from the plan, and why

- No separate `nfa/engine.py`: the whole machine is the nested text machine with the N options on, and
  the governed MPO engine on the chain. A wrapper would have added a second path to the same code.
- The plan cut the chain's bonds at the Page edge; that edge separates sampling noise from structure,
  and the chain's state is exact (see L5 verified growth above). The entropy-governed bond realizes the
  area law instead, and the sinks realize the purge.
- The text Page analysis counts only verified contexts (D15).
