> Historical research record, preserved for context. Session-specific instructions, branch restrictions and pause/resume notes describe the original research workflow; they are not instructions for users of this standalone repository. Private session identifiers have been removed.

# VAM handoff: continuing the nested fractal (N) work

## Paused (2026-10-22, owner's request; manager)

Branch `claude/vam-nfa-continue`, last pushed work `ced8b93` (no pushes since).
- **Done:**
  - Phase 0; E6 (G6 failed, stopping rule); E7 (G7 passed); E6b run (`d6ce3de`, C-E6b failed, E4+K loses on
    A-test); E6b diagnosis (`9198356`).
  - Development check (`d894a5b`, A-validation only), which decided Amendment 1 = A, the top mixture; built at
    `ced8b93`.
- **E6b complete:** the Amendment 1 run finished after the pause began (`78b8ddc`, resumed after a worker restart, peak
  4.2 GiB). The mixture M also fails C-E6b: class 0 −0.0375 against E4 and −0.0287 against M random; U2 +0.0062 and
  +0.0041 (bar 0.02); A-test M 2.1197 against E4 2.1057. G6b fails; E6b is closed with no further amendment. The
  reading: word classes help new text (U1, U2, unseen books against the random control) and hurt contexts already
  known.
- **Next ledger steps:** the E6b report to the planner (G6b). Phase 3 (E8: loop, feedback port, capped law, with the
  Phase 3 note in DESIGN §12) needs the owner's go, since G6 and G6b failed.
- **Open owner items:**
  - (1) The throughput-benchmark change (commit kept local in the engineer's session; local `85fa6c3`, based on
    `d894a5b`, to be rebased on resume) needs the owner's confirmation in the engineer's own session; until then
    the receipt is FAIL on that wall-clock test only.
  - (2) Whether Phase 3 starts after G6's failure.
- **Sessions:** planner `[private session reference removed]`, engineer `[private session reference removed]`, helper
  `[private session reference removed]`.


**Read this first, every session.** It is the single source of truth for this branch: what VAM is, who
does what, and what is done. Only the manager edits it, and it is updated on every push.

- Work branch: `claude/vam-nfa-continue`, started from `6fbfdc7` (the head of `claude/project-thread-20ve0y`).
  Draft PR #3, for visibility only.
- **Scope.** Every session here commits and pushes only to `claude/vam-nfa-continue` and comments only on PR
  #3. Nobody pushes to, merges into, or comments on `claude/project-thread-20ve0y`, PR #2, `main` or any other
  branch.
- **For the first account (owner of `claude/project-thread-20ve0y`):** this file is optional reference. This
  branch is a separate continuation, and its manager session keeps it on the specification. You may reuse
  a result here that is verified (tests, receipt) instead of repeating it. Do not merge it unless the
  owner says so.

## What VAM is (read before any work)

- VAM is standalone. It is built only from the owner's documents in `spec/`:
  - F, `VAM_Framework_Foundational_Principles.md`;
  - T, `VAM_Coupled_Theory_and_Proofs.md`;
  - N, `VAM_Nested_Fractal_Architecture.md`.

  `DESIGN.md` maps them to code, and `experiments/PROTOCOL.md` fixes every experiment before it runs.
- SERA and VAM are different projects with the same end goal (the owner's). VAM is treated
  separately: it is built, tested and judged only from `spec/`. Nothing from SERA, SERA-U or SERA-Field
  is used here: not `workspaces/`, not `sessions/`, not the raw archive. Their terms (SCORE, ledgers,
  kernels, credit) are not VAM terms.
- **No scoring.** The output comes from the machine's own output instrument, one propagator, "not an
  unrelated answer-scoring system" (T7.1). Path sums "are not an additional stochastic answer
  generator" (T10.1). "Do not convert the complex phases into unrelated positive path costs" (T14
  step 7). Replacing amplitudes "by softmax or shortest-path costs" was rejected (T15.3).
  - No session adds answer scores, path costs, beams, softmax selection or quality thresholds to the
    machine.
  - The experiments' bits per byte is a measurement taken outside the machine: each bit is charged the
    probability the machine gave it before seeing it (PROTOCOL, line 7).
- **The loop is the event loop.** It is F's master loop and T10.1's event map
  `R_new = Π E_τ J_τ B_b(R) / Tr(J_τ B_b(R))`, run as the 15 steps of T14: take the next bit at the
  sensory port, evolve, emit at the active port, re-enter the emitted bit, canalize, take the next bit.
  - It is built and tested: the T10 theorem tests, the Markov-property audit, and emission and re-entry
    after every external bit (audit finding 2). At zero mass the nested machine computes the full loop
    exactly.
  - Repetition in E2's greedy writing is an observation about the massless text machine with frozen
    weights. There, re-entry only shifts the 64-bit register: the re-entered bit cannot reach the
    context. Re-entered outputs are also not evidence (T10.3). It is reported, not patched.
- Every new element is labelled source statement, derived result or chosen realization (T1.1, DESIGN
  section 2).

## State at `6fbfdc7` (measured, 2026-10-19)

| Item | State |
|---|---|
| Phase 1: core mathematics, every exact result of T | done (receipt at `2eb4f70`: 267 tests, 26/26 headline results) |
| Phase 2: tensor networks, VAM-1, bit kernel, nested levels, MPO engine | done |
| Phase 3: E1, E1b, E2 | run, in `RESULTS.md` |
| Audit 1 | answered in `AUDIT.md` |
| N-L1 to N-L7 (`src/vam/nfa/*`, options in `nested.py`, `collapse.py`) | built and tested (`bfc1e1b..6fbfdc7`) |
| E3 script and protocol | written before any run |
| E3 run | **not run**: no `results/e3.json` and no `results/e3_smoke.json` |
| Audit of the N work | **not done** |
| `src/vam/nfa/engine.py` (promised in DESIGN section 10) | **missing** |
| Receipt and README | out of date (327 test functions, 267 in the receipt) |
| PG-19 corpus files | not in git; `data/pg19/MANIFEST.json` and `selection.json` (URLs, SHA-256) are |

E2's numbers: the best machine gets 2.271 bpb on A-test, against 2.003 for lzma. The pilot of E3's
options reaches 1.905 on 1 MiB of A-validation. Transfer was +0.002 bpb (none).

## Engineering notes

1. E3 is ready, but the N work has never been audited. Audit first, then the smoke run, then the full
   run, so a finding cannot waste a 3-5 h run.
2. Part 2 of E3 stops unless E2's start value `2.0316126330797104` reproduces to 1e-9. Use E2's
   environment: Python 3.13, numpy 2.5.3, scipy 1.18.1, numba 0.68.0, sympy 1.14.0. If the value still
   differs, the engineer and the advisor rule on it, and the manager writes the ruling into PROTOCOL as an
   amendment before the run.
3. `part1_sink_cost` runs untruncated MPOs at 5 cells (11 qubits) for 2,048 events. The smoke run gives
   seconds per part. A part projected above about 1 h gets a smaller stream, as a protocol amendment
   before the run.
4. `nfa/engine.py` is promised in DESIGN section 10 and missing. It is a thin composition, with no new
   mathematics.

## Roles (three sessions)

| Role | Does | Never does |
|---|---|---|
| Manager, Opus low | edits this file, `DESIGN.md`, `PROTOCOL.md` and `AUDIT.md`; assigns ledger tasks; runs the drift check on every push; runs the gates | write large code; read whole result JSONs |
| Engineer, Sonnet medium with Opus via `/advisor` | Audit 2, the fixes, the engine, checkpoint C2; consults the advisor on mathematical and design calls | edit this file or PROTOCOL; run the long experiments |
| Helper, Haiku medium | environment, data restore, smoke and full runs with `--resume`, result commits, `report.py`, receipt, ruff; owns `verify/RUNLOG.md` | change `src/`; interpret results |

The advisor's advice is not visible to other sessions, so every advisor-informed decision is written
into `audits/*.md` or a commit message.

## Drift check (the manager, on every push from the other sessions, before a ledger row is marked done)

1. **Scope:** commits only on `claude/vam-nfa-continue`, and only in the files the task's brief names.
2. **Specification:**
   - nothing that scores, costs or selects outputs is added to the machine (T7.1, T10.1, T14 step 7,
     T15.3);
   - the event loop (T10, T14) is unchanged unless an audit finding requires the change;
   - every new element is labelled source, derived or chosen (T1.1);
   - every protocol amendment is dated before its run;
   - every reported number comes from a saved file.
3. **Plan:** the work is the ledger task. Anything else becomes a proposal for the owner, not code.
4. **Cost:** nobody read the excluded files (below); advisor-informed decisions are written down.

A push that fails the check is reverted on this branch (`git revert`), with the reason in the ledger.

## Rules for every session (they keep cost down)

- Read only this file and your brief first.
- Never read `WORKSPACE_FILES.json`, `TRANSPORT.json`, `sessions/*.jsonl` or `workspaces/**`.
- Never read a whole `results/*.json` file: extract fields with python. Never read all of spec T: grep
  the section you need.
- Run tests per module; the full suite only at gates.
- Before work: `git pull --ff-only`. Before pushing: `git pull --rebase`, rebasing only your own unpushed
  commits.
- Commit messages list what changed and the numbers measured, ending with the session link.

## Ledger

| ID | Task | Owner | Status | Commit / notes |
|---|---|---|---|---|
| T0.1 | Branch, this file, stacked draft PR, notice on PR #2 | planning session | done | `5c90ddb` |
| T0.2 | This file revised after the owner's corrections (VAM as specified, three sessions, scope) | planning session | done | this commit |
| T1.1 | Environment with pinned versions; `pytest -q -m "not slow"`; ruff | helper | done | `8ee56c9`: Python 3.13.16, pinned versions; not-slow tests exit 0 (408 collected, 609 s); ruff exit 0 |
| T1.2 | Restore the PG-19 corpus from `selection.json`; MANIFEST byte-identical | helper | done | `8ee56c9`: 241 raw SHA-256 match; MANIFEST byte-identical; Corpus.load verifies 237 books. Download path is `<source>/<split>/<id>.txt`, not `url` (a catalogue page) |
| T1.3 | E3 smoke run; seconds per part to RUNLOG | helper | done | `cac24c0`: 200.3 s; Part 1 41.1 s, Part 2 16.7 s, Part 4 50.9 s, Part 3 unrecorded; E2 start NOT verified: smoke forces `reproduced` True and reads 2.0316126330797104 back from e2.json (check T1.6). Provenance commit `6bfde8f` is not in the branch (rebased away): T2.1 must run from a pushed commit. Smoke sizes (5-6 cells, 256 KiB) do not project the full run (8-12 cells) |
| T1.4 | Audit 2 of the N work | engineer | done | `a5118ed`, `audits/audit2.md`: A1 high, A2-A4 medium, A5-A7 low, A8-A9 info; answered in AUDIT.md, PROTOCOL E3 Amendment 1 |
| T1.5 | Fix confirmed Audit 2 findings, one test each | engineer | done | `38536a0`: CACHE_BOND = 90 + host_bounds in provenance; report.py "read" column and Part 4 instrument labels; A6 test over 5 option sets; docstrings only in src |
| T1.6 | Verify E2's start value (engineering note 2): one command, logged in RUNLOG | helper | done | `8639026`: printed 2.0316126330797104 True (4 min 11 s); engineering note 2 verified |
| T1.7 | `test_measured_throughput`: find the cost the N options add to the default path (options off) and remove it, or show it is host noise with repeated paired runs; the test and its threshold are not weakened without a written ruling | engineer | done | `362256d` (nested.py only): doorkeeper hash and admission checks moved behind `door_on`; paired interleaved runs: gap to 45d1209 from 11-14% to 2.7-3%; defaults bit-identical. Ruling: threshold 1.5e6 kept; the test may take the median of 3 runs (not weakened). The helper reruns the full suite at T5 |
| G1 | Gate: smoke green, fixes merged, full suite green, time per part acceptable | manager | passed for T2.1 (ruling) | 2026-10-20: full suite at `3a22339`: 640 passed, 1 failed (`test_measured_throughput`), ruff 0; engine tests 8 passed at `87e692f`. The failure is speed only (head median 1.26e6 bits/s, E2 code 1.52e6, same host, runs overlap; `36f91c7`); no test of correctness fails and no E3 number depends on speed, so T2.1 proceeds. The speed gap is T1.7, to close before T5 |
| T2.1 | E3 full run with `--resume` and checkpoint commits; `e3.json`, `RESULTS.md` | helper | done | `2a0d531` e3.json, `b4bc7e3` RESULTS.md, `7d3400d` RUNLOG provenance: Parts 1-2 at `1029e49`, Parts 3-4 per Amendment 3 (46 min, peak 2.6 GiB). Stops: Part 2 time (Amendment 2), Part 4 memory (T2.3, Amendment 3) |
| T2.3 | E3 Parts 3-4 memory fix (OOM at 13.9 GB in Part 4 writing; `9ec3a88`): release copies, save Part 4 on completion; results bit-identical | engineer | done | `f87184d`: gc after each prompt (cycle in collapse.paths kept copies alive); Part 4 saved per setting; peak 12.6 to 1.87 GiB (grid), about 3.3 GiB whole; old vs new outputs identical (test). Amendment 3 |
| T2.2 | `src/vam/nfa/engine.py` and its tests | engineer | done | `b10b461`: engine.py + 8 tests; read and bpb bit-identical to e2 make+read+frozen on 64 KiB (3 specs); write equals collapse.generate; constants checked equal to e2 |
| T3 | Checkpoint C2: read E3 | engineer | done | `ef2f633` `audits/c2.md`: R1 A-test 2.131 vs lzma 2.003 (not below); R3 -0.0143 vs shuffled -0.0167 (not beyond control); Part 1 criteria pass; four uncaught gaps; T4.2, T4.3 approved, T4.1 not (Amendment 4) |
| T4 | Only the fixes C2 names | engineer + helper | done | `7ca5ec8` tooling, `cf28245` run (158 s): T4.2 text Page rank 19-51 with floor 256 vs null 1 (Part 1: 0); T4.3 cache trigger at half the cap discards 5.7e-7 to 5.7e-6 vs 0.371 for the Page trigger |
| T5 | Close: receipt, README, DESIGN, AUDIT answers | all | done | full suite 691 passed, ruff 0; receipt `e4d230f` PASS (691/0, 26/26) at `9c97d83`, an earlier receipt at `1207400` FAILed only the wall-clock test (logged); README, DESIGN §10, AUDIT (Audit 2, C2), `audits/claims.md` (98 claims, 0 numeric mismatches); E4 Amendment 2 control `9c97d83` |
| T6.1 | E4 design: DESIGN section 11, "N above one equation, on text" (decision D11, a chosen realization) | engineer | done | `7e007a5` DESIGN §11, D11 |
| T6.2 | E4 protocol in PROTOCOL.md, written before any E4 run | engineer | done | `7e007a5` PROTOCOL E4 (Parts 0-6, R1-R6, budget about 4 h, 4.1 GiB) |
| G2 | Gate: the manager checks T6.1 and T6.2 against N and spec T; nothing is built before this passes | manager | passed | 2026-10-20: present-state registers only (T4.1, T11.5); no scoring, cost or selection; event loop and instrument unchanged; every element labelled; Part 0 bit-identity gate; controls and reading rules fixed before code. The three owner questions of §11.8 run as designed (both segmentation arms labelled) and go in the final report |
| T6.3 | Build E4 to the approved design, with tests | engineer | done | `4a27a0e`: hier.py, e4_hier.py, tests; upper levels off = NestedMachine bit for bit; compiled loop = reference on 8 option sets; E4 Amendment 1 |
| T6.4 | Run E4; commit results; regenerate RESULTS.md | helper | done | `22116ce` e4.json + RESULTS.md, `9454606`: 3 h 38 min, peak 5.0 GiB (estimate 4.2), slowest row 690 s. R1 A-test 2.1057 (E3 2.1309, lzma 2.0031); R2 pass; R3 pass; R4 fail; R5 fail; R6 transfer -0.0205 vs control -0.0238 |
| T7.1 | E5 draft (design and protocol only): `experiments/E5_DRAFT.md`, source table, question set with hash in `experiments/data/e5/`, owner decisions, risks | engineer | done | `eacf59e`: `experiments/E5_DRAFT.md`; `data/e5/` (questions.json SHA-256 2706322f..., verified; 79 recall, 61 inference); D1-D11, R1-R13 |
| G3 | Gate: the manager's drift check of T7.1 (spec rules, controls, pre-registration, no overlap) | manager | passed | 2026-10-21: draft and data only (no src, PROTOCOL.md untouched); spec rules restated; arms T and C; criteria and question hash fixed before any run; flags for the owner in the report |
| T7.2 | Owner approval of E5; only then is PROTOCOL.md edited | owner | done | approved "yes" via the planner, 2026-10-21; PROTOCOL E5 added before the run |
| T7.3 | Build E5 as the protocol names (`e5_*.py` run script, pre-run gates: set_state vs copy, storage hashes), tests | engineer | done | `a9107cd`: e5_taught.py, e5_restore.py, report.py E5 section, 16 tests; no src change; gates abort the run on failure |
| T7.4 | Run E5, arms T and C; commit results; regenerate RESULTS.md | helper | done | `9f25b82` e5.json, `afa06e8` RESULTS.md: restore 375/375 at `0bc6261`; 58 min, peak 1.85 GiB; gates pass. P1 pass (29/79), P2 pass (T 29 vs C 13, p 0.0004), P3 fail (11 vs 11). Recall gain shrinks to 9 vs 4 (p 0.11) where the answer is not the most frequent option |
| T8.1 | Phase 0: DESIGN §12 (verdicts V1-V13 with proofs, Phases 1-4, owner decisions, Not doing), advisor conclusions recorded | engineer | done | `bc2156e` DESIGN §12 (DESIGN.md only). Narrowed or failed as stated, with fixes: V2(a), V2(b) numbers, V4 (lattice condition, not connectivity), V5, V8, V10 (forms S/U with restriction R), V13 (last sentence fails; occupancy in the readiness rule); V7, V9, V11 qualified. Proposals P1-P7 and decisions 6-7 go to the owner at G5. Earlier note: plan "from memorizing to learning", owner-approved via the planner 2026-10-21. The explainer's earlier E6 request (error then correction) is superseded: it becomes E8(iii) and E9 |
| G5 | Gate: owner approval of §12; nothing built or run before it | owner | passed | 2026-10-21 via the planner: decisions 1-6 at defaults, 7 as a trial; P1-P7 ruled in DESIGN §12.6 |
| T9.x | Phase 1, E6: shared operator level O (`nfa/operator.py`, mixture column), tests, protocol, run | engineer + helper | done | E6 `c4b6fab`: C1 fails (class 0 +0.0336 vs E4, -0.0036 vs rank 1); diagnosis `66bd7b6`; Amendment 1 reruns `53e82b8`: R-a -0.0048, R-b -0.0000 vs rank 1: stopping rule, "spectral O at L <= 4 learns nothing beyond the byte unigram on this corpus" |
| G6 | Gate: E6 criterion (class-0 bytes at least 0.02 bpb better than E4 and the rank-1 arm) | manager | failed | 2026-10-22: C1 fails as registered and in both diagnostic reruns; Phase 2 does not start without the owner. Receipt FAIL only on the wall-clock test; the benchmark change `bf41002` awaits the owner |
| T10.x | Phase 2, E7: rotation families, synthetic sums | engineer + helper | done | protocol `300c282` + Amendment 1 `fdf569f`; build `a2a59a2`; run `0c73964` (20 min, 1.7 GiB): C-E7 pass (stage 1 0.948, stage 2 1.000; E4 0.000); linear O (V3 control) 0.000; random-relation control at chance |
| G7 | Gate: E7 (at least 90% held-out sums at 20% held out) | manager | passed | 2026-10-22 |
| T9b.x | E6b: word-class levels K1, K2, K4 (V2′ in §12, design check with the advisor), random-class control, protocol, run | engineer + helper | in progress | owner 2026-10-22; after E7 |
| G6b | Gate: E6b criteria on class-0 and unseen-2-word bytes against E4 and random classes | manager | failed (also with Amendment 1, `78b8ddc`) | 2026-10-22, `d6ce3de`: class 0 -0.0380 vs E4, -0.0174 vs random; U2 +0.0059, +0.0125 (bar 0.02). E4+K worse than E4 on A-test (2.1200 vs 2.1057) although train improves (1.7678 vs 1.7961): the reserved fallback amendment (E6b.7) is triggered |
| T-bench | Throughput tests as benchmarks, out of the verdict, thresholds unchanged (`bf41002`) | engineer | in progress | owner approved 2026-10-22 |
| T11.x | Phase 3, E8: loop, feedback port, capped law; (i) worked steps, (ii) exam feedback, (iii) correction by reading | engineer + helper | todo | after G7 |
| G8 | Gate: E8 | manager | todo | |
| T12.x | Phase 4, E9: teaching order with the readiness rule | engineer + helper | todo | after G8 |
| G9 | Gate: E9 | manager | todo | |

## Final state (2026-10-20, manager)

Every ledger row is done and drift-checked; G1 (by ruling) and G2 passed; the receipt passes.
- E3: A-test 2.131 bpb; E4: 2.106 (lzma 2.003). E4's gain over E3 at the same sink rate: +0.019 A-test,
  +0.023 unseen books; beats its capacity control (R3); R4 (order) and R5 (long matches) fail as registered.
- Open for the owner: segmentation (the text-rule arm won on A-validation; the modality-free arm has no test
  number), the sentence level, arity and moduli (D11, chosen not tuned); a second shuffle seed for R6;
  the Part 2 grid extension (not approved: tuning after the test split); N-L5 fails in E4 (W1, S0); why lzma
  still wins is not established; the wall-clock throughput test is noisy on a shared host.

## E5 (2026-10-21, manager)

Recall above the control holds as registered (P1, P2), but most of it sits on questions whose answer is the
option most frequent in the taught text; without those, 9 against 4 (p 0.11). Inference is at chance (P3
fails), as the protocol expected. E5 bounds "uses taught knowledge" to frequency-weighted recall. Stem-free reading (RESULTS, E5 table): T
recalls 29/79 with the question stem and 27/79 without it; C 13/79 with it and 28/79 without it. P2's gap
therefore comes from the stem pulling the control away from the right answers, not from T using the stem.

## Briefs

### T1.1 Helper: environment
From `vam/`:
```sh
python3.13 -m venv .venv && . .venv/bin/activate
pip install numpy==2.5.3 scipy==1.18.1 numba==0.68.0 sympy==1.14.0 pytest hypothesis ruff
pip install --no-deps -e .
python -m pytest -q -m "not slow"
ruff check src tests
```
Pass: both exit 0. Append the versions, test counts and seconds to `verify/RUNLOG.md`.

### T1.2 Helper: restore the corpus
- Write `experiments/fetch_selection.py`, standard library only:
  - for every book in `experiments/data/pg19/selection.json`, download its `url`;
  - check the SHA-256 of the raw bytes against the recorded one, and stop on any mismatch;
  - write the file where `build_corpus.py` expects it (read `selection.json` and `build_corpus.py` for the
    layout).
- Then run `python experiments/build_corpus.py <download dir> experiments/data/pg19`.
- Pass: `git diff --exit-code experiments/data/pg19/MANIFEST.json`, and
  `python -c "from vam.corpus import Corpus; Corpus.load('experiments/data/pg19/MANIFEST.json')"` (it
  verifies every book's hash).
- Commit the script only; the data stays out of git.

### T1.3 Helper: smoke run
```sh
python experiments/e3_nfa.py --manifest experiments/data/pg19/MANIFEST.json --smoke
```
Pass: it completes and writes `results/e3_smoke.json`. Commit that file. Log each part's `seconds`
(extract them with python) in RUNLOG.

### T1.4 Engineer: Audit 2 (read-only on code; writes `audits/audit2.md`)
- Scope: `git diff 45d1209..6fbfdc7 -- vam/` against spec N, the relevant sections of T, DESIGN section 10
  and PROTOCOL E3.
- Checks:
  - `coherent_mix`: its bounds; isometric nesting for every `c` (T11.4); and that it is an explicit
    enlarged state, not a mixture presented as interference (T13.1).
  - The doorkeeper and governor semantics: admission, protection, the sink `Re lambda = -2`.
  - N-L7 against T7.1, T14 step 7 and T15.3: is the action `S = -ln p` the machine's own
    observation-generator integral (`nfa/flow.py`)? Which Part 4 setting is the specification's own
    output instrument (`J_tau = O_tau`: hbar = 1, euclidean, sample)?
  - The Page edge on text (split-half), and the reading rule of the area law with four chain sizes.
  - E3's runtime and reproduction risks (engineering notes 2 and 3).
- Each finding: severity, evidence (a file and line, or a script and its output), and the T, F or N text
  it is judged against. Fixes go only as far as a finding needs.

### T1.5 Engineer: Audit 2 fixes
One fix and one test per confirmed finding, in the files the finding names; run the module's tests and
ruff. The manager answers every finding in `AUDIT.md` (Audit 2) and writes any PROTOCOL amendment,
dated before the run.

### T1.6 Helper: verify E2's start value
From `vam/experiments`, with the T1.1 venv:
```sh
python -c "from e2_curriculum import make, read, frozen, streams; from e3_nfa import START; from pathlib import Path; S = streams(Path('data/pg19/MANIFEST.json')); m = make(START); read(m, S['A_train'], learning=True); v = frozen(m, S['A_validation']); print(repr(v), abs(v - 2.0316126330797104) < 1e-9)"
```
Log the printed line and the seconds in RUNLOG. If it prints False, stop: the engineer and the advisor
rule on it, and the manager writes a dated PROTOCOL amendment before the full run.

### T2.1 Helper: E3 full run
```sh
python experiments/e3_nfa.py --manifest experiments/data/pg19/MANIFEST.json --resume
```
- Run it in the background. After each part ends:
  `git add -f experiments/results/e3_partial.json && git commit && git push`. A reclaimed container then
  loses one part at most; restart with the same command.
- At the end: commit `results/e3.json`, run `python experiments/report.py` and commit `RESULTS.md`, then
  `git rm --cached experiments/results/e3_partial.json`.
- If any part runs past about 1 h without finishing, stop the run and report. The engineer, with the
  advisor, proposes a smaller setting for that part only; the manager writes it as a dated PROTOCOL
  amendment; the run resumes with `--resume`.
- Pass: the provenance commit in `e3.json` is the run's start commit.

### T2.2 Engineer: the composed engine
- `src/vam/nfa/engine.py`:
  - `Engine(spec)` builds the machine through the same path as `e2_curriculum.make`. If that path must
    move into `src/` to be shared, move it with identical behaviour.
  - `read(data, learning=True) -> bpb`; `bpb(data)` (frozen, on a copy).
  - `write(prompt, n_bytes, ...)` over `collapse.generate`, defaulting to the machine's own instrument
    (hbar = 1, euclidean, sample).
  - `layers()` returns the bounds and caps of N-L1 and N-L3 and the N options in force.
- `tests/test_nfa_engine.py`: `read` and `bpb` equal `make(spec)` + `read` + `frozen` on 64 KiB; `write`
  equals `collapse.generate` with the same arguments and seed.
- Do not touch `e3_nfa.py` while T2.1 runs.

### T3 Engineer: checkpoint C2 (writes `audits/c2.md`; items fixed now, before E3 results exist)
- R1: the chosen configuration's A-test bpb against lzma's 2.003, reported as the headline either way.
- R3: transfer (`A_then_B - B_only`) with the governor, beside the shuffled control, reported.
- Every Part 1 criterion that fails: what the layer does instead, and the fix that makes it do what N
  says, within T.
- Writing: Part 4 reported verbatim, as the protocol says, with the specification's own instrument
  labelled as such.

### T4 Only the fixes C2 names
Each fix gets a dated PROTOCOL amendment before any rerun.

### T5 Close
- Helper: `python verify/run_all.py` (writes `verify/RECEIPT.json`) and ruff.
- Manager: README status, DESIGN section 10, AUDIT.md answers, the final ledger and the next steps.

### T6.1-T6.2 Engineer: the E4 design and protocol (no code until G2)
- Map each of N-L1 to N-L7 to one mechanism on text, above the 64-bit equation. Reuse what exists before
  writing anything new:
  - nfa/flow.py: an event as an equation, with trajectory() and trajectory_basis();
  - nfa/page.py: Page sizing of a basis;
  - coherent_mix in nested.py: a density over hypotheses;
  - tensornet.py and vam1_mpo.py: tensors and MPOs;
  - nfa/governor.py and the nested governor;
  - nfa/bounds.py: the register and cache caps.
- Constraints, all from spec:
  - present state only, no transcript (F Part 1, T4.1);
  - learning forward only (T12);
  - new distinctions by isometric nesting (T11.4);
  - storage fixed and declared (T11.5);
  - no scoring, path costs or selection rules; output through the machine's own instrument (T7.1, T10.1,
    T14 step 7, T15.3);
  - the event loop unchanged (T10, T14);
  - every element labelled source, derived or chosen (T1.1). T15.4: the coupling law between layers is a
    declared choice.
- The protocol fixes before any run:
  - what each layer must be shown to do;
  - bits per byte against the E3 machine, on the same splits;
  - the controls;
  - the time budget.
- Use the Opus advisor on the mathematics, and write its conclusions into DESIGN section 11.

## Session openers (paste into a new session; each names the branch)

- **Manager (Opus low):** "Repo sera-research-raw-20261005, branch `claude/vam-nfa-continue`; push only there.
  Read `vam/HANDOFF.md` only. You are the manager. Your main job is the drift check on every push. Then
  assign the next ledger task, run the gates, and update HANDOFF on every push."
- **Engineer (Sonnet medium):** "Run `/advisor` with Opus first. Branch `claude/vam-nfa-continue`; push only
  there. Read `vam/HANDOFF.md` ('What VAM is' first) and brief <T…>. Work only in the files it names; run
  the module's tests and ruff; write advisor-informed decisions into the repo; commit and push."
- **Helper (Haiku medium):** "Branch `claude/vam-nfa-continue`; push only there. Read `vam/HANDOFF.md` and brief
  <T…>. Follow its commands exactly; log results in `vam/verify/RUNLOG.md`; commit and push."

## Decided after E3: E4, N's layers above one equation, on text

On text, the machine realizes only N-L1: all six nested levels sit inside one 64-bit register (DESIGN §10, L1
row: "one context word per level"). N and F Part 9/10 put structure above one equation: bits, then words, then
sentences as equations of at most one register each, then vectors spanning the equations (N-L2), then densities
over the vectors (N-L3), then tensors of densities with blankets between them (N-L4/L5), under the global
governor (N-L6), ending in the collapse (N-L7). E4 realizes that on text. The owner's instruction with N decided
it (decision delegated by the owner to the planning session, 2026-10-20).

## Unknowns (stated, not guessed)

- How the Pro plan meters each model; the role split assumes the API price ratios (Opus 2x Sonnet, Haiku
  about 1/20 of Sonnet).
- Whether a cloud container survives a multi-hour run while idle (covered by the checkpoint commits).
- Whether E2's start value reproduces exactly in a fresh container (pinned versions; otherwise a ruling
  before the run).
- The real runtime of each E3 part; the smoke run measures it.
