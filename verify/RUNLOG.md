# Helper run log

## T1.1 Environment (helper, Haiku medium)

- Python 3.13.16 (`python3.13 -m venv .venv`), `--no-deps -e .` installed.
- numpy 2.5.3, scipy 1.18.1, numba 0.68.0, sympy 1.14.0, pytest 9.1.1, hypothesis 6.170.5, ruff 0.17.1.
- `python -m pytest -q -m "not slow"`: exit 0, 408 tests collected, 609 s wall time (run in the background).
  The pass count line was not captured; the exit code is the pass criterion.
- `ruff check src tests`: exit 0, "All checks passed!".

## T1.2 Corpus restore (helper, Haiku medium)

- `experiments/fetch_selection.py` downloads each book from `<source>/<pg19_split>/<id>.txt` (the `url` field
  is a catalogue page, so it is not fetched) and checks the raw SHA-256 of all 241 books: all match.
- `python experiments/build_corpus.py <download dir> experiments/data/pg19`: exit 0; stage A 92 books
  (42.1 MiB, 9 held out), stage B 145 books (62.8 MiB, 14 held out).
- `git diff --exit-code experiments/data/pg19/MANIFEST.json`: exit 0 (byte-identical).
- `Corpus.load('experiments/data/pg19/MANIFEST.json')`: exit 0, 237 books, every hash verified.

## T1.3 E3 smoke run (helper, Haiku medium)

- `python experiments/e3_nfa.py --manifest experiments/data/pg19/MANIFEST.json --smoke`: exit 0, 201 s wall
  (`real 3m21s`), writes `experiments/results/e3_smoke.json` (total `seconds` 200.3).
- Seconds per part, summed from the recorded `seconds` fields of `e3_smoke.json` (top level of each part;
  Part 1 counts each top-level item once, not its nested engines):
  - Part 1 (nested layers N-L1 to N-L7): 41.1 s
  - Part 2 (selection, 5 rows): 16.7 s
  - Part 3 (arms, transfer, generation): not recorded in the JSON; the remainder is 91.6 s (200.3 - 41.1 - 16.7 - 50.9)
    and includes Part 3 and setup.
  - Part 4 (16 writing settings): 50.9 s
- Part 2 `e2_start_bpb` = 2.0316126330797104 is the value saved in e2.json, not recomputed; `reproduced` is forced True in smoke mode (e3_nfa.py line 424). The smoke run's own start value is on 256 KiB, not comparable. Engineering note 2 is NOT verified by the smoke run. (Corrected by the manager, 2026-10-20.)

## T1.6 E2 start value (helper, Haiku medium)

- Command from `vam/experiments` with the T1.1 venv (the T1.6 brief command), on `make(START)` after
  `read(S['A_train'], learning=True)`, then `frozen(S['A_validation'])`.
- Printed: `2.0316126330797104 True`; the value equals 2.0316126330797104 and `abs(v - 2.0316126330797104) < 1e-9` is True.
- Wall time 4 min 11 s (`real 4m11.316s`).
- Engineering note 2 is verified in this environment, unlike the smoke run, which forces `reproduced` True.

## G1 gate checks (helper, Haiku medium), head 38536a0 (T1.5)

- `python -m pytest -q` (full suite, slow included): exit 1, 2954 s wall. 641 collected; 1 failed, 640 passed
  (derived from the collection count and the short summary; the pytest count line was not in the log).
  - FAILED `tests/test_nested.py::test_measured_throughput`: `assert 1256892.17 > 1500000.0` in the first run;
    rerun alone: `assert 1385772.01 > 1500000.0`. It fails on both runs, so it is not a single flake.
    The test measures wall-clock throughput with a hard threshold; the machine load average was about 3.
    Not changed here: the threshold and the code are the engineer's and the manager's to rule on.
- `ruff check src tests`: exit 0, "All checks passed!".
- G1 status: NOT PASSED on the test suite (one failing timing test). T2.1 not started.

## G1 addendum: engine tests at b10b461 (helper, Haiku medium)

- The G1 full run started at 38536a0, before b10b461 (T2.2). Rerun: `python -m pytest -q tests/test_nfa_engine.py`
  on head f035446: exit 0, 8 tests passed (dot count; 36 s wall).
- The full-suite result above still stands for 38536a0; the full suite at head f035446 has not been rerun.

## Throughput measurement for the G1 ruling (helper, Haiku medium)

- Host CPU: `Intel(R) Xeon(R) Processor @ 2.80GHz` (lscpu). Same host and venv (T1.1) for all runs.
- Body of `test_measured_throughput`: NestedMachine(CONS, depths (8,16,24,32,48,64), table_bits=22), 1000 warm-up
  bits, then timed `run` of 4,000,000 bits from `np.random.default_rng(0)`. Script kept outside the repo.
- Current head (src at the repo's working tree, `f035446`), bits/s, three runs one at a time:
  1,245,524.0; 1,261,933.5; 1,414,374.8
- Old code (src tree at `45d1209`, scratch worktree, PYTHONPATH on its `src`), bits/s, three runs:
  1,517,521.3; 1,859,322.6; 1,413,039.7
- No file under src/ or tests/ was changed. The scratch worktree was removed afterwards.
- Note: the test's own seed is 20261018; this measurement used seed 0 as asked.

## T2.1 E3 full run, Amendment 2 resume (helper, Haiku medium)

- Start commit 1029e49. The working tree's `src/` had moved on (T1.7, 362256d and 44cd999) through my pulls,
  so the resumed run used a scratch worktree at 1029e49 (`src/` and `experiments/` from that commit; `data/pg19`
  linked to the verified corpus). The partial file was copied back and committed in this repo.
- Part 1: complete (17c1cd2). Part 2: complete under Amendment 2 (6 min per configuration). 32 configurations
  saved in total (12 before the stop, 20 in this resume); the slowest configuration took 303.9 s.
  The commit message of e95a62c says 20 configurations and 293.6 s; the saved file shows 32 and 303.9 s.
- Part 3 and Part 4: the process was killed by the kernel's OOM killer (cgroup memory limit, python RSS
  about 13.9 GB) after 14 writing settings of Part 4 had printed, at about 11:44 (started about 11:20). No
  `part3` key was saved, so Part 3 did not complete. No traceback. `e3.json` was not written.
- Not done: `e3.json`, `report.py`, `RESULTS.md`. The run needs a memory ruling before it can resume.

## T2.1 Parts 3-4, Amendment 3 (helper, Haiku medium)

- Provenance: Parts 1-2 from 1029e49. Parts 3-4 with the `src/` and `e2_curriculum.py` of 1029e49 and the
  `e3_nfa.py` of f87184d (Amendment 3). The run used a scratch worktree at 1029e49 with that one file replaced.
  `e3.json` provenance records `1029e49+uncommitted` (the uncommitted change is the `e3_nfa.py` swap).
- Resumed 12:04:04Z with `--resume` from the partial file (Parts 1-2 present). Finished 12:50:47Z.
  `seconds` in `e3.json`: 2778.4 (46 min, under the 1 h rule). All 32 Part 2 configurations were reused.
- Peak memory: sampled RSS of the python process every 15 s, maximum 2,711,056 KiB (about 2.6 GiB). Amendment 3
  quoted about 3.3 GiB; the 13.9 GiB of the killed run is not comparable (that run did not release copies).
- `report.py` (1029e49 version, run in the worktree): `RESULTS.md` written, 794 lines.
- The partial file `experiments/results/e3_partial.json` was untracked after this commit.

## T4 run (helper, Haiku medium), scripts at 7ca5ec8, head ca39c85

- `python experiments/e3_t4.py --manifest experiments/data/pg19/MANIFEST.json --smoke`: exit 0, 14.7 s wall
  (smoke file `results/e3_t4_smoke.json`, not committed).
- `python experiments/e3_t4.py --manifest experiments/data/pg19/MANIFEST.json`: exit 0, 2 min 38.5 s wall;
  `seconds` in `results/e3_t4.json`: 157.7.
- `python experiments/report.py`: exit 0; `RESULTS.md` regenerated.

## T6.4 E4 run (helper, Haiku medium), head e4c8671 (E4 Amendment 1)

- Smoke: `python experiments/e4_hier.py --manifest experiments/data/pg19/MANIFEST.json --smoke`: exit 0,
  7 min 2 s wall. Smoke file not committed.
- Full: same manifest, no `--seed2`, in background, one job at a time. Started 14:48Z, exit 0 at 18:26Z.
  `seconds` in `results/e4.json`: 13072.2 (3 h 38 min, under the expected 4.4 h).
- Memory: RSS sampled every 15 s. Peak 5,236,872 KiB (about 5.0 GiB), under the 10 GiB stop rule.
- Part 2 rows (9 configurations): slowest 690.3 s, under the 900 s stop rule.
- Partial file committed after parts 0-1 (e4e649d) and after part 2 (5ff49a7); the final partial was untracked
  after the results commit.
- `report.py`: exit 0; `RESULTS.md` regenerated.

## T5 close checks (helper, Haiku medium), head 1207400

- `python -m pytest -q` (full suite, slow included): exit 0, 3281 s wall. 691 collected; all passed (the -q
  run prints no count line; the exit code is the criterion).
- `ruff check src tests`: exit 0, 1 s.
- `python verify/run_all.py`: exit 1, 3216 s wall. Verdict FAIL: "690 tests passed, 1 failed, 26/26 headline
  results". The failure is `tests/test_nested.py::test_measured_throughput`: `assert 1421378.86 > 1500000.0`
  (median of three runs after the T1.7 warm-up). The same test passed in the pytest step above (exit 0), so
  the throughput threshold is flaky on this host under the verify run's load. Not fixed here, per the brief.
- `verify/RECEIPT.json` written (verdict FAIL, commit 1207400).

## E4-T run (helper, Haiku medium), head 20d2a8d

- Smoke: `python experiments/e4_t.py --manifest experiments/data/pg19/MANIFEST.json --smoke`: exit 0, smoke file
  not committed (moved out of the repo).
- Full: same manifest: exit 0, 7 min 28.8 s wall. `seconds` in `results/e4_t.json`: 447.9.
- `python experiments/report.py`: exit 0; `RESULTS.md` regenerated.
- The T5 full suite ran at 1207400, before 20d2a8d, so `python -m pytest -q tests/test_e4.py` was rerun at
  head 20d2a8d: exit 0, 6 tests collected and passed, 5 min 46.6 s wall.

## Verify receipts after the e4_t control (helper, Haiku medium)

- e4_t control: done first (9c97d83 at 20d2a8d; smoke exit 0, full exit 0, report exit 0).
- Receipt 1 (T5, head 1207400, `a525097`): FAIL. 690 passed, 1 failed (`test_measured_throughput`, 1,421,379 < 1.5e6);
  3216 s.
- Receipt 2 (rerun at head 9c97d83): PASS. 691 passed, 0 failed, 26/26 headline results; 3104.6 s (`seconds` in
  the receipt); `runall_exit=0`.
- The rerun was first started in the foreground-wrapped background shell, which the tool killed at its 30-minute
  limit before the run finished; no receipt came from that attempt. It was restarted once, detached, and that
  restart is the rerun reported above.

## T7.4 E5 run (helper, Haiku medium), head 26e8cf6: STOPPED at step 1

- `python experiments/e5_restore.py --data <scratchpad>/e5data`: exit 1, 55.6 s. "375 files named, 375 to build";
  "rebuilt 366 files; 9 could not be reproduced": S3_33283, S3_37354, S3_38993, S3_71655, S3_75107, S3_78586,
  S3_79080, S3_41654, S4_47464. The brief requires 375 of 375, so the run was stopped there; steps 2 and 3 were
  not run. No E5 outputs were committed.

## T7.4 E5 restore investigation (helper, Haiku medium), read-only except the rerun

- Attempt 1 (`e5_restore.py --data <scratchpad>/e5data`, head 26e8cf6): exit 1, 366 rebuilt, 9 not reproduced.
- Probe, read-only (same URLs and cleaning as the script; nothing written to the data dir). Every candidate
  URL for each of the nine ebook ids, with HTTP status:
  - PG-19 bucket (train, validation, test) and Project Gutenberg `pg<id>.txt`: HTTP 404 for all nine.
  - Project Gutenberg `pg<id>-images.html`: HTTP 200 for all nine (no proxy refusal; curl rc 0).
  - The converted HTML text (script's own steps: html_to_text, START/END markers, strip, final newline) has
    the START/END markers, and its SHA-256 differs from the recorded one for all nine. Its size is larger than
    the recorded `bytes`, by 1.4%–2.2%:

    | file | recorded bytes | converted bytes |
    |---|---|---|
    | S3_33283 | 195,639 | 199,328 |
    | S3_37354 | 10,032 | 10,189 |
    | S3_38993 | 100,282 | 102,453 |
    | S3_71655 | 95,891 | 97,430 |
    | S3_75107 | 177,492 | 180,717 |
    | S3_78586 | 121,462 | 123,189 |
    | S3_79080 | 88,365 | 89,593 |
    | S3_41654 | 403,622 | 410,880 |
    | S4_47464 | 288,620 | 293,160 |

  So the fetch did not fail. The bytes arrived, and they differ from the recorded ones. The cause is not
  determined here (for example, a different HTML revision of the ebook or a different conversion). The script
  and `sources.json` were not changed.
- Attempt 2 (rerun, same directory, same head): exit 1, "375 files named, 9 to build; rebuilt 0 files; 9 could
  not be reproduced" (the same nine files).
- Consequence: 375 of 375 is not reached, so T7.4 steps 2 and 3 were not run, per the brief.

## T7.4 E5 run (helper, Haiku medium), head 0bc6261 (restore fix), then a9107cd scripts

- Step 1, `e5_restore.py --data <scratchpad>/e5data`: exit 0, 375 files named, 9 rebuilt, 0 not reproduced (after
  the fix in 0bc6261; the failing attempts before it are logged above).
- Step 2, `e5_taught.py --data <scratchpad>/e5data --smoke`: exit 0, 2 min 22.3 s wall. Smoke outputs not committed.
- Step 3, full run, detached, `e5_taught.py --data <scratchpad>/e5data`: exit 0, started 01:35Z, finished 02:34Z.
  `seconds` in `results/e5.json`: 3500.0. Arms: T and C; both under the 1 h rule. Partial committed after arm T
  (5b196af) and after arm C (47db9c7); the partial was untracked in 7b81489.
- Peak memory: RSS sampled every 15 s, maximum 1,942,272 KiB (about 1.85 GiB), under the 10 GiB rule.
- `report.py`: exit 0; `RESULTS.md` regenerated.
- Numbers are not read here (per the brief).

## E2 corpus check for E6 prep (helper, Haiku medium), head 8706e21

- `Corpus.load('experiments/data/pg19/MANIFEST.json')`: exit 0, "corpus ok" (no restore needed).
- `git diff --exit-code vam/experiments/data/pg19/MANIFEST.json` against the committed manifest: exit 0 (byte-identical).

## T E6 run (helper, Haiku medium), build 7a1e08e

- Smoke: `e6_operator.py --smoke`: exit 0, 2 min 55 s wall; smoke outputs not committed.
- Full run (detached, no --resume needed): exit 0, started 21:53Z, finished 22:46Z. `seconds` in `results/e6.json`: 3210.5.
  Arms E4, E4+O and E4+O at rank 1 (each saved in partial; partial committed after arms E4 and E4+O, then final).
  The third arm ran from about 22:29Z to 22:46Z, under the 45 min rule.
- Peak RSS (sampled every 15 s): 2,815,324 KiB (about 2.7 GiB), under the 10 GiB rule.
- `report.py`: exit 0; `RESULTS.md` regenerated.
- Numbers not interpreted.

## Final checks after E6, head f001d25 (helper, Haiku medium), run alone and in order

- `python -m pytest -q`: exit 1, 3745 s. Two timing failures, both `test_measured_throughput`:
  `tests/test_bitkernel.py`: 8,880,674 < 10,000,000 bits/s (threshold 1e7);
  `tests/test_nested.py`: 1,250,542 < 1,500,000 bits/s. The rest of the suite passed. Not fixed here.
- `ruff check src tests`: exit 0, 2 s.
- `python verify/run_all.py`: exit 1, 3752 s. Verdict FAIL: "722 tests passed, 1 failed, 26/26 headline results".
  The one failure: `tests/test_nested.py::test_measured_throughput`, 1,198,785 < 1.5e6. `test_bitkernel` passed in this run.
- `verify/RECEIPT.json` written at this head (verdict FAIL), committed as the receipt for this head.

## Verify rerun at c4b6fab (helper, Haiku medium), alone, once, detached

- `python verify/run_all.py`: exit 1, 3752 s wall in the log's own clock (the receipt's `seconds`: 3774.1). Verdict FAIL:
  "722 tests passed, 1 failed, 26/26 headline results". The one failure: `tests/test_nested.py::test_measured_throughput`,
  1,232,117 < 1.5e6 bits/s. No other test failed. No test changed.
- Receipt for this head (c4b6fab) committed, verdict FAIL.
- Receipts logged: (1) the receipt at head c4b6fab before this rerun: FAIL, 722 passed, 1 failed (`test_nested` throughput,
  1,198,785 < 1.5e6); (2) this rerun: FAIL, 722 passed, 1 failed (`test_nested` throughput, 1,232,117 < 1.5e6).
- No further reruns, per the manager's instruction.

## E6 Amendment 1 (helper, Haiku medium), build 891d790 (pulled at b94f7b8)

- `e6_operator.py --arm O --past-bits 14` (results/e6_ra.json): exit 0; `seconds` 1452.8 (about 24 min wall, 01:56:36Z
  to 02:20:50Z); peak RSS 4,359,840 KiB (about 4.2 GiB), sampled every 15 s. Under the 10 GiB and 45 min rules.
- `e6_operator.py --arm O --k-force 24` (results/e6_rb.json): exit 0; `seconds` 1141.7 (about 19 min wall,
  02:20:50Z to 02:39:53Z); peak RSS 2,885,240 KiB (about 2.8 GiB). Under the rules.
- `report.py`: exit 0; `RESULTS.md` regenerated.
- The partial files are gitignored and were not committed.
- Receipt: no `run_all.py` was run in this step. The receipt for the current head is the one committed in b94f7b8
  (FAIL, 722 passed, 1 failed: `test_nested.py::test_measured_throughput`).

## E7 run (helper, Haiku medium), build a2a59a2

- Smoke: `e7_families.py --smoke`: exit 0, 1 min 58 s; smoke outputs not committed (moved out of the repo).
- Full: `e7_families.py`: exit 0, started 04:27Z, finished 04:47Z. `seconds` in `results/e7.json`: 1196.4 (about 20 min).
  No unit passed 15 min (the run took about 20 min in total). Peak RSS (sampled every 15 s): 1,771,916 KiB (about 1.7 GiB).
- After0: `e7_families.py --slot after0 --stages 1`: exit 0, 7 min 32 s wall; `seconds` in `results/e7_after0.json`: 451.2.
  Its RSS was not sampled.
- `report.py`: exit 0; `RESULTS.md` regenerated.
- Numbers not interpreted.

## E6b run (helper, Haiku medium), build d250c8a (pulled at 43e5425)

- Corpus: `Corpus.load('experiments/data/pg19/MANIFEST.json')`: ok.
- Smoke: `e6b_classes.py --smoke`: exit 0, 1 min 45 s; smoke outputs not committed (moved out of the repo).
- Full: `e6b_classes.py`, detached, one job: exit 0, started 16:11Z, finished 16:50Z. `seconds` in `results/e6b.json`:
  2366.1 (about 39 min). No arm passed 45 min; no gate failed.
- Peak RSS (sampled every 15 s): 3,724,732 KiB (about 3.6 GiB), under the 10 GiB rule.
- `report.py`: exit 0; `RESULTS.md` regenerated.
- Numbers not interpreted.

## E6b Amendment 1 run (helper, Haiku medium), build ced8b93

- Corpus: `Corpus.load` ok. Smoke: `e6b_mix.py --smoke`: exit 0, 1 min 24 s; smoke outputs not committed.
- Full run: started 17:19Z. The worker process was restarted at about 17:57Z and the run stopped (no exit line).
  Resumed with `--resume` at 20:57Z; exit 0 at 21:43Z. `seconds` in `results/e6b_mix.json`: 2763.8 (run-time clock as
  reported by the script; the wall time spans the restart).
- Peak RSS (sampled every 15 s): 4,404,352 KiB (about 4.2 GiB) across both processes, under the 10 GiB rule.
- The stop at 17:57Z was the worker restart, not a rule; no arm passed 45 min in the resumed run.
- `report.py`: exit 0; `RESULTS.md` regenerated.
- Numbers not interpreted.
