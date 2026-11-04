# Publication validation, 2026-11-04

The root continuation snapshot passed a fresh quick verification on Linux:

- 545 selected tests passed; no failures.
- 225 tests marked slow were excluded. This is not a full-suite pass.
- All 26 mathematical headline checks passed.
- Runtime: 536.34 seconds.
- Ruff check of `src` and `tests`: passed.
- Editable installation and wheel build: passed.
- Exact source Git blob hashes, published provenance hashes and the fixed E5
  question-set SHA-256 were checked successfully.

Command: `python verify/run_all.py --quick --out ../validation_receipt.json`.
The resulting receipt is preserved as
[`verify/PUBLICATION_RECEIPT.json`](verify/PUBLICATION_RECEIPT.json).

Environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, SymPy 1.14.0,
Numba 0.68.1, pytest 9.2.0, Hypothesis 6.170.14 and Ruff 0.17.3.
The receipt's commit field is null because validation used the exported source
copy before the new repository's initial commit. Its exact source revision is
`b350cf647a39b094e6b179a49512f7628e562ac6`; publication file hashes and any
publication-only documentation changes are in `provenance/source-files.json`.
Executable source, tests, mathematical specifications and saved experimental
results match that revision byte for byte.

The original-branch archive was hash-checked but its tests were not rerun.
The full slow suite, corpus downloads and E1-E7 benchmark experiments were not
rerun for publication. No CI run is implied. The preserved historical FAIL
receipt remains at `verify/RECEIPT.json` and must not be confused with this quick
publication check.
