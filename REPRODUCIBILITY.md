# Reproduction guide

## Core package and verification

Install in a fresh environment with `python -m pip install -e ".[test,fast]"`.
The runtime dependencies are NumPy, SciPy and SymPy. Numba supplies the compiled
fast engines; many experiment runs and throughput expectations assume it.
Python 3.11+ is declared. A system C compiler is not part of the package itself.

From the repository root:

```sh
python verify/run_all.py --quick --out verify/LOCAL_RECEIPT.json
python -m pytest
ruff check src tests
```

The first command skips tests marked `slow`; it is not the full suite. The
second runs all tests, including slow numerical grids. To compute a full receipt,
run `python verify/run_all.py --out verify/LOCAL_FULL_RECEIPT.json`. Throughput
tests use wall-clock thresholds and can fail on slower or shared hosts even when
numerical tests pass. Do not silently remove or weaken them to obtain a pass.

`verify/RECEIPT.json` and `verify/RUNLOG.md` are preserved historical evidence.
The receipt dated 2026-10-22 records 722 passed, one failed throughput test and
26/26 headline checks. It is tied to source commit
`c4b6fab44dd72a075d6bc9d4f4dd9a93855a2609`, not the final continuation head.
Later result files have their own provenance fields. Review those fields before
comparing outputs or describing a result as reproduced.

## Historical numerical environment

The recorded E2/E3 continuation environment was Python 3.13.16, NumPy 2.5.3,
SciPy 1.18.1, Numba 0.68.0 and SymPy 1.14.0. This information is preserved from
the research records; it is not a claim that every platform can install those
versions or that other versions reproduce the results exactly. The package's
broad minimum versions are installation constraints, not a numerical lockfile.

## Corpus restoration

Read THIRD_PARTY_NOTICES.md and the external source terms before downloading.
No full books are tracked. For the PG-19 selection used in E2 onward:

```sh
python experiments/fetch_selection.py experiments/data/pg19/selection.json /path/to/pg19-download
python experiments/build_corpus.py /path/to/pg19-download experiments/data/pg19
```

`fetch_selection.py` verifies every selected raw file against its recorded
SHA-256. `build_corpus.py` generates the cleaned corpus and manifest. Compare the
new manifest with the tracked one before treating it as the historical corpus.
If upstream content changed, stop on a hash mismatch rather than replacing the
recorded hash. Network access and enough disk space are required.

For E5's stage texts, which are also not tracked:

```sh
python experiments/e5_restore.py --data /path/to/e5-data
python experiments/e5_restore.py --data /path/to/e5-data --check
```

This downloader requires curl, uses the recorded derivation rules and checks
every restored file. The 140-question evaluation set and its fixed hash are
already included; they must remain unchanged for a comparable E5 run.

## Experiments and reports

Consult each script's `--help` and the preregistered protocol before running.
Full E2-E7 experiments can take hours and significant memory; smoke runs have
smaller settings and must never be reported as the full experiment. Resume files
and selected historical partial results are explicitly distinguished from final
result JSONs.

`experiments/report.py` reads saved results and regenerates RESULTS.md. It does
not perform new experiments. Preserve the original report before regeneration
and inspect any differences. The publication's existing results are historical
measurements, not newly executed benchmark results.

The `archive/original-branch/` snapshot is for comparison and provenance. Its
code, protocol and receipt belong together. Do not mix its implementation into
the root snapshot while claiming to reproduce either revision unchanged.
