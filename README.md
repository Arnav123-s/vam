# Variational Action Machine (VAM)

VAM is an experimental present-state learning machine built from three original
specification documents. This repository contains the mathematical framework,
Python implementations, tests, preregistered experiments, saved results, audits,
and research decisions. The project studies a density-state event loop with
input, evolution, output, re-entry, canalization and forward plasticity.

**Research status:** the implementation and experimental record are published
for inspection and reproduction. The text-generalization problem remains open.
Synthetic arithmetic results do not establish general reasoning, language
understanding, or a broadly capable autonomous learner.

## Start here

- [Foundational principles](spec/VAM_Framework_Foundational_Principles.md)
- [Coupled theory and proofs](spec/VAM_Coupled_Theory_and_Proofs.md)
- [Nested fractal architecture](spec/VAM_Nested_Fractal_Architecture.md)
- [Full design and decision log](DESIGN.md)
- [Experiment protocol and amendments](experiments/PROTOCOL.md)
- [Complete experiment report](RESULTS.md)
- [Audit findings and responses](AUDIT.md), plus [detailed audits](audits/)
- [Reproduction guide](REPRODUCIBILITY.md)
- [Publication scope and source provenance](PUBLICATION.md)
- [Original project overview](RESEARCH_README.md)

VAM is standalone. SERA, SERA-U and SERA-Field sources, datasets, transcripts,
environments and raw backup archives are not part of this repository.

## Install and verify

Python 3.11 or newer is declared by the package. A clean virtual environment is
recommended. NumPy, SciPy and SymPy are runtime dependencies; the optional fast
extra installs Numba. The test extra installs pytest, Hypothesis and Ruff.

```sh
python -m venv .venv
# Linux/macOS:
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test,fast]"
python verify/run_all.py --quick --out verify/LOCAL_RECEIPT.json
ruff check src tests
```

For the full mathematical test suite, omit `--quick`. Verification can take
substantial time; the full suite includes explicitly marked slow tests. The
quick mode skips those tests but still includes machine-dependent timing tests.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact scope and historical
environment versions. No training corpus is downloaded by the installation.

`verify/RECEIPT.json` is a historical receipt, **not a fresh verification of this
publication**. It records 722 passes and one nested-throughput failure on
2026-10-22, with all 26 mathematical headline checks passing. It predates some
later E6b/E7 work. Keep newly generated receipts under a distinct filename.

A separate [publication validation](VALIDATION.md) passed 545 quick tests and
26/26 headline checks on 2026-11-04. The 225 slow tests and full experiments were
not rerun. The fresh receipt is kept separately from the historical record.

## What the saved experiments show

All values below come from the preserved results and their accompanying report.
Lower bits per byte (bpb) is better. These runs were not repeated for publication.

| Experiment | Recorded outcome | Scope / limitation |
|---|---|---|
| E1 / E1b | Known-source and nested-source experiments | Controlled sources with exact references |
| E2 | Best machine 2.271 bpb on A-test | lzma baseline 2.003 bpb |
| E3 | 2.131 bpb on A-test | Nested-fractal text configuration; still behind lzma |
| E4 | 2.1057 bpb on A-test | Equal-capacity criterion passes; order and long-match criteria fail as registered; lzma 2.0031 |
| E5 | Recall 29/79 versus 13/79 control | Inference 11/61 versus 11/61; strong answer-frequency confounding |
| E6 | Shared-operator criterion fails | No improvement beyond the byte-unigram control under the tested settings |
| E6b | Word-class and fallback-mixture criteria fail | Mixture A-test 2.1197 versus E4 2.1057; some unseen-context gains but overall text generalization unresolved |
| E7 | Synthetic held-out sums: stage 1 0.948, stage 2 1.000 | Restricted synthetic setting and identifiable held-out sums; not general mathematical reasoning |

For E5, after excluding questions whose correct answer was the most frequent
option in the taught text, recall is 9 versus 4 (p = 0.11). The stem-free checks
also narrow the interpretation of the registered recall advantage. The full
report retains these checks and failed outcomes rather than only successes.

E8 (extended loop/feedback) and E9 (developmental teaching) remain planned work
in the preserved record. Publishing this snapshot does not claim they were
implemented or run, or that paused research has resumed.

## Repository map

| Path | Contents |
|---|---|
| `spec/` | The three source specifications |
| `src/vam/` | Density and channel mathematics, event loop, plasticity, tensor networks, nested engines and NFA components |
| `tests/` | Mathematical, behavioral and implementation tests |
| `experiments/` | Protocol, corpus restoration tools, experiment scripts, diagnostics and saved results |
| `experiments/data/` | Selection manifests, hashes and E5 question/evidence metadata; no full training books |
| `audits/` | Detailed audit, claim-check and diagnostic records |
| `verify/` | Verification runner, historical run log and receipt |
| `HANDOFF.md`, `PLANNER.md` | Historical decisions, gates, limitations and continuation plans |
| `archive/original-branch/` | Complete earlier VAM branch snapshot, including divergent audits and partial results |
| `provenance/` | Source-to-publication file hashes and revision identifiers |

The root package is the continuation snapshot. The original-branch archive is
historical, not a second supported installation or a merged alternate engine.

## License and external material

Original VAM code, specifications and documentation are offered under the
[Apache License 2.0](LICENSE). Third-party quotations, corpus-derived excerpts,
source metadata and dependencies retain their applicable rights and notices;
the VAM license does not relicense them. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Full PG-19 and Project Gutenberg books are not bundled. Restoration scripts
download the exact recorded sources and verify hashes. Rights can vary by
jurisdiction; consult the original source notices before downloading or reusing
external texts.
