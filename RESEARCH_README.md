# Variational Action Machine (VAM)

A standalone build of the machine specified in the two documents in [`spec/`](spec/):

* `VAM_Coupled_Theory_and_Proofs.md` (cited **T**, sections 1-15)
* `VAM_Framework_Foundational_Principles.md` (cited **F**, Parts 1-10 and the master loop)

The documents are the specification. Nothing from earlier projects is used. [`DESIGN.md`](DESIGN.md) is
the full build plan: architecture, the map from every section of the documents to code and tests,
the reference machine (VAM-1), the verification plan and the decision log for the few places where the
documents leave a choice open.

## In simple words

The machine keeps one thing: its present state, a density matrix that holds every possibility it is
still considering, plus a few slow numbers that act like its habits. Each incoming bit nudges that
state and teaches the habits a little. The machine then says one output bit at its boundary (its own
guess of what comes next), feeds that bit straight back into itself through a second port, and waits
for the next bit. A tidying step that removes distinctions that can never matter again is proved safe;
at zero mass the fast engines keep only what that step keeps. It never keeps a transcript. Every rule it
follows comes from the documents, and every number the documents calculate is recomputed here and
checked automatically.

## Run it

```bash
pip install -e ".[test]"
python verify/run_all.py          # all tests + the headline results, writes verify/RECEIPT.json
python -m pytest                  # tests only
ruff check src tests              # lint (the standard is declared in pyproject.toml)
```

`verify/RECEIPT.json` records the environment, the SHA-256 of both specification documents, every
test outcome, and each headline result of T15.2 with its numerical error.

## Layout

| Path | What it holds |
|---|---|
| `src/vam/linalg.py`, `space.py`, `density.py` | Linear algebra, named tensor-product spaces, density checks and conditioning |
| `src/vam/channels.py`, `lindblad.py` | Kraus maps, instruments, collision channels, Lindblad generators, dynamical locks |
| `src/vam/dirac.py`, `hamiltonian.py`, `geometry.py` | Dirac Hamiltonians (Wilson, staggered chain, split-step walk), joint vector-weight Hamiltonians, density geometry |
| `src/vam/boundary.py`, `gaussian_blanket.py` | Generator-level boundaries, statistical blankets, the exact spiral blanket |
| `src/vam/paths.py`, `semiclassical.py`, `readout.py` | Path sums, influence functionals, stationary phase, Newton from Dirac, nested readout |
| `src/vam/machine.py` | The present-state event loop and macro/micro generation |
| `src/vam/canalization.py`, `plasticity.py`, `constraints.py` | Canalization and retention, forward plasticity, hard constraints |
| `src/vam/bits.py` | The universal bit substrate (bytes to bits, most significant first) |
| `src/vam/realizations/` | The worked examples of T as runnable machines, VAM-1 (`vam1.py`) and VAM-1 on tensor networks (`vam1_tn.py`, a purification; `vam1_mpo.py`, the density as an MPO) |
| `src/vam/tensornet.py` | Matrix-product and locally purified densities, local channels, truncation |
| `src/vam/bitkernel.py`, `nested.py` | VAM-1 at zero mass on packed bits, with one window or nested levels |
| `src/vam/sources.py`, `baselines.py`, `corpus.py` | Known test sources with exact references, the KT baseline, real-text corpora |
| `experiments/` | Pre-registered protocol, experiment scripts and saved results |
| `RESULTS.md` | The experiment report, generated from the saved results by `experiments/report.py` |
| `tests/` | One file per section of T (plus F-Part 7) and one per module; every exact value is asserted |
| `verify/run_all.py` | One-command verification and receipt |
| `AUDIT.md` | Independent audit findings and the response to each |

## Status

- Phase 1 (core mathematics and every exact result in T2-T13, F-Parts 1-7): complete and verified.
- Phase 2 (tensor networks, VAM-1, bit kernel, nested levels): complete; the engines agree to 3e-16.
  The purified tensor-network engine is exact when untruncated, but with mass its truncation is
  not faithful. The MPO-density engine is faithful when the chain is fully einselected (1e-5 at
  10 cells, bond 64) and approximate with context einselection, where the chain stays coherent
  (about 1e-3 at 12 cells, bond 64); DESIGN section 5 has the measurements.
- Phase 3: experiments E1 and E1b on known sources and E2 on real books (92 textbooks, then 145
  other books from PG-19) have run (`RESULTS.md`).
- Phase 4: the first independent audit and the response to every finding are in `AUDIT.md`.
- Continuation (2026-10-20, branch `claude/vam-nfa-continue`): the nested fractal design (N) is built, audited
  (Audit 2, answered in `AUDIT.md`), composed (`src/vam/nfa/engine.py`) and measured. E3 (N on the 64-bit
  text machine) reads A-test at 2.131 bpb; E4 (N above one equation, DESIGN section 11) at 2.106, against
  lzma's 2.003. E4's chosen machine also uses the text-rule segmentation (a tokenizer arm), no sinks and upper
  verify 4; the modality-free arm stopped at 1.8605 on A-validation and has no test number. Against its
  equal-capacity control E4 gains (R3 passes). Order helps on A-test but not reliably on unseen books (R4
  fails as registered). The 16- and 32-byte classes hold 40% of the gain (R5 fails as registered); why lzma
  still wins is not established. N-L5 fails in E4 (W1 and S0 occupancy above 1%). R6's shuffled control has
  one seed. Checkpoint C2 is `audits/c2.md`, the claims check `audits/claims.md`; every number is in `RESULTS.md`;
  E5 (2026-10-21; a taught stream of a dictionary, prose, mathematics, physics and primary works, against a
  control stream): recall above the control passes as registered (29 against 13 of 79), but on the 56 questions
  whose answer is not the most frequent option in the taught text it is 9 against 4 (p 0.11); inference is at
  chance (11 against 11 of 61).
  the ledger is [`HANDOFF.md`](HANDOFF.md).
