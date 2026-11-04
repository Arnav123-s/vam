# E6b development check (E6b Amendment 1, option C): a short reading

Run from `experiments/e6b_classes.py --dev-check`; data in `experiments/results/e6b_dev.json`. **A-validation only: A-test
was not read.** Training: the train parts of the first 9 whole A books (2.69 MB, 3 compiles of the map at most); frozen
read on the validation parts of the same books (336 KB), the structure of A-test against A-train in small. Sets and window
classes are recomputed against that prefix.

| validation bits per byte | E4 | E4+K − E4 | fixed map − E4 | K after S1 − E4 |
|---|---|---|---|---|
| all (336 KB) | 1.9512 | **+0.0009** | +0.0029 | **+0.0196** |
| class 0 (28.9%) | 2.3435 | −0.0233 | −0.0268 | −0.0152 |
| class 8 (60.5%) | 1.8231 | +0.0053 | +0.0070 | +0.0254 |
| class 16 (10.3%) | 1.6145 | +0.0414 | +0.0609 | +0.0811 |
| class 32 (0.2%) | 1.4639 | +0.0569 | +0.0738 | +0.1171 |
| U1 (23.4%) | 1.9147 | −0.0246 | −0.0293 | −0.0244 |
| U2 (60.3%) | 1.8539 | −0.0159 | −0.0173 | −0.0135 |
| class 0 inside U2 (25.1%) | 2.2923 | −0.0290 | −0.0324 | −0.0267 |
| class 0 outside U2 (3.9%) | 2.6764 | +0.0135 | +0.0092 | +0.0596 |
| prequential read of the prefix | 2.0083 | −0.0406 | −0.0407 | −0.0378 |

## What it shows (F)

- **The staleness candidate is not supported.** Fixing the map after its first compile is no better than the map as
  built (+0.0029 against +0.0009 overall; classes 16 and 32 are worse, +0.061 and +0.074): old class ids in the K tables
  are not what costs bits. (Caveat: with at most 3 compiles in 2.7 MB the maps barely drift; this tests staleness weakly.)
- **Moving the K levels after S1 is worse, not better** (+0.0196 overall; +0.117 on class 32, +0.060 on class 0
  outside U2). It does not remove the loss at contexts the tables hold; it enlarges it. At the top the K sectors have the
  last word, with no W or S level after them to correct the readout. This is consistent with the K sectors' never-drained
  evidence pulling the readout where the exact levels are strong, and says that the E7 top-slot cure does not transfer
  (there the stage was right at the answer bytes; here it is partly wrong). Option B is rejected by this check.
- **The pattern of E6b reappears in small:** K helps where the word context is new (class 0 inside U2 −0.029, U1 −0.025,
  U2 −0.016, prequential −0.041) and hurts at long exact windows (class 16 +0.041, class 32 +0.057), and it is at
  break-even overall (+0.0009).

## Limits (U)

- **The big loss of E6b is not present at this scale.** At 31 MiB class 0 got worse by 0.038, and class 0 outside U2
  (8.6% of A-test bytes) by 0.098; here class 0 gets better (−0.023) and class 0 outside U2 is only 3.9% of the bytes,
  +0.0135. The 3 MiB prefix does not reproduce the full-scale loss, which probably needs the larger tables that 31 MiB
  builds. So the check refutes B (the loss does not move to zero at the top, it grows) and weakens staleness, but it
  does not locate the cause of the full-scale class-0 loss.
- Small and unreplicated: 9 books, 336 KB of validation, one seed, no by-book interval; differences under about 0.01
  should not be read.

## Recommendation (for the manager's ruling)

**A, the reserved fallback (the top mixture).** B is rejected by the check. Staleness is not supported. The remaining
candidate, dilution by the K sectors' evidence at contexts the exact levels hold, is what a mixture keyed by the deepest
contributing level can gate: in the data above K is better in the shallow-context classes (class 0, U1, U2) and worse
in the deep ones (16, 32), which is the split a per-deepest-level node can follow. At most 28 bits against the better
of E4 and E4+K. It is cause-agnostic and unlikely to reach 0.02 on both sets at 31 MiB (`e6b_diag.md`, section 3); the
alternative is to stop (option D) and record that K helps new text and hurts known contexts.
