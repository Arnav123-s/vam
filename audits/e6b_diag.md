# E6b diagnosis: why training improves while A-test and class 0 get worse (saved numbers only)

Analysis only: no code changed, nothing rerun. Numbers are from `experiments/results/e6b.json` (run at `d6ce3de`) by the
read-only script `audits/e6b_diag.py` (it reads that file and nothing else). F fact, A assumption, U unknown.

## 0. The result

| bits per byte | E4 | E4+K | E4+K random | K − E4 | control − E4 |
|---|---|---|---|---|---|
| train (read once with learning, each byte predicted before it is seen) | 1.7961 | 1.7678 | 1.7905 | **−0.0283** | −0.0056 |
| A-test (tail of the training books, frozen) | 2.1057 | 2.1200 | 2.1271 | **+0.0143** | +0.0214 |
| A-heldout (unseen books, frozen) | 1.9237 | 1.9144 | 1.9426 | **−0.0093** | +0.0189 |
| class 0 (23.4% of A-test) | 3.0868 | 3.1248 | 3.1074 | +0.0380 | +0.0206 |
| U2 (42.2%) | 2.0477 | 2.0417 | 2.0542 | −0.0059 | +0.0065 |
| U1 (12.9%) | 2.2429 | 2.2061 | 2.2198 | −0.0368 | −0.0231 |

C-E6b fails (class 0: −0.0380 against E4, −0.0174 against the control; U2: +0.0059 and +0.0125; the bar is 0.02) and E4+K
loses on A-test by more than the tie band, which triggers the one reserved amendment (E6b.7).

## 1. The question: training improves, A-test gets worse (F)

- **The improvement in training is not a late or a transient effect.** The saved learning curves show E4+K ahead of E4 at
  **every one of the 32 MiB** (−0.062 at the first, then −0.019 to −0.046); the random control is ahead at 23 MiB and
  behind (by at most 0.006) at nine. So while it learns, the machine with K predicts each next byte better throughout, and
  it does so late in the stream, when many contexts are seen, as well as early.
- **The loss appears only in the frozen read at contexts the tables hold.** By window class (positive = worse than E4; share
  × difference is the contribution to the A-test total of +0.0143): class 0 +0.038 (+0.0089), class 8 +0.003 (+0.0016),
  class 16 +0.010 (+0.0018), class 32 **+0.086** (+0.0020). Class 0 split by U2: **inside U2 (exact 2-unit context unseen)
  +0.0028**, **outside U2 (the exact 2-unit context seen, no 8-byte window) +0.098** (3.178 → 3.277); the control is
  +0.0618 outside and −0.0034 inside. On U2 outside class 0 the K levels win by 0.0107. They win on U1 (−0.037) and on
  unseen books (−0.009), where the context is new.
- So the loss is not "seen contexts in general" (they are plentiful in the learning read, where K wins) but specific to
  **the frozen read of A-test**, the tail of books that A-train holds (77% of its bytes have a long window that A-train
  contains), at contexts the final tables hold.

## 2. Candidate causes, none resolved by the saved numbers (A / U)

1. **Dilution at the final, never-drained evidences.** The K levels are interleaved, K1 before W1, K2 between W1 and W2, K4
   between W2 and W4 (declared order K1, W1, K2, W2, K4, W4, S0, S1), so on class 0 outside U2, where the exact context is
   W2's, K4 follows it, and K1 and K2 feed it. A K sector's evidence is the running log-ratio of its own prediction against
   its input and is never drained (sink rate 0); it accumulates at the many positions where the input is weak and persists
   where the input is strong, as at O's slot in E7. Fits: loss at seen contexts, more for real classes (class 32 +0.086,
   class 0 +0.038) than for random ones (+0.031, +0.021), which carry less evidence. Does not by itself explain why the
   learning read, which has the same evidences, is ahead throughout.
2. **Stale class ids in the frozen read.** K tables are written under 31 successive maps and read under the final one. The
   prequential read always reads under the map it is writing; the frozen read of A-test does not. Churn between compiles is
   0.115 at the second compile, 0.025 at the median and 0.011 at the last (real map), so ids drift by a few percent per
   compile, cumulatively more. This acts on the real map only: **the control's churn is not measured** (`control_map`
   copies the real map's `info`, so `report.py` and RESULTS.md show the control's churn equal to the real map's; by
   construction its assignments persist, so its true churn is small). That the control still loses **+0.062** on class 0
   outside U2 with a persistent map says staleness cannot be all of it; for the real map (+0.098) it may add.
3. **Differences between the frozen and the learning read** (the sectors do not adapt during the frozen read; the
   composition of A-test against the stream). Unmeasured.

**One development check would separate them (not run).** On A-validation (unread by E6b): train on a 3 MiB prefix, read
A-validation frozen, with E4 and three variants of E4+K: as built; with the map fixed after its first compile (no staleness:
removes candidate 2); with the K levels after S1 (removes candidate 1, as E7's top slot did). The E6b sets (U2, class 0)
recomputed on A-validation. If the loss follows the first compile-fixed variant, it is staleness; if it follows the order,
it is dilution; if neither moves it, candidate 3.

**Capacity and map quality are not the cause.** Occupancy of the E4 levels is unchanged in E4+K (hashed levels 2 to 5: 0.09,
0.30, 0.73, 0.87 in both; W1, W2, W4, S0, S1: 0.88, 0.83, 0.36, 0.24, 0.03 in both, at shifted positions); evictions rise 39%
and refusals 34% in total because of the three new tables (K1 0.84, K2 0.88, K4 0.64 full). 90.5% of A-test unit tokens
are in the map.

## 3. Does the reserved fallback address it?

The fallback (E6b.7): E4 and E4+K read side by side, their final readouts mixed by the two-hypothesis coherent mixture, one
node per class of the deepest contributing level of the E4+K machine (level 0 always contributes, so at most 14 classes for
its 14 levels). By V2a it costs at most 2 bits per node against the better of the two: **at most 28 bits, so it cannot
lose to E4 by more than that** on any stream.

It is **cause-agnostic**: whichever of 1 to 3 acts inside E4+K, the mixture can recover E4's exactness at the contexts where
the deepest level is deep and keep K where it is shallow. Limits, from the numbers:
1. **The largest loss, class 0 outside U2 (+0.098), is not identified by the deepest level.** Its bytes have no 8-byte
   window but an exact word context, so their deepest contributing level may be a W level that U2-inside bytes also have.
   (U: the deepest level per byte is not in the file.)
2. **Selection between the two readouts is bounded by the better of them.** On class 0 the K levels are at best equal to E4
   where they do not lose (class 0 inside U2 +0.003); on U2 outside class 0 they win by 0.0107, in all of U2 by 0.0059.
   A mixture of two different predictors can beat both, which the saved numbers cannot bound; so **0.02 on both sets
   against E4 is unlikely to pass** (A), and the likely reading is "no loss to E4 on A-test, part of the unseen-books
   gain kept".
3. **Cost.** Two machines side by side: about 5 to 6 GiB and about 20 minutes per mixture arm (E4 426 s plus K 731 s
   of reading), two arms (real and random), 40 to 60 minutes.

## 4. What the data say beyond the criterion (F)

On text that is new to the machine, real word classes help, and by more than a size-matched random map. Real minus control,
in bits per byte (negative = real better): unseen books **−0.028**, U1 **−0.014**, U2 **−0.0125**, train **−0.023**,
A-test −0.0071. The learning-read gain over E4 is −0.028 (the control −0.006). That is the generalization V2′ names, at a
smaller size than the criterion's 0.02 on A-test sets, and on A-test the real map loses to E4 by 0.0143 while the control
loses by 0.0214.

## 5. Options for the manager (not built)

- **A. Run the reserved fallback as written.** The cause-agnostic way to stop losing to E4 (at most 28 bits). Neither A
  nor B is likely to reach 0.02 on both sets. Cost 40 to 60 minutes. Spends the one amendment.
- **B. Spend the amendment on one candidate: move the K levels after S1** (a declared order change, no new loop code). It
  tests the dilution candidate only, and only if that is the cause; if staleness is the cause it changes little. Same two
  arms, about 50 minutes.
- **C. Run the development check of section 2 first** (A-validation, 3 MiB prefix, a few minutes per variant), then choose
  between A and B, or stop. It costs no amendment.
- **D. Stop here** and record that C-E6b fails, that K helps new text, and why the criterion's sets (A-test) are where
  exact memory dominates.

I recommend C then the manager's choice, because the three candidates make A and B bets on different causes. The manager
rules.
