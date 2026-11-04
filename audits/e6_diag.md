# E6 diagnosis (T9.d): why C1 fails, against V2's proof (DESIGN §12.2)

Analysis only. No code changed, nothing rerun. Numbers are from `experiments/results/e6.json` (run at `63687db`) unless a script below says otherwise. Facts, assumptions and unknowns are marked F, A and U.

## 0. The result in one table

| bits per byte, A-test, frozen | E4 | E4+O | E4+O rank 1 (unigram) | O − E4 | rank 1 − E4 | O − rank 1 |
|---|---|---|---|---|---|---|
| class 0 (23.4% of bytes) | 3.0868 | 3.0532 | 3.0496 | −0.0336 | −0.0371 | **+0.0036** |
| class 8 | 1.8852 | 1.8832 | 1.8830 | −0.0020 | −0.0022 | +0.0003 |
| class 16 | 1.6219 | 1.6209 | 1.6207 | −0.0009 | −0.0011 | +0.0002 |
| class 32 | 1.2699 | 1.2690 | 1.2687 | −0.0009 | −0.0012 | +0.0003 |
| all bytes | 2.1057 | 2.0965 | 2.0954 | −0.0092 | −0.0102 | +0.0010 |
| A-heldout (unseen books) | 1.9237 | 1.9209 | 1.9204 | −0.0028 | −0.0033 | +0.0005 |
| train (learning pass) | 1.7961 | 1.7911 | 1.7912 | −0.0050 | −0.0049 | −0.0001 |

C1 needs class 0 at least 0.02 below both comparators. Against E4 it passes (0.0336). Against the unigram arm it fails, and by the wrong sign: O is 0.0036 worse (by-book bootstrap [+0.0014, +0.0057], 83 books). The fallback top mixture is not triggered: E4+O is better than E4 on A-test by 0.0092 (the trigger is "worse by more than 0.0005").

## 1. Is E4+O's gain over E4 explained by the unigram node alone?

Yes, on every number saved (F):

- The state-independent byte unigram, put through the same stage with the same 255 mixing nodes, takes **110%** of O's gain on class 0 (0.0371 against 0.0336), on all classes and on unseen books. O keeps 90% of the unigram's class-0 gain. During learning the two are indistinguishable (−0.0050 and −0.0049).
- O's state-dependent part is the difference O − rank 1: +0.0036 on class 0, positive (worse) in every window class, and worse than the unigram on 50 of 83 books (script 1).
- Per book, (O − rank 1) is negatively correlated with (rank 1 − E4) (r = −0.55, script 1). This is partly mechanical (the two differences share the rank-1 term with opposite signs), so I do not read it as evidence; it is not used below.

What the data cannot say (U): why a byte unigram helps at all. The unigram stage lowers class 0 by 0.037 bpb, which is the size of the effect E6 was meant to attribute to O. I have not read what E4's coarsest readout conditions on, so I do not claim a reason. The measured fact is only that O's context dependence adds nothing.

## 2. Which step of V2's argument does the run violate?

**(a) The bound: not violated, and not the issue (F).** Worst node excess 1.967 bits against the bound of 2 over 187 nodes used, and the summed excess is −49.7 bits (O) and −56.8 bits (rank 1) over 8.7 million bits (`stage_regret`: stage output against the better of its input and O). The bound is a statement relative to the stage's input (V2a, "scope, stated plainly"). It limits the loss and says nothing about the size of a gain, so a pass here is compatible with a zero gain and does not explain the failure.

**(b) The rank: the instability is real, but it cannot carry the gap (F, script 3).** V2(b) cited Page ranks of 19 to 51 for frequent rows (E3 T4, 81 columns). Here the Page rule on the 4096 hashed columns chose k = 17, 26, 5, 5, 8, 13, … 41, 41, then 24 from compile 22 on, then **7 at the last compile** (all 31 in `arms["E4+O"]["compiles"]`). The operators in force on A-test and A-heldout are that last compile, k = 7 (A-train is 31.56 MiB).

But the learning pass already shows O is no better than the unigram at any k. Script 3 lines up the per-MiB learning bpb (all bytes, so diluted against class 0) with the k in force: (O − rank 1) is between −0.0009 and +0.0012 at every MiB, mean −0.0001, with k anywhere from 5 to 41. At MiB 16 to 21 (k = 35 to 41) it is −0.0002 to −0.0007; in the last, k = 7 block it is −0.0002. So rank 7 against rank 41 makes no visible difference, and the k = 7 last compile cannot be what puts the A-test gap at +0.0010 (all bytes) and +0.0036 (class 0). The rank is therefore not the explanation of O's failure; it is a stability defect to fix, no more. (A: learning-pass bpb is a dilute, all-bytes proxy; class-0 values were not saved per MiB.)

(U) The file has k, rows, columns and samples per compile, not the singular values, so I cannot say whether the rank rule's jumps follow the data.

**(c) Consistency: this is where the run departs most, in three ways, none of them the theorem's fault (F for the numbers, A for the attribution).** Hsu–Kakade–Zhang (2009) is a statement about a finite-rank HMM observed through its own symbols and with enough samples. V2(c) already says text is not of finite rank, so the theorem was only ever a mechanism. The run adds:

1. **Hashed pasts.** 310,721 distinct 4-byte contexts are hashed into P = 4096 columns: all 4096 occupied, **75.9 contexts per column** (F, `store`). Each column of C2 and C3 is a mixture of about 76 unrelated contexts, so the Hankel block the spectral step sees is not the Hankel block of the process. The compiled operators describe hash buckets. Counts per cell (F, from `store` and `rows` = 167): C2 has 33.09 M counts over 167 × 4096 cells, **about 48 per cell**; C3 has 33.09 M over 167² × 4096 = 114 M cells, **about 0.29 per cell**. C3, the three-index tensor the operators B_x are built from, is sparse: a typical cell holds zero or one count. That is the sample-complexity term of HKZ (variance of the estimated operators) and it works against a bigger P: at 2^14 columns C3 falls to about 0.07 per cell.
2. **Negative mass is large.** A mean 2.4% of the predicted mass is negative on A-test and **79% of A-test bytes have a negative part** (97.7% during learning, mean share 3.1%). V2 expected negative entries and floors them at ε; the share logged is the error the floor absorbs. Where the floor acts, O's output is the floor, not an estimate.
3. **State resets.** The bad-update rule resets the state to b1 on 3.5% of bytes while learning (1.11 M of 32.0 M) and **4.4% of A-test bytes (183,548)**. The unigram arm triggers the same rule on 0.38% and 0.43% (122,430 and 17,572; A: rare bytes whose raw mass is at most ε·Σ|raw|, no-ops there since its state is constant). So O's state becomes invalid on about 10 times as many bytes as rare bytes explain. b1 = Ũᵀπ (`compile_operator`), the state of the stationary past distribution, so a reset moves O toward the byte marginal, which is the comparator that wins. Resets therefore cannot explain O losing to the unigram; they are a symptom that the state leaves the region where the operators are valid on 1 byte in 23.

Together these describe an estimate near the byte marginal whose state-dependent part is noise: sparse C3, mixed columns, invalid states on 4.4% of bytes. Its excess over the unigram is small (+0.0036 on class 0) because the floor and the resets pull it back to the marginal.

**A fourth fact that applies to the design of E6, not to V2's proof (F, script 2).** O's past is the last 4 bytes. Of the A-test bytes in class 0, **only 6.5%** have a 4-byte context unseen in A-train (62,752 of about 969,000 class-0 bytes; 1.5% of all A-test bytes). For the other 93.5% the context O would generalize from is one that O (through C3) and E4's own order-4 sectors have both seen, so O is a lossy, low-rank copy of counts E4 already holds. V2's claim was O's value "on unseen contexts" (the 8-byte window), and the 8-byte window is absent for class 0 by definition; but O reads only 4 bytes, and at that horizon class-0 bytes are mostly seen contexts. The ceiling on what O's generalization could save is therefore small: 0.02 bpb on class 0 needs 19.4 kbit saved, which is 0.31 bit per byte over just the 62,752 unseen-context bytes if nothing is gained elsewhere. This does not change the verdict on (a), (b), (c); it says E6 as specified gave O little room, and that a larger L trades against the hash.

**Answer.** (a) holds and is not the issue. (b) is excluded as the cause (script 3: no difference at k = 41 or k = 7), though the rank rule is unstable. The failure is (c) in practice: hashed pasts at 76 contexts per column, C3 at 0.29 counts per cell, a negative part on 79% of bytes, invalid states on 4.4%. That is the best-supported reading (A); the saved numbers do not separate the hash from the sparsity, and reruns 2 and 3 below do.

## 3. Ranked fixes within T

Every rerun needs a **dated PROTOCOL amendment before it runs** (E6.1 fixes the configuration and the rank rule), and the run script needs changes: `e6_operator.py` has no flag for `k_force` or for a single arm, and `--resume` needs `e6_partial.json`, which the helper untracked. So each fix is a small script change plus one O arm (≈1300 s from the timing check; the E6 run took 3210 s for three arms plus the gate); E4 and the rank-1 arm are deterministic and are reused from `e6.json` by the script change, except in fix 4.

| # | change | what it changes | cost | what the rerun tests |
|---|---|---|---|---|
| 1 | `k_force = 24` (the median) at every compile | removes the rank instability and the k = 7 last compile | script flag only | E4+O at fixed k. Script 3 predicts no change in class 0; the rerun confirms or refutes that (b) is excluded. Cheap, so worth it, but it is not the likely fix. |
| 2 | past size alone: keep L = 4, `past_bits = 14` (16,384 columns, about 19 contexts per column), C3 stored over the 167 observed rows (167² × 16,384 × 4 B ≈ 1.8 GB) | cuts hash mixing 4-fold; makes C3 sparser (≈0.07 per cell) | `MomentStore` change; memory ≈ 1.8 GB + the machine (helper peak was 2.7 GiB) | whether hash mixing or C3 sparsity dominates: if class 0 improves, mixing was the cause; if not, sparsity or the hash are not the lever. Not L = 3: only 1.25% of class-0 bytes have an unseen 3-byte context (script 2), which would shrink O's room further. Distinct contexts: 62,267 at L = 3, 310,721 at L = 4 (script 2'). |
| 3 | replace the bad-update reset with a state update that stays valid: renormalize **the state** by the signed mass Σ|raw| rather than resetting, or restart from b1 only when that is also bad. Exact rule to be written and tested before the amendment. | cuts the 4.4% resets, but the state may then carry a bad vector | a few lines in `_o_pass`; gate (g) is rewritten | resets are a symptom (b1 is the marginal), so I expect little change in class 0; this may make O worse. Lowest priority of the three. |
| 4 | key O's mixing nodes by (partial byte u, deepest stored level class), at most nine classes (V2a bound 2·255·9 = 4,590 bits) | O is weighed by whether E4 knows the context | a column in the loop; **the rank-1 arm must be rerun with the same 2295 nodes**, or node keying and O are confounded: two arms | whether O's weight, not its estimate, is the problem. Not a fix for this failure: the stage regret is already negative. It is the one change that targets class 0, which C1 names. |
| 5 | L = 5 or 6 with the past hashed to 2^14 or 2^16 | more class-0 positions have an unseen context (6.5% at L = 4) | memory as in 2; fewer counts per cell | after 2. |

Order: 2 first (the best-supported cause), then 1 only if the manager wants the rank excluded cleanly, then 3 if the reset rate stays high after 2. Fix 4 is for a different question.

Not proposed: more compile frequency (the problem is not staleness), tuning ε (the floor is not binding on rank 1), changing C1 (fixed in advance).

**Stopping rule I would propose for the manager** (A): if reruns 1 and 2 leave O above the unigram arm on class 0 with the by-book interval above 0, the finding is "the spectral O at L ≤ 4 learns nothing beyond the byte unigram on this corpus", reported as that and not retried with further tuning.

## 4. Does this result trigger the density (Kraus) form of O?

DESIGN §12.7 puts the Kraus form under "not doing until linear O works". **No, this result does not trigger it** (A, the manager rules):

- "Linear O works" has not been tested: C1 failed with an estimate that is hashed 76 to 1, whose rank moves between compiles, and whose last compile is k = 7. That is a failure of the estimator on this input, not a bound on the linear form (V3 bounds the linear form only on held-out sums, which E6 does not test).
- The one symptom a positive form would cure is the negative share (79% of bytes). The Kraus form needs a compile that is not the spectral one (a positive fit, for example by nonnegative factorization or EM), so it is a new algorithm and a new protocol, not a variation of this run; the cost is larger than fixes 1 to 3 together, and it inherits the hashed past and the same rank question.
- It becomes relevant only if fixes 1 to 3 leave the negative share high (say above 0.1 of mass, which I have not defined in advance; the manager would set it) while O's state-dependent part is still positive; then the positive form addresses the remaining measured defect.

## 5. Scripts (read-only; run from `vam/`)

Scripts 1 and 3 read only `e6.json`. Script 2 and 2' also read the corpus streams (the E2 corpus restored), which goes beyond "saved numbers"; they change nothing, and the manager may rule it out of the diagnosis. The conclusions that depend on them are the 6.5% ceiling and the contexts-per-column figures.

Script 1, per-book comparison from `e6.json`:

```python
import json, numpy as np
d = json.load(open('experiments/results/e6.json'))['arms']
def c0(a):
    b = d[a]['class0_per_book']; return np.array(b['bits']), np.array(b['bytes'], float)
(bo, n), (br, _), (be, _) = c0('E4+O'), c0('E4+O rank 1'), c0('E4')
dv, dr = (bo - br) / n, (br - be) / n
print((dv > 0).sum(), len(n), np.corrcoef(dv, dr)[0, 1], (be - bo).sum() / (be - br).sum())
# -> 50 of 83 books O worse than rank 1; r = -0.548; O keeps 0.904 of the unigram gain on class 0
```

Script 2, how many A-test bytes have a context unseen in A-train (run in `experiments/`):

```python
import sys, numpy as np
sys.path.insert(0, '.')
from pathlib import Path
from e2_curriculum import streams
from e4_hier import window_classes
S = streams(Path('data/pg19/MANIFEST.json'), False)
tr = np.frombuffer(S['A_train'], np.uint8); te = np.frombuffer(S['A_test'], np.uint8)
cls = window_classes(S['A_train'], S['A_test'])
def keys(a, L):
    a = a.astype(np.uint64); v = np.zeros(len(a) - L + 1, np.uint64)
    for i in range(L): v = (v << np.uint64(8)) | a[i:len(a) - L + 1 + i]
    return v
for L in (1, 2, 3, 4):
    ktr = np.unique(keys(tr, L)); ctx = keys(te, L)[:-1]; pos = np.arange(L, len(te))
    seen = np.isin(ctx[:len(pos)], ktr); c0 = cls[pos] == 0
    print(L, 1 - seen.mean(), 1 - seen[c0].mean(), int((~seen).sum()))
# L=4: 0.0152 of all A-test positions, 0.0647 of class-0 positions, 62,752 positions; every unseen one is class 0
```

Script 2', distinct contexts at L = 3 and 4 in A-train (same setup): `np.unique(keys(tr, L))` gives 62,267 (L = 3) and 310,721 (L = 4).

Script 3, learning-pass bpb per MiB against the k in force (MiB i uses `compiles[i-1]`; MiB 0 has no operator):

```python
import json, numpy as np
d = json.load(open('experiments/results/e6.json'))['arms']
o = np.array(d['E4+O']['learning_curve_bpb_per_MiB']); r = np.array(d['E4+O rank 1']['learning_curve_bpb_per_MiB'])
ks = [c['k'] for c in d['E4+O']['compiles']]
for i in range(1, len(o)): print(i, ks[i - 1], round(o[i] - r[i], 4))
print((o - r)[1:].mean(), ((o - r)[1:] < 0).sum())
# O - rank 1 per MiB lies in [-0.0009, +0.0012], mean -0.0001, negative at 22 of 31 MiB; k from 5 to 41; the last block has k = 7 and -0.0002
```
