# Claims check (T5): E4 in RESULTS.md, the README status paragraph, AUDIT.md's C2 answer

- Author: the engineer session (`[private session reference removed]`), with the Opus advisor on the readings of R4, R5, the headline's composition and the memory figure
  (each conclusion it informed is marked *advisor* below). Date 2026-10-20. Read-only: no file other than this one was changed.
- Checked at `6de8f7c` against `experiments/results/e4.json`, `e3.json`, `e3_t4.json` and `e2.json` (python extractions; no whole JSON read), RUNLOG for the memory figure.

## Bottom line

**98 claims: 94 matched, 0 numerically mismatched, 4 qualified** (the number is right, the wording is off). Besides those, six material points are in the
documents' silence ("not claimed but material", section 3). **None of this blocks anything**; the README and AUDIT wording is the manager's to change.

- The E4 section of `RESULTS.md` is identical, line for line, to a fresh render of `report.e4_section(e4, e3, e2)`: it is current with the saved files.
- Every table row, every derived number (gains, shares, R2 to R6) and the tie rule behind each Part 2 stage were recomputed independently of `report.py` (the script is in the appendix) and match.
- The E4 provenance commit `5ff49a7` is a partial-save commit made while the run was in progress. `git diff e4c8671..5ff49a7 --stat` changes only
  `vam/experiments/results/e4_partial.json`, so the code that ran is the start commit's (`e4c8671`, RUNLOG): **matched**.
- The R-rule verdicts are as registered: R2 pass, R3 pass, R4 fail, R5 fail.

## 1. Qualified claims

| where | claim | what the data say |
|---|---|---|
| README | "E4's gain is from span ... but not from order" (R4 fails) | R4 fails as registered (it needs 0.005 on A-test **and** unseen books): C2 (order destroyed) is worse than E4 by +0.0103 on A-test and +0.0030 on unseen books. Order accounts for about 41% of the A-test gain (0.0103 of 0.0252) and about 10% of the unseen-books gain (0.0030 of 0.0303). *Advisor:* "not from order" overstates it; C2 removes order across units and sentences only, not within a unit. Suggested wording: order helps on A-test, not reliably on unseen books; R4 fails as registered |
| README | "... or long matches" (R5 fails) | R5 fails as registered (0.395 of the gain from classes 16 and 32, below 0.5). Those classes hold 20.2% of the bytes but 39.5% of the gain; the per-byte gain there is 0.051 and 0.039 bpb against 0.013 and 0.022 in classes 0 and 8. *Advisor:* R5 says where E4 gains, not why lzma wins: class membership is "occurs anywhere in A-train", lzma's per-class cost is unmeasured, and the gap to lzma shrank from 0.128 to 0.103 bpb. Nothing here confirms or rules out long matches as its source |
| AUDIT C2 | "the inert halt is replaced in E4 by a raised admission bar" | True of the design (DESIGN section 11) and of Part 1 (the raise changes refusals at verify 2, 3, 4). In Part 2 global scope lost, with and without the raise (1.8924 and 1.8881 against 1.8524): the chosen machine runs E3's per-level governor, whose halt is inert |
| AUDIT C2 | "the cost of sinks is reported, not fixed" | True of E3. E4's Part 2 sink ablation (rate 0) won (1.8412 against 1.8524) and the chosen machine runs without sinks |

## 2. Mismatches

None numerically. (One cosmetic point in `RESULTS.md`: the N-L2 table prints `None` for W4 and S1, whose contexts never reach the count floor, where the protocol's tables use "n/a".)

## 3. Not claimed but material (omissions)

1. **The headline's composition (README).** E4's 2.106 is the chosen row, which is also the text rule (the tokenizer arm, labelled in PROTOCOL as a departure from F Part 10),
   `sink_rate 0` and upper verify 4. The sequential A-validation steps from E3's 1.8676 to E4's 1.8374 are (validation only, order-dependent, no test-split numbers):

   | step | A-validation | change |
   |---|---|---|
   | E3 chosen | 1.8676 | |
   | content-defined hierarchy (modality-free) | 1.8605 | -0.0071 |
   | text rule instead of content-defined | 1.8524 | -0.0081 |
   | sinks removed (rate 0) | 1.8412 | -0.0112 |
   | upper verify 4 | 1.8374 | -0.0038 |

   The modality-free arm stopped at 1.8605 on A-validation and has no test number. This is the result of owner question 1 of DESIGN 11.8 and the README is silent on it.
2. **Which comparisons are clean (advisor).** R3 (C1) and R4 (C2) are clean: both controls share sink 0, upper verify 4 and the segmentation of the chosen row. R2, R5 and the README's E3-against-E4
   numbers compare with E3 at sink rate 0.1, so part of the "gain" is sink removal. Do not reverse R2: the protocol made sink 0 selectable. What it attributes is "E4 as chosen against E3 as chosen".
   C1 is 0.0030 *worse* than E3 on A-test (2.134 against 2.131) although it also has sink 0; with no "E3 with sink 0" arm one cannot tell whether sink removal fails to transfer to the test split or the duplicated levels cost
   something. **Proposed (not run), labelled "added after the result":** E3's base with sink rate 0, one A-only arm plus the Part 5 decomposition.
3. **N-L5 fails (RESULTS) and the README does not say so.** Registered criterion: every upper level below 1% occupancy at verify 3 on 8 MiB of i.i.d. bytes. Measured W1 0.0452 and S0 0.0145 (W2 0.0002, W4 0.0000, S1 0.0001). Short units repeat in random bytes by construction
   (one-byte units have 256 codes). **Disclosure:** the unit test I wrote for T6.3 (`tests/test_hier.py::test_verified_growth_keeps_noise_out_of_the_upper_levels`) was written to tolerate this (it requires only W2, W4 and S1 below 1%, at 2^16 slots), so it
   passed while the registered criterion at full size failed. The mismatch between my test and the protocol is mine; the run's criterion is the one that counts.
4. **Label entropy per cut is numerically zero (4.6e-16 mean).** The RESULTS line says the area law is vacuous at bond 2; the value is zero because the chosen coherence is 1.0, which makes the label state pure, not because of the bond-2 bound.
5. **Part 3c (second shuffle seed) was not run** (it was optional), so R6's shuffled control (-0.0238 against a transfer of -0.0205) has one seed and no spread; RESULTS does not say that.
6. **Peak memory (RUNLOG).** 5,236,872 KiB = 4.99 GiB against about 4.1 GiB in the protocol and 4.2 GiB in my T6.3 estimate: 5.45 machine-equivalents against the 4.5 assumed from E3. The stop rule (10 GiB) held. *Advisor:* does not block. The
   cause is not attributed (RSS was sampled every 15 s, so the true peak may be higher, and there is no per-unit RSS); not guessed here. If a later run needs headroom, sample RSS per unit.

## 4. Items for the advisor's view (answered above, listed for the record)

- R4 and R5 failures and how the README should read them (section 1). *Advisor: both fail as registered; neither supports "not from".*
- Composition of the headline and the missing sink-0 control (section 3, items 1 and 2).
- The memory figure (section 3, item 6). *Advisor: not blocking, unattributed.*

## 5. Every claim checked

| group | claim | expected text or value | source field | verdict |
|---|---|---|---|---|
| header | run commit | `5ff49a71e8fffe483b7ceb694615e794d58a8c52` | provenance.commit | matched |
| header | run time | `2026-10-20T18:26:32Z` | provenance.utc | matched |
| header | hours | `3.6 hours` | seconds | matched |
| Part 0 | E3 saved A-validation | `1.8676168321` | part0.e3_saved_A_validation_bpb | matched |
| Part 0 | recomputed A-validation | `1.8676168321` | part0.recomputed_A_validation_bpb | matched |
| Part 0 | identity and reproduction pass | `bit for bit: **pass**` | part0.criteria.identity | matched |
| Part 0 | upper-off rate ratio | `0.988 of the nested` | part0.median_rate_ratio_upper_off_to_nested | matched |
| Part 0 | saved E3 value equals e3.json chosen row | `1.867616832051096 vs 1.867616832051096` | e3.part2.rows / part0 | matched |
| Part 1 | N-L1 state, units, mean unit, collisions | `state 88 bytes; units 663,054, mean 6.33 bytes, code collisions 0` | part1.N_L1 | matched |
| Part 1 | N-L3 bounds | `[0.000244141, 0.999756]` | part1.N_L3 | matched |
| Part 1 | N-L4 spans | `median span 8 bytes, largest 14` | part1.N_L4 | matched |
| Part 1 | N-L5 verify 3 occupancy W1 | `W1 0.0452` | part1.N_L5.occupancy_verify3 | matched |
| Part 1 | N-L5 verify 3 occupancy W2 | `W2 0.0002` | part1.N_L5.occupancy_verify3 | matched |
| Part 1 | N-L5 verify 3 occupancy W4 | `W4 0.0000` | part1.N_L5.occupancy_verify3 | matched |
| Part 1 | N-L5 verify 3 occupancy S0 | `S0 0.0145` | part1.N_L5.occupancy_verify3 | matched |
| Part 1 | N-L5 verify 3 occupancy S1 | `S1 0.0001` | part1.N_L5.occupancy_verify3 | matched |
| Part 1 | N-L5 verify 1 occupancy W1 | `W1 1.00` | part1.N_L5.occupancy_verify1 | matched |
| Part 1 | N-L5 verify 1 occupancy W2 | `W2 1.00` | part1.N_L5.occupancy_verify1 | matched |
| Part 1 | N-L5 verify 1 occupancy W4 | `W4 1.00` | part1.N_L5.occupancy_verify1 | matched |
| Part 1 | N-L5 verify 1 occupancy S0 | `S0 1.00` | part1.N_L5.occupancy_verify1 | matched |
| Part 1 | N-L5 verify 1 occupancy S1 | `S1 1.00` | part1.N_L5.occupancy_verify1 | matched |
| Part 1 | N-L5 criterion (every level below 1% at verify 3) recomputed | `fails for ['W1', 'S0']` | part1.N_L5 | matched |
| Part 1 | N-L6 survival | `0/257 plain, 252/254 verify 2, 255/257 governor` | part1.N_L6.survival | matched |
| Part 1 | N-L7 | `1.1e-16` | part1.N_L7 | matched |
| Part 1 | N-L2 W1 row | `\| W1 \| 33 \| 0.273 \| 622 of 2,382,714 \| 1 \|` | part1.N_L2.rows | matched |
| Part 1 | N-L2 W2 row | `\| W2 \| 23 \| 0.138 \| 85 of 3,406,071 \| 1 \|` | part1.N_L2.rows | matched |
| Part 1 | N-L2 W4 row | `\| W4 \| None \|  \| 0 of 4,092,974 \| None \|` | part1.N_L2.rows | matched |
| Part 1 | N-L2 S0 row | `\| S0 \| 17 \| 0.185 \| 58 of 3,990,516 \| 1 \|` | part1.N_L2.rows | matched |
| Part 1 | N-L2 S1 row | `\| S1 \| None \|  \| 0 of 4,189,392 \| None \|` | part1.N_L2.rows | matched |
| Part 1 | label entropy | `mean 4.562507634522007e-16, largest 4.60786666215941e-15 bits over 9,582` | part1.N_L5_cuts | matched |
| Part 2 | segmentation row | ``{'sink_rate': 0.1, 'hier_levels': 'WS'}`: 1.8605, 690.3 s` | part2.rows | matched |
| Part 2 | segmentation row | ``{'sink_rate': 0.1, 'hier_levels': 'WS', 'hier_segment': 'text'}`: 1.8524, 622.8 s (chosen)` | part2.rows | matched |
| Part 2 | levels row | ``{'sink_rate': 0.1, 'hier_levels': 'WS', 'hier_segment': 'text'}`: 1.8524, 622.8 s (chosen)` | part2.rows | matched |
| Part 2 | levels row | ``{'sink_rate': 0.1, 'hier_levels': 'W', 'hier_segment': 'text'}`: 1.8557, 531.5 s` | part2.rows | matched |
| Part 2 | levels row | ``{'sink_rate': 0.1, 'hier_levels': 'S', 'hier_segment': 'text'}`: 1.8647, 449.9 s` | part2.rows | matched |
| Part 2 | governor row | ``{'sink_rate': 0.1, 'hier_levels': 'WS', 'hier_segment': 'text'}`: 1.8524, 622.8 s (chosen)` | part2.rows | matched |
| Part 2 | governor row | ``{'sink_rate': 0.1, 'hier_levels': 'WS', 'hier_segment': 'text', 'hier_scope': 'global'}`: 1.8924, …` | part2.rows | matched |
| Part 2 | governor row | ``{'sink_rate': 0.1, 'hier_levels': 'WS', 'hier_segment': 'text', 'hier_scope': 'global', 'hier_rais…` | part2.rows | matched |
| Part 2 | sink_ablation row | ``{'sink_rate': 0.1, 'hier_levels': 'WS', 'hier_segment': 'text'}`: 1.8524, 622.8 s` | part2.rows | matched |
| Part 2 | sink_ablation row | ``{'sink_rate': 0.0, 'hier_levels': 'WS', 'hier_segment': 'text'}`: 1.8412, 583.4 s (chosen)` | part2.rows | matched |
| Part 2 | upper_verify row | ``{'sink_rate': 0.0, 'hier_levels': 'WS', 'hier_segment': 'text'}`: 1.8412, 583.4 s` | part2.rows | matched |
| Part 2 | upper_verify row | ``{'sink_rate': 0.0, 'hier_levels': 'WS', 'hier_segment': 'text', 'hier_verify': 2}`: 1.8494, 577.2 s` | part2.rows | matched |
| Part 2 | upper_verify row | ``{'sink_rate': 0.0, 'hier_levels': 'WS', 'hier_segment': 'text', 'hier_verify': 4}`: 1.8374, 573.9 …` | part2.rows | matched |
| Part 2 | chosen | `Chosen: `{'machine': 'hier', 'law': 'counting', 'prior': 4.0, 'coherence': 1.0, 'governor': True, '…` | part2.chosen | matched |
| Part 2 | stage segmentation: chosen follows the tie rule (best + 0.0005, then … | `{'sink_rate': 0.1, 'hier_levels': 'WS', 'hier_segment': 'text'}` | part2.stages | matched |
| Part 2 | stage levels: chosen follows the tie rule (best + 0.0005, then fewer … | `{'sink_rate': 0.1, 'hier_levels': 'WS', 'hier_segment': 'text'}` | part2.stages | matched |
| Part 2 | stage governor: chosen follows the tie rule (best + 0.0005, then fewe… | `{'sink_rate': 0.1, 'hier_levels': 'WS', 'hier_segment': 'text'}` | part2.stages | matched |
| Part 2 | stage sink_ablation: chosen follows the tie rule (best + 0.0005, then… | `{'sink_rate': 0.0, 'hier_levels': 'WS', 'hier_segment': 'text'}` | part2.stages | matched |
| Part 2 | stage upper_verify: chosen follows the tie rule (best + 0.0005, then … | `{'sink_rate': 0.0, 'hier_levels': 'WS', 'hier_segment': 'text', 'hier_verify': 4}` | part2.stages | matched |
| Part 3 | A_only row | `\| A_only \| 2.106 [2.131] \| 1.924 [1.954] \| 2.450 [2.468] \| 2.337 [2.360] \|` | part3a/3b.A_only and e3.part3.arms.A_only | matched |
| Part 3 | A_then_B row | `\| A_then_B \| 2.148 [2.186] \| 1.976 [2.021] \| 2.059 [2.090] \| 2.031 [2.063] \|` | part3a/3b.A_then_B and e3.part3.arms.A_then_B | matched |
| Part 3 | B_only row | `\| B_only \| 2.370 [2.382] \| 2.123 [2.143] \| 2.080 [2.105] \| 2.046 [2.074] \|` | part3a/3b.B_only and e3.part3.arms.B_only | matched |
| Part 3 | shuffled_A_then_B row | `\| shuffled_A_then_B \| 2.147 [2.185] \| 1.977 [2.021] \| 2.056 [2.088] \| 2.031 [2.063] \|` | part3a/3b.shuffled_A_then_B and e3.part3.arms.shuffled_A_th… | matched |
| Part 4 | E4 chosen | `\| E4 chosen \| 2.106 \| 1.924 \| 2.450 \| 2.337 \|` | part4 / part3a / e3.part3 | matched |
| Part 4 | C1 equal storage and compute, no extra span | `\| C1 equal storage and compute, no extra span \| 2.134 \| 1.957 \| 2.472 \| 2.361 \|` | part4 / part3a / e3.part3 | matched |
| Part 4 | C2 order destroyed | `\| C2 order destroyed \| 2.116 \| 1.927 \| 2.455 \| 2.341 \|` | part4 / part3a / e3.part3 | matched |
| Part 4 | C0 upper levels off (E3, saved) | `\| C0 upper levels off (E3, saved) \| 2.131 \| 1.954 \| 2.468 \| 2.360 \|` | part4 / part3a / e3.part3 | matched |
| Part 5 | class 0 | `\| 0 \| 0.2345 \| 3.100 \| 3.087 \| 3,004,455 \| 2,992,107 \| 12,348 \|` | part5 | matched |
| Part 5 | class 8 | `\| 8 \| 0.5639 \| 1.907 \| 1.885 \| 4,445,460 \| 4,394,719 \| 50,742 \|` | part5 | matched |
| Part 5 | class 16 | `\| 16 \| 0.1782 \| 1.673 \| 1.622 \| 1,231,892 \| 1,194,546 \| 37,347 \|` | part5 | matched |
| Part 5 | class 32 | `\| 32 \| 0.0234 \| 1.309 \| 1.270 \| 126,567 \| 122,800 \| 3,767 \|` | part5 | matched |
| Part 5 | class all | `\| all \| 1.0000 \| 2.131 \| 2.106 \| 8,808,375 \| 8,704,171 \| 104,203 \|` | part5 | matched |
| Part 5 | class gains add up to the total | `104,203.3 vs 104,203.3` | part5 | matched |
| Part 5 | E4 per-byte bpb total equals A_only A_test (frozen) | `2.105656 vs 2.105656` | part5.E4.all vs part3a.A_only.A_test | matched |
| Part 5 | E3 per-byte bpb total equals E3 A_only A_test | `2.130865 vs 2.130865` | part5.E3.all vs e3.part3 | matched |
| Reading rules | R1 numbers | `E4 A-test 2.1057 bpb; E3 2.1309; lzma 2.0031` | part3a.A_only.A_test; e3.part3; e2.reference_lzma | matched |
| Reading rules | R2 gains and verdict | `+0.0252 and +0.0303: **pass**` | e3.part3.arms.A_only - part3a.A_only | matched |
| Reading rules | R3 gains and verdict | `+0.0282 and +0.0329: **pass**` | part4.C1 - part3a.A_only | matched |
| Reading rules | R4 gains and verdict | `+0.0103 and +0.0030: **fail**` | part4.C2 - part3a.A_only | matched |
| Reading rules | R5 share and verdict | `0.39 of 104,203 bits: **fail**` | part5 | matched |
| Reading rules | R6 transfer, control, forgetting | `transfer (A then B minus B only) -0.0205; sentence-shuffled control -0.0238; forgetting of A after …` | part3a/3b | matched |
| Part 6 | hbar 1 row metrics | `0.000 \| 0.906 \| 1.000 \| 1.000 \| 1.731 \|` | part6.settings | matched |
| Part 6 | hbar 0 row metrics | `0.244 \| 1.000 \| 1.000 \| 1.000 \| 0.600 \|` | part6.settings | matched |
| Part 6 | prompt 1 E4 output verbatim | `- E4 `"\npractical utting them ree points of\nanother, to"`` | part6.settings.outputs | matched |
| Part 6 | prompt 1 E3 output verbatim | `- E3 `"\npractical utting them ree points of animals,\nto"`` | e3.part4.settings.outputs | matched |
| Part 6 | prompt 2 E4 output verbatim | `- E4 `"   _A_, and _n_). Hence\nthe seventeese: general "`` | part6.settings.outputs | matched |
| Part 6 | prompt 2 E3 output verbatim | `- E3 `"   _A_, and _look forth the seventeese: general "`` | e3.part4.settings.outputs | matched |
| Part 6 | prompt 3 E4 output verbatim | `- E4 `" of the teacher.\n Becognise have the high relief"`` | part6.settings.outputs | matched |
| Part 6 | prompt 3 E3 output verbatim | `- E3 `" of the woman and B by means of apples of politi"`` | e3.part4.settings.outputs | matched |
| Part 6 | prompt 4 E4 output verbatim | `- E4 `" we can also the good.[187 \|3 as an After a mean"`` | part6.settings.outputs | matched |
| Part 6 | prompt 4 E3 output verbatim | `- E3 `" we can also the good.[187 \|3 as an After diffic"`` | e3.part4.settings.outputs | matched |
| Part 6 | prompt 5 E4 output verbatim | `- E4 `" felt iron oxide, It is undesigns which the work"`` | part6.settings.outputs | matched |
| Part 6 | prompt 5 E3 output verbatim | `- E3 `" felt iron will leather, I must table proposed t"`` | e3.part4.settings.outputs | matched |
| Part 6 | prompt 6 E4 output verbatim | `- E4 `", we should train utilities value.\n\nThere is no "`` | part6.settings.outputs | matched |
| Part 6 | prompt 6 E3 output verbatim | `- E3 `", we should train utilities the origin of much l"`` | e3.part4.settings.outputs | matched |
| Part 6 | prompt 7 E4 output verbatim | `- E4 `"     3. Under the action of parcommon. In\ncausae"`` | part6.settings.outputs | matched |
| Part 6 | prompt 7 E3 output verbatim | `- E3 `"     3/4!\\xa9\u0002[\u000bs\\xa1\u0003\\xa3C)\u0003+K;C\\xa3++q\u0003#\u000b\u02d9\u0003{1\…` | e3.part4.settings.outputs | matched |
| Part 6 | prompt 8 E4 output verbatim | `- E4 `" indefinite placed that the modern\nphenomenon of"`` | part6.settings.outputs | matched |
| Part 6 | prompt 8 E3 output verbatim | `- E3 `" indeed it is plain the popularly upon here, and"`` | e3.part4.settings.outputs | matched |
| README | E3 A-test 2.131 | `reads A-test at 2.131 bpb` | e3.part3.arms.A_only.A_test | matched |
| README | E4 A-test 2.106 | `at 2.106` | part3a.A_only.A_test | matched |
| README | lzma 2.003 | `lzma's 2.003` | e2.reference_lzma | matched |
| README | E4 beats the equal-capacity control (R3) | `+0.0282, +0.0329` | part4.C1 | matched |
| README | "not from order": R4 fails | `R4 gains +0.0103 (A-test) and +0.0030 (unseen books); the rule needs both at 0.005` | part4.C2 | QUALIFIED |
| README | "not from long matches": R5 fails | `share 0.395 < 0.5, but classes 16 and 32 hold 0.202 of the bytes` | part5 | QUALIFIED |
| AUDIT C2 | T4.2 and T4.3 exist in e3_t4.json | `keys ['T4_2_text', 'T4_3_chain']` | e3_t4.json | matched |
| AUDIT C2 | "the inert halt is replaced in E4 by a raised admission bar" | `design: yes; chosen configuration: scope level, raise False` | part2.chosen | QUALIFIED |
| AUDIT C2 | "the cost of sinks is reported, not fixed" | `E4 Part 2 ablation chose sink_rate 0.0` | part2.chosen | QUALIFIED |

## Appendix: the checking script (run from `vam/`: `python check.py rows.json`)

```python
"""Independent check of RESULTS.md (E4), README status and AUDIT.md C2 against e4.json, e3.json, e3_t4.json, e2.json."""
import json, re, sys
R = 'experiments/results/'
e4, e3, e2, t4 = (json.load(open(R + f + '.json')) for f in ('e4', 'e3', 'e2', 'e3_t4'))
res = open('RESULTS.md').read()
sec = res[res.index('## E4.'):]
readme = open('README.md').read()
audit = open('AUDIT.md').read()
rows = []          # (group, claim, expected text, source, verdict)

def chk(group, claim, expected, source, text=None):
    text = sec if text is None else text
    ok = expected in text
    rows.append((group, claim, expected, source, 'matched' if ok else 'MISMATCHED'))
    return ok

f = lambda x, n=3: f"{x:.{n}f}"
p0, p1, p2, p3a, p3b, p4, p5, p6 = (e4[k] for k in ('part0', 'part1', 'part2', 'part3a', 'part3b', 'part4', 'part5', 'part6'))
# header
chk('header', 'run commit', e4['provenance']['commit'], 'provenance.commit')
chk('header', 'run time', e4['provenance']['utc'], 'provenance.utc')
chk('header', 'hours', f"{e4['seconds'] / 3600:.1f} hours", 'seconds')
# part 0
chk('Part 0', 'E3 saved A-validation', f"{p0['e3_saved_A_validation_bpb']:.10f}", 'part0.e3_saved_A_validation_bpb')
chk('Part 0', 'recomputed A-validation', f"{p0['recomputed_A_validation_bpb']:.10f}", 'part0.recomputed_A_validation_bpb')
chk('Part 0', 'identity and reproduction pass', 'bit for bit: **pass**', 'part0.criteria.identity')
chk('Part 0', 'upper-off rate ratio', f"{p0['median_rate_ratio_upper_off_to_nested']:.3f} of the nested", 'part0.median_rate_ratio_upper_off_to_nested')
e3row = next(r for r in e3['part2']['rows'] if r['spec'] == e3['part2']['chosen'])
rows.append(('Part 0', 'saved E3 value equals e3.json chosen row', f"{e3row['A_validation_bpb']!r} vs {p0['recomputed_A_validation_bpb']!r}", 'e3.part2.rows / part0',
             'matched' if abs(e3row['A_validation_bpb'] - p0['recomputed_A_validation_bpb']) < 1e-9 else 'MISMATCHED'))
# part 1
l1, l2, l3, l4, l5, l6, l7 = (p1[k] for k in ('N_L1', 'N_L2', 'N_L3', 'N_L4', 'N_L5', 'N_L6', 'N_L7'))
chk('Part 1', 'N-L1 state, units, mean unit, collisions', f"state {l1['state_bytes']} bytes; units {l1['units']:,}, mean {l1['mean_unit_bytes']:.2f} bytes, code collisions {l1['code_collisions']}", 'part1.N_L1')
chk('Part 1', 'N-L3 bounds', f"[{l3['min']:.6g}, {l3['max']:.6g}]", 'part1.N_L3')
chk('Part 1', 'N-L4 spans', f"median span {l4['median_span_bytes']:.0f} bytes, largest {l4['max_span_bytes']}", 'part1.N_L4')
for k, v in l5['occupancy_verify3'].items():
    chk('Part 1', f'N-L5 verify 3 occupancy {k}', f"{k} {v:.4f}", 'part1.N_L5.occupancy_verify3')
for k, v in l5['occupancy_verify1'].items():
    chk('Part 1', f'N-L5 verify 1 occupancy {k}', f"{k} {v:.2f}", 'part1.N_L5.occupancy_verify1')
n5 = all(v < 0.01 for v in l5['occupancy_verify3'].values()) and any(v > 0.5 for v in l5['occupancy_verify1'].values())
rows.append(('Part 1', 'N-L5 criterion (every level below 1% at verify 3) recomputed', f"fails for {[k for k,v in l5['occupancy_verify3'].items() if v >= 0.01]}", 'part1.N_L5',
             'matched' if (not n5 and '**fail**' in sec.split('| N-L5')[1].split('\n')[0]) else 'MISMATCHED'))
chk('Part 1', 'N-L6 survival', f"{l6['survival']['plain']['after']}/{l6['survival']['plain']['before']} plain, {l6['survival']['verify_2']['after']}/{l6['survival']['verify_2']['before']} verify 2, {l6['survival']['governor']['after']}/{l6['survival']['governor']['before']} governor", 'part1.N_L6.survival')
chk('Part 1', 'N-L7', f"{l7['hbar1_Z_minus_prediction']:.1e}", 'part1.N_L7')
for r in l2['rows']:
    if r['data'] != 'A_train': continue
    null = next(x for x in l2['rows'] if x['data'] != 'A_train' and x['level'] == r['level'])
    cut = '' if r['cutoff'] is None else f"{r['cutoff']:.3f}"
    chk('Part 1', f"N-L2 {r['level']} row", f"| {r['level']} | {r['rank']} | {cut} | {r['rows_kept']:,} of {r['rows']:,} | {null['rank']} |", 'part1.N_L2.rows')
cuts = p1['N_L5_cuts']
chk('Part 1', 'label entropy', f"mean {cuts['mean_label_entropy_bits']}, largest {cuts['max_label_entropy_bits']} bits over {cuts['samples']:,}", 'part1.N_L5_cuts')
# part 2
for st in p2['stages']:
    for c in st['candidates']:
        r = next(x for x in p2['rows'] if x['spec'] == c)
        mark = ' (chosen)' if c == st['chosen'] else ''
        shown = {k: v for k, v in c.items() if k.startswith('hier_') or k == 'sink_rate'}
        chk('Part 2', f"{st['stage']} row", f"`{shown}`: {r['A_validation_bpb']:.4f}, {r['seconds']} s{mark}", 'part2.rows')
chk('Part 2', 'chosen', f"Chosen: `{p2['chosen']}`", 'part2.chosen')
# tie rule recomputation: independent
TIE = 0.0005
def simpler(c):
    lv = {'WS': 5, 'W': 3, 'S': 2, '': 0}[c.get('hier_levels', 'WS')]
    return (lv + (c.get('hier_segment', 'content') == 'text') + (c.get('hier_scope') == 'global') + bool(c.get('hier_raise'))
            + (c.get('hier_verify', 3) != 3) + (c.get('sink_rate', 0.1) > 0))
for st in p2['stages']:
    sc = [(next(x for x in p2['rows'] if x['spec'] == c)['A_validation_bpb'], c) for c in st['candidates']]
    best = min(s for s, _ in sc)
    pick = min([(simpler(c), s, c) for s, c in sc if s <= best + TIE], key=lambda t: (t[0], t[1]))[2]
    rows.append(('Part 2', f"stage {st['stage']}: chosen follows the tie rule (best + 0.0005, then fewer options)", str({k: v for k, v in pick.items() if k.startswith('hier_') or k == 'sink_rate'}), 'part2.stages', 'matched' if pick == st['chosen'] else 'MISMATCHED'))
# part 3
for arm, r in {'A_only': p3a['A_only'], 'A_then_B': p3a['A_then_B'], 'B_only': p3b['B_only'], 'shuffled_A_then_B': p3b['shuffled_A_then_B']}.items():
    r3 = e3['part3']['arms'][arm]
    cell = lambda k: f"{r[k]:.3f} [{r3[k]:.3f}]"
    chk('Part 3', f'{arm} row', f"| {arm} | {cell('A_test')} | {cell('A_heldout')} | {cell('B_test')} | {cell('B_heldout')} |", f'part3a/3b.{arm} and e3.part3.arms.{arm}')
# part 4
c1, c2 = p4['C1_equal_storage_no_extra_span'], p4['C2_order_destroyed']
for name, r in (('E4 chosen', p3a['A_only']), ('C1 equal storage and compute, no extra span', c1), ('C2 order destroyed', c2), ('C0 upper levels off (E3, saved)', e3['part3']['arms']['A_only'])):
    chk('Part 4', name, f"| {name} | {r['A_test']:.3f} | {r['A_heldout']:.3f} | {r['B_test']:.3f} | {r['B_heldout']:.3f} |", 'part4 / part3a / e3.part3')
# part 5
tot_gain = 0
gains = {}
for c in ('0', '8', '16', '32', 'all'):
    a, b = p5['E3'][c], p5['E4'][c]
    gains[c] = a['bits'] - b['bits']
    chk('Part 5', f'class {c}', f"| {c} | {a['share']:.4f} | {a['bpb']:.3f} | {b['bpb']:.3f} | {a['bits']:,.0f} | {b['bits']:,.0f} | {gains[c]:,.0f} |", 'part5')
rows.append(('Part 5', 'class gains add up to the total', f"{sum(gains[c] for c in ('0','8','16','32')):,.1f} vs {gains['all']:,.1f}", 'part5', 'matched' if abs(sum(gains[c] for c in ('0','8','16','32')) - gains['all']) < 1e-3 else 'MISMATCHED'))
rows.append(('Part 5', 'E4 per-byte bpb total equals A_only A_test (frozen)', f"{p5['E4']['all']['bpb']:.6f} vs {p3a['A_only']['A_test']:.6f}", 'part5.E4.all vs part3a.A_only.A_test', 'matched' if abs(p5['E4']['all']['bpb'] - p3a['A_only']['A_test']) < 1e-9 else 'MISMATCHED'))
rows.append(('Part 5', 'E3 per-byte bpb total equals E3 A_only A_test', f"{p5['E3']['all']['bpb']:.6f} vs {e3['part3']['arms']['A_only']['A_test']:.6f}", 'part5.E3.all vs e3.part3', 'matched' if abs(p5['E3']['all']['bpb'] - e3['part3']['arms']['A_only']['A_test']) < 1e-6 else 'MISMATCHED'))
# reading rules, recomputed independently
a4, a3_ = p3a['A_only'], e3['part3']['arms']['A_only']
lz = e2['reference_lzma']['lzma_A_test_given_A_train']
R2 = (a3_['A_test'] - a4['A_test'], a3_['A_heldout'] - a4['A_heldout'])
R3 = (c1['A_test'] - a4['A_test'], c1['A_heldout'] - a4['A_heldout'])
R4 = (c2['A_test'] - a4['A_test'], c2['A_heldout'] - a4['A_heldout'])
share = (gains['16'] + gains['32']) / gains['all']
tr = p3a['A_then_B']['B_test'] - p3b['B_only']['B_test']
trs = p3b['shuffled_A_then_B']['B_test'] - p3b['B_only']['B_test']
forg = p3a['A_then_B']['A_test'] - a4['A_test']
chk('Reading rules', 'R1 numbers', f"E4 A-test {a4['A_test']:.4f} bpb; E3 {a3_['A_test']:.4f}; lzma {lz:.4f}", 'part3a.A_only.A_test; e3.part3; e2.reference_lzma')
chk('Reading rules', 'R2 gains and verdict', f"{R2[0]:+.4f} and {R2[1]:+.4f}: **{'pass' if min(R2) >= 0.01 else 'fail'}**", 'e3.part3.arms.A_only - part3a.A_only')
chk('Reading rules', 'R3 gains and verdict', f"{R3[0]:+.4f} and {R3[1]:+.4f}: **{'pass' if min(R3) >= 0.005 else 'fail'}**", 'part4.C1 - part3a.A_only')
chk('Reading rules', 'R4 gains and verdict', f"{R4[0]:+.4f} and {R4[1]:+.4f}: **{'pass' if min(R4) >= 0.005 else 'fail'}**", 'part4.C2 - part3a.A_only')
chk('Reading rules', 'R5 share and verdict', f"{share:.2f} of {gains['all']:,.0f} bits: **{'pass' if share > 0.5 else 'fail'}**", 'part5')
chk('Reading rules', 'R6 transfer, control, forgetting', f"transfer (A then B minus B only) {tr:+.4f}; sentence-shuffled control {trs:+.4f}; forgetting of A after B {forg:+.4f}", 'part3a/3b')
# part 6
for s in p6['settings']:
    w = '' if s['words_in_A_train_vocabulary'] is None else f"{s['words_in_A_train_vocabulary']:.3f}"
    chk('Part 6', f"hbar {s['hbar']:g} row metrics", f"{s['repeated_8byte_windows']:.3f} | {w} | {s['printable_bytes']:.3f} | {s['valid_utf8']:.3f} | {s['own_bits_per_byte']:.3f} |", 'part6.settings')
inst = next(s for s in p6['settings'] if s['hbar'] == 1.0 and s['mode'] == 'sample')
e3inst = [s for s in e3['part4']['settings'] if s['hbar'] == 1.0 and s['form'] == 'euclidean' and s['mode'] == 'sample' and not s['verified']][0]
for i, o in enumerate(inst['outputs']):
    chk('Part 6', f'prompt {i+1} E4 output verbatim', f"- E4 `{json.dumps(o['text'])}`", 'part6.settings.outputs')
    chk('Part 6', f'prompt {i+1} E3 output verbatim', f"- E3 `{json.dumps(e3inst['outputs'][i]['text'])}`", 'e3.part4.settings.outputs')
# README
rd = readme[readme.index('Continuation (2026-10-20'):]
chk('README', 'E3 A-test 2.131', f"reads A-test at {a3_['A_test']:.3f} bpb", 'e3.part3.arms.A_only.A_test', rd)
chk('README', 'E4 A-test 2.106', f"at {a4['A_test']:.3f}", 'part3a.A_only.A_test', rd)
chk('README', 'lzma 2.003', f"lzma's {lz:.3f}", 'e2.reference_lzma', rd)
rows.append(('README', 'E4 beats the equal-capacity control (R3)', f"{R3[0]:+.4f}, {R3[1]:+.4f}", 'part4.C1', 'matched' if min(R3) >= 0.005 else 'MISMATCHED'))
rows.append(('README', '"not from order": R4 fails', f"R4 gains {R4[0]:+.4f} (A-test) and {R4[1]:+.4f} (unseen books); the rule needs both at 0.005", 'part4.C2', 'QUALIFIED'))
rows.append(('README', '"not from long matches": R5 fails', f"share {share:.3f} < 0.5, but classes 16 and 32 hold {p5['E4']['16']['share'] + p5['E4']['32']['share']:.3f} of the bytes", 'part5', 'QUALIFIED'))
# AUDIT C2
au = audit[audit.index('## Checkpoint C2'):]
rows.append(('AUDIT C2', 'T4.2 and T4.3 exist in e3_t4.json', f"keys {sorted(k for k in t4 if k.startswith('T4'))}", 'e3_t4.json', 'matched' if 'T4_2_text' in t4 and 'T4_3_chain' in t4 else 'MISMATCHED'))
rows.append(('AUDIT C2', '"the inert halt is replaced in E4 by a raised admission bar"', f"design: yes; chosen configuration: scope {p2['chosen'].get('hier_scope', 'level')}, raise {p2['chosen'].get('hier_raise', False)}", 'part2.chosen', 'QUALIFIED'))
rows.append(('AUDIT C2', '"the cost of sinks is reported, not fixed"', f"E4 Part 2 ablation chose sink_rate {p2['chosen']['sink_rate']}", 'part2.chosen', 'QUALIFIED'))
json.dump(rows, open(sys.argv[1], 'w'))
print(len(rows), 'claims;', sum(r[4] == 'matched' for r in rows), 'matched;', sum(r[4] == 'MISMATCHED' for r in rows), 'mismatched;', sum(r[4] == 'QUALIFIED' for r in rows), 'qualified')
for r in rows:
    if r[4] != 'matched': print(r[4], r[0], '|', r[1], '|', r[2])
```
