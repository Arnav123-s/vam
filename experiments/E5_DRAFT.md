# E5 draft: does the taught machine use what it was taught? (design and protocol, not run)

Drafted 2026-10-21 by the engineer (task T7.1), before any E5 run. The selection and checking scripts were run to build the question set and the source table; the machine was not run. **Nothing here is registered:** `PROTOCOL.md` is not edited and no machine code changed. The draft becomes protocol only after gate G3 and the owner's approval (T7.2). If approved, it is copied into `PROTOCOL.md` as section E5 with its date, and changes after that are dated amendments.

Labels (T1.1): **S** = taken from a source (a named book, a stated catalogue, a stated prior experiment); **D** = derived by a rule or a computation from S and C, with the rule given; **C** = chosen by the designer, with the reason given. Every table row and rule below carries one.

## 1. Question and claim (S: manager's task, owner's instruction)

After one fixed teaching stream (dictionary, general prose, mathematics, physics, then named primary works and the authors' own accounts of their reasoning), does E4's chosen machine, unchanged, give lower cost to the correct answer of held-out questions that follow from the taught material than the same machine that was not taught stages 3 to 5?

Bound on the claim (S: T15.4): the state is a 64-bit register, its context statistics and E4's upper levels. A fact reaches an answer only if question and answer appear close together in a form like the teaching text. E5 therefore bounds "uses taught knowledge" from one side. It cannot show understanding. Either outcome is reported.

## 2. Rules kept (S: HANDOFF, T7.1, T10.1, T14, T15.3)

- The machine is E4's chosen configuration, read from `results/e4.json` `part2.chosen` (S): `{"machine": "hier", "law": "counting", "prior": 4.0, "coherence": 1.0, "governor": true, "protect": 0.5, "sink_rate": 0.0, "verify": 3, "hier_levels": "WS", "hier_segment": "text", "hier_verify": 4}`. Code at E4's head; no change.
- No scoring, path costs, beams, softmax or thresholds are added to the machine. The event loop is unchanged. Learning is forward only: no transcript, no replay, no second pass over any stage.
- Everything, the dictionary included, is read as raw bytes, as bits. Cleaning is E2's (`vam.corpus.clean_text`: UTF-8, `\n`, and the text between the `*** START OF` and `*** END OF` Project Gutenberg markers when both are present) and nothing else (S). Files with an older header format keep it: a scan (`data/e5/header_scan.json`, script A.e5_headerscan.py) finds the old "SMALL PRINT" boilerplate in 4 S2 files, 2 S4, 2 S5 and 2 C files, and Project Gutenberg text in all stages. Those bytes are taught like any other and are not in a question source or a stem (checks, section 5).
- The cost of an option is a measurement taken outside the machine: each bit is charged the probability the machine gave it before seeing it (PROTOCOL line 7; `m.run(bits, learning=False)` returns those probabilities).
- Numbers in any report come only from saved JSON that records the commit and seeds.

## 3. Curriculum, one continuous stream, no reset (C: order from the task; D: sizes)

Stages in order: **1** dictionary, **2** general prose, **3** mathematics, **4** physics, **5a** primary works, **5b** the authors' own accounts of their reasoning (5a and 5b are read as the single stage 5, 5a first). Stage 5 comes last, so recency favours it; this is a risk (R2), not a hidden choice.

### 3.1 Source table

Bytes are bytes of cleaned text. "Trained" excludes the held-out parts; "held out" is never read by the machine except in the frozen held-out measurement (section 8).

| Stage | Source | Licence | Rule | Bytes | SHA-256 | Flags |
|---|---|---|---|---|---|---|
| 1 dictionary | Webster's Unabridged Dictionary (text marked 1913 Webster, with a 1913 supplement), Project Gutenberg #29765 (PG-19 train-split copy; fixed by name, not by rule) | public domain (1913); PG-19 dataset Apache-2.0 | whole book, cut into 64 KiB blocks at line ends, every 20th block held out | 26585810 trained + 1376846 held out | cleaned file `7139d6bba7079f521662846364be0e72da8c30dfe1b02f481c3a1528bb5c0100` | licence not independently verified (PG header only); Webster has three PG-19 copies, #29765 chosen, the others excluded by the repeat rule |
| 2 general prose | PG-19 stage-B rule of E2 (`fetch_pg19.py`): title matches neither FORM nor SUBJECT; seeded shuffle, seed 20261018; English check; repeat rule; whole books until >= 32 MiB | public domain (US) per PG header; PG-19 dataset Apache-2.0 where fetched from PG-19 | 69 books trained, every 10th book held out (7 held out) | 30206436 trained + 3443380 held out | list hash `9cf9de102fed5dccaf1d811d327396557548a433f90d8c9366dd45478e094aba` (per-file hashes in `sources.json`) | 76 of 76 books are also in E2's MANIFEST.json (same rule and seed, so the same first books) |
| 3 mathematics | Project Gutenberg catalog, English texts with LoCC class QA, ascending ebook number, excluding stage-5 ids, foreign-language teaching, repeats, books under 60% letters (digit dumps); cap 24 MiB | public domain (US) per PG header; PG-19 dataset Apache-2.0 where fetched from PG-19 | 54 books trained, every 10th book held out (6 held out) | 16716924 trained + 2438489 held out | list hash `a77e8f2719f9e9b4e5d6b8c3c90ae5f062f97e9cb7afe2607feb4e24165e1bf6` (per-file hashes in `sources.json`) | supply-limited: 19155413 bytes available, under the cap; 53 QA books had no fetchable text; 6 books also in E2's MANIFEST |
| 4 physics | Project Gutenberg catalog, English texts with LoCC class QC, ascending ebook number, same exclusions, cap 24 MiB | public domain (US) per PG header; PG-19 dataset Apache-2.0 where fetched from PG-19 | 74 books trained, every 10th book held out (8 held out) | 22128698 trained + 3113447 held out | list hash `d3263ecb324094821f428cb0553a386c8ae3f85673ba5c4042ecbc368368409b` (per-file hashes in `sources.json`) | cap reached (last book overshoots); 10 QC books had no fetchable text; 7 books also in E2's MANIFEST |
| 5a primary | Newton, Principia (Motte translation, with the Life of Newton), PG #76404 | public domain (PG header) | named work; last 3% held out | cleaned file 1457485 B | `459ecf96ae4b8405d11bc79ac2ac9c71d77e9df609edcb0a96ecc14560e39572` | html (converted); licence from PG header only |
| 5a primary | Newton, Opticks (4th ed., 1730), PG #33504 | public domain (PG header) | named work; last 3% held out | cleaned file 567127 B | `38fc7c50ef7e802da45d8016cd32ad0281db6836f16a98ff38c8c8a85393aa52` | txt; licence from PG header only |
| 5a primary | Einstein, Relativity: The Special and General Theory (Lawson transl.), PG #30155 | public domain (PG header) | named work; last 3% held out | cleaned file 186921 B | `824ee60bad8ed05c7e876cc614fe734fdbeb78afd05d0befc3a4068f96f70c5d` | txt; licence from PG header only |
| 5a primary | Leonardo da Vinci, The Notebooks (Richter transl.; PG "Complete" = vols 1+2), PG #5000 | public domain (PG header) | named work; last 3% held out | cleaned file 1381410 B | `d9414526fd5181425aa27466ffe534f0ba4e5df10787726aa5933129b847ebb3` | txt; licence from PG header only |
| 5a primary | Euclid, The First Six Books of the Elements, ed. Casey, PG #21076 | public domain (PG header) | named work; last 3% held out | cleaned file 476650 B | `338bf7fa1cbcb6cbc4ed617b486fc5024b994d4f45a2bb0bbeb953effc68b547` | html (converted); licence from PG header only |
| 5a primary | Plato, Republic (Jowett), PG #1497 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 1195161 B | `816fa05bbb4baadfcfc1e03da2bb332ed0fae30c16e8393b74639da129883392` | PG-19 copy |
| 5a primary | Plato, Charmides (Jowett), PG #1580 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 121657 B | `9d8388d0007de3408f7c42557b8625f0bf547280e3ba442b9733b0d5434508ab` | PG-19 copy |
| 5a primary | Plato, Laches (Jowett), PG #1584 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 66688 B | `ac409cfd670cf7279f81d2b5f95ea4d1305d27ed5afbc5988e8717ba6ad3cbdc` | PG-19 copy |
| 5a primary | Plato, Euthydemus (Jowett), PG #1598 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 114304 B | `aecb2daa57a2c61b5b1aa650cb1de47770e4b245c8570ff3775bcdb8e50acfdf` | PG-19 copy |
| 5a primary | Plato, Cratylus (Jowett), PG #1616 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 299964 B | `2f29b47091fc37a48d5470b62d117a84c023042f1649f46a96385e6522e59f11` | PG-19 copy |
| 5a primary | Plato, Ion (Jowett), PG #1635 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 35490 B | `6a78795ff4c083762baea1618ff6849b2d4800785caba3054124e81bdcdd87b4` | PG-19 copy |
| 5a primary | Plato, Meno (Jowett), PG #1643 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 123020 B | `59c8ace60d24b61bf0453deb131371ae5ad9858cea46365d603829e0383c117e` | PG-19 copy |
| 5a primary | Plato, Apology (Jowett), PG #1656 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 85851 B | `8052b747de1fe25715ea96f48937df69521754a8ca83aacf92034100b8943b90` | PG-19 copy |
| 5a primary | Plato, Crito (Jowett), PG #1657 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 47257 B | `f0edc62362a155eb3fb0c8137d9b19e38bed9de7d6ede00b1d1a3a17b3898d85` | PG-19 copy |
| 5a primary | Plato, Phaedo (Jowett), PG #1658 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 232976 B | `3d31d86bbc5a1f6bed45850357ef843442b7b8b53d381e6692d5c37dd08d2b2c` | PG-19 copy |
| 5a primary | Plato, Gorgias (Jowett), PG #1672 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 328352 B | `ae83d821db959ae4c8b33419e7b754b37d04da42c3dacb4d35d2f318832551f5` | PG-19 copy |
| 5a primary | Plato, Parmenides (Jowett), PG #1687 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 197261 B | `66aeb8eeccd854420abcf39d7769fbd7281684f23087f6d06be9ffc8e505428e` | PG-19 copy |
| 5a primary | Plato, Philebus (Jowett), PG #1744 (PG-19) | public domain; PG-19 Apache-2.0 | one of the 13 Jowett dialogues in PG-19; last 3% held out | cleaned file 261522 B | `b18fd1cf01d2726ae6ba52b17ca9ef767f8081472177b3a5200815b617a6cb39` | PG-19 copy |
| 5b own accounts | Einstein, Sidelights on Relativity (Ether and Relativity; Geometry and Experience), PG #7333 | public domain (US) per PG header | added at the owner's request; last 3% held out | cleaned file 51862 B | `343b983ad9e86ed635dd4a6bfa63f51fb570a844bef53379dff566d582128af8` | gutenberg direct; licence from PG header only |
| 5b own accounts | Einstein, The Meaning of Relativity (four lectures, Princeton, 1921), PG #36276 | public domain (US) per PG header | added at the owner's request; last 3% held out | cleaned file 145229 B | `d62efc44dd0b7608bb051ee32cec626ff271fd81e6faf5572b4e32d99dcd3e6b` | gutenberg direct; licence from PG header only |
| 5b own accounts | Einstein, Fundamental Ideas and Problems of the Theory of Relativity (Nobel lecture), PG #69572 | public domain (US) per PG header | added at the owner's request; last 3% held out | cleaned file 23268 B | `c567dee8ff660929de929baef1b788fcfc3139ab0c5c13a9e53a34b64fbd95a1` | gutenberg direct; licence from PG header only |
| 5b own accounts | The Principle of Relativity (papers by Lorentz, Einstein, Minkowski, Weyl; several authors, not Einstein alone), PG #66944 | public domain (US) per PG header | added at the owner's request; last 3% held out | cleaned file 408764 B | `bc9d5acc5e2da31b165899a28ba7968f9d7d71fae11b2b0a1fd3bdcaa9359f70` | gutenberg direct; licence from PG header only |
| 5b own accounts | Leonardo da Vinci, Thoughts on Art and Life (extracts from the Notebooks; transl. Maurice Baring; the "Einstein" in the catalogue entry is Lewis Einstein, the editor, not Albert), PG #29904 | public domain (US) per PG header | added at the owner's request; last 3% held out | cleaned file 267795 B | `4e0d47bc68bb4aab78845487d607b613c4e5a41c1dac5f195a8f51f0202f7d69` | gutenberg direct; licence from PG header only |
| 5b own accounts | Xenophon, The Memorabilia (Dakyns): Xenophon's account of Socrates, not Socrates' or Plato's own words, PG #1177 | public domain (US) per PG header | added at the owner's request; last 3% held out | cleaned file 412124 B | `422d8edfb90b4e8e2e4d5880b8fe94df594584738cbaf8e04cedc196a5294634` | gutenberg direct; licence from PG header only |
| 5b own accounts | Xenophon, The Apology (Dakyns): Xenophon's account of Socrates' defence, PG #1171 | public domain (US) per PG header | added at the owner's request; last 3% held out | cleaned file 47107 B | `27c3595a9b455c4f5518920d04251245283520decabfdd443d1cf33b65449ba9` | gutenberg direct; licence from PG header only |

| C extra prose (replaces 3-5 in arm C) | PG-19 stage-B order continued after stage 2; same exclusions; last book cut at a line end so that bytes equal the trained bytes of stages 3-5 | 105 books, 47125859 bytes (= 47125859) | list hash `488313e65a1839ef49cd67cd9bc56f38a209bac9c44bc75729d86613b75c164e` | 69 of 105 books also in E2's MANIFEST |

Totals (D, from `sources.json`): trained bytes by stage `{"S1": 26585810, "S2": 30206436, "S3": 16716924, "S4": 22128698, "S5": 8280237, "C_extra": 47125859, "need": 47125859}`; held-out bytes by stage `{"S1": 1376846, "S2": 3443380, "S3": 2438489, "S4": 3113447, "S5": 255008}`. Trained bytes of one T arm: 103918105.

Notes on the table:
- Stage 5 is called by what it is. No work is called a diary. Newton and Leonardo left no separate diaries on Gutenberg; Leonardo's Notebooks are his own notes, and Newton's accounts of his method are in the Opticks Queries and the Principia's rules of reasoning, which are inside the works above.
- **Stage 5b was added at the owner's direct request during this task, not by the manager's specification.** The owner asked for the authors' own descriptions of their ideas and how they reached them. What exists in public domain on Gutenberg is the set above. Entries that are not what the label might suggest: #66944 is a collection by several authors; #29904 is extracts from the Notebooks, whose catalogue "Einstein" is Lewis Einstein, the editor; the Xenophon works are accounts of Socrates by Xenophon. No question is drawn from stage 5b (section 5), so 5b changes only what the machine has read.
- **Heath's Euclid is not on Project Gutenberg** (the catalogue has Casey's edition and Heath's Archimedes). Casey is used (owner decision D3).
- **Licences.** Every row's licence is the Project Gutenberg header and catalogue entry (public domain in the USA) and, for PG-19 copies, the dataset's Apache-2.0. None is independently verified outside the USA. Flagged unverified for owner review: all Einstein items (translations), #29904, and the converted-HTML works.
- **Frozen supply.** `sources.json` records the exact files and hashes. The run reads those files and never re-selects. This matters because the fetch depended on network state: Gutenberg returns 406 to clients that do not accept gzip, so the first fetch (`fetchlist.py`, urllib) marked 99 QA/QC books unavailable; `curl --compressed` (`fetch_html2.py`) recovered 36 of them, and 53 QA and 10 QC books still have no text at either URL. The fetch of every stage-5 and 5b file and of the dictionary is `fetch_s5.py`; re-running it on 2026-10-21 reproduced every cleaned-file hash in the table.
- **Digit filter (C, added after seeing the supply).** The first QA selection included files such as "Catalan's Constant to 1,500,000 places" and "The Value of Zeta(3)" (megabytes of digits). A rule was added before the final build: a book needs at least 60% letters among non-space characters. It dropped 13 QA books. It was chosen after seeing the data, which is why it is labelled C.
- **S3 is supply-limited** (all eligible QA books are used and the stage is under its 24 MiB cap); **S4 reached the cap** (the last book overshoots it).
- **Overlap with E2's `MANIFEST.json`** (D, `data/e5/overlap.json`, by id and normalised title; sources in MANIFEST / sources): S1: 0/1; S2: 76/76; S3: 6/60; S4: 7/82; C: 69/105; S5: 0/25. Stage 2 uses E2's stage-B rule and seed, so its books are E2's stage-B books. No stage-5 source and not the dictionary is in the MANIFEST. Overlap with the question sources is tested in section 5. Whether overlap with E2 matters is owner decision D6.

### 3.2 Rules (D unless marked)

- **S1** fixed by name (C).
- **S2** (D): PG-19 books whose title matches neither the instructional-form nor the school-subject pattern of `fetch_pg19.py`, in the seeded shuffle of seed 20261018 (S: E2), skipping ids excluded below, non-English books (E2's 12% function-word test) and repeats (same cleaned SHA-256 or same normalised title); whole books until at least 32 MiB.
- **S3, S4** (D): Project Gutenberg catalogue (`pg_catalog.csv`, fetched 2026-10-20), `Language=en`, `Type=Text`, LoCC class QA (mathematics) or QC (physics); ascending ebook number; skip stage-5 ids, other editions of stage-5 works, books for learning a foreign language (E2's FOREIGN pattern), repeats, non-English, digit-heavy; stop at 24 MiB. Book of a class in both goes to QA.
- **Exclusion list** (C): the ids of every stage-5 work and every other edition of one (`excl_ids.json`: 29765, 33504, the Plato dialogues, 30155, 5001, 36114, 28801, 76404, 5000, 4998, 4999, 21076 and the 5b ids), applied to S2, S3, S4 and C.
- **Held out** (C): S1 every 20th 64 KiB block; S2, S3, S4 every 10th book of the stage order (E2's rule); S5 the last 3% of each work, cut at a line end. A question whose source quote lies in a held-out part is dropped (this removed 3 items).
- **Why 32 MiB for stage 2 and not E2's 64 MiB per stage (C):** E4's A-train was 33.1 MB, a size at which E4's tables were at 49% global occupancy (`results/e4.json`, A-only arm). The whole T stream is 99 MiB. A 64 MiB stage 2 would put the total near 130 MiB and over an hour per arm for a stage that carries no question. Owner decision D1.

### 3.3 Arms (S: manager)

- **T (taught):** stages 1, 2, 3, 4, 5a, 5b.
- **C (control):** stages 1 and 2 as in T, then the "C extra prose" stream in place of stages 3 to 5. It has the same bytes as T's trained stages 3 to 5 (cut at a line end), is chosen by the same rule, uses the same seed, and is disjoint from T's prose by construction (it continues the order after stage 2). For the held-out measurement the C stream is paused at the byte offsets where T's stage 3, 4 and 5 end, with no reset.
- **C0 (optional, reference):** E4's end-of-A machine. `results/` holds only JSON, so the state would be rebuilt from A-train (about 10 minutes at E4's rate). Recommendation: skip (D8).

## 4. Machine facts the run needs, from the code (S: `vam/src/vam/nfa/hier.py`, `experiments/e4_hier.py`)

- Capacity counters: `counters(m)` in `e4_hier.py` returns `evictions`, `refused`, `protected`, per-level `occupancy` and `global_occupancy` (from `m.evictions`, `m.refused`, `m.protected`, `m.occupancy()`, `m.global_occupancy()`). They are recorded at the end of every stage, and again as deltas per stage.
- Register snapshot and restore without copying the 937 MiB tables: `m.state` returns the register list and ring state; `m.set_state(list)` restores it. `m.copy()` duplicates `_wn`, `_tags`, `_door`, `_occ`, `_stats` and `_st`. That frozen reads leave the tables alone, and that `state`/`set_state` is a complete snapshot, are **read from the code and not yet tested**; they are pre-run gates in section 6.
- `m.run(bits, learning=False)` returns per-bit probabilities of the bits read. Cost of a byte string = `-log2` sum over its bits.

## 5. Questions (S for quotes, C for questions and distractors, D for checks)

File: `data/e5/questions.json`; SHA-256 `2706322fb9c371ff0c12cfc302bb5f8a193bf06bb650c091076e9467892d5e62` (also in `questions.sha256`). Authored 2026-10-20 and frozen 2026-10-21, before any run. The file records, per item: id, kind, source work, stem, four options, answer index, option byte length, the source quotes with locators, the written-out reasoning, the key terms, the largest overlap of the stem with any teaching text, the option counts in T and in C, and whether the answer is the most frequent option in T.

**Kinds.** Recall (a): the stem is a lead sentence plus the words just before a fact in the source, the options are four strings of equal byte length (numbers or words from the same work or same shape), one correct. Inference (b): the stem gives a premise of the taught material and asks for a quantity that follows by a one-step computation from two taught facts (or one fact and one definition); the options are equal-length numbers or phrases. The written-out chain of each is in `reasoning`. After the checks: **79 recall and 61 inference questions** from Opticks, the Notebooks, the Principia and the Life of Newton, Relativity, Euclid (Casey) and Plato's dialogues. No question comes from stage 5b, from the dictionary, mathematics or physics.

**Common knowledge.** Facts are chosen for being specific to these editions (Newton's measured numbers, Leonardo's rules in braccia, the Life of Newton's dates, Casey's wording of terms, Jowett's wording). That a fact is not common knowledge is a judgement, not a check; the stem-free baseline (section 6) measures how far the options alone give it away.

**Mechanical checks (D, `e5_questions.py`, log in `data/e5/checks.json` and `dropped.json`; thresholds fixed before they were applied):**
1. The four options have equal byte length and are distinct (all 166 authored items pass).
2. The exact stem and every stem plus option, whitespace-normalised, occur in no stage text of T and not in the C extra prose (all pass).
3. Largest common substring (bytes) between the stem and any stage text of T or C: at most 80 for recall and 48 for inference. This bounds how much of a question can be a verbatim continuation of a passage; items above it are dropped.
4. Every source quote of an item is found in the trained part of its stage-5 text (not held out, not in S1 to S4, not in C).
5. For inference: the answer string is searched, as a whole word, in all T text and in C within 100 bytes of any of the item's key terms. For answers of at least 4 bytes, **any hit in T, including one in the item's own source quote, drops the item** (this removed I-PLA-02, I-PLA-03, I-PLA-04, I-PLA-13 and I-EUC-08). Answers under 4 bytes ("two", "ten", "six", short numerals) are exempt, because they co-occur with any key by chance; counts for all items are logged. **14 of the 61 kept inference items are exempt** (I-NOT-01, I-NOT-03, I-PLA-08, I-PLA-10, I-PLA-11, I-PLA-12, I-PLA-14, I-EUC-03, I-EUC-09, I-EUC-10, I-EUC-11, I-EUC-12, I-OPT-07, I-PRI-12); I-PLA-08 and I-EUC-10 have hundreds of key co-occurrences in T and pass only through the exemption.
6. Option counts in T and in C are logged. They are not a criterion; they define the familiarity subsets in section 7.

Authored 166 (92 recall, 74 inference); dropped 26: R-OPT-05 (lcs>80); R-OPT-09 (lcs>80); R-OPT-10 (lcs>80); R-NOT-12 (lcs>80); R-PRI-03 (lcs>80); R-PRI-09 (lcs>80); R-PRI-12 (lcs>80); R-PRI-13 (lcs>80); I-PRI-04 (lcs>48); I-PLA-02 (cooccurrence); I-PLA-03 (cooccurrence); I-PLA-04 (cooccurrence); R-REL-05 (quote-not-in-trained-S5); R-EUC-11 (lcs>80); R-EUC-14 (lcs>80); R-EUC-19 (lcs>80); R-EUC-20 (lcs>80); I-EUC-01 (lcs>48); I-EUC-04 (lcs>48); I-EUC-08 (cooccurrence); I-NOT-14 (lcs>48); I-NOT-16 (lcs>48); I-PRI-09 (lcs>48); I-REL-08 (quote-not-in-trained-S5); I-PLA-13 (cooccurrence); I-REL-10 (quote-not-in-trained-S5). Kept **79 recall, 61 inference.** The answer is the most frequent option in T for 23 of the recall and 11 of the inference items.

**Positions** are shuffled once with `random.Random(f"{seed}/{id}")`, seed 20261020, and fixed in the file.

## 6. Measurement (D, specified, not run)

For each arm, at the end of the stream (after all stages, learning frozen from then on):
0. **Pre-run gate (both must pass before any question is scored):** (a) on a few questions, the cost of each option through `set_state(S_end)` equals, bit for bit, its cost on a `m.copy()` of the end-of-stream machine; (b) a SHA-256 of each of `_wn`, `_tags`, `_door` and `_occ` is equal before the first frozen read and after the last (the doorkeeper and occupancy arrays are the ones that could move). If either fails the measurement is redesigned before the run.
1. Take `S_end = m.state`. Define the fixed neutral preamble `PRE = b"\n\nQuestion and answer.\n\n"` (C): the W and S levels do not flush at `\n\n`, so the last bytes of the previous stage (T: the 5b text; C: prose) otherwise leak into every question. PRE is identical in both arms.
2. For each question and each option `o`: `m.set_state(S_end)`; read `PRE + stem` with `learning=False` and discard the probabilities; read `o`'s bytes with `learning=False`; the **cost** is the sum of `-log2 p` over the bits of `o` only. Option lengths are equal by construction, so bits and bits per byte rank options identically; both are saved.
3. The chosen option is the one with the lowest cost. A tie counts as wrong. Nothing is tuned: no scaling, no priors, no calibration.
4. **Stem-free baseline (pre-registered):** the same with `PRE + o` and no stem. The share of items the baseline already gets right, per arm and kind, is reported; the stem's contribution is `cost(o | stem) - cost(o | no stem)`.
5. Save per item and per arm: the four costs, the baseline costs, the choice. All reported figures are computed from that JSON.

Held-out and capacity records (**measurements, not criteria**): at the end of each stage (and, for C, at the matching byte offsets), frozen bits per byte on the held-out passages of every stage up to and including it (total 34.5 MB of frozen reads); `counters(m)` and its per-stage deltas; trained bytes; seconds. This gives, per stage, transfer to unseen passages, forgetting of earlier stages, and how much capacity the stream used.

## 7. Criteria (D from the numbers below; C for the choices)

Unit: one question = one trial. Chance is 1/4 (four options, ties wrong, so chance is at most 1/4). All tests are exact; one-sided; alpha = 0.025 per kind (Bonferroni over the two kinds, so the family is 0.05). The three criteria are reported separately and are not combined.

- **P1 (recall, taught above chance):** T is correct on at least X of the N = 79 recall questions, X = 28: the smallest count whose binomial tail under p = 0.25 is at most 0.025 (`stats.json`). It is the same test for inference with N = 61, X = 23, reported as P1-inference (descriptive; the manager's P1 is on recall).
- **P2 (recall, T beats C):** among questions where exactly one arm is correct, T is the correct one at least as often as an exact sign test at alpha = 0.025 requires, **and** T's accuracy minus C's is at least D = 0.10 (8 of 79 items).
- **P3 (inference, T beats C):** the same on the 61 inference questions, D = 0.10 (7 of 61 items).

**Why these N, X, D.** N is what survived the checks; the recommended 60 per kind is met, inference only just (61). X comes from the exact binomial: 28 of 79 is 35.4%, 23 of 61 is 37.7%. D: with about 20 discordant items the smallest gap the sign test can reject is 10 items (12.7% of 79), so the sign test is the binding criterion and D = 0.10 only forbids a "significant" but trivial gap. **Power** (exact, T and C independent, C at chance 0.25; `stats.json`): recall `{"T=0.3,C=0.25": 0.073, "T=0.35,C=0.25": 0.217, "T=0.4,C=0.25": 0.449, "T=0.45,C=0.25": 0.696, "T=0.5,C=0.25": 0.875, "T=0.6,C=0.25": 0.993}`; inference `{"T=0.3,C=0.25": 0.059, "T=0.35,C=0.25": 0.163, "T=0.4,C=0.25": 0.339, "T=0.45,C=0.25": 0.557, "T=0.5,C=0.25": 0.757, "T=0.6,C=0.25": 0.966}`. A true T accuracy of 0.40 is detected with probability 0.45 (recall) and 0.34 (inference); 0.50 with 0.88 and 0.76. A null here is a statement about gaps larger than about 15 points.

**Pre-registered secondary analyses** (reported with the criteria, never replacing them): (i) P2 and P3 on the subset where the answer is not the most frequent option in T (familiarity-matched subset: the numbers in section 5 minus the most-frequent ones); (ii) the stem-free baseline accuracy per arm; (iii) per source work, descriptive only; (iv) accuracy and the T minus C gap with the 28 items that have a distractor absent from all of T excluded (the `option_counts_T` field); (v) P3 on the 47 inference items whose answers are at least 4 bytes (the items that went through the co-occurrence check).

## 8. Expectation (S: T15.4; the manager's wording)

Recall above C is possible: the stem is close in form to the teaching text, and E4's byte and word levels can carry a fact over a few dozen bytes. Inference is expected near chance: the premises are in the stem, the answer is a computed quantity that usually appears nowhere in the teaching text, and nothing in the machine adds or compares numbers. Both results are reported. If T beats C on inference, the first suspicion is exposure of the answer's digits and words (section 7 secondary analyses), not reasoning.

## 9. Budget (D from saved E4 data)

- Rate: E4 Part 3a read 112.3 MB of training (A-train once, then A-train and B-train) and took 2105.4 s with its evaluations (`results/e4.json` `part3a.seconds`): 18.8 s/MB all-in. Part 2 rows took 531 to 690 s.
- One arm: 103918105 trained bytes plus 34.5 MB of frozen held-out reads at the same rate: about 2600 s, 45 minutes. The questions add 140 x 5 short reads (about 0.9 MB of bits): seconds. Two arms about 1.5 hours; C0 would add about 10 minutes.
- Memory: E4's recorded peak is about 5.0 GiB (HANDOFF T6.4); the register snapshots add kilobytes. Stop above 10 GiB.
- One job at a time. A stage that runs past 1 hour is stopped and a dated amendment written.

## 10. Owner decisions, each with a recommendation

- **D1. Stage sizes.** 32 MiB general prose, 24 MiB caps for mathematics and physics (E2 used 64 MiB per stage). Recommend as drafted (about 99 MiB total); raising stage 2 to 64 MiB adds about 10 minutes per arm and moves no question.
- **D2. Mathematics supply.** Only 19.2 MB of mathematics qualifies by the QA rule. Recommend accepting it (a smaller stage 3 is a fact about the supply) rather than adding books by hand.
- **D3. Casey instead of Heath for Euclid.** Heath's *Elements* is not on Project Gutenberg. Recommend Casey (public domain, on Gutenberg, quotes verified). If the owner wants Heath, a file with a verified licence must be supplied and new questions authored.
- **D4. Plato.** The 13 Jowett dialogues in PG-19. Recommend these; the other Jowett dialogues (Timaeus, Symposium, Theaetetus, Laws and others) are on Gutenberg and can be added at the cost of the 5a size, with no question drawn from them.
- **D5. Stage 5b (the authors' own accounts).** Included at the owner's request. Recommend keeping it as drafted: it carries the authors' descriptions of method (Einstein's lectures, Xenophon's Socrates, Leonardo's thoughts) but no question depends on it. If the owner wants questions about stated reasoning, a new batch must be written and frozen before the run, and this draft's hash changes.
- **D6. Overlap with E2.** Stage 2 is E2's stage B; some stage 3 and 4 books are in E2's MANIFEST. Recommend accepting: E5 never uses E2's held-out books as a test, and the overlap is logged (`overlap.json`). Alternative: exclude MANIFEST ids (cost: stage 2 starts later in the same order).
- **D7. Digit filter.** Recommend keeping it, labelled C.
- **D8. C0.** Recommend skipping (no saved E4 end-of-A machine in `results/`; the comparison that answers the question is T against C).
- **D9. Where the fetched texts live.** About 270 raw files and the built streams are not in git. Recommend committing hashes only (`sources.json`) and keeping one tarball outside git; the run verifies every hash before reading.
- **D10. Alpha and multiplicity.** 0.025 per kind, one-sided. Recommend as drafted.
- **D11. Licences.** Recommend the owner confirm the items flagged in section 3.1 (Einstein translations, #29904, #66944) or drop stage 5b.

## 11. Risks, each with the cheapest check

| # | Risk | Cheapest check |
|---|---|---|
| R1 | Table saturation: 99 MiB into E4's tables (49% global occupancy after 33 MB, 1.2e9 refusals in E4's A-only arm); late facts may be refused or evicted | The per-stage counters in section 6, read before any question is scored; a run on the first 40 MiB of T shows the trend |
| R2 | Recency: stage 5 is last, so T's facts are the freshest; T versus C does not separate "taught" from "recent" | Report accuracy by source position in stage 5; a later T-reordered arm (stage 5 first) is the clean control and is not in this draft |
| R3 | Familiarity or frequency: answers may be cheap because of their string frequency, not their link to the stem | Stem-free baseline and the familiarity-matched subset (section 7) |
| R4 | Leak through state: last bytes of the previous stage condition every question | The fixed preamble and the restore from `S_end`; check that the cost of a fixed neutral string is the same after each restore |
| R5 | Frozen reads change tables or the snapshot is incomplete | The two pre-run gates of section 6 |
| R6 | Question text taught elsewhere (another edition, an excerpt, a quotation inside a stage 3 or 4 book) | The exact-substring and largest-overlap checks already ran against every stage text and C; rerun on any change of sources |
| R7 | Distractors unseen in T make the answer cheap by familiarity | Option counts logged; the 28-item exclusion analysis |
| R8 | Inference items test arithmetic the machine cannot do, so a gap reflects digit exposure | Section 7 secondary analyses; read P3 with them |
| R9 | Power: inference N is 61; a 40% true accuracy is detected 34% of the time | State the detectable gap with the result; add questions before the run only by a dated amendment |
| R10 | Licences unverified; converted HTML (Principia, Euclid, three Einstein items) may carry conversion artefacts | Owner confirmation; the converter is in the appendix and the converted text is hashed |
| R11 | Network dependence of the fetch | `sources.json` freezes files; verify hashes before the run |
| R12 | Time or memory over budget | One arm first; stop rules in section 9 |
| R13 | The Life of Newton is a nineteenth-century biography inside the Principia file, not Newton's words | Report per source work |

## 12. Reproduction

All scripts are inline in the appendix; none is stored under `experiments/`. Order: `fetchlist.py` (catalogue, `qaqc.json` and the first QA/QC fetch), `fetch_html2.py` (recovers books `fetchlist.py` missed), `fetch_s5.py` (dictionary, stage-5 and 5b files), `prose_fetch.py` (prose; needs `excl_ids.json`, listed below), `conv.py` (HTML to text, imported by the fetchers), `e5_texts.py`, `e5_assemble.py`, `e5_overlap.py` (streams), the `items_*.py` files (questions, authored), `e5_questions.py` (checks), `e5_freeze.py` (file and hash), `e5_stats.py`, `e5_headerscan.py`. `excl_ids.json` is the list in section 3.2. Paths in the scripts are those of the drafting machine (`/path/to/e5-work`). Scratch outputs not committed: the fetched texts and built streams (see D9). Committed: this file and `experiments/data/e5/` (`questions.json`, `questions.sha256`, `checks.json`, `dropped.json`, `overlap.json`, `stats.json`, `header_scan.json`, `sources.json`).

## Appendix: scripts (as run on 2026-10-20 and 2026-10-21 to produce the files above)

### A.conv.py

```python
import re, html, sys
sys.path.insert(0,'/path/to/vam/src')
from html.parser import HTMLParser
from vam.corpus import clean_text
BLOCK={'p','div','br','h1','h2','h3','h4','h5','h6','li','tr','table','blockquote','pre','section','hr','dd','dt','center'}
class P(HTMLParser):
    def __init__(s): super().__init__(convert_charrefs=True); s.out=[]; s.skip=0
    def handle_starttag(s,t,a):
        if t in('script','style','head'): s.skip+=1
        if t in BLOCK: s.out.append('\n')
    def handle_endtag(s,t):
        if t in('script','style','head'): s.skip=max(0,s.skip-1)
        if t in BLOCK: s.out.append('\n')
    def handle_data(s,d):
        if not s.skip: s.out.append(d)
def html_to_text(raw: bytes) -> bytes:
    p=P(); p.feed(raw.decode('utf-8','replace'))
    t=''.join(p.out).replace('\xa0',' ')
    t=re.sub(r'[ \t]+',' ',t); t=re.sub(r' ?\n ?','\n',t); t=re.sub(r'\n\s*\n(\s*\n)+','\n\n',t)
    return t.strip('\n').encode('utf-8')+b'\n'
def build(name, path, kind):
    raw=open(path,'rb').read()
    if kind=='html':
        t=html_to_text(raw)
        txt=t.decode('utf-8')
        m1=re.search(r'\*\*\* START OF.*?\*\*\*',txt); m2=re.search(r'\*\*\* END OF.*?\*\*\*',txt)
        if m1 and m2: txt=txt[m1.end():m2.start()]
        out=txt.strip('\n').encode('utf-8')+b'\n'
    else:
        out=clean_text(raw)
    open(f'clean/{name}.txt','wb').write(out); return len(raw),len(out)
if __name__=='__main__':
    for name,path,kind in [('principia','raw/gut_76404_principia.html','html'),('euclid_casey','raw/gut_21076_euclid_casey.html','html'),('relativity','raw/gut_30155_relativity.txt','txt'),('notebooks','raw/gut_5000_leonardo_notebooks.txt','txt'),('opticks','raw/gut_33504_opticks.txt','txt'),('webster','raw/gut_29765_webster.txt','txt')]:
        print(name,build(name,path,kind))
```

### A.fetchlist.py

```python
import csv, json, os, sys, urllib.request, hashlib, time
rows=list(csv.DictReader(open('pg_catalog.csv',encoding='utf-8')))
where=json.load(open('where.json'))
os.makedirs('stg',exist_ok=True)
def want(r,cls):
    return r['Language']=='en' and r['Type']=='Text' and cls in [x.strip() for x in (r['LoCC'] or '').split(';')]
out={}
for cls in ('QA','QC'):
    out[cls]=[r for r in rows if want(r,cls)]
json.dump({k:[{'id':r['Text#'],'title':r['Title'],'issued':r['Issued'],'authors':r['Authors'],'locc':r['LoCC'],'in_pg19':r['Text#'] in where} for r in v] for k,v in out.items()},open('qaqc.json','w'))
print({k:(len(v),sum(r['Text#'] in where for r in v)) for k,v in out.items()})
BK="https://storage.googleapis.com/deepmind-gutenberg"
def get(u):
    for k in range(3):
        try:
            with urllib.request.urlopen(u,timeout=120) as r: return r.read()
        except Exception as e: err=e; time.sleep(1+k)
    return None
res=[]
for cls,v in out.items():
    for r in v:
        i=r['Text#']; p=f'stg/{i}.txt'
        if not os.path.exists(p):
            b=get(f"{BK}/{where[i]}/{i}.txt") if i in where else None
            src='pg19' if b is not None else None
            if b is None:
                for u in (f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt",f"https://www.gutenberg.org/ebooks/{i}.txt.utf-8"):
                    b=get(u)
                    if b: src='gutenberg'; break
            if b is None: res.append({'id':i,'cls':cls,'src':None}); continue
            open(p,'wb').write(b); open(p+'.src','w').write(src)
        else: src=open(p+'.src').read()
        b=open(p,'rb').read()
        res.append({'id':i,'cls':cls,'src':src,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'title':r['Title'][:90]})
json.dump(res,open('qaqc_fetch.json','w'))
print('done',len(res),sum(x['src'] is None for x in res),'unavailable')
```

### A.fetch_html2.py

```python
import json,sys,os,re,subprocess
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'.')
import conv
fet=json.load(open('qaqc_fetch.json'))
ids=[r['id'] for r in fet if r['src'] is None and not os.path.exists(f"stg/{r['id']}.txt")]
def GET(u):
    for k in range(2):
        r=subprocess.run(['curl','-s','-f','--compressed','--max-time','90',u],capture_output=True)
        if r.returncode==0: return r.stdout
        if r.returncode==22: return None
    return None
def one(i):
    b=GET(f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt"); kind='txt'
    if b is None: b=GET(f"https://www.gutenberg.org/cache/epub/{i}/pg{i}-images.html"); kind='html'
    if b is None: return i,None
    t=conv.html_to_text(b).decode('utf-8') if kind=='html' else b.decode('utf-8-sig').replace('\r\n','\n')
    m1=re.search(r'\*\*\* START OF.*?\*\*\*',t); m2=re.search(r'\*\*\* END OF.*?\*\*\*',t)
    if m1 and m2 and m1.end()<m2.start(): t=t[m1.end():m2.start()]
    open(f'stg/{i}.txt','wb').write(t.strip('\n').encode()+b'\n'); open(f'stg/{i}.txt.src','w').write('gutenberg-'+kind)
    return i,len(t)
with ThreadPoolExecutor(8) as ex:
    for i,n in ex.map(one,ids): print(i,n,flush=True)
print('done')
```

### A.fetch_s5.py

```python
"""Fetch and clean the stage-5 works (5a, 5b) and the dictionary; prints each cleaned file's SHA-256. Usage: python3 fetch_s5.py OUTDIR"""
import sys, os, re, json, hashlib, subprocess
sys.path.insert(0,'/path/to/vam/src'); sys.path.insert(0,'/path/to/e5-work')
from vam.corpus import clean_text
import conv
OUT=sys.argv[1]; os.makedirs(OUT,exist_ok=True)
BK="https://storage.googleapis.com/deepmind-gutenberg"
where=json.load(open('/path/to/e5-work/where.json'))
def curl(u):
    r=subprocess.run(['curl','-s','-f','--compressed','--max-time','180',u],capture_output=True)
    return r.stdout if r.returncode==0 else None
def gut(i,kind):
    return curl(f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt") if kind=='txt' else curl(f"https://www.gutenberg.org/cache/epub/{i}/pg{i}-images.html")
def clean_html(b):
    t=conv.html_to_text(b).decode('utf-8'); m1=re.search(r'\*\*\* START OF.*?\*\*\*',t); m2=re.search(r'\*\*\* END OF.*?\*\*\*',t)
    if m1 and m2: t=t[m1.end():m2.start()]
    return t.strip('\n').encode()+b'\n'
JOBS=[('webster','gut','29765','txt'),('principia','gut','76404','html'),('euclid_casey','gut','21076','html'),('relativity','gut','30155','txt'),('notebooks','gut','5000','txt'),('opticks','gut','33504','txt')]
JOBS+=[(n,'pg19',i,'txt') for i,n in (('1497','republic'),('1580','charmides'),('1584','laches'),('1598','euthydemus'),('1616','cratylus'),('1635','ion'),('1643','meno'),('1656','apology'),('1657','crito'),('1658','phaedo'),('1672','gorgias'),('1687','parmenides'),('1744','philebus'))]
JOBS+=[(n,'gut',i,k) for i,n,k in (('7333','einstein_sidelights','txt'),('36276','einstein_meaning','html'),('69572','einstein_fundamental','html'),('66944','einstein_principle','txt'),('29904','leonardo_thoughts','txt'),('1177','xen_memorabilia','txt'),('1171','xen_apology','txt'))]
for name,src,i,kind in JOBS:
    raw=curl(f"{BK}/{where[i]}/{i}.txt") if src=='pg19' else gut(i,kind)
    if raw is None: print(name,'FETCH FAILED'); continue
    c=clean_html(raw) if kind=='html' else clean_text(raw)
    open(f'{OUT}/{name}.txt','wb').write(c); print(name,len(c),hashlib.sha256(c).hexdigest())
```

### A.prose_fetch.py

```python
import csv,json,random,re,sys,hashlib,time,urllib.request,os
sys.path.insert(0,'/path/to/vam/experiments'); sys.path.insert(0,'/path/to/vam/src')
import fetch_pg19 as F
from vam.corpus import clean_text
meta=open('metadata.csv',encoding='utf-8').read().splitlines()
rows=[r for r in (l.split(',') for l in meta) if len(r)>=4]
rows=[[r[0],','.join(r[1:-2]),r[-2],r[-1]] for r in rows]
where=json.load(open('where.json'))
rows=[r for r in rows if r[0] in where]
titled={r[0] for r in rows if F.FORM.search(r[1]) or F.SUBJECT.search(r[1])}
b=sorted((r for r in rows if r[0] not in titled),key=lambda r:int(r[0]))
random.Random(F.SEED).shuffle(b)
json.dump([r[0] for r in b],open('prose_order.json','w'))
EXCL=set(json.load(open('excl_ids.json'))) if os.path.exists('excl_ids.json') else set()
used=0; log=[]
for ebook,title,year,url in b:
    if used>=100*2**20: break
    if ebook in EXCL: continue
    p=f'prose/{ebook}.txt'
    if os.path.exists(p): raw=open(p,'rb').read()
    else:
        raw=None
        for k in range(3):
            try: raw=urllib.request.urlopen(f"{F.BUCKET}/{where[ebook]}/{ebook}.txt",timeout=60).read(); break
            except Exception: time.sleep(2)
        if raw is None: continue
        open(p,'wb').write(raw)
    used+=len(raw); print(ebook,len(raw),title[:50],flush=True)
```

### A.e5_texts.py

```python
"""Stage texts for E5 (selection by rule). Output: stages.json with ordered files, bytes, SHA-256 per stage."""
import json, re, hashlib, sys, os, glob
sys.path.insert(0,'/path/to/vam/experiments'); sys.path.insert(0,'/path/to/vam/src')
import fetch_pg19 as F
from vam.corpus import clean_text
E5='/path/to/e5-work/'
MiB=2**20
S2_MIB=32; M_CAP=24; Q_CAP=24
def sha(b): return hashlib.sha256(b).hexdigest()
def normkey(title): return re.sub(r'[^a-z0-9]','',title.lower())
EXC=set(json.load(open(E5+'excl_ids.json')))
stages={}
# S1 dictionary
web=open(E5+'clean/webster.txt','rb').read()
stages['S1']=[dict(id='29765',file='clean/webster.txt',bytes=len(web),sha256=sha(web))]
# S3/S4 by LoCC (QA then QC), ascending ebook number, caps
cat=json.load(open(E5+'qaqc.json'))
fet=[]
for cls in ('QA','QC'):
    for r in cat[cls]:
        p=E5+f"stg/{r['id']}.txt"
        src=open(p+'.src').read() if os.path.exists(p) else None
        fet.append(dict(id=r['id'],cls=cls,src=src,title=r['title'].replace('\n',' ')))
seen_sha={sha(web)}; seen_title=set(); taken=set(); log={'skipped':[]}
def eligible(rec):
    if rec['src'] is None: return 'unavailable'
    if rec['id'] in EXC: return 'stage-5/other'
    if F.FOREIGN.search(rec['title']): return 'foreign'
    raw=open(E5+f"stg/{rec['id']}.txt",'rb').read()
    if not F.english(raw): return 'not english'
    c=clean_text(raw)
    t=c.decode('utf-8','replace'); ns=[ch for ch in t if not ch.isspace()]
    if sum(ch.isalpha() for ch in ns)<0.6*len(ns): return 'digit-heavy'
    if sha(c) in seen_sha or normkey(rec['title']) in seen_title: return 'repeat'
    return None
for cls,cap,name in (('QA',M_CAP,'S3'),('QC',Q_CAP,'S4')):
    used=0; lst=[]
    for rec in sorted((r for r in fet if r['cls']==cls and r['id'] not in taken),key=lambda r:int(r['id'])):
        if used>=cap*MiB: break
        why=eligible(rec)
        if why: log['skipped'].append((rec['id'],cls,why)); continue
        c=clean_text(open(E5+f"stg/{rec['id']}.txt",'rb').read())
        p=E5+f"build/{name}_{rec['id']}.txt"; open(p,'wb').write(c)
        seen_sha.add(sha(c)); seen_title.add(normkey(rec['title'])); taken.add(rec['id']); used+=len(c)
        lst.append(dict(id=rec['id'],title=rec['title'],file=p,bytes=len(c),sha256=sha(c),src=rec['src']))
    stages[name]=lst
json.dump({'stages':stages,'log':log},open(E5+'build/stages_partial.json','w'),indent=1)
print({k:(len(v),sum(x['bytes'] for x in v)) for k,v in stages.items()}, len(log['skipped']))
```

### A.e5_assemble.py

```python
"""Assemble the E5 stage streams (T and C), held-out passages, and the sources record."""
import json, re, sys, os, hashlib
sys.path.insert(0,'/path/to/vam/experiments'); sys.path.insert(0,'/path/to/vam/src')
import fetch_pg19 as F
from vam.corpus import clean_text
E5='/path/to/e5-work/'; B=E5+'build/'; MiB=2**20
S2_MIB=32
def sha(b): return hashlib.sha256(b).hexdigest()
def nk(t): return re.sub(r'[^a-z0-9]','',t.lower())
part=json.load(open(B+'stages_partial.json'))['stages']
meta={}
for l in open(E5+'metadata.csv',encoding='utf-8').read().splitlines():
    r=l.split(',')
    if len(r)>=4: meta[r[0]]=','.join(r[1:-2])
EXC=set(json.load(open(E5+'excl_ids.json')))|{x['id'] for s in ('S3','S4') for x in part[s]}
order=json.load(open(E5+'prose_order.json'))
seen_sha=set(); seen_t=set(); prose=[]
for i in order:
    if i in EXC or not os.path.exists(E5+f'prose/{i}.txt'): continue
    raw=open(E5+f'prose/{i}.txt','rb').read()
    if not F.english(raw): continue
    c=clean_text(raw)
    if sha(c) in seen_sha or nk(meta[i]) in seen_t: continue
    seen_sha.add(sha(c)); seen_t.add(nk(meta[i])); prose.append((i,meta[i],c))
def holdout_books(lst): return [(x,k%10==9) for k,x in enumerate(lst)]   # every tenth book is held out
# ---- S2: whole books until >= 32 MiB of (train+held-out) bytes
s2=[];used=0
for p in prose:
    if used>=S2_MIB*MiB: break
    s2.append(p); used+=len(p[2])
rest=prose[len(s2):]
out={'T':{},'C':{},'heldout':{},'sources':{}}
def wr(name,b): 
    open(B+name,'wb').write(b); return B+name
# S1: Webster in blocks of 64 KiB cut at line ends; every 20th block held out
web=open(E5+'clean/webster.txt','rb').read()
blocks=[];i=0
while i<len(web):
    j=min(len(web),i+65536); k=web.find(b'\n',j); j=len(web) if k<0 else k+1; blocks.append(web[i:j]); i=j
tr=b''.join(b for n,b in enumerate(blocks) if n%20!=19); ho=b''.join(b for n,b in enumerate(blocks) if n%20==19)
out['T']['S1']=[wr('S1_train.txt',tr)]; out['heldout']['S1']=[wr('S1_heldout.txt',ho)]
def stage_books(name,lst):
    tr=[];ho=[]
    for n,(i,t,c) in enumerate(lst):
        (ho if n%10==9 else tr).append(wr(f'{name}_{i}.txt',c) if not os.path.exists(B+f'{name}_{i}.txt') else B+f'{name}_{i}.txt')
    return tr,ho
t,h=stage_books('S2p',s2); out['T']['S2']=t; out['heldout']['S2']=h
for name in ('S3','S4'):
    lst=[(x['id'],x['title'],open(x['file'],'rb').read()) for x in part[name]]
    t,h=[],[]
    for n,(i,tt,c) in enumerate(lst): (h if n%10==9 else t).append(B+f'{name}_{i}.txt')
    out['T'][name]=t; out['heldout'][name]=h
# S5: primary works + accounts; last 3% of each work held out (cut at a line end)
S5=[('principia','clean/principia.txt'),('opticks','clean/opticks.txt'),('relativity','clean/relativity.txt'),('notebooks','clean/notebooks.txt'),('euclid_casey','clean/euclid_casey.txt')]
S5+=[(n,f'clean/pg19_{i}_{n}.txt') for i,n in (('1497','republic'),('1580','charmides'),('1584','laches'),('1598','euthydemus'),('1616','cratylus'),('1635','ion'),('1643','meno'),('1656','apology'),('1657','crito'),('1658','phaedo'),('1672','gorgias'),('1687','parmenides'),('1744','philebus'))]
S5b=[(n,f'clean5b/{n}.txt') for n in ('einstein_sidelights','einstein_meaning','einstein_fundamental','einstein_principle','leonardo_thoughts','xen_memorabilia','xen_apology')]
out['T']['S5']=[];out['heldout']['S5']=[]
for grp,lst in (('5a',S5),('5b',S5b)):
    for n,p in lst:
        b=open(E5+p,'rb').read(); cut=int(len(b)*.97); k=b.find(b'\n',cut); cut=len(b) if k<0 else k+1
        out['T']['S5'].append(wr(f'S5{grp}_{n}.txt',b[:cut])); 
        if cut<len(b): out['heldout']['S5'].append(wr(f'S5{grp}_{n}_ho.txt',b[cut:]))
# C: the next prose books, same bytes as the trained S3+S4+S5 bytes, cut at a line end
need=sum(os.path.getsize(f) for s in ('S3','S4','S5') for f in out['T'][s])
cf=[];got=0;n=0
for (i,t,c) in rest:
    if got>=need: break
    if got+len(c)>need:
        k=c.rfind(b'\n',0,need-got); c=c[:k+1] if k>0 else c[:need-got]
    cf.append(wr(f'C_{i}.txt',c)); got+=len(c)
out['C']['prose_extra']=cf
out['bytes']={s:sum(os.path.getsize(f) for f in fs) for s,fs in out['T'].items()}; out['bytes']['C_extra']=got; out['bytes']['need']=need
out['bytes']['heldout']={s:sum(os.path.getsize(f) for f in fs) for s,fs in out['heldout'].items()}
out['counts']={'S2_books':len(s2),'C_books':len(cf)}
json.dump(out,open(B+'stages_final.json','w'),indent=1)
print(json.dumps({k:out[k] for k in ('bytes','counts')},indent=1))
```

### A.e5_overlap.py

```python
"""Overlap of every E5 source with E2's MANIFEST.json (by id, by normalised title) and with the question sources."""
import json,re,sys,os
E5='/path/to/e5-work/'; B=E5+'build/'
man=json.load(open('/path/to/vam/experiments/data/pg19/MANIFEST.json'))['books']
nk=lambda t: re.sub(r'[^a-z0-9]','',t.lower())
mid={b['id']:b for b in man}; mt={nk(b['title']):b for b in man}
meta={}
for l in open(E5+'metadata.csv',encoding='utf-8').read().splitlines():
    r=l.split(',')
    if len(r)>=4: meta[r[0]]=','.join(r[1:-2])
st=json.load(open(B+'stages_final.json'))
def ids(files,prefix): 
    return sorted({re.match(prefix+r'_?(\d+)',os.path.basename(f)).group(1) for f in files if re.match(prefix+r'_?(\d+)',os.path.basename(f))})
rows={'S1':['29765'],'S2':ids(st['T']['S2']+st['heldout']['S2'],'S2p'),'S3':ids(st['T']['S3']+st['heldout']['S3'],'S3'),'S4':ids(st['T']['S4']+st['heldout']['S4'],'S4'),'C':ids(st['C']['prose_extra'],'C')}
S5={'principia':'76404','opticks':'33504','relativity':'30155','notebooks':'5000','euclid_casey':'21076','republic':'1497','charmides':'1580','laches':'1584','euthydemus':'1598','cratylus':'1616','ion':'1635','meno':'1643','apology':'1656','crito':'1657','phaedo':'1658','gorgias':'1672','parmenides':'1687','philebus':'1744','einstein_sidelights':'7333','einstein_meaning':'36276','einstein_fundamental':'69572','einstein_principle':'66944','leonardo_thoughts':'29904','xen_memorabilia':'1177','xen_apology':'1171'}
rows['S5']=sorted(S5.values())
out={}
for s,l in rows.items():
    hit=[]
    for i in l:
        t=meta.get(i)
        if i in mid: hit.append((i,'id',mid[i]['stage'],mid[i]['role']))
        elif t and nk(t) in mt: hit.append((i,'title',mt[nk(t)]['id']))
    out[s]={'sources':len(l),'in_manifest':len(hit),'matches':hit}
json.dump(out,open(sys.argv[1],'w'),indent=1)
print({s:(v['sources'],v['in_manifest']) for s,v in out.items()})
```

### A.q/qt.py

```python
import re, json, hashlib, itertools, os
CL='/path/to/e5-work/clean/'
WORKS={'opticks':'pg19_33504_opticks.txt','principia':'principia.txt','relativity':'relativity.txt','notebooks':'notebooks.txt','euclid':'euclid_casey.txt',
 'republic':'pg19_1497_republic.txt','apology':'pg19_1656_apology.txt','crito':'pg19_1657_crito.txt','phaedo':'pg19_1658_phaedo.txt','meno':'pg19_1643_meno.txt','gorgias':'pg19_1672_gorgias.txt','charmides':'pg19_1580_charmides.txt','laches':'pg19_1584_laches.txt','euthydemus':'pg19_1598_euthydemus.txt','cratylus':'pg19_1616_cratylus.txt','ion':'pg19_1635_ion.txt','parmenides':'pg19_1687_parmenides.txt','philebus':'pg19_1744_philebus.txt'}
_raw={}; _norm={}
def raw(w):
    if w not in _raw: _raw[w]=open(CL+WORKS[w],encoding='utf-8').read()
    return _raw[w]
def norm(s): return re.sub(r'\s+',' ',s)
def nt(w):
    if w not in _norm: _norm[w]=norm(raw(w))
    return _norm[w]
def sents(w):
    return re.split(r'(?<=[.;:?!])\s+',nt(w))
def grep(w,pat,n=30,lo=50,hi=300,flags=re.I):
    out=[s for s in sents(w) if lo<len(s)<hi and re.search(pat,s,flags)]
    return out[:n]

NUM=re.compile(r'(?<![\w/.-])(\d[\d,]*(?:-\d+/\d+|/\d+|\.\d+)?|\d+\'\d*)(?![\w/])')
def candidates(w, nmax=60, minw=8, seed=0, pat=None, hi=260):
    import random
    out=[]
    for s in sents(w):
        if not (70<len(s)<hi): continue
        if re.search(r'Fig\.|\[Pg|\bPage\b|Section|Chapter',s): continue
        for m in NUM.finditer(s):
            pre=s[:m.start()]
            if len(pre.split())<minw: continue
            if pat and not re.search(pat,s,re.I): continue
            out.append((s,m.start(),m.group(0)))
    random.Random(seed).shuffle(out)
    return out[:nmax]
def show(w,n=40,seed=0,pat=None,minw=8):
    for k,(s,i,a) in enumerate(candidates(w,n,minw,seed,pat)):
        words=s[:i].split()
        print(f'[{k}] ...{" ".join(words[-12:])} <<{a}>> {" ".join(s[i+len(a):].split()[:4])}')
```

### A.q/qdef.py

```python
"""Question definitions and mechanical checks (draft tooling; the final script goes inline in E5_DRAFT.md)."""
import re, random, json, math
import qt
ITEMS=[]
def tok_vocab(work):
    t=qt.nt(work)
    return [m.group(0) for m in qt.NUM.finditer(t)]
_vocab={}
def vocab(work):
    if work not in _vocab:
        from collections import Counter
        _vocab[work]=Counter(tok_vocab(work))
    return _vocab[work]
def shape(tok): return re.sub(r'\d','9',re.sub(r'[A-Za-z]','a',tok))
def val(tok):
    try: return float(tok.replace(',','').rstrip('.'))
    except: return None
def pick_distractors(work, ans, k=3, seed=0):
    v=vocab(work); sh=shape(ans); a=val(ans)
    pool=[t for t in v if t!=ans and shape(t)==sh and len(t.encode())==len(ans.encode())]
    if a is not None and a>0:
        pool=[t for t in pool if val(t) is not None and val(t)>0]
        pool.sort(key=lambda t: abs(math.log(val(t)/a)))
        near=pool[:max(k,6)]
    else: near=pool
    r=random.Random(f'{work}{ans}{seed}'); r.shuffle(near)
    return near[:k]
def R(work, qid, quote, lead='', words=12, dist=None, locator=''):
    """Recall item. quote contains one ⟦answer⟧ marker; stem = lead + the last `words` words before it."""
    m=re.search(r'⟦(.+?)⟧',quote); ans=m.group(1)
    full=quote.replace('⟦','').replace('⟧','')
    pre=quote[:m.start()]
    tail=' '.join(pre.split()[-words:] if words else []) + (' ' if pre.endswith(' ') or True else '')
    stem=re.sub(r'\s+',' ',(lead.strip()+' ' if lead else '')+tail).strip()+' '
    d=dist if dist else pick_distractors(work,ans)
    ITEMS.append(dict(id=qid,kind='recall',work=work,stem=stem,answer=ans,distractors=list(d),quotes=[{'work':work,'quote':full,'locator':locator}],reasoning=f'The passage states {ans!r} at this point.'))
def I(work, qid, stem, answer, distractors, quotes, reasoning, keys=None, lead=''):
    """Inference item: quotes = list of (work, quote, locator); keys = distinctive stem terms for the co-occurrence check."""
    ITEMS.append(dict(id=qid,kind='inference',work=work,stem=stem,answer=answer,distractors=list(distractors),quotes=[{'work':w,'quote':q,'locator':l} for w,q,l in quotes],reasoning=reasoning,keys=keys or []))
```

### A.q/items_opticks.py

```python
from qdef import R, I
W='opticks'
L='In Newton\'s Opticks:'
R(W,'R-OPT-01',"For if the Sine of Incidence out of Air into Glass be to the Sine of Refraction as 17 to ⟦11⟧, the Sine of Incidence out of Glass into Air must on the contrary be to the Sine of Refraction as 11 to 17",lead='According to the Opticks,',words=14,locator='Book I, Part I')
R(W,'R-OPT-02',"In passing out of Glass into Air, where the Refraction is measured by the Ratio of the Sines 20 to 31, the total Reflexion begins when the Angle of Incidence is 40 Degrees ⟦10⟧ Minutes;",lead=L,words=14,locator='Book II, Part III')
R(W,'R-OPT-03',"Thus in the passing of Light out of Water into Air, where the Refraction is measured by the Ratio of the Sines 3 to 4, the total Reflexion begins when the Angle of Incidence is about 48 Degrees ⟦35⟧ Minutes.",lead=L,words=14,locator='Book II, Part III')
R(W,'R-OPT-04',"Sounds move about ⟦1140⟧ _English_ Feet in a second Minute of Time, and in seven or eight Minutes of Time they move about one hundred _English_ Miles.",lead='Concerning the motion of sounds the Opticks says:',words=3,locator='Book III, Query 21',dist=None)
R(W,'R-OPT-05',"For were Mercury transparent, its Density is such that the Sine of Incidence upon it (by my Computation) would be to the Sine of its Refraction, as ⟦71⟧ to 20, or 7 to 2.",lead=L,words=22,locator='Book II, Part III')
R(W,'R-OPT-06',"the Intervals of their Fits of easy Reflexion are the 1/⟦89000⟧th part of an Inch.",lead='For the rays at the confine of yellow and orange passing perpendicularly out of any medium into air,',words=12,dist=['88739','88850','88952'],locator='Book II, Part III, Prop. 16')
R(W,'R-OPT-07',"the Length of this Image at the same distance of ⟦18-1/2⟧ Feet was also about 10 Inches, or 10-1/8.",lead=L,words=12,locator='Book I, Part I, Exp. 3')
R(W,'R-OPT-08',"and the refracting Angle of the Prism, whereby so great a Length was made, was ⟦64⟧ degrees.",lead=L,words=12,locator='Book I, Part I, Exp. 3')
R(W,'R-OPT-09',"The Eye-glass was Plano-convex, and the diameter of the Sphere to which the convex side was ground was about ⟦1/5⟧ of an Inch, or a little less",lead='Of his reflecting telescope the Opticks says:',words=16,locator='Book I, Part I, Prop. 7')
R(W,'R-OPT-10',"a Telescope of 64 Feet in length, with an Aperture of 2-2/3 Inches, magnifies about ⟦120⟧ times, with as much distinctness as one of a Foot in length",lead=L,words=18,locator='Book I, Part I, Prop. 7')
R(W,'R-OPT-11',"The Atmosphere by its weight presses the Quick-silver into the Glass, to the height of ⟦29⟧ or 30 Inches.",lead=L,words=12,locator='Book III, Query 28')
R(W,'R-OPT-12',"and at the height of ⟦15⟧ Miles it is sixteen times rarer than that at the Surface of the Earth;",lead='Of the air above the earth the Opticks says,',words=8,locator='Book III, Query 28')
R(W,'R-OPT-13',"These diameters are therefore to one another as 75 to 61-1/2 or 50 to ⟦41⟧, and their Squares as 2500 to 1681, that is, as 3 to 2 very nearly.",lead=L,words=18,locator='Book II, Part IV')
R(W,'R-OPT-14',"The obtuse Angles of the Parallelograms are each of them 101 Degrees and ⟦52⟧ Minutes;",lead=L,words=10,locator='Book II, Part I')
R(W,'R-OPT-15',"At the same time there appear'd a Halo about 22 Degrees ⟦35⟧' distant from the center of the Moon.",lead=L,words=10,locator='Book II, Part IV')

# ---- inference ----
I(W,'I-OPT-01',"Light passes out of air into water with the sines of incidence and refraction as 4 to 3, and out of air into glass as 17 to 11. For a ray passing out of water into glass the sine of incidence is to the sine of refraction as ",
  "51 to 44",["68 to 33","44 to 51","33 to 68"],
  [(W,"Thus if the Refraction be made out of Air into Water, the Sine of Incidence of the red Light is to the Sine of its Refraction as 4 to 3.","Book I, Part I, Prop. 6"),
   (W,"if the Sine of Incidence out of Air into Glass be to the Sine of Refraction as 17 to 11","Book I, Part I, Prop. 1"),
   (W,"the Sine of Incidence of that Ray out of Glass into Water will be to its Sine of Refraction as 20 to 31 and 4 to 3 jointly, that is, as the Factum of 20 and 4 to the Factum of 31 and 3","Book II, Part III, Prop. 10")],
  "Compound the ratios by Newton's rule: incidence in water is 3/4 of the sine in air and refraction in glass is 11/17 of it, so the ratio is (3/4)/(11/17) = 51/44.",keys=['water','glass'])
I(W,'I-OPT-02',"Total reflexion begins at a smaller angle of incidence for light passing out of glass into air than for light passing out of water into air; the angle for water exceeds the angle for glass by ",
  "8 Degrees 25 Minutes",["8 Degrees 45 Minutes","9 Degrees 25 Minutes","7 Degrees 35 Minutes"],
  [(W,"Thus in the passing of Light out of Water into Air, where the Refraction is measured by the Ratio of the Sines 3 to 4, the total Reflexion begins when the Angle of Incidence is about 48 Degrees 35 Minutes.","Book II, Part III, Prop. 8"),
   (W,"In passing out of Glass into Air, where the Refraction is measured by the Ratio of the Sines 20 to 31, the total Reflexion begins when the Angle of Incidence is 40 Degrees 10 Minutes;","Book II, Part III, Prop. 8")],
  "48 degrees 35 minutes minus 40 degrees 10 minutes is 8 degrees 25 minutes.",keys=['total Reflexion','Water into Air'])
I(W,'I-OPT-03',"In the rainbow construction, the least angle AYS for the most refrangible rays is larger than for the least refrangible rays by ",
  "3 Degrees 10 Minutes",["3 Degrees 50 Minutes","4 Degrees 10 Minutes","2 Degrees 50 Minutes"],
  [(W,"thence by Computation the greatest Angle AXR will be found 42 Degrees and 2 Minutes, and the least Angle AYS, 50 Degrees and 57 Minutes.","Book I, Part II, Prop. 9"),
   (W,"thence by Computation the greatest Angle AXR will be found 40 Degrees and 17 Minutes, and the least Angle AYS 54 Degrees and 7 Minutes.","Book I, Part II, Prop. 9")],
  "54 degrees 7 minutes minus 50 degrees 57 minutes is 3 degrees 10 minutes.",keys=['AYS','refrangible'])
I(W,'I-OPT-04',"The greatest angle AXR for the least refrangible rays is larger than the greatest angle for the most refrangible rays by ",
  "1 Degree 45 Minutes",["1 Degree 55 Minutes","2 Degree 15 Minutes","1 Degree 15 Minutes"],
  [(W,"thence by Computation the greatest Angle AXR will be found 42 Degrees and 2 Minutes, and the least Angle AYS, 50 Degrees and 57 Minutes.","Book I, Part II, Prop. 9"),
   (W,"thence by Computation the greatest Angle AXR will be found 40 Degrees and 17 Minutes, and the least Angle AYS 54 Degrees and 7 Minutes.","Book I, Part II, Prop. 9")],
  "42 degrees 2 minutes minus 40 degrees 17 minutes is 1 degree 45 minutes.",keys=['AXR','refrangible'])
I(W,'I-OPT-05',"Between two glasses the thickness of air at the most luminous parts of the first six rings was 1, 3, 5, 7, 9 and 11 parts of an inch over 178000; at the seventh ring the thickness is ",
  "13/178000",["12/178000","14/178000","15/178000"],
  [(W,"the thickness of the thinned Air, which between two Glasses exhibited the most luminous parts of the first six Rings were 1/178000, 3/178000, 5/178000, 7/178000, 9/178000, 11/178000 parts of an Inch.","Book II, Part I, Obs. 10")],
  "The thicknesses follow the odd numbers; the next odd number after 11 is 13.",keys=['178000'])
I(W,'I-OPT-06',"Newton computes that the air is four times rarer at seven and a half miles and sixteen times rarer at 15 miles above the earth; at 45 miles it is about this many times rarer: ",
  "4096",["2048","8192","3072"],
  [(W,"at the height of about seven and a half _English_ Miles from the Earth, the Air is four times rarer than at the Surface of the Earth; and at the height of 15 Miles it is sixteen times rarer than that at the Surface of the Earth; and at the height of 22-1/2, 30, or 38 Miles, it is respectively 64, 256, or 1024 times rarer, or thereabouts","Book III, Query 28")],
  "Each seven and a half miles multiplies the rarity by four; 45 miles is six steps, 4 to the sixth power is 4096.",keys=['rarer','Miles'])
```

### A.q/items_notebooks.py

```python
from qdef import R, I
W='notebooks'
L="In the Notebooks of Leonardo da Vinci:"
R(W,'R-NOT-01',"The length of the hand is 1/3 of a braccio [8 inches] and this is found ⟦9⟧ times in man.",lead=L,words=14,locator='Proportion of the human body')
R(W,'R-NOT-02',"The four smaller toes are all equally thick from the nail at the top to the bottom, and are ⟦1/13⟧ of the foot.",lead=L,words=14,locator='Proportion of the human body')
R(W,'R-NOT-03',"Every man, at ⟦three⟧ years old is half the full height he will grow to at last.",lead=L+' Of the growth of man, Leonardo says:',words=6,dist=['seven','eight','forty'],locator='Preliminary observations (308)')
R(W,'R-NOT-04',"TO DRAW A FIGURE ON A WALL 12 BRACCIA HIGH WHICH SHALL LOOK ⟦24⟧ BRACCIA HIGH.",lead=L,words=12,locator='526')
R(W,'R-NOT-05',"For instance: Look at a man at a distance of ⟦300⟧ braccia and judge attentively whether he be handsome or ugly",lead=L,words=10,locator='On perspective')
R(W,'R-NOT-06',"When you draw from nature stand at a distance of ⟦3⟧ times the height of the object you wish to draw.",lead=L,words=12,locator='540')
R(W,'R-NOT-07',"It should be 10 braccia wide and ⟦20⟧ braccia long and 10 braccia high and covered with a linen awning;",lead=L,words=10,locator='On painting')
R(W,'R-NOT-08',"Put the white into an earthen pot, and lay it no thicker than a string, and let it stand in the sun undisturbed for ⟦2⟧ days;",lead=L,words=16,locator='Recipes')
R(W,'R-NOT-09',"Four ounces virgin wax, four ounces Greek pitch, ⟦two⟧ ounces incense, one ounce oil of roses, first melt the wax and oil",lead=L,words=10,dist=['six','ten','one'],locator='644')
R(W,'R-NOT-10',"and the whole army came after him--⟦40⟧ squadrons of cavalry, and 2000 foot soldiers went with him.",lead=L,words=12,locator='Letters and notes')
R(W,'R-NOT-11',"But I have found that as a rule the second is ⟦4/5⟧ of the first when it is 20 braccia beyond it.",lead=L,words=14,locator='On aerial perspective')
R(W,'R-NOT-12',"Although practical painters attribute to all shaded objects--trees, fields, hair, beards and skin--⟦four⟧ degrees of darkness in each colour they use:",lead=L,words=12,dist=['nine','five','zero'],locator='On light and shade')
R(W,'R-NOT-13',"If a man 2 braccia high is too small, one of ⟦four⟧ is too tall, the medium being what is admirable.",lead=L,words=12,dist=['five','nine','zero'],locator='309')
I(W,'I-NOT-01',"Vitruvius says that 4 fingers make a palm, 6 palms make a cubit and 4 cubits make a man's height; in fingers a man's height is ",
  "96",["24","16","64"],
  [(W,"4 fingers make 1 palm, and 4 palms make 1 foot, 6 palms make 1 cubit; 4 cubits make a man's height.","Proportion of the human body")],
  "4 cubits of 6 palms of 4 fingers: 4 x 6 x 4 = 96 fingers.",keys=['fingers','cubit'])
I(W,'I-NOT-02',"Leonardo divides the head into 12 degrees, each degree into 12 points and each point into 12 minutes; the number of minutes in the head is ",
  "1728",["1584","1440","1296"],
  [(W,"Divide the head into 12 degrees, and each degree divide into 12 points, and each point into 12 minutes","708. Measurement and division of a statue")],
  "12 x 12 x 12 = 1728.",keys=['degrees','minutes'])
I(W,'I-NOT-03',"A man's hand is 8 inches and this is found 9 times in his height; in inches a man's height is ",
  "72",["64","81","96"],
  [(W,"The length of the hand is 1/3 of a braccio [8 inches] and this is found 9 times in man.","Proportion of the human body")],
  "9 times 8 inches is 72 inches.",keys=['hand','times in man'])
I(W,'I-NOT-04',"Leonardo takes a man of 3 braccia in height as the model, and says a man at three years old is half his final height; such a child is ",
  "1-1/2 braccia",["1-1/3 braccia","2-1/2 braccia","1-1/4 braccia"],
  [(W,"Every man, at three years old is half the full height he will grow to at last.","Preliminary observations (308)"),
   (W,"therefore take a man of 3 braccia in height and measure him by the rule I will give you.","Preliminary observations (309)")],
  "Half of 3 braccia is 1-1/2 braccia.",keys=['three years','braccia'])
I(W,'I-NOT-05',"The wax, Greek pitch, incense and oil of roses of Leonardo's varnish are four ounces, four ounces, two ounces and one ounce; the total weight is ",
  "11 ounces",["12 ounces","10 ounces","13 ounces"],
  [(W,"Four ounces virgin wax, four ounces Greek pitch, two ounces incense, one ounce oil of roses","644")],
  "4 + 4 + 2 + 1 = 11 ounces.",keys=['ounces'])
I(W,'I-NOT-06',"Leonardo says to stand at 3 times the height of the object you draw from nature; for a figure 12 braccia high you stand at ",
  "36 braccia",["24 braccia","48 braccia","30 braccia"],
  [(W,"When you draw from nature stand at a distance of 3 times the height of the object you wish to draw.","540"),
   (W,"TO DRAW A FIGURE ON A WALL 12 BRACCIA HIGH WHICH SHALL LOOK 24 BRACCIA HIGH.","526")],
  "3 times 12 braccia is 36 braccia.",keys=['braccia high','distance'])
I(W,'I-NOT-07',"A man 2 braccia high is too small and one of four too tall; the measure Leonardo takes from them is ",
  "3 braccia",["2 braccia","4 braccia","5 braccia"],
  [(W,"If a man 2 braccia high is too small, one of four is too tall, the medium being what is admirable.","309")],
  "The medium between 2 and 4 braccia is 3 braccia.",keys=['too small'])
```

### A.q/items_principia.py

```python
from qdef import R, I
W='principia'
L="In the Life of Newton and the text of Motte's Principia:"
R(W,'R-PRI-01',"was duly admitted, on the 5th day of June ⟦1660⟧, in the eighteenth year of his age.",lead='Of Newton at Trinity College, Cambridge, the Life says he',words=10,locator='Life of Newton')
R(W,'R-PRI-02',"In the year ⟦1665⟧, he received the degree of Bachelor of Arts;",lead=L,words=8,locator='Life of Newton')
R(W,'R-PRI-03',"Barrow, having resolved to devote himself to Theology, resigned the Lucasian Professorship of Mathematics, in ⟦1669⟧, in favour of Newton",lead=L,words=14,locator='Life of Newton')
R(W,'R-PRI-04',"was made Lucasian Professor of Mathematics in ⟦1663⟧, and soon afterward delivered his Optical Lectures:",lead='Of Barrow the Life says he',words=8,locator='Life of Newton')
R(W,'R-PRI-05',"In ⟦1703⟧, he was chosen President of the Royal Society of London, to which office he was annually re-elected",lead='Of Newton and the Royal Society, the Life says:',words=6,locator='Life of Newton')
R(W,'R-PRI-06',"a few weeks after being elected a member thereof, in the spring of ⟦1671-2⟧.",lead='Newton communicated his discoveries on refrangibility to the Royal Society',words=12,dist=['1668-9','1691-2','1685-6'],locator='Life of Newton')
R(W,'R-PRI-07',"dedicated to the Royal Society, and presented thereto on the ⟦28th⟧ of April, 1685-6.",lead='Of the Principia the Life says it was',words=12,dist=['20th','25th','30th'],locator='Life of Newton')
R(W,'R-PRI-08',"and the council resolved, on the ⟦19th⟧ of May, to print it at the expense of the Society",lead='Of the printing of the Principia:',words=12,dist=['20th','21st','31st'],locator='Life of Newton')
R(W,'R-PRI-09',"Now by experiments it actually appears that sounds do really advance in one second of time about ⟦1142⟧ feet of English measure, or 1070 feet of French measure.",lead=L,words=16,locator='Book II')
R(W,'R-PRI-10',"and the moon in ⟦27⟧ days, 7 hours, 43'.",lead='In the Principia, the periodic time of the moon:',words=4,locator='Book III')
R(W,'R-PRI-11',"for the greatest semi-diameter of the earth was ⟦19658600⟧ Paris feet, and the mean distance of the centres of the earth and moon",lead=L,words=10,locator='Book III, Prop. 4')
R(W,'R-PRI-12',"The mean distance of the centres of the earth and moon in the syzygies of the moon is ⟦60⟧ mean semi-diameters of the earth, and a 10th part of one semi-diameter;",lead=L,words=18,locator='Book III, Prop. 37')
R(W,'R-PRI-13',"Therefore, by this computation, heavy bodies in the latitude of Paris falling in vacuo will describe ⟦15⟧ Paris feet, 1 inch, lines, very nearly, in one second of time.",lead=L,words=14,locator='Book III, Prop. 4')
R(W,'R-PRI-14',"In ⟦1656⟧, Newton's mother was again left a widow, and took up her abode once more at Woolsthorpe.",lead='Of his early years the Life says:',words=2,locator='Life of Newton')
I(W,'I-PRI-01',"Newton was admitted to Trinity College in 1660 and took his Bachelor of Arts degree in 1665; the years between were ",
  "5 years",["6 years","4 years","7 years"],
  [(W,"was duly admitted, on the 5th day of June 1660, in the eighteenth year of his age.","Life of Newton"),(W,"In the year 1665, he received the degree of Bachelor of Arts;","Life of Newton")],
  "1665 minus 1660 is 5 years.",keys=['Bachelor of Arts','Trinity'])
I(W,'I-PRI-02',"Newton received his Bachelor of Arts in 1665 and the Lucasian chair in 1669; the interval was ",
  "4 years",["5 years","3 years","6 years"],
  [(W,"In the year 1665, he received the degree of Bachelor of Arts;","Life of Newton"),(W,"resigned the Lucasian Professorship of Mathematics, in 1669, in favour of Newton","Life of Newton")],
  "1669 minus 1665 is 4 years.",keys=['Bachelor of Arts','Lucasian'])
I(W,'I-PRI-03',"Newton took his Bachelor of Arts in 1665 and his Master of Arts in the year following, that is in ",
  "1666",["1667","1665","1668"],
  [(W,"In the year 1665, he received the degree of Bachelor of Arts;","Life of Newton"),(W,"and, in the year following, he took his degree of Master of Arts, and was appointed to a Senior Fellowship.","Life of Newton")],
  "The year following 1665 is 1666.",keys=['Master of Arts'])
I(W,'I-PRI-04',"Sounds advance in one second about 1142 feet of English measure, or 1070 feet of French measure; the English measure exceeds the French by ",
  "72 feet",["62 feet","82 feet","78 feet"],
  [(W,"Now by experiments it actually appears that sounds do really advance in one second of time about 1142 feet of English measure, or 1070 feet of French measure.","Book II")],
  "1142 minus 1070 is 72.",keys=['English measure','French measure'])
I(W,'I-PRI-05',"The moon's periodic time is 27 days, 7 hours, 43 minutes; expressed in hours it is ",
  "655 hours 43 minutes",["654 hours 43 minutes","665 hours 43 minutes","645 hours 43 minutes"],
  [(W,"and the moon in 27 days, 7 hours, 43'.","Book III")],
  "27 x 24 + 7 = 655 hours.",keys=['periodic','moon'])
I(W,'I-PRI-06',"A heavy body in Paris falls 15 Paris feet, 1 inch in one second, and the spaces described are in the duplicate ratio of the times; in two seconds it falls ",
  "60 feet 4 inches",["30 feet 2 inches","45 feet 3 inches","60 feet 2 inches"],
  [(W,"heavy bodies in the latitude of Paris falling in vacuo will describe 15 Paris feet, 1 inch, lines, very nearly, in one second of time.","Book III, Prop. 4"),(W,"are in the very beginning of the motion one to the other in the duplicate ratio of the times.","Book I, Lemma 10")],
  "Twice the time gives four times the space: 4 x 15 feet 1 inch = 60 feet 4 inches.",keys=['Paris','duplicate ratio'])
I(W,'I-PRI-07',"Barrow was made Lucasian Professor in 1663 and resigned the chair in 1669 in favour of Newton; he held it for ",
  "6 years",["7 years","5 years","9 years"],
  [(W,"was made Lucasian Professor of Mathematics in 1663, and soon afterward delivered his Optical Lectures:","Life of Newton"),(W,"resigned the Lucasian Professorship of Mathematics, in 1669, in favour of Newton","Life of Newton")],
  "1669 minus 1663 is 6 years.",keys=['Barrow','Lucasian'])
I(W,'I-PRI-08',"Newton was born on 25 December 1642 and admitted to Trinity College on 5 June 1660, in the eighteenth year of his age; the years he had completed were ",
  "17 years",["18 years","19 years","16 years"],
  [(W,"was born on the 25th of December, O. S. 1642","Life of Newton"),(W,"was duly admitted, on the 5th day of June 1660, in the eighteenth year of his age.","Life of Newton")],
  "From December 1642 to June 1660 is 17 years and a few months, which is the eighteenth year of his age.",keys=['eighteenth year'])
```

### A.q/items_plato.py

```python
from qdef import R, I
L="In Jowett's Plato:"
W3=lambda *a:a
R('apology','R-PLA-01',"and I have a family, yes, and sons, O Athenians, ⟦three⟧ in number, one almost a man, and two others who are still young;",lead='In the Apology, Socrates says of his sons:',words=9,dist=['seven','eight','forty'],locator='Apology')
R('apology','R-PLA-02',"my friends here, bid me say ⟦thirty⟧ minae, and they will be the sureties.",lead='In the Apology, the friends of Socrates bid him name a penalty:',words=10,dist=['twenty','eighty','ninety'],locator='Apology')
R('apology','R-PLA-03',"but now, had ⟦thirty⟧ votes gone over to the other side, I should have been acquitted.",lead='In the Apology, Socrates says that, had a few votes gone the other way, he would be acquitted:',words=8,dist=['twenty','eighty','ninety'],locator='Apology')
R('meno','R-PLA-04',"Therefore the double line, boy, has given a space, not twice, but ⟦four times⟧ as much.",lead='In the Meno:',words=10,dist=['five times','nine times','many times'],locator='Meno')
R('meno','R-PLA-05',"and is not a space of ⟦eight⟧ feet twice the size of this, and half the size of the other?",lead='In the Meno, Socrates asks the boy:',words=8,dist=['seven','three','sixty'],locator='Meno')
R('meno','R-PLA-06',"Then if we add a half to this line of two, that will be the line of ⟦three⟧.",lead='In the Meno:',words=12,dist=['seven','eight','forty'],locator='Meno')
R('philebus','R-PLA-07',"There are three criteria of goodness--beauty, symmetry, ⟦truth⟧.",lead='In the Philebus Introduction:',words=6,dist=['pride','proof','power'],locator='Philebus Introduction')
R('phaedo','R-PLA-08',"These rivers are many and mighty, and there are four principal ones, Oceanus, Acheron, Pyriphlegethon, and ⟦Cocytus⟧.",lead='In the Phaedo:',words=9,dist=['Alpheus','Acheron','Oceanus'],locator='Phaedo')
R('phaedo','R-PLA-09',"and still more clearly did I seem to perceive that ten is two more than eight, and that two cubits are more than one, because two is the ⟦double⟧ of one.",lead='In the Phaedo:',words=10,dist=['triple','single','halved'],locator='Phaedo')
R('republic','R-PLA-10',"There are five of the State, and five of the ⟦soul⟧, I said.",lead='In the Republic:',words=7,dist=['body','city','hand'],locator='Republic')
R('republic','R-PLA-11',"Thus there are five forms in all, both of states and of ⟦souls⟧...",lead='In the Republic:',words=7,dist=['stars','towns','cases'],locator='Republic')
R('republic','R-PLA-12',"My notion is, that our State being perfect will contain all the four virtues--wisdom, courage, temperance, ⟦justice⟧.",lead=L,words=8,dist=['bravery','honesty','harmony'],locator='Republic')
R('republic','R-PLA-14',"He said that for every evil deed they were punished tenfold--now the journey was of a thousand years' duration, because the life of man was reckoned as a ⟦hundred⟧ years",lead='In the Republic, the life of man was reckoned as a',words=0,dist=['seventy','fifteen','sixteen'][:3],locator='Republic')
R('charmides','R-PLA-15',"and, in the spirit of Socrates and of Greek life generally, proposes as a ⟦fifth⟧ definition, (5) Temperance is self-knowledge.",lead='In the Charmides Introduction, Critias proposes as a',words=0,dist=['sixth','third','ninth'],locator='Charmides Introduction')
R('gorgias','R-PLA-16',"But you, who are the ⟦three⟧ wisest men in Hellas, have nothing better to say",lead='In the Gorgias, Socrates addresses Callicles, Polus and Gorgias, the',words=0,dist=['seven','eight','forty'],locator='Gorgias')
R('euthydemus','R-PLA-17',"Yes, I said, he is my half-brother, the son of my ⟦mother⟧, but not of my father.",lead='In the Euthydemus:',words=10,dist=['father','sister','master'],locator='Euthydemus')
R('republic','R-PLA-18',"a man may begin at five-and-twenty, when he has passed the point at which the pulse of life beats quickest, and continue to beget children until he be ⟦fifty-five⟧.",lead='In the Republic:',words=10,dist=['sixty-five','forty-five','fifty-nine'],locator='Republic')
R('republic','R-PLA-19',"A woman, I said, at twenty years of age may begin to bear children to the State, and continue to bear them until ⟦forty⟧;",lead='In the Republic:',words=12,dist=['fifty','sixty','seven'],locator='Republic')
# ---- inference
I('meno','I-PLA-01',"In the Meno, the boy's square has a side of two feet and an area of four feet, and Socrates says that a line of three feet gives a square of three times three feet; the line of three feet gives the space of ",
  "nine feet",["five feet","four feet","many feet"],
  [('meno',"But if there are three feet this way and three feet that way, the whole space will be three times three feet?","Meno")],"Three times three is nine.",keys=['three times three'])
I('meno','I-PLA-02',"In the Meno, a space of eight feet is wanted; the side of two feet gives four feet and the side of four feet gives sixteen feet, so the side of the space of eight feet is more than two feet and less than ",
  "three",["seven","eight","fifty"],
  [('meno',"Then the line which forms the side of eight feet ought to be more than this line of two feet, and less than the other of four feet?","Meno")],"Eight lies between four and sixteen.",keys=['eight feet'])
I('republic','I-PLA-03',"In the Republic, the State contains four virtues, wisdom, courage, temperance, justice, and the first three correspond to the three classes; the virtue that is not one of the first three is ",
  "justice",["courage","modesty","insight"],
  [('republic',"My notion is, that our State being perfect will contain all the four virtues--wisdom, courage, temperance, justice.","Republic"),('republic',"Each of the first three virtues corresponds to one of the three parts of the soul and one of the three classes in the State","Republic Introduction")],"The fourth listed is justice.",keys=['four virtues'])
I('republic','I-PLA-04',"In the Republic a woman may bear children from twenty years of age until forty, so the period is ",
  "twenty years",["thirty years","twelve years","ninety years"],
  [('republic',"A woman, I said, at twenty years of age may begin to bear children to the State, and continue to bear them until forty;","Republic")],"Forty minus twenty is twenty.",keys=['bear children'])
I('republic','I-PLA-05',"In the Republic a man may begin at five-and-twenty and continue to beget children until fifty-five, so the period is ",
  "thirty years",["twenty years","twelve years","ninety years"],
  [('republic',"a man may begin at five-and-twenty, when he has passed the point at which the pulse of life beats quickest, and continue to beget children until he be fifty-five.","Republic")],"Fifty-five minus twenty-five is thirty.",keys=['beget children'])
I('republic','I-PLA-06',"In the Republic the journey of the souls was a thousand years and the life of man was reckoned at a hundred years, so the journey was ",
  "ten lives",["two lives","six lives","all lives"],
  [('republic',"He said that for every evil deed they were punished tenfold--now the journey was of a thousand years' duration, because the life of man was reckoned as a hundred years","Republic")],"1000 / 100 = 10.",keys=['thousand years'])
I('apology','I-PLA-07',"In the Apology Evenus teaches at five minae and Socrates proposes a penalty of thirty minae; the penalty exceeds the charge by ",
  "twenty-five minae",["thirty-five minae","twenty-nine minae","twenty-four minae"],
  [('apology',"he is the man, and his charge is five minae.","Apology"),('apology',"Let that be the penalty, or, if his friends wish, thirty minae;","Apology")],"Thirty minus five is twenty-five.",keys=['five minae'])
I('apology','I-PLA-08',"In the Apology Socrates has three sons, one almost a man, and two who are still young; the number still young is ",
  "two",["one","six","ten"],
  [('apology',"and I have a family, yes, and sons, O Athenians, three in number, one almost a man, and two others who are still young;","Apology")],"Three less one is two.",keys=['sons'])
I('phaedo','I-PLA-09',"In the Phaedo ten is two more than eight, so eight is two ",
  "less than ten",["more than ten","less than six","more than six"],
  [('phaedo',"I seem to perceive that ten is two more than eight","Phaedo")],"Converse of the stated relation.",keys=['two more than eight'])
I('phaedo','I-PLA-10',"In the Phaedo two is the double of one and two cubits exceed one cubit not by a half but by a magnitude of ",
  "one",["two","six","ten"],
  [('phaedo',"two cubits are more than one, because two is the double of one","Phaedo")],"Two minus one is one.",keys=['two cubits'])
```

### A.q/items_relativity.py

```python
from qdef import R, I
W='relativity'
L="In Einstein's Relativity:"
R(W,'R-REL-01',"takes place in straight lines with a velocity _c_ = ⟦300,000⟧ km./sec.",lead='Of light in Einstein\'s Relativity:',words=12,dist=['200,000','250,000','350,000'],locator='Part I')
R(W,'R-REL-02',"comparable with a railway carriage travelling with a velocity of about ⟦30⟧ kilometres per second",lead='Of the earth in its orbit round the sun:',words=8,dist=['20','40','60'],locator='Part I')
R(W,'R-REL-03',"that in the case of Mercury it must amount to ⟦43⟧ seconds of arc per century",lead=L,words=10,dist=['34','47','53'],locator='Part II')
R(W,'R-REL-04',"First observed by Eddington and others in ⟦1919⟧.",lead='Of the deflection of light by the sun:',words=6,dist=['1917','1921','1927'],locator='Part II notes')
R(W,'R-REL-05',"towards the red end of the spectrum was definitely established by Adams in ⟦1924⟧",lead=L,words=11,dist=['1914','1934','1944'],locator='Part II notes')
R(W,'R-REL-06',"it was found (Leverrier: ⟦1859⟧; and Newcomb: 1895) that an unexplained perihelial movement of the orbit of Mercury remained over",lead=L,words=11,dist=['1869','1849','1839'],locator='Appendix')
R(W,'R-REL-07',"he will not obtain as quotient the familiar number π = ⟦3.14⟧",lead=L,words=11,dist=['3.41','2.14','3.24'],locator='Part II')
R(W,'R-REL-08',"an arrangement (_e.g._ two mirrors inclined at ⟦90⟧°",lead=L,words=8,dist=['45','60','30'],locator='Part I')
R(W,'R-REL-09',"The stellar universe ought to be a finite island in the infinite ⟦ocean⟧ of space.",lead=L,words=10,dist=['plain','beach','court'],locator='Part III')
R(W,'R-REL-10',"[8] _e.g._ the middle of the first and of the ⟦hundredth⟧ carriage",lead=L,words=10,dist=['twentieth','thirtieth','ninetieth'],locator='Part I')
R(W,'R-REL-11',"the point _P_ in the diagram has the Gaussian co-ordinates _u_ = ⟦3⟧, _v_ = 1",lead=L,words=10,dist=['2','4','5'],locator='Part II')
R(W,'R-REL-12',"would differ from ⟦360⟧°",lead='Of the angle described by the line joining sun and planet from perihelion to perihelion:',words=4,dist=['180','270','720'],locator='Appendix')
I(W,'I-REL-01',"The rotation of the orbit of Mercury is 43 seconds of arc per century; in two centuries it amounts to ",
  "86 seconds",["43 seconds","96 seconds","66 seconds"],
  [(W,"in the case of Mercury it must amount to 43 seconds of arc per century","Part II")],"2 x 43 = 86.",keys=['seconds of arc'])
I(W,'I-REL-02',"Light travels at 300,000 km./sec. and the earth in its orbit at about 30 kilometres per second; the speed of light is greater by the factor ",
  "10,000",["20,000","30,000","90,000"],
  [(W,"a velocity _c_ = 300,000 km./sec.","Part I"),(W,"a velocity of about 30 kilometres per second","Part I")],"300,000 / 30 = 10,000.",keys=['300,000'])
I(W,'I-REL-03',"The unexplained motion of Mercury was reported by Leverrier in 1859 and by Newcomb in 1895; the interval is ",
  "36 years",["26 years","46 years","34 years"],
  [(W,"(Leverrier: 1859; and Newcomb: 1895)","Appendix")],"1895 - 1859 = 36.",keys=['Leverrier'])
I(W,'I-REL-04',"The deflection of light was first observed in 1919 and the red-shift was established by Adams in 1924; the interval is ",
  "5 years",["4 years","6 years","9 years"],
  [(W,"First observed by Eddington and others in 1919.","Part II notes"),(W,"Established by Adams in 1924","Part II notes")],"1924 - 1919 = 5.",keys=['Adams','Eddington'])
I(W,'I-REL-05',"The book was first published in December, 1916, and Eddington and others first observed the deflection of light in 1919; the interval is about ",
  "3 years",["2 years","4 years","6 years"],
  [(W,"First Published: December, 1916","Colophon"),(W,"First observed by Eddington and others in 1919.","Part II notes")],"1919 - 1916 = 3.",keys=['December, 1916'])
```

### A.q/items_euclid.py

```python
from qdef import R, I
W='euclid'
L="In Casey's First Six Books of Euclid:"
R(W,'R-EUC-01',"A geometrical magnitude which has three dimensions, that is, length, breadth, and ⟦thickness⟧, is a solid;",lead=L,words=10,dist=['direction','curvature','magnitude'],locator='Introduction')
R(W,'R-EUC-02',"A quadrilateral whose four sides are equal is called a ⟦lozenge⟧.",lead=L,words=8,dist=['rhombus','polygon','scalene'],locator='Definitions')
R(W,'R-EUC-03',"A polygon which has five sides is called a ⟦pentagon⟧;",lead=L,words=8,dist=['triangle','quadrant','diagonal'],locator='Definitions')
R(W,'R-EUC-04',"one which has six sides, a ⟦hexagon⟧, and so on.",lead='In Casey\'s Euclid, of a polygon:',words=5,dist=['octagon','polygon','decagon'],locator='Definitions')
R(W,'R-EUC-05',"A system of more than three concurrent lines is called a ⟦pencil⟧ of lines.",lead=L,words=11,dist=['bundle','radial','sector'],locator='Definitions')
R(W,'R-EUC-06',"they are called ⟦supplements⟧ of each other.",lead='In Casey\'s Euclid, when the legs BA, AD form one right line, the two angles',words=3,dist=['complements','congruences','concurrents'][:3],locator='Definitions')
R(W,'R-EUC-07',"When the sum of two angles is a right angle, each is called the ⟦complement⟧ of the other.",lead=L,words=11,dist=['supplement','congruence','concurrent'],locator='Definitions')
R(W,'R-EUC-08',"Hence a triangle has ⟦six⟧ exterior angles;",lead=L,words=5,dist=['two','ten','one'],locator='Definitions')
R(W,'R-EUC-09',"1.—The sum of two supplemental angles is ⟦two⟧ right angles.",lead=L,words=8,dist=['six','ten','one'],locator='Book I')
R(W,'R-EUC-10',"A triangle whose three sides are unequal is said to be ⟦scalene⟧,",lead=L,words=10,dist=['regular','general','complex'],locator='Definitions')
R(W,'R-EUC-11',"A theorem consists of two parts, the hypothesis, or that which is assumed, and the ⟦conclusion⟧, or that which is asserted to follow therefrom.",lead=L,words=15,dist=['assumption','definition','conversion'],locator='Introduction')
R(W,'R-EUC-12',"Two theorems are said to be ⟦converse⟧, each of the other, when the hypothesis of either is the conclusion of the other.",lead=L,words=6,dist=['contrary','inverses','adjacent'],locator='Introduction')
R(W,'R-EUC-13',"The Exercises, of which there are over ⟦eight⟧ hundred, have been all selected with great care.",lead='In the Preface to Casey\'s Euclid:',words=7,dist=['seven','three','forty'],locator='Preface')
R(W,'R-EUC-14',"vii.—When a right line intersects two other right lines in two distinct points it makes with them ⟦eight⟧ angles, which have received special names in relation to one another.",lead=L,words=18,dist=['seven','three','forty'],locator='Book I')
R(W,'R-EUC-15',"6.—Each angle of an equilateral triangle is ⟦two-thirds⟧ of a right angle.",lead=L,words=9,dist=['two-fifths','one-fourth','two-sixths'],locator='Book I')
R(W,'R-EUC-16',"The three lines are called its ⟦sides⟧.",lead='In Casey\'s Euclid, of a triangle formed by three right lines joined end to end:',words=5,dist=['edges','faces','spans'],locator='Definitions')
R(W,'R-EUC-17',"The two lines are called the legs, and the point the ⟦vertex⟧ of the angle.",lead=L,words=11,dist=['summit','centre','corner'],locator='Definitions')
R(W,'R-EUC-18',"An acute-angled triangle is one that has its three angles ⟦acute⟧,",lead=L,words=10,dist=['right','equal','small'],locator='Definitions')
R(W,'R-EUC-19',"it is called a ⟦plane⟧.",lead='In Casey\'s Euclid, of a surface such that the right line joining any two arbitrary points in it lies wholly in the surface,',words=4,dist=['solid','curve','point'],locator='Introduction')
R(W,'R-EUC-20',"is called a rectilineal ⟦angle⟧.",lead='In Casey\'s Euclid, the inclination of two right lines extending out from one point in different directions',words=4,dist=['curve','plane','solid'],locator='Definitions')
def S(q,loc='Book I'): return (W,q,loc)
Q_SUP=S("1.—The sum of two supplemental angles is two right angles.")
Q_COMP=S("When the sum of two angles is a right angle, each is called the complement of the other.","Definitions")
Q_QUAD=S("3.—Since a quadrilateral can be divided into two triangles, the sum of its angles is equal to four right angles.")
Q_EQ=S("6.—Each angle of an equilateral triangle is two-thirds of a right angle.")
Q_ISO=S("1.—If a right-angled triangle be isosceles, each base angle is half a right angle.")
Q_TRI=S("the sum of the three internal angles is equal to two right angles")
Q_EXT=S("5.—If all the sides of any convex polygon be produced, the sum of the external angles is equal to four right angles.")
Q_PEN=S("A system of more than three concurrent lines is called a pencil of lines.","Definitions")
Q_SIX=S("Hence a triangle has six exterior angles","Definitions")
Q_FIVE=S("A polygon which has five sides is called a pentagon;","Definitions")
I(W,'I-EUC-01',"The sum of two supplemental angles is two right angles; if one of them is one right angle, the other is right angles numbering ","one",["two","six","ten"],[Q_SUP],"2 - 1 = 1 right angle.",keys=['supplemental'])
I(W,'I-EUC-02',"Two angles whose sum is a right angle are complements; the complement of one-third of a right angle is ","two-thirds",["two-fifths","one-fourth","two-sixths"],[Q_COMP],"1 - 1/3 = 2/3 of a right angle.",keys=['complement'])
I(W,'I-EUC-03',"A quadrilateral can be divided into two triangles and the sum of its angles is four right angles; a pentagon can be divided into three triangles, so the sum of its angles is equal to right angles numbering ","six",["two","ten","one"],[Q_QUAD,Q_TRI],"3 x 2 = 6.",keys=['pentagon'])
I(W,'I-EUC-04',"Each angle of an equilateral triangle is two-thirds of a right angle; the sum of the three angles is right angles numbering ","two",["six","ten","one"],[Q_EQ],"3 x 2/3 = 2.",keys=['equilateral'])
I(W,'I-EUC-05',"In a right-angled isosceles triangle each base angle is half a right angle; the sum of the base angles is ","one right angle",["two right angle","six right angle","ten right angle"],[Q_ISO],"1/2 + 1/2 = 1.",keys=['isosceles'])
I(W,'I-EUC-06',"The sum of the three internal angles of a triangle is two right angles; in a right-angled triangle the sum of the other two angles is ","one right angle",["two right angle","six right angle","ten right angle"],[Q_TRI],"2 - 1 = 1.",keys=['internal angles'])
I(W,'I-EUC-07',"The sum of the external angles of a convex polygon is four right angles and the sum of the internal angles of a quadrilateral is four right angles; for a quadrilateral the two sums together are right angles numbering ","eight",["seven","three","forty"],[Q_EXT,Q_QUAD],"4 + 4 = 8.",keys=['convex polygon'])
I(W,'I-EUC-08',"A pencil is a system of more than three concurrent lines; the least number of lines in a pencil is ","four lines",["five lines","nine lines","many lines"],[Q_PEN],"More than three means at least four.",keys=['pencil'])
I(W,'I-EUC-09',"A triangle has six exterior angles, formed by each side and the continuation of another; the number of exterior angles at each vertex is ","two",["six","ten","one"],[Q_SIX,S("An exterior angle of a triangle is one that is formed by any side and the continuation of another side.","Definitions")],"6 / 3 = 2.",keys=['exterior angles'])
I(W,'I-EUC-10',"A triangle has three sides and a pentagon has five sides; a pentagon has more sides than a triangle by ","two",["six","ten","one"],[Q_FIVE,S("A triangle is a figure formed by three right lines joined end to end.","Definitions")],"5 - 3 = 2.",keys=['pentagon'])
```

### A.q/items_more.py

```python
from qdef import I
N='notebooks'; O='opticks'; P='principia'; X='relativity'; E='euclid'
Q_COMP=(E,"When the sum of two angles is a right angle, each is called the complement of the other.","Definitions")
Q_SUP=(E,"1.—The sum of two supplemental angles is two right angles.","Book I")
# Notebooks
I(N,'I-NOT-12',"The stair of Vigevano has 130 steps, each 1/4 braccio high; its total height in braccia is ","32 1/2",["31 1/2","33 1/2","30 1/2"],[(N,"Stair of Vigevano below La Sforzesca, 130 steps, 1/4 braccio high and 1/2 braccio wide","Stair of Vigevano")],"130 x 1/4 = 32 1/2.",keys=['Vigevano'])
I(N,'I-NOT-13',"At Sforzesca Leonardo drew twenty five steps, 2/3 braccia to each; their total height in braccia is ","16 2/3",["15 2/3","17 2/3","18 2/3"],[(N,"At Sforzesca I drew twenty five steps, 2/3 braccia to each, and 8 braccia wide.","Sforzesca")],"25 x 2/3 = 16 2/3.",keys=['Sforzesca'])
I(N,'I-NOT-14',"When drawing from nature one stands at a distance of 3 times the height of the object; for an object 2 braccia high the distance is ","6 braccia",["5 braccia","4 braccia","8 braccia"],[(N,"When you draw from nature stand at a distance of 3 times the height of the object you wish to draw.","540. Of painting")],"3 x 2 = 6.",keys=['draw from nature'])
I(N,'I-NOT-15',"From Bonconventi to Casa Nova are 10 miles and from Casa Nova to Chiusi 9 miles; from Bonconventi to Chiusi by Casa Nova is ","19 miles",["18 miles","20 miles","29 miles"],[(N,"From Bonconventi to Casa Nova are 10 miles, from Casa Nova to Chiusi 9 miles","1052")],"10 + 9 = 19.",keys=['Bonconventi'])
I(N,'I-NOT-16',"Leonardo says 14 inches of fall are necessary in two thousand eight hundred braccia of flow; in 1400 braccia of flow the fall needed is ","7 inches",["6 inches","8 inches","9 inches"],[(N,"14 inches of fall are necessary in two thousand eight hundred braccia of flow in these canals","Canals")],"14 / 2 = 7.",keys=['inches of fall'])
I(N,'I-NOT-17',"At 3 dinari the square braccio the mile of canal amounts to 506 1/4 ducats; two miles amount to ","1012 1/2",["1006 1/4","1112 1/2","1013 1/2"],[(N,"At 3 dinari the square braccio, the mile will amount to 506 1/4 ducats","Canals")],"2 x 506 1/4 = 1012 1/2.",keys=['dinari'])
# Opticks
I(O,'I-OPT-07',"A telescope of 64 feet magnifies about 120 times and one of a foot magnifies 15 times; the larger magnifies more by the factor ","8",["6","4","9"],[(O,"a Telescope of 64 Feet in length, with an Aperture of 2-2/3 Inches, magnifies about 120 times, with as much distinctness as one of a Foot in length, with 1/3 of an Inch aperture, magnifies 15 times","Book I, Part I")],"120 / 15 = 8.",keys=['Telescope'])
I(O,'I-OPT-08',"Total reflexion out of glass into air begins at an angle of incidence of 40 Degrees 10 Minutes; the complement of that angle is ","49 Degrees 50 Minutes",["48 Degrees 50 Minutes","50 Degrees 10 Minutes","49 Degrees 10 Minutes"],[(O,"In passing out of Glass into Air, where the Refraction is measured by the Ratio of the Sines 20 to 31, the total Reflexion begins when the Angle of Incidence is 40 Degrees 10 Minutes;","Book II, Part III, Prop. 8"),Q_COMP],"90 deg - 40 deg 10' = 49 deg 50'.",keys=['total Reflexion'])
I(O,'I-OPT-09',"Total reflexion out of water into air begins at an angle of incidence of 48 Degrees 35 Minutes; the complement of that angle is ","41 Degrees 25 Minutes",["41 Degrees 35 Minutes","42 Degrees 25 Minutes","40 Degrees 25 Minutes"],[(O,"the total Reflexion begins when the Angle of Incidence is about 48 Degrees 35 Minutes.","Book II, Part III, Prop. 8"),Q_COMP],"90 deg - 48 deg 35' = 41 deg 25'.",keys=['Water into Air'])
I(O,'I-OPT-10',"Total reflexion out of glass into air begins at an angle of incidence of 40 Degrees 10 Minutes; the supplement of that angle is ","139 Degrees 50 Minutes",["138 Degrees 50 Minutes","140 Degrees 10 Minutes","139 Degrees 10 Minutes"],[(O,"In passing out of Glass into Air, where the Refraction is measured by the Ratio of the Sines 20 to 31, the total Reflexion begins when the Angle of Incidence is 40 Degrees 10 Minutes;","Book II, Part III, Prop. 8"),Q_SUP,(E,"When the sum of two angles BAC, CAD is such that the legs BA, AD form one right line, they are called supplements of each other.","Definitions")],"180 deg - 40 deg 10' = 139 deg 50'.",keys=['40 Degrees 10 Minutes'])
# Principia
I(P,'I-PRI-09',"Barrow was made Lucasian Professor of Mathematics in 1663 and resigned the chair in 1669; he held it for ","6 years",["5 years","7 years","9 years"],[(P,"was made Lucasian Professor of Mathematics in 1663, and soon afterward delivered his Optical Lectures:","Life of Newton"),(P,"Barrow, having resolved to devote himself to Theology, resigned the Lucasian Professorship of Mathematics, in 1669, in favour of Newton","Life of Newton")],"1669 - 1663 = 6.",keys=['Barrow'])
I(P,'I-PRI-10',"Newton was born on the 25th of December, 1642, and received the degree of Bachelor of Arts in 1665; he was then aged about ","23 years",["22 years","24 years","25 years"],[(P,"was born on the 25th of December, O. S. 1642","Life of Newton"),(P,"In the year 1665, he received the degree of Bachelor of Arts;","Life of Newton")],"1665 - 1642 = 23.",keys=['Bachelor of Arts','1642'])
I(P,'I-PRI-11',"Newton was born in 1642 and chosen President of the Royal Society in 1703; he was then aged about ","61 years",["60 years","62 years","63 years"],[(P,"was born on the 25th of December, O. S. 1642","Life of Newton"),(P,"In 1703, he was chosen President of the Royal Society of London","Life of Newton")],"1703 - 1642 = 61.",keys=['President'])
I(P,'I-PRI-12',"A wooden vessel was 4 feet long, 1 foot broad and 1 foot high; its content in cubic feet is ","4",["3","2","6"],[(P,"I procured a wooden vessel 4 feet long, 1 foot broad, and 1 foot high.","Book II")],"4 x 1 x 1 = 4.",keys=['wooden vessel'])
I(P,'I-PRI-13',"The clock went slower at the rate of 2' 13'' in 24 hours; in 48 hours it goes slower by ","4' 26''",["4' 16''","4' 36''","5' 26''"],[(P,"he found his clock to go slower than before at the rate of 2' 13'' in 24 hours.","Book III")],"2 x 2' 13'' = 4' 26''.",keys=['clock'])
# Relativity
I(X,'I-REL-06',"The earth travels in its orbit at about 30 kilometres per second; in ten seconds it travels about ","300 km",["200 km","400 km","600 km"],[(X,"a velocity of about 30 kilometres per second","Part I")],"10 x 30 = 300.",keys=['30 kilometres'])
I(X,'I-REL-07',"Light travels with a velocity of 300,000 km./sec.; in two seconds it travels ","600,000 km",["300,000 km","900,000 km","200,000 km"],[(X,"a velocity _c_ = 300,000 km./sec.","Part I")],"2 x 300,000 = 600,000.",keys=['300,000'])
I(X,'I-REL-08',"Newcomb reported in 1895 the unexplained motion of Mercury, and the book says astronomers were still working on the light deflection at the present time (1920); the interval is ","25 years",["35 years","15 years","24 years"],[(X,"(Leverrier: 1859; and Newcomb: 1895)","Appendix"),(X,"at the present time (1920) astronomers are working with great zeal","Appendix")],"1920 - 1895 = 25.",keys=['Newcomb'])
I(X,'I-REL-09',"Mercury's orbit rotates 43 seconds of arc per century; in half a century it is about ","21.5 seconds",["20.5 seconds","22.5 seconds","23.5 seconds"],[(X,"in the case of Mercury it must amount to 43 seconds of arc per century","Part II")],"43 / 2 = 21.5.",keys=['seconds of arc'])
# Plato
I('republic','I-PLA-11',"In the Republic there are five forms of the State and five of the soul; the total number of forms is ","ten",["six","two","one"][:3],[('republic',"There are five of the State, and five of the soul, I said.","Republic"),('republic',"Thus there are five forms in all, both of states and of souls...","Republic")],"5 + 5 = 10.",keys=['five forms'])
I('republic','I-PLA-12',"In the Republic every wrong is punished tenfold in a journey of a thousand years, and the life of man is reckoned at a hundred years; the penalty is paid in times numbering ","ten",["six","two","one"],[('republic',"He said that for every evil deed they were punished tenfold--now the journey was of a thousand years' duration, because the life of man was reckoned as a hundred years","Republic")],"1000 / 100 = 10.",keys=['tenfold'])
I('apology','I-PLA-13',"In the Apology Socrates is more than seventy years old; ten years more would make him at least ","eighty",["thirty","ninety","twenty"],[('apology',"For I am more than seventy years of age, and appearing now for the first time in a court of law","Apology")],"70 + 10 = 80.",keys=['seventy years'])
# Euclid
I(E,'I-EUC-11',"A pentagon has five sides and a hexagon six sides; the hexagon has more sides than the pentagon by ","one",["two","six","ten"],[(E,"A polygon which has five sides is called a pentagon;","Definitions"),(E,"one which has six sides, a hexagon, and so on.","Definitions")],"6 - 5 = 1.",keys=['hexagon'])
I(E,'I-EUC-12',"A triangle has six exterior angles and a triangle has three angles; the exterior angles outnumber the interior angles by a factor of ","two",["six","ten","one"],[(E,"Hence a triangle has six exterior angles","Definitions"),(E,"An acute-angled triangle is one that has its three angles acute","Definitions")],"6 / 3 = 2.",keys=['exterior angles'])
I('republic','I-PLA-14',"In the Republic a woman bears children from twenty until forty years of age and a man begets children from five-and-twenty until fifty-five; the man's period exceeds the woman's by years numbering ","ten",["six","two","one"],[('republic',"A woman, I said, at twenty years of age may begin to bear children to the State, and continue to bear them until forty;","Republic"),('republic',"a man may begin at five-and-twenty, when he has passed the point at which the pulse of life beats quickest, and continue to beget children until he be fifty-five.","Republic")],"30 - 20 = 10.",keys=['bear children'])
I(P,'I-PRI-14',"Newton was born on the 25th of December, 1642, and admitted to Trinity College on the 5th of June, 1660; his age in completed years at his admission was ","17 years",["16 years","18 years","19 years"],[(P,"was born on the 25th of December, O. S. 1642","Life of Newton"),(P,"was duly admitted, on the 5th day of June 1660, in the eighteenth year of his age.","Life of Newton")],"June 1660 minus Dec 1642 is 17 years and a half.",keys=['admitted'])
I(O,'I-OPT-11',"The Halo is at a distance of 22-1/2 Degrees from the Moon; the angle across the Halo from side to side is ","45 Degrees",["35 Degrees","55 Degrees","65 Degrees"],[(O,"The Halo at the distance of 22-1/2 Degrees from the Moon is of another sort.","Book II, Part IV")],"2 x 22-1/2 = 45.",keys=['Halo'])
I(X,'I-REL-10',"The book was first published in December, 1916, and speaks of the present time as (1920); the interval is ","4 years",["3 years","5 years","6 years"],[(X,"First Published: December, 1916","Colophon"),(X,"at the present time (1920) astronomers are working with great zeal","Appendix")],"1920 - 1916 = 4.",keys=['December, 1916'])
```

### A.e5_questions.py

```python
"""Mechanical checks, seeded position shuffle and hashing of the E5 question set (run after e5_texts.py and stage lists exist)."""
import json, re, sys, random, hashlib, os
import numpy as np
sys.path.insert(0,'/path/to/e5-work/q')
import qdef, items_opticks, items_notebooks, items_principia, items_plato, items_relativity, items_euclid, items_more, qt
E5='/path/to/e5-work/'
SEED=20261020
CAP={'recall':80,'inference':48}
K=12
def norm(s): return re.sub(r'\s+',' ',s)
st=json.load(open(E5+'build/stages_final.json'))   # {'T':{stage:[files]},'C':{...}}
def load(files): return norm(''.join(open(f,'rb').read().decode('utf-8','replace')+'\n' for f in files))
T={s:load(fs) for s,fs in st['T'].items()}; C=load(st['C']['prose_extra'])
allT=' '.join(T.values()); allC=' '.join([T['S1'],T['S2'],C])
# ---- rolling 12-byte hash index of the stems
def windows(b):
    a=np.frombuffer(b,dtype=np.uint8).astype(np.uint64); n=len(a)-K+1
    h=np.zeros(n,dtype=np.uint64)
    for j in range(K): h=h*np.uint64(1000003)+a[j:j+n]
    return h
items=qdef.ITEMS; stemb=[it['stem'].encode() for it in items]
def lcs_all(stems,texts):
    """longest common substring (bytes) of every stem against every text: 12-byte hash anchors over all texts in one pass, then extension."""
    index={}
    for n,st in enumerate(stems):
        if len(st)>=K:
            for i,h in enumerate(windows(st).tolist()): index.setdefault(h,[]).append((n,i))
    keys=np.array(sorted(index),dtype=np.uint64)
    best={name:[0]*len(stems) for name in texts}
    CH=1<<24
    for name,tb in texts.items():
        for off in range(0,len(tb),CH):
            seg=tb[off:off+CH+K]
            if len(seg)<K: break
            w=windows(seg); pos=np.nonzero(np.isin(w,keys))[0]
            for p in pos.tolist():
                for n,i in index.get(int(w[p]),[]):
                    st=stems[n]
                    if seg[p:p+K]!=st[i:i+K]: continue
                    a=i;b=p
                    while a>0 and b>0 and st[a-1]==seg[b-1]: a-=1;b-=1
                    e=i+K;f=p+K
                    while e<len(st) and f<len(seg) and st[e]==seg[f]: e+=1;f+=1
                    if e-a>best[name][n]: best[name][n]=e-a
    return best
Tb={k:v.encode() for k,v in T.items()}; Tb['C']=C.encode()
LC=lcs_all(stemb,Tb)
rows=[]; rng=random.Random(SEED)
for n,(it,sb) in enumerate(zip(items,stemb)):
    opts=[it['answer']]+it['distractors']; r={'id':it['id'],'kind':it['kind'],'fail':[]}
    if len(set(len(o.encode()) for o in opts))!=1: r['fail'].append('option-length')
    if len(set(opts))!=4: r['fail'].append('duplicate-option')
    # the full stem and stem+option absent from every teaching text (T and C)
    full=norm(it['stem'])
    if full in allT or full in allC: r['fail'].append('stem-in-stage')
    if any(norm(it['stem']+o) in allT or norm(it['stem']+o) in allC for o in opts): r['fail'].append('stem+option-in-stage')
    lc={k:LC[k][n] for k in Tb}; r['lcs']=lc
    if max(lc.values())>CAP[it['kind']]: r['fail'].append('lcs>%d'%CAP[it['kind']])
    # source quotes are in stage 5 and nowhere else in T (not in S1-S4, not in C)
    for q in it['quotes']:
        nq=norm(q['quote'])
        r.setdefault('quote_in',[]).append([k for k,v in T.items() if nq in v]+(['C'] if nq in C else []))
    # option counts in T-stage-5, all T, and C-prose
    r['count']={o:{'S5':T['S5'].count(o),'T':allT.count(o),'C':allC.count(o)} for o in opts}
    if it['kind']=='inference':
        ans=it['answer']; keys=[k.lower() for k in it.get('keys',[])]
        pat=re.compile(r'(?<![\w])'+re.escape(ans)+r'(?![\w])')
        co={}
        for name,txt in (('T',allT),('C',allC)):
            low=txt.lower(); n=0; tot=0
            for m in pat.finditer(txt):
                tot+=1
                w=low[max(0,m.start()-100):m.end()+100]
                if any(k in w for k in keys): n+=1
            co[name]={'answer_occurrences':tot,'with_key_within_100':n}
        r['cooccur']=co
        if co['T']['with_key_within_100']>0 and len(ans.encode())>=4: r['fail'].append('cooccurrence')
    rows.append(r)
json.dump(rows,open(E5+'build/q_checks.json','w'),indent=1)
```

### A.e5_freeze.py

```python
"""Freeze: drop failures, seeded option shuffle, write questions.json + SHA-256, logs, and sources.json."""
import json, sys, random, hashlib, os
sys.path.insert(0,'/path/to/e5-work/q')
import qdef, items_opticks, items_notebooks, items_principia, items_plato, items_relativity, items_euclid, items_more
E5='/path/to/e5-work/'; B=E5+'build/'; OUT=sys.argv[1]; SEED=20261020
chk={x['id']:x for x in json.load(open(B+'q_checks.json'))}
keep=[];drop=[]
for it in qdef.ITEMS:
    c=chk[it['id']]; why=list(c['fail'])
    if any((not q) or (set(q)-{'S5'}) for q in c['quote_in']): why.append('quote-not-in-trained-S5')
    (drop if why else keep).append((it,c,why))
qs=[]
for it,c,_ in keep:
    r=random.Random(f"{SEED}/{it['id']}"); opts=[it['answer']]+it['distractors']; r.shuffle(opts)
    cnt=[c['count'][o]['T'] for o in opts]; a=c['count'][it['answer']]['T']
    qs.append(dict(id=it['id'],kind=it['kind'],work=it['work'],stem=it['stem'],options=opts,answer_index=opts.index(it['answer']),
      option_bytes=len(opts[0].encode()),quotes=it['quotes'],reasoning=it['reasoning'],keys=it.get('keys',[]),
      lcs_bytes=max(c['lcs'].values()),option_counts_T=cnt,option_counts_C=[c['count'][o]['C'] for o in opts],
      answer_most_frequent_in_T=bool(a>max(v['T'] for k,v in c['count'].items() if k!=it['answer']))))
doc=dict(protocol_date='2026-10-20',seed=SEED,rule='positions shuffled by random.Random(f"{seed}/{id}")',questions=qs)
b=json.dumps(doc,indent=1,ensure_ascii=False).encode('utf-8')
os.makedirs(OUT,exist_ok=True)
open(OUT+'/questions.json','wb').write(b)
open(OUT+'/questions.sha256','w').write(hashlib.sha256(b).hexdigest()+'  questions.json\n')
json.dump(dict(dropped=[dict(id=i['id'],kind=i['kind'],why=w) for i,c,w in drop],kept={k:sum(q['kind']==k for q in qs) for k in ('recall','inference')}),open(OUT+'/dropped.json','w'),indent=1)
json.dump(list(chk.values()),open(OUT+'/checks.json','w'),indent=1)
st=json.load(open(B+'stages_final.json'))
def rec(f): d=open(f,'rb').read(); return dict(file=os.path.basename(f),bytes=len(d),sha256=hashlib.sha256(d).hexdigest())
json.dump(dict(bytes=st['bytes'],counts=st['counts'],T={s:[rec(f) for f in fs] for s,fs in st['T'].items()},C={s:[rec(f) for f in fs] for s,fs in st['C'].items()},heldout={s:[rec(f) for f in fs] for s,fs in st['heldout'].items()}),open(OUT+'/sources.json','w'),indent=1)
print(len(qs),doc and {k:sum(q['kind']==k for q in qs) for k in ('recall','inference')},[ (i['id'],w) for i,c,w in drop])
```

### A.e5_stats.py

```python
"""Exact statistics for the E5 criteria (binomial thresholds, sign-test power); writes stats.json."""
import json, sys
from math import comb
def pmf(n,k,p): return comb(n,k)*p**k*(1-p)**(n-k)
def tail(n,k,p): return sum(pmf(n,i,p) for i in range(k,n+1))
A=0.025; CH=0.25
q=json.load(open(sys.argv[1]+'/questions.json'))['questions']
N={k:sum(x['kind']==k for x in q) for k in ('recall','inference')}
out={'alpha_one_sided':A,'chance':CH,'N':N,'X':{},'power':{}}
for k,n in N.items(): out['X'][k]=next(i for i in range(n+1) if tail(n,i,CH)<=A)
def kmin(n): return next((i for i in range(n+2) if tail(n,i,.5)<=A),n+1)
def power(n,pT,pC):
    pb=pT*(1-pC); pc=(1-pT)*pC; d=pb+pc; tot=0
    for m in range(n+1):
        pm=pmf(n,m,d)
        if m==0 or pm<1e-12: continue
        k=kmin(m); tot+=pm*tail(m,k,pb/d)
    return tot
for k,n in N.items():
    out['power'][k]={f'T={pT},C=0.25':round(power(n,pT,.25),3) for pT in (.30,.35,.40,.45,.50,.60)}
out['min_gap_items']={k:{str(m):2*kmin(m)-m for m in (10,15,20,25,30)} for k in N}
json.dump(out,open(sys.argv[1]+'/stats.json','w'),indent=1); print(json.dumps(out))
```

### A.e5_headerscan.py

```python
"""Count leftover Project Gutenberg header/footer text in every trained stage text and in the C extra prose (writes header_scan.json)."""
import json,re,sys
st=json.load(open('/path/to/e5-work/build/stages_final.json')); res={}
for grp,files in list(st['T'].items())+[('C',st['C']['prose_extra'])]:
    sp=pr=0
    for f in files:
        t=open(f,'rb').read().decode('utf-8','replace')
        sp+=len(re.findall(r'SMALL PRINT',t)); pr+=len(re.findall(r'(?i)project gutenberg',t))
    res[grp]={'files':len(files),'SMALL PRINT':sp,'Project Gutenberg':pr}
json.dump(res,open(sys.argv[1],'w'),indent=1); print(json.dumps(res))
```

### A.excl_ids.json

```json
["29765","33504","1497","1580","1584","1598","1616","1635","1643","1656","1657","1658","1672","1687","1744","30155","5001","36114","28801","76404","5000","4998","4999","21076","7333","36276","69572","66944","29904","1177","1171"]
```

