"""Phase 17: FIND GAPS -- why the population loses at 2MB, and the fix.

Phase 16: pop 2.98 vs flat5 2.03 at 2MB. Gaps found by diagnosis:

Gap 1 (CRITICAL): the word path HURTS. At word boundaries the bigram
  predicts at 3.89 bpb vs the char path's 2.61. The bigram is too diffuse
  (mean 22.9 distinct next-words, top-1 accuracy only 11.4%) and the model
  commits to it raw instead of interpolating with the char model.
Gap 2: interpolation does NOT fix it. Tested: bigram 0.5 + char-4 0.5 at
  boundaries gives 2.8284 vs baseline 2.8285 on the same slice -- no
  change. The bigram is so diffuse that mixing cannot rescue it, and the
  char dist at a word boundary is itself uncertain (first letters are
  inherently hard to predict).
Gap 3: char backoff is order-4 while the flat baseline is order-5. The
  population's char component is weaker than the baseline it is compared
  against.
Gap 4: the bigram only fires on the first character of each word (17.7% of
  chars). The rest is deterministic spelling or char backoff, so the word
  level's influence is small AND negative.
Gap 5 (fundamental): at 2MB the char 5-gram is too strong. The hierarchy's
  advantage is a low-data phenomenon (Phase 13). This is a boundary, not a
  bug.

Fix tested: interpolate bigram 0.5 + char-4 0.5 at boundaries.
  Same-slice result: baseline 2.8285 -> interp 2.8284 (no change).
  An earlier 2.98 -> 2.83 comparison mixed different test slices and was
  wrong; corrected here. The gap is structural, not a smoothing problem.
  Conclusion: the word bigram is too weak at this scale. The architecture
  needs richer word context (trigram) or a different approach entirely.
  The hierarchy remains a low-data architecture (Phase 13).
"""

import json
import time
from collections import Counter, defaultdict

import numpy as np

from runs.phase7b_realtext import WordPopulation, flat_bpb
from runs.phase16_train import clean


class CharModel:
    def __init__(self, text, n=4):
        self.n = n
        self.counts = defaultdict(Counter)
        for i in range(n, len(text)):
            self.counts[text[i - n:i]][text[i]] += 1

    def dist(self, hist):
        ctx = hist[-self.n:]
        c = self.counts.get(ctx, Counter())
        tot = sum(c.values())
        if tot == 0:
            return {}
        return {ch: v / tot for ch, v in c.items()}


def bpb_mixed(pop, cm, test, alpha=0.5):
    lps = []
    hist = ""
    for ch in test:
        d = pop.predict_dist(hist)
        last_space = hist.rfind(" ")
        prefix = hist[last_space + 1:]
        at_boundary = prefix == "" and last_space > 0
        if at_boundary:
            pw = hist[:last_space].split()
            if pw and pw[-1] in pop.vocab and pop.vocab[pw[-1]] in pop.bigram:
                cd = cm.dist(hist)
                mixed = {}
                for k in set(d) | set(cd):
                    mixed[k] = alpha * d.get(k, 0) + (1 - alpha) * cd.get(k, 0)
                d = mixed
        lps.append(-np.log2(max(d.get(ch, 1e-6), 1e-6)))
        hist = (hist + ch)[-50:]
    return float(np.mean(lps))


def main():
    t0 = time.time()
    books = {k: clean(f"/tmp/{p}") for k, p in [
        ("pride", "pride.txt"), ("holmes", "book_1661.txt"),
        ("frank", "book_84.txt"), ("moby", "book_2701.txt")]}
    train = "".join(books[k][:500000] for k in ("pride", "holmes", "frank", "moby"))
    test = books["pride"][500000:525000]
    pop = WordPopulation(min_count=5)
    pop.train(train)
    cm = CharModel(train, 4)
    base = pop.bpb(test[:20000])
    interp = bpb_mixed(pop, cm, test[:20000], alpha=0.5)
    f5 = flat_bpb(train, test[:20000], 5)
    # diagnostics
    words = test.split()
    known = sum(1 for w in words[:-1] if w in pop.vocab and pop.vocab[w] in pop.bigram)
    sizes = [len(c) for c in pop.bigram.values()]
    correct = tot = 0
    for a, b in zip(words[:-1], words[1:]):
        if a in pop.vocab and pop.vocab[a] in pop.bigram:
            top = pop.bigram[pop.vocab[a]].most_common(1)[0][0]
            tot += 1
            if pop.words[top] == b:
                correct += 1
    out = {
        "protocol": "phase17_find_gaps",
        "baseline_bpb": round(base, 4),
        "interp_bpb": round(interp, 4),
        "flat5_bpb": round(f5, 4),
        "bigram_coverage": round(known / (len(words) - 1), 4),
        "bigram_mean_distinct_next": round(float(np.mean(sizes)), 1),
        "bigram_top1_acc": round(correct / tot, 4),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"baseline: {base:.4f}  interp: {interp:.4f}  flat5: {f5:.4f}")
    print(f"bigram coverage {known/(len(words)-1):.1%}, mean distinct next {np.mean(sizes):.1f}, top-1 {correct/tot:.1%}")
    with open("runs/results_phase17.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase17.json")


if __name__ == "__main__":
    main()
