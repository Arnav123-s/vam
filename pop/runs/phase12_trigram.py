"""Phase 12: trigram word model -- richer slow-scale structure.

Phase 11: bigram word model wins (2.75 vs 2.81 flat5). Question: does a
richer slow scale (trigram: p(word | prev 2 words)) help further, or does
sparsity kill it? Back off trigram -> bigram -> unigram.

Protocol (written before the run): same 150KB/20KB split as Phase 11.
Expectation: trigram <= bigram (more context helps when seen, backoff
covers the rest). If trigram is worse, sparsity dominates at this scale.
"""

import json
import time
from collections import Counter

import numpy as np

from runs.phase7b_realtext import WordPopulation, flat_bpb, load_corpus


class TriPopulation(WordPopulation):
    def __init__(self, min_count=2):
        super().__init__(min_count)
        self.trigram = {}  # (a, b) -> Counter(c)

    def train(self, text):
        super().train(text)
        words = text.split()
        ids = [self.vocab[w] for w in words if w in self.vocab]
        for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
            self.trigram.setdefault((a, b), Counter())[c] += 1

    def predict_dist(self, hist):
        # At word boundary, use trigram if available, else fall back to
        # parent (bigram) logic.
        last_space = hist.rfind(" ")
        prefix = hist[last_space + 1:]
        if prefix == "":
            prev_words = hist[:last_space].split() if last_space > 0 else []
            if len(prev_words) >= 2:
                a = prev_words[-2]
                b = prev_words[-1]
                if a in self.vocab and b in self.vocab:
                    key = (self.vocab[a], self.vocab[b])
                    if key in self.trigram:
                        tg = self.trigram[key]
                        tot = sum(tg.values())
                        dist = {}
                        for ni, cnt in tg.items():
                            ch = self.words[ni][0]
                            dist[ch] = dist.get(ch, 0) + cnt / tot
                        return self._smooth(dist)
            return super().predict_dist(hist)
        return super().predict_dist(hist)


def main():
    t0 = time.time()
    train, test = load_corpus(n_train=150000, n_test=20000)
    tri = TriPopulation(min_count=2)
    tri.train(train)
    print(f"vocab {len(tri.words)} words, {len(tri.trigram)} trigrams")
    pb = tri.bpb(test)
    gen_w = tri.generate_words(40, seed=3)
    out = {
        "protocol": "phase12_trigram",
        "trigram_bpb": round(pb, 4),
        "bigram_bpb_phase11": 2.7506,
        "flat5_bpb_phase11": 2.8111,
        "vocab_size": len(tri.words),
        "n_trigrams": len(tri.trigram),
        "pop_word_generation": gen_w,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"trigram bpb: {pb:.4f} (bigram was 2.7506)")
    print(f"--- word generation ---\n{gen_w}")
    with open("runs/results_phase12.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase12.json")


if __name__ == "__main__":
    main()
