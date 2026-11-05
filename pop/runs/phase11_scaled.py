"""Phase 11: scaled text -- does the hierarchy hold at realistic scale?

Phase 7b used 30KB train / 311-word vocab. This scales up: 150KB train,
20KB test, min_count=2 (larger vocab). Stronger baseline too (flat7).

Question: does the word-level population still beat flat n-grams when the
vocab is larger and the bigram is sparser? Or does sparsity kill it?

Protocol (written before the run): same WordPopulation as 7b, min_count=2.
Baselines flat5 and flat7. Expectation: population still wins (word
structure beats char n-grams even at scale), but the margin may shrink.
"""

import json
import time

import numpy as np

from runs.phase7b_realtext import WordPopulation, flat_bpb, load_corpus


def main():
    t0 = time.time()
    train, test = load_corpus(n_train=150000, n_test=20000)
    print(f"train {len(train)} chars, test {len(test)} chars")
    pop = WordPopulation(min_count=2)
    pop.train(train)
    print(f"vocab {len(pop.words)} words")
    pb = pop.bpb(test)
    f5 = flat_bpb(train, test, 5)
    f7 = flat_bpb(train, test, 7)
    gen_w = pop.generate_words(40, seed=3)
    out = {
        "protocol": "phase11_scaled_text",
        "train_chars": len(train),
        "test_chars": len(test),
        "vocab_size": len(pop.words),
        "population_bpb": round(pb, 4),
        "flat5_bpb": round(f5, 4),
        "flat7_bpb": round(f7, 4),
        "pop_word_generation": gen_w,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"population bpb: {pb:.4f}")
    print(f"flat5 bpb:      {f5:.4f}")
    print(f"flat7 bpb:      {f7:.4f}")
    print(f"--- word generation ---\n{gen_w}")
    with open("runs/results_phase11.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase11.json")


if __name__ == "__main__":
    main()
