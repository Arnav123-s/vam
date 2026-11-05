"""Phase 13: full-book text -- the hierarchy's boundary.

Phase 11 (150KB): population 2.75 vs flat5 2.81 (wins narrowly).
This scales to the full book: 500KB train / 50KB test.
Question: does the hierarchy hold, or does abundant data let flat win?
Result (logged): flat5 2.27 beats population 2.71 -- the advantage reverses.
Structure helps most under scarcity; data helps most under abundance.
"""

import json
import time

from runs.phase7b_realtext import WordPopulation, flat_bpb, load_corpus


def main():
    t0 = time.time()
    train, test = load_corpus(n_train=500000, n_test=50000)
    print(f"train {len(train)} chars, test {len(test)} chars")
    pop = WordPopulation(min_count=3)
    pop.train(train)
    print(f"vocab {len(pop.words)} words")
    pb = pop.bpb(test)
    f5 = flat_bpb(train, test, 5)
    gen = pop.generate_words(40, seed=5)
    out = {
        "protocol": "phase13_fullbook",
        "train_chars": len(train), "test_chars": len(test),
        "vocab_size": len(pop.words),
        "population_bpb": round(pb, 4), "flat5_bpb": round(f5, 4),
        "pop_word_generation": gen,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"population: {pb:.4f}  flat5: {f5:.4f}")
    print(gen)
    with open("runs/results_phase13.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase13.json")


if __name__ == "__main__":
    main()
