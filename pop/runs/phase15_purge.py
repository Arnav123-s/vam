"""Phase 15: unified text -- governor purge of rare words.

Phase 11 word population at 150KB. Purge = drop words with count < N.
Question: does purging the rare tail help (less noise) or hurt (less
coverage)? Sweep min_count over {2, 5, 10, 20}.
Result (logged): sweet spot at 5 (2.727) -- prune the tail, keep the core.
"""

import json
import time

from runs.phase7b_realtext import WordPopulation, load_corpus


def main():
    t0 = time.time()
    train, test = load_corpus(n_train=150000, n_test=20000)
    out = {"protocol": "phase15_unified_text"}
    for mc in (2, 5, 10, 20):
        pop = WordPopulation(min_count=mc)
        pop.train(train)
        pb = pop.bpb(test)
        out[f"min_count_{mc}"] = {"vocab": len(pop.words), "bpb": round(pb, 4)}
        print(f"min_count={mc:2d}: vocab={len(pop.words):4d} bpb={pb:.4f}")
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase15.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase15.json")


if __name__ == "__main__":
    main()
