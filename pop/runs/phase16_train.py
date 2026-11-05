"""Phase 16: TRAIN IT -- unified machine on a multi-book corpus.

Corpus: 5 Gutenberg books (Pride & Prejudice, Sherlock Holmes, Frankenstein,
Moby Dick, Alice in Wonderland) ~3.1MB raw.
Architecture: word population (bigram + deterministic spelling + char-4
backoff) with governor purge min_count=5 (Phase 15 sweet spot).
Splits:
  - train: 2MB from P&P + Holmes + Frankenstein + Moby Dick.
  - test-in: held-out 50KB from the same 4 books.
  - test-out: Alice in Wonderland (never trained on) -- generalization test.
Baselines: flat5 char model, and Phase 11 single-book population (2.75).
Protocol (written before the run): measure bpb on both test sets.
"""

import json
import re
import time

from runs.phase7b_realtext import WordPopulation, flat_bpb


def clean(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        t = f.read()
    t = t.lower()
    # strip Gutenberg header/footer
    start = t.find("*** start of")
    if start >= 0:
        t = t[t.find("\n", start) + 1:]
    end = t.find("*** end of")
    if end >= 0:
        t = t[:end]
    t = re.sub(r"[^a-z ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    t0 = time.time()
    books = {
        "pride": clean("/tmp/pride.txt"),
        "holmes": clean("/tmp/book_1661.txt"),
        "frank": clean("/tmp/book_84.txt"),
        "moby": clean("/tmp/book_2701.txt"),
        "alice": clean("/tmp/book_11.txt"),
    }
    for k, v in books.items():
        print(f"{k}: {len(v)} chars")
    # train: 500KB from each of the 4 training books (2MB total)
    train = "".join(books[k][:500000] for k in ("pride", "holmes", "frank", "moby"))
    # test-in: next 50KB from pride + holmes (held out)
    test_in = books["pride"][500000:525000] + books["holmes"][500000:525000]
    test_in = test_in[:50000]
    # test-out: alice (never seen)
    test_out = books["alice"][:50000]
    print(f"train {len(train)} chars, test-in {len(test_in)}, test-out {len(test_out)}")

    pop = WordPopulation(min_count=5)
    pop.train(train)
    print(f"vocab {len(pop.words)} words")

    pb_in = pop.bpb(test_in)
    pb_out = pop.bpb(test_out)
    f5_in = flat_bpb(train, test_in, 5)
    f5_out = flat_bpb(train, test_out, 5)
    gen = pop.generate_words(40, seed=11)
    out = {
        "protocol": "phase16_train_multibook",
        "train_chars": len(train),
        "vocab_size": len(pop.words),
        "pop_bpb_test_in": round(pb_in, 4),
        "flat5_bpb_test_in": round(f5_in, 4),
        "pop_bpb_test_out": round(pb_out, 4),
        "flat5_bpb_test_out": round(f5_out, 4),
        "pop_word_generation": gen,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"test-in : pop {pb_in:.4f}  flat5 {f5_in:.4f}")
    print(f"test-out: pop {pb_out:.4f}  flat5 {f5_out:.4f}")
    print(f"--- generation ---\n{gen}")
    with open("runs/results_phase16.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase16.json")


if __name__ == "__main__":
    main()
