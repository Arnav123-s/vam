"""Phase 18b: present-state learner at 2MB -- the decisive scale test.

Phase 18 (150KB): present-state 2.58 beats population 2.75.
Phase 16 (2MB): population 2.98 vs flat5 2.03 -- the population fails.
Question: does the faithful present-state learner also fail at 2MB, or
does the scale limit belong to the count-based implementation only?
Result (logged): buckets=65536, eta=0.3: test-in 2.074, test-out 2.115.
  vs flat5 2.028 / 2.261. Essentially tied in-distribution, BEATS flat5
  out-of-distribution. The scale limit was my artifact.
"""

import json
import time

from runs.phase18_present_state import PresentStateLearner
from runs.phase16_train import clean


def main():
    t0 = time.time()
    books = {k: clean(f"/tmp/{p}") for k, p in [
        ("pride", "pride.txt"), ("holmes", "book_1661.txt"),
        ("frank", "book_84.txt"), ("moby", "book_2701.txt"),
        ("alice", "book_11.txt")]}
    train = "".join(books[k][:500000] for k in ("pride", "holmes", "frank", "moby"))
    test_in = books["pride"][500000:525000]
    test_out = books["alice"][:50000]
    out = {"protocol": "phase18b_present_state_2MB"}
    for buckets, eta in [(16384, 0.3), (65536, 0.3)]:
        m = PresentStateLearner(n_buckets=buckets, eta=eta)
        m.bpb(train)
        pb_in = m.bpb(test_in)
        m2 = PresentStateLearner(n_buckets=buckets, eta=eta)
        m2.bpb(train)
        pb_out = m2.bpb(test_out)
        out[f"buckets_{buckets}_eta_{eta}"] = {
            "test_in": round(pb_in, 4), "test_out": round(pb_out, 4)}
        print(f"buckets={buckets} eta={eta}: test-in {pb_in:.4f}  test-out {pb_out:.4f}")
    out["phase16_pop_test_in"] = 2.9777
    out["phase16_flat5_test_in"] = 2.0281
    out["phase16_pop_test_out"] = 3.2156
    out["phase16_flat5_test_out"] = 2.2613
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase18b.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase18b.json")


if __name__ == "__main__":
    main()
