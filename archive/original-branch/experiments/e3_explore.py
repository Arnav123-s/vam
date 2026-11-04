"""E3 exploration past the grid's edge (experiments/PROTOCOL.md, E3, amendment of 2026-10-22, post hoc).

    python experiments/e3_explore.py --manifest experiments/data/pg19/MANIFEST.json

Part 2 chose the smallest sink rate and the smallest protection in its grid. This run scores, on
A-validation only and after Part 2 had ended, the chosen configuration with weaker sinks (down to none)
and weaker protection. It is reported apart from E3 and changes nothing E3 chose or measured. The first
row recomputes the chosen configuration, which must give Part 2's value exactly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from _common import RESULTS, commit, provenance, save
from e2_curriculum import frozen, make, read, streams
from e3_nfa import key, plain

GRID = [{"sink_rate": 0.05}, {"sink_rate": 0.025}, {"sink_rate": 0.0},
        {"protect": 0.25, "sink_rate": 0.1}, {"protect": 0.25, "sink_rate": 0.05}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    args = ap.parse_args()
    e3 = json.loads((RESULTS / "e3.json").read_text())
    chosen, rows = e3["part2"]["chosen"], {r["key"]: r for r in e3["part2"]["rows"]}
    S = streams(args.manifest)
    path = RESULTS / "e3_explore_partial.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    out = []
    for change in [{}, *GRID]:
        spec = {**chosen, **change}
        k = key(spec)
        if k not in done:
            t0 = time.perf_counter()
            m = make(spec)
            train = read(m, S["A_train"], learning=True)
            done[k] = {"spec": spec, "A_train_bpb": train, "A_validation_bpb": frozen(m, S["A_validation"]),
                       "evictions": m.evictions, "refused": m.refused, "protected": m.protected,
                       "seconds": round(time.perf_counter() - t0, 1), "commit": commit()}
            path.write_text(json.dumps(plain(done)))
            print(" ", {x: done[k][x] for x in ("spec", "A_validation_bpb", "seconds")}, flush=True)
        out.append(done[k])
    reproduced = out[0]["A_validation_bpb"] == rows[key(chosen)]["A_validation_bpb"]
    if not reproduced:
        raise RuntimeError("the chosen configuration does not reproduce Part 2's value")
    print(save("e3_explore", plain({"experiment": "E3 exploration past the grid's edge (post hoc)",
                                    "provenance": provenance(manifest=str(args.manifest)),
                                    "chosen": chosen, "reproduced": reproduced, "rows": out})))


if __name__ == "__main__":
    main()
