"""Experiment E1b: nested levels on the E1 streams (experiments/PROTOCOL.md, E1b).

    python experiments/e1b_nested.py          # writes experiments/results/e1b.json
"""

from __future__ import annotations

import time

import numpy as np
from _common import losses, provenance, save
from e1_known_sources import SEED, sources

from vam.nested import NestedConfig, NestedMachine
from vam.realizations.vam1 import Config

LAW = Config(cells=1, context=1, consolidating=True, dt=1.0)
MACHINES = {
    "N16": NestedConfig(LAW, depths=(2, 4, 8, 12, 16), table_bits=22),
    "N64": NestedConfig(LAW, depths=(8, 16, 24, 32, 48, 64), table_bits=22),
}
LOG2_BITS = 22


def main() -> None:
    t0 = time.perf_counter()
    rows = []
    for i, (name, src) in enumerate(sources().items()):
        sample = src.sample(1 << LOG2_BITS, np.random.default_rng(SEED + i))      # the E1 stream
        row = {"source": name, "seed": SEED + i, "oracle": losses(sample.oracle_q)}
        for label, cfg in MACHINES.items():
            nm = NestedMachine(cfg)
            row[label] = losses(nm.run(sample.bits))
            row[label + "_evictions"] = nm.evictions
        rows.append(row)
        print(name, {k: v for k, v in row.items() if k.startswith("N")}, flush=True)
    payload = {"experiment": "E1b nested levels on the known sources",
               "protocol": "experiments/PROTOCOL.md#e1b-nested-levels-on-the-known-sources",
               "provenance": provenance(seed=SEED, log2_bits=LOG2_BITS,
                                        machines={k: {"depths": list(c.depths), "table_bits": c.table_bits,
                                                      "readout": c.readout, "prior": c.prior}
                                                  for k, c in MACHINES.items()}),
               "rows": rows, "seconds": round(time.perf_counter() - t0, 1)}
    print(save("e1b", payload))


if __name__ == "__main__":
    main()
