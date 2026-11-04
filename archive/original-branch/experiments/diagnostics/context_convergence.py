"""Context einselection at 12 cells: do predictions converge as the bond grows? (no exact engine reaches here)

    python experiments/diagnostics/context_convergence.py 12 100

Run once on 2026-10-18 after vam1_mpo was written; output in context_convergence.txt. Four engines with
bonds 16, 32, 64 and 128 read the same text with equal generators until their emissions first differ.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vam.realizations.vam1 import Config
from vam.realizations.vam1_mpo import OperatorEngine

TEXT = (b"A textbook states a rule, shows it at work on an example, and then asks the reader to use it "
        b"alone. The machine reads every byte as eight bits, oldest bit first along the chain. ")
bits = [int(b) for b in np.unpackbits(np.frombuffer(TEXT, np.uint8))[:int(sys.argv[2])]]
cells = int(sys.argv[1])
for theta in (0.3, 0.6):
    cfg = Config(cells, 4, theta=theta)
    eng = {D: OperatorEngine(cfg, rng=np.random.default_rng(7), max_bond=D) for D in (16, 32, 64, 128)}
    q = {D: [] for D in eng}
    t0 = time.time()
    for b in bits:
        for D, e in eng.items():
            q[D].append(e.step(b))
        outs = {e.last_output for e in eng.values()}
        if len(outs) > 1:
            print("emissions differ at", len(q[16]), {D: e.last_output for D, e in eng.items()}, flush=True)
            break
    ref = np.array(q[128])
    row = {D: float(np.abs(np.array(q[D]) - ref).max()) for D in (16, 32, 64)}
    disc = {D: round(e.discarded / len(q[D]), 4) for D, e in eng.items()}
    print(f"c={cells} theta={theta}: max |q_D - q_128| over {len(ref)} bits: {row}; discarded HS per event {disc} "
          f"[{time.time() - t0:.0f}s]", flush=True)
