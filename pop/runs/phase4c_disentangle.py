"""Phase 4c experiment: disentangled structural radiation.

Protocol (written before the run):
- Source: order-5 (32 micros). TRUE p(1|m) = sigmoid(a + b*s[m]), s = fixed
  32-dim shape. R0: a=0.5, b=1.0 (20k bits). R1: a=-0.3, b=0.7 (5k bits).
  Same shape s, different calibration -- the invariant/variant split.
- Phase A: table slot learns R0 (32 params) -> disentangle -> radiate s.
- Phase B (R1):
    T: TransferSlot with radiated s, SGD on (alpha, beta) only (2 params).
    B: fresh 32-param table from scratch.
- Sanity: disentangled s correlates with true s (>0.95); (alpha,beta) converge.
- Metrics: bpb on Phase B. Expectation: T wins clearly (2 params vs 32).
"""

import json
import time

import numpy as np

from proto.disentangle import disentangle, TransferSlot
from proto.governor import Slot


def gen(seed=0, seg0=20000, seg1=5000, order=5):
    rng = np.random.default_rng(seed)
    nm = 2 ** order
    s_true = rng.normal(size=nm)
    s_true = (s_true - s_true.mean()) / s_true.std()
    bits = []
    m = 0
    for a, b, seg in [(0.5, 1.0, seg0), (-0.3, 0.7, seg1)]:
        for _ in range(seg):
            p = 1 / (1 + np.exp(-(a + b * s_true[m])))
            bit = 1 if rng.random() < p else 0
            bits.append(bit)
            m = ((m << 1) | bit) & (nm - 1)
    return np.array(bits), s_true, nm, seg0


def micro_at(bits, t, order=5):
    m = 0
    for b in bits[max(0, t - order):t]:
        m = (m << 1) | int(b)
    return m


def run(seed=0):
    bits, s_true, nm, seg0 = gen(seed)
    order = 5
    # Phase A: learn R0's table
    slot = Slot(nm)
    for t in range(seg0):
        b = int(bits[t])
        m = micro_at(bits, t, order)
        slot.observe(m, b)
    # Disentangle and radiate the shape
    alpha0, beta0, s_rad = disentangle(slot.counts)
    corr = float(np.corrcoef(s_rad, s_true)[0, 1])
    # Phase B: transfer vs baseline on R1
    tr = TransferSlot(s_rad, lr=0.3)
    base = Slot(nm)
    lt, lb = [], []
    for t in range(seg0, len(bits)):
        b = int(bits[t])
        m = micro_at(bits, t, order)
        pt = tr.predict(m)
        pb = min(max(base.predict(m), 1e-6), 1 - 1e-6)
        lt.append(-np.log2(pt if b else 1 - pt))
        lb.append(-np.log2(pb if b else 1 - pb))
        tr.observe(m, b)
        base.observe(m, b)
    return (float(np.mean(lt)), float(np.mean(lb)), corr,
            (float(tr.alpha), float(tr.beta)))


def main():
    out = {"protocol": "phase4c_disentangle", "seeds": {}}
    t0 = time.time()
    for seed in (7, 8, 9):
        lt, lb, corr, (a, b) = run(seed)
        out["seeds"][seed] = {
            "transfer_bpb": round(lt, 4), "baseline_bpb": round(lb, 4),
            "shape_corr": round(corr, 4),
            "alpha_beta": [round(a, 3), round(b, 3)],
        }
        print(f"seed {seed}: transfer={lt:.4f} baseline={lb:.4f} | "
              f"shape_corr={corr:.3f} (alpha,beta)=({a:.2f},{b:.2f}) "
              f"[true (-0.3, 0.7)]")
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase4c.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase4c.json")


if __name__ == "__main__":
    main()
