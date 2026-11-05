"""Phase 4d: hard transfer test (H2 x H4).

Phase 4c's honest limit: at toy scale (32 params), exact counting matches
transfer. This is the harder test where transfer MUST win if the principle
is right -- or the principle is wrong.

Design:
- 1024 micros (order 10). Shared 1024-dim shape s (the structural invariant).
- R0: 100k bits, abundant -> disentangle -> radiate s.
- R1, R2, R3: 3k bits each (scarce, ~3 visits/micro), different (alpha, beta).
- Transfer: 2-param SGD per regime with FIXED radiated s.
- Baseline: 1024-param tables per regime, learned from scratch.
- Metric: bpb on R1/R2/R3. Expectation: transfer wins BIG (baseline is
  hopelessly noisy at ~3 visits/micro; transfer's 2 params see all 3k bits).

Protocol (written before the run): single seed set (7,8,9); no tuning.
If transfer ties here, the disentangled-radiation principle fails outright.
"""

import json
import time

import numpy as np

from proto.disentangle import disentangle, TransferSlot
from proto.governor import Slot


def gen(seed=0, order=10, seg0=100_000, seg_rare=3_000):
    rng = np.random.default_rng(seed)
    nm = 2 ** order
    s_true = rng.normal(size=nm)
    s_true = (s_true - s_true.mean()) / s_true.std()
    # (alpha, beta) per regime: R0 common, R1-R3 rare and different
    params = [(0.5, 1.0), (-0.4, 0.8), (0.2, 1.2), (-0.1, 0.6)]
    segs = [seg0, seg_rare, seg_rare, seg_rare]
    bits, regs = [], []
    m = 0
    for ri, ((a, b), seg) in enumerate(zip(params, segs)):
        for _ in range(seg):
            p = 1 / (1 + np.exp(-(a + b * s_true[m])))
            bit = 1 if rng.random() < p else 0
            bits.append(bit)
            regs.append(ri)
            m = ((m << 1) | bit) & (nm - 1)
    return np.array(bits), np.array(regs), s_true, nm, seg0


def micro_at(bits, t, order=10):
    m = 0
    for b in bits[max(0, t - order):t]:
        m = (m << 1) | int(b)
    return m


def run(seed=0):
    order = 10
    bits, regs, s_true, nm, seg0 = gen(seed)
    # Phase A: learn R0 abundantly, disentangle, radiate shape
    slot = Slot(nm)
    for t in range(seg0):
        slot.observe(micro_at(bits, t, order), int(bits[t]))
    _, _, s_rad = disentangle(slot.counts)
    corr = float(np.corrcoef(s_rad, s_true)[0, 1])
    # Phase B: transfer vs baseline on the three scarce regimes
    lt, lb = [], []
    for ri in (1, 2, 3):
        tr = TransferSlot(s_rad, lr=0.3)
        base = Slot(nm)
        idx = np.where(regs == ri)[0]
        for t in idx:
            b = int(bits[t])
            m = micro_at(bits, t, order)
            pt = tr.predict(m)
            pb = min(max(base.predict(m), 1e-6), 1 - 1e-6)
            lt.append(-np.log2(pt if b else 1 - pt))
            lb.append(-np.log2(pb if b else 1 - pb))
            tr.observe(m, b)
            base.observe(m, b)
    return float(np.mean(lt)), float(np.mean(lb)), corr


def main():
    out = {"protocol": "phase4d_hard_transfer", "seeds": {}}
    t0 = time.time()
    for seed in (7, 8, 9):
        lt, lb, corr = run(seed)
        out["seeds"][seed] = {
            "transfer_bpb": round(lt, 4), "baseline_bpb": round(lb, 4),
            "shape_corr": round(corr, 4),
        }
        print(f"seed {seed}: transfer={lt:.4f} baseline={lb:.4f} "
              f"(gap={lb-lt:.4f}) shape_corr={corr:.3f}")
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase4d.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase4d.json")


if __name__ == "__main__":
    main()
