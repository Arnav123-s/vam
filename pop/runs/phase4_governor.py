"""Phase 4 experiment: governor as criticality (H2) under slot pressure.

Protocol (written before the run):
- Source: 6 regimes with biases {0.15,0.3,0.4,0.6,0.7,0.85}, i.i.d. bits per
  regime; sequence 1,2,3,4,5,6,1,2,3 (each 20k bits; 180k total). Regimes RETURN.
- Population: 4 slots (fewer than regimes). Oracle regime id as slow key --
  the test is slot management, not regime detection.
- Models:
    G:  full governor (entropy monitor, purge stalest, radiate converged,
        restore invariants on return).
    B1: slot reuse without radiate/restore (purge stalest, reset counts).
    B2: static (first 4 regimes keep slots; rest contaminate via mod).
- Metrics: bpb per segment and total; governor action log; entropy vs capacity.
- Expectation: G wins on regimes 5,6 (clean slots) and dominates on returns
  1,2,3 (instant restore). B1 vs G isolates radiate/restore; B2 vs B1 isolates
  purging.
"""

import json
import time

import numpy as np

from proto.governor import Governor, Slot


BIAS = {1: 0.15, 2: 0.30, 3: 0.40, 4: 0.60, 5: 0.70, 6: 0.85}
SEQ = [1, 2, 3, 4, 5, 6, 1, 2, 3]
SEG = 20_000


def gen(seed=0):
    rng = np.random.default_rng(seed)
    bits, regs = [], []
    for r in SEQ:
        b = (rng.random(SEG) < BIAS[r]).astype(int)
        bits.extend(b)
        regs.extend([r] * SEG)
    return np.array(bits), np.array(regs)


def run_governed(bits, regs):
    gov = Governor(n_slots=4, n_micro=4)
    seg_losses = [[] for _ in SEQ]
    total = []
    cur = None
    slot_i = None
    seg_i = -1
    for t, (b, r) in enumerate(zip(bits, regs)):
        b = int(b)
        if r != cur:
            slot_i = gov.on_regime(t, r)
            cur = r
            seg_i += 1
        s = gov.slots[slot_i]
        m = int(bits[t - 2]) * 2 + int(bits[t - 1]) if t >= 2 else 0
        p = min(max(s.predict(m), 1e-6), 1 - 1e-6)
        l = -np.log2(p if b else 1 - p)
        total.append(l)
        seg_losses[seg_i].append(l)
        s.observe(m, b)
        gov.tick(t)
    return float(np.mean(total)), [float(np.mean(s)) for s in seg_losses], gov


def run_b1(bits, regs):
    # Slot reuse: purge stalest (reset counts), no radiate/restore.
    slots = [Slot(4) for _ in range(4)]
    total = []
    cur = None
    slot_i = None
    recent = [0] * 4
    for t, (b, r) in enumerate(zip(bits, regs)):
        b = int(b)
        if r != cur:
            hit = next((i for i, s in enumerate(slots) if s.regime == r), None)
            if hit is None:
                free = next((i for i, s in enumerate(slots)
                             if s.regime is None), None)
                slot_i = free if free is not None else int(np.argmin(recent))
                slots[slot_i] = Slot(4)
                slots[slot_i].regime = r
                recent[slot_i] = 0
            else:
                slot_i = hit
            cur = r
        s = slots[slot_i]
        m = int(bits[t - 2]) * 2 + int(bits[t - 1]) if t >= 2 else 0
        p = min(max(s.predict(m), 1e-6), 1 - 1e-6)
        total.append(-np.log2(p if b else 1 - p))
        s.observe(m, b)
        recent[slot_i] += 1
        if t % 5000 == 0:
            recent = [0] * 4
    return float(np.mean(total))


def run_b2(bits, regs):
    # Static: first 4 regimes keep slots; others contaminate via mod.
    slots = [Slot(4) for _ in range(4)]
    for i, r in enumerate([1, 2, 3, 4]):
        slots[i].regime = r
    total = []
    for t, (b, r) in enumerate(zip(bits, regs)):
        b = int(b)
        slot_i = (r - 1) % 4
        s = slots[slot_i]
        m = int(bits[t - 2]) * 2 + int(bits[t - 1]) if t >= 2 else 0
        p = min(max(s.predict(m), 1e-6), 1 - 1e-6)
        total.append(-np.log2(p if b else 1 - p))
        s.observe(m, b)
    return float(np.mean(total))


def main():
    out = {"protocol": "phase4_governor", "segment_bpb": {}}
    t0 = time.time()
    bits, regs = gen(seed=0)
    g, g_seg_list, gov = run_governed(bits, regs)
    b1 = run_b1(bits, regs)
    b2 = run_b2(bits, regs)
    # per-segment bpb for G: segment i <-> regime SEQ[i]
    g_seg = {}
    for i, r in enumerate(SEQ):
        g_seg.setdefault(str(r), []).append(round(g_seg_list[i], 4))
    out["segment_bpb"] = g_seg
    out["total"] = {"governor": round(g, 4), "reuse_no_radiate": round(b1, 4),
                    "static": round(b2, 4)}
    out["actions"] = {}
    for t, a, r, i in gov.log:
        out["actions"][a] = out["actions"].get(a, 0) + 1
    out["invariants_radiated"] = sorted(int(k) for k in gov.invariants.keys())
    s0, s1 = gov.entropy_trace[0], gov.entropy_trace[-1]
    out["entropy_vs_capacity"] = {
        "start": [round(s0[1], 2), round(s0[2], 2)],
        "end": [round(s1[1], 2), round(s1[2], 2)]}
    print(f"total bpb: G={g:.4f} B1={b1:.4f} B2={b2:.4f}")
    print("per-segment G bpb:", {r: g_seg[str(r)] for r in sorted(set(SEQ))})
    print("actions:", out["actions"], "| radiated:", out["invariants_radiated"])
    print(f"entropy S/C: {s0[1]:.1f}/{s0[2]:.1f} -> {s1[1]:.1f}/{s1[2]:.1f}")
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase4.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase4.json")


if __name__ == "__main__":
    main()
