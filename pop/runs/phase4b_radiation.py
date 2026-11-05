"""Phase 4b: structural radiation (H2 x H4).

Phase 4's reframing: rebuilding COUNTS is cheap; the radiated invariant must be
STRUCTURAL. Here the invariant is the micro-pattern SHAPE: regime 0's XOR-like
p(1|micro) = [0.55, 0.95, 0.95, 0.55], centered -> [-0.2,+0.2,+0.2,-0.2].
Regime 1's true shape is the FLIP: [+0.2,-0.2,-0.2,+0.2].

Transfer: when regime 1 arrives, the new slot doesn't learn 4 micros from
scratch. It tests TWO structural hypotheses (radiated shape, flipped shape) as
pseudo-counts, runs a 200-bit shootout, keeps the winner. One bit of structure
+ the marginal vs 4 independent micros.

Protocol (written before the run):
- regime_flip source, oracle regime key. Sequence: R0 (20k), R1 (20k),
  R0 return (20k), R1 return (20k).
- Models:
    T: transfer -- radiate R0's shape at convergence; R1 gets shape/-shape
       shootout (200 bits); returns restore radiated tables directly.
    B: baseline -- fresh slots per regime, relearn from scratch; returns relearn.
- Metrics: bpb per segment and total. Expectation: T wins big on R1's segment
  (structure transfer) and on returns (restore).
"""

import json
import time

import numpy as np

from proto.governor import Slot


def get_shape(slot):
    """Centered p(1|micro) = the structural invariant."""
    p = np.array([slot.predict(m) for m in range(4)])
    return p - p.mean()


def shape_to_counts(shape, scale=40.0, base=0.5):
    """Pseudo-counts encoding a shape hypothesis around base rate."""
    probs = np.clip(base + shape, 0.05, 0.95)
    counts = np.zeros((4, 2))
    counts[:, 1] = probs * scale
    counts[:, 0] = (1 - probs) * scale
    return counts


def run_transfer(seed=0, seg=20_000):
    bits, regs = gen_forced(seed, seg)

    # --- Transfer model ---
    slots = {}            # regime -> active Slot
    radiated_shape = None  # structural invariant from R0
    radiated_tables = {}  # regime -> frozen counts
    seg_bpb_T = []
    cur, shoot = None, None
    for t, (b, r) in enumerate(zip(bits, regs)):
        b = int(b)
        m = int(bits[t - 2]) * 2 + int(bits[t - 1]) if t >= 2 else 0
        if r != cur:
            # radiate the outgoing regime
            if cur is not None:
                radiated_tables[cur] = slots[cur].counts.copy()
                if cur == 0 and radiated_shape is None:
                    radiated_shape = get_shape(slots[cur])
            # incoming regime
            if r in radiated_tables and not (r == 1 and radiated_shape is not None
                                             and 1 not in radiated_tables):
                slots[r] = Slot(4)
                slots[r].counts = radiated_tables[r].copy()  # direct restore
            elif r == 1 and radiated_shape is not None:
                # STRUCTURAL TRANSFER: shape/-shape shootout, 200 bits
                cA, cB = Slot(4), Slot(4)
                cA.counts = shape_to_counts(radiated_shape)
                cB.counts = shape_to_counts(-radiated_shape)
                shoot = [cA, cB, 0.0, 0.0, t]
                slots[r] = cA  # placeholder; winner takes over
            else:
                slots[r] = Slot(4)
            cur = r
            seg_losses = []
        if shoot is not None:
            cA, cB, la, lb, t0s = shoot
            pa = min(max(cA.predict(m), 1e-6), 1 - 1e-6)
            pb = min(max(cB.predict(m), 1e-6), 1 - 1e-6)
            la += -np.log2(pa if b else 1 - pa)
            lb += -np.log2(pb if b else 1 - pb)
            cA.observe(m, b)
            cB.observe(m, b)
            l = -np.log2(pa if b else 1 - pa)
            shoot = [cA, cB, la, lb, t0s] if t - t0s < 200 else None
            if shoot is None:
                slots[1] = cA if la < lb else cB  # winner takes over
        else:
            act = slots[r]
            p = min(max(act.predict(m), 1e-6), 1 - 1e-6)
            l = -np.log2(p if b else 1 - p)
            act.observe(m, b)
        seg_losses.append(l)
        if (t + 1) % seg == 0:
            seg_bpb_T.append(float(np.mean(seg_losses)))

    # --- Baseline: fresh slots, relearn everything ---
    seg_bpb_B = []
    slots = {}
    cur = None
    for t, (b, r) in enumerate(zip(bits, regs)):
        b = int(b)
        m = int(bits[t - 2]) * 2 + int(bits[t - 1]) if t >= 2 else 0
        if r != cur:
            slots[r] = Slot(4)
            cur = r
            seg_losses = []
        p = min(max(slots[r].predict(m), 1e-6), 1 - 1e-6)
        seg_losses.append(-np.log2(p if b else 1 - p))
        slots[r].observe(m, b)
        if (t + 1) % seg == 0:
            seg_bpb_B.append(float(np.mean(seg_losses)))
    return seg_bpb_T, seg_bpb_B


def gen_forced(seed=0, seg=20_000):
    rng = np.random.default_rng(seed)
    regs = np.array([0] * seg + [1] * seg + [0] * seg + [1] * seg)
    T = {0: {(0, 0): 0.55, (0, 1): 0.95, (1, 0): 0.95, (1, 1): 0.55},
         1: {(0, 0): 0.45, (0, 1): 0.05, (1, 0): 0.05, (1, 1): 0.45}}
    bits, prev = [], (0, 0)
    for r in regs:
        b = 1 if rng.random() < T[r][prev] else 0
        bits.append(b)
        prev = (prev[1], b)
    return np.array(bits), regs


def main():
    out = {"protocol": "phase4b_structural_radiation", "seeds": {}}
    t0 = time.time()
    for seed in (7, 8, 9):
        st, sb = run_transfer(seed=seed)
        out["seeds"][seed] = {"transfer": [round(x, 4) for x in st],
                              "baseline": [round(x, 4) for x in sb]}
        print(f"seed {seed}: T={[round(x,3) for x in st]} B={[round(x,3) for x in sb]}")
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase4b.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase4b.json")


if __name__ == "__main__":
    main()
