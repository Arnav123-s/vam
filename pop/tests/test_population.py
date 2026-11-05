"""Smoke tests for the population prototype."""

import numpy as np

from proto.page import page_rank, mp_upper_edge
from proto.population import Node, Population
from proto.sources import regime_flip, make_key


def test_page_rank_recovers_known_rank():
    rng = np.random.default_rng(0)
    # Rank-2 signal + noise: 20x30 counts (constant shift, removed by centering).
    U = rng.normal(size=(20, 2))
    V = rng.normal(size=(2, 30))
    counts = U @ V * 30 + rng.normal(size=(20, 30)) * 2 + 50
    rank, sv, edge = page_rank(counts)
    assert rank == 2, f"rank={rank}"
    print(f"  page_rank on rank-2 signal: {rank} (edge {edge:.2f}) ok")
    # Pure noise -> rank 0 (larger matrix: MP edge is tight asymptotically).
    # Note: the MP edge is a statistical threshold -- Tracy-Widom fluctuations put
    # ~1 direction above it on some draws.  Seed fixed for determinism.
    nrng = np.random.default_rng(100)
    noise = np.abs(nrng.normal(size=(100, 200)) * 5) + 1
    r0, _, _ = page_rank(noise)
    assert r0 == 0, f"noise rank={r0}"
    print("  pure noise -> rank 0 ok")
    # NOTE (known limitation): on tiny matrices (e.g. 20x30) finite-size effects
    # can put 1 noise direction above the edge.  Per-node 4x2 count matrices use
    # the same criterion; their ranks are reported as measured, not trusted blindly.


def test_page_rank_dormant_when_scarce():
    rank, _, _ = page_rank(np.zeros((4, 2)))
    assert rank == 0
    print("  dormant node ok")


def test_node_learns_bias():
    rng = np.random.default_rng(1)
    n = Node(1, rng, dephase=1.0)
    for _ in range(300):
        n.observe(0, 1 if rng.random() < 0.8 else 0)
    p = n.predict(0)
    assert 0.65 < p < 0.95, f"p={p}"
    print(f"  node learns bias: p={p:.3f} ok")


def test_population_beats_flat_on_regime_flip():
    bits, _ = regime_flip(20000, seed=7)
    key_fn = make_key()
    pop = Population(8, 2, 4, key_fn, dephase=1.0)
    losses = []
    hist = []
    for t, b in enumerate(bits):
        losses.append(pop.step(list(hist), int(b)))
        hist.append(int(b))
    pop_bpb = float(np.mean(losses[1000:]))
    # Flat order-2 counting baseline.
    counts = np.zeros((4, 2))
    flat_losses = []
    for t in range(2, len(bits)):
        micro = int(bits[t - 2]) * 2 + int(bits[t - 1])
        tot = counts[micro].sum()
        p = counts[micro, 1] / tot if tot > 0 else 0.5
        p = min(max(p, 1e-6), 1 - 1e-6)
        b = int(bits[t])
        flat_losses.append(-np.log2(p if b else 1 - p))
        counts[micro, b] += 1
    flat_bpb = float(np.mean(flat_losses[1000:]))
    print(f"  population {pop_bpb:.3f} vs flat {flat_bpb:.3f}")
    assert pop_bpb < flat_bpb - 0.1, "population should resolve the regime flip"
    print("  population beats flat ok")


if __name__ == "__main__":
    test_page_rank_recovers_known_rank()
    test_page_rank_dormant_when_scarce()
    test_node_learns_bias()
    test_population_beats_flat_on_regime_flip()
    print("all smoke tests pass")
