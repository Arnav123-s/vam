"""N-L2 Page limit: Page's exact average entropy, the random-state edge, and the cutoff that counts the
verified directions of a stream (the hidden states of known sources, 1 for i.i.d. bits)."""

import math

import numpy as np
import pytest

from vam import sources as S
from vam.linalg import random_density
from vam.nfa import page


def test_page_entropy_closed_forms():
    assert page.page_entropy(2, 2) == pytest.approx(1 / 3, abs=1e-14)                     # two qubits: 1/3 nat
    for m, n in [(2, 3), (3, 7), (4, 4), (5, 40)]:
        exact = sum(1.0 / k for k in range(n + 1, m * n + 1)) - (m - 1) / (2 * n)
        assert page.page_entropy(m, n) == pytest.approx(exact, rel=1e-12)
        assert page.page_entropy(n, m) == page.page_entropy(m, n)
    # information about the small side is about m / 2n nats (Page 1993)
    assert page.page_information(8, 4096) == pytest.approx(8 / (2 * 4096), rel=0.02)


def test_page_curve_turns_at_half():
    c = page.page_curve(10)
    assert c[0] == 0.0 and c[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(c, c[::-1])
    assert int(np.argmax(c)) == 5 and page.page_turnaround(10) == 5.0
    assert c[5] == pytest.approx(5 - 1 / (2 * math.log(2)), abs=0.01)                    # Page: n/2 - 1/(2 ln 2)


def test_page_entropy_matches_random_states(rng):
    m, n = 4, 16
    S_mc, top = [], []
    for _ in range(400):
        v = rng.normal(size=(m, n)) + 1j * rng.normal(size=(m, n))
        lam = np.linalg.svd(v / np.linalg.norm(v), compute_uv=False) ** 2
        S_mc.append(-np.sum(lam * np.log(lam)))
        top.append(lam[0])
    assert np.mean(S_mc) == pytest.approx(page.page_entropy(m, n), abs=0.01)
    assert np.mean(top) < page.random_state_edge(m, n) * 1.1                           # the MP edge bounds the top


def test_noise_edge_of_a_gaussian_matrix(rng):
    m, n, N = 300, 500, 1.0e4
    s = np.linalg.svd(rng.normal(size=(m, n)) / math.sqrt(N), compute_uv=False)
    assert s[0] == pytest.approx(page.noise_edge(m, n, N), rel=0.03)
    assert page.tracy_widom_width(m, n) < 0.01


def test_hankel_counts_and_exact_hankel_agree(rng):
    src = S.MarkovBits()
    h = src.hmm()
    H = page.exact_hankel(h.T, h.stationary, 3, 2)
    assert H.sum() == pytest.approx(1.0) and H.shape == (8, 4)
    bits = src.sample(200_000, rng).bits
    Hc = page.hankel_counts(bits, 3, 2).toarray()
    assert np.abs(Hc / Hc.sum() - H).max() < 0.005
    # the row index is the past, oldest bit first
    assert page.windows(np.array([1, 0, 1, 1]), 3).tolist() == [0b101, 0b011]


@pytest.mark.parametrize("name,src,states", [
    ("markov2", S.MarkovBits(), 4), ("codes", S.IncompleteCodes(), 3),
    ("echo3", S.DelayedEcho(3, 0.1), 8), ("echo4", S.DelayedEcho(4, 0.15), 16)])
def test_exact_hankel_rank_is_the_number_of_states(name, src, states):
    h = src.hmm()
    for k in (4, 6):
        assert page.exact_rank(page.exact_hankel(h.T, h.stationary, k, k)) == states


@pytest.mark.parametrize("name,src,states,N,k", [
    ("markov2", S.MarkovBits(), 4, 400_000, 6), ("codes", S.IncompleteCodes(), 3, 100_000, 8),
    ("echo3", S.DelayedEcho(3, 0.1), 8, 100_000, 8), ("echo4", S.DelayedEcho(4, 0.15), 16, 400_000, 8)])
def test_page_rank_recovers_the_hidden_states(name, src, states, N, k, rng):
    full, halves = page.hankel_halves(src.sample(N, rng).bits, k, k)
    sp = page.page_spectrum(full, halves)
    assert sp["rank"] == states
    assert sp["split_edge"] >= sp["edge"] * 0.9                    # the empirical edge is never far under MP


def test_iid_bits_have_rank_one(rng):
    for k in (6, 8, 10):
        full, halves = page.hankel_halves(rng.integers(0, 2, 200_000), k, k)
        assert page.page_rank(full, halves) == 1
        assert page.page_rank(full) == 1


def test_marchenko_pastur_alone_overcounts_structured_sources(rng):
    """The flat-noise edge sits under the real noise of a strongly structured source with overlapping
    windows; the split-half edge does not (why page_rank takes halves)."""
    full, halves = page.hankel_halves(S.DelayedEcho(3, 0.1).sample(100_000, rng).bits, 8, 8)
    sp = page.page_spectrum(full, halves)
    assert sp["rank"] == 8 and page.page_rank(full) > 8 and sp["split_edge"] > 1.5 * sp["edge"]


def test_never_more_directions_than_the_source_has():
    """No overcount over many independent samples, including the sparse 256 x 256 case."""
    for seed in range(8):
        r = np.random.default_rng(seed)
        full, halves = page.hankel_halves(S.DelayedEcho(3, 0.1).sample(50_000, r).bits, 8, 8)
        assert page.page_rank(full, halves) <= 8


def test_independent_past_and_future_have_one_direction(rng):
    """A product table (past independent of future) whitens to one direction of singular value 1."""
    a, b = random_density(4, rng=rng).real.diagonal(), random_density(8, rng=rng).real.diagonal()
    C, _, _, _ = page.whiten(np.outer(a, b) * 1e6)
    s = page.singular_values(C)
    assert s[0] == pytest.approx(1.0) and s[1] < 1e-6


# -- byte streams: the depth of the nesting -------------------------------------------------------

def skip_one_bytes(n: int, rng, letters: int = 4, keep: float = 0.9) -> bytes:
    """``x_t = x_{t-2}`` with probability ``keep``, else a fresh letter: two independent interleaved
    chains, so the previous byte says nothing and the one before it says almost everything."""
    x = np.zeros(n, dtype=np.uint8)
    x[:2] = rng.integers(0, letters, 2)
    fresh = rng.integers(0, letters, n)
    stay = rng.random(n) < keep
    for t in range(2, n):
        x[t] = x[t - 2] if stay[t] else fresh[t]
    return (x + ord("a")).tobytes()


def test_byte_hankel_counts_every_position_once(rng):
    data = skip_one_bytes(20_000, rng)
    full, halves = page.byte_hankel_halves(data, past=3, future=1, block=1024)
    assert full.sum() == len(data) - 3
    assert full.shape == (64, 4)                                   # every 3-letter context is seen
    for h in halves:
        assert 0 < h.sum() < full.sum()
        assert (full - h).min() >= 0                               # each half is part of the sample
    with pytest.raises(ValueError):
        page.byte_hankel_halves(data, past=8, future=1)


def test_iid_bytes_have_one_direction_at_every_depth(rng):
    data = (rng.integers(0, 8, 40_000).astype(np.uint8) + ord("a")).tobytes()
    assert [r["rank"] for r in page.depth_profile(data, depths=(1, 2, 3))] == [1, 1, 1]


def test_depth_profile_finds_the_depth_the_source_needs(rng):
    """Depth 1 sees nothing, depth 2 sees all four directions, and the turnaround is depth 2."""
    profile = page.depth_profile(skip_one_bytes(40_000, rng), depths=(1, 2, 3, 4))
    ranks = [r["rank"] for r in profile]
    assert ranks[0] == 1 and ranks[1] == 4 and max(ranks) == 4
    assert page.turnaround(profile) == 2
