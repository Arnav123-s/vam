"""Design check: Euclidean and Feynman forms working together, with a certified error bound.

Reference: full enumeration of every h-bit path (no cut). Certified: best-first expansion of prefixes in order of
probability (the Euclidean weights), with a bound on everything not yet expanded:
    sum over the 2^r completions of a prefix of sqrt(p(gamma)) <= sqrt(2^r * P(prefix))   (Cauchy-Schwarz),
so |A_b - A_b(expanded)| <= E_b. Stop when the decision is certified (collapse) or when Z is known to tol (sample).
Also: hbar -> 0 least action by best-first search (Dijkstra on S = -ln p, nonnegative step costs: exact).
"""
from __future__ import annotations

import heapq
import math
import sys
import time
from pathlib import Path

import numpy as np

from vam.bits import bytes_to_bits
from vam.nested import NestedConfig, NestedMachine
from vam.nfa.collapse import Readout
from vam.realizations.vam1 import Config


def full(readout, state, h, hbar):
    """Every path, no cut. Returns Feynman A_0, A_1, Euclidean least path, nodes visited."""
    A = [0j, 0j]
    best = (math.inf, None)
    nodes = 0

    def walk(st, first, s, depth, prefix):
        nonlocal best, nodes
        nodes += 1
        if depth == h:
            A[first] += math.exp(-0.5 * s) * complex(math.cos(s / hbar), math.sin(s / hbar))
            if s < best[0]:
                best = (s, prefix)
            return
        p0 = readout.p0(st)
        for b, pb in ((0, p0), (1, 1.0 - p0)):
            walk(readout.advance(st, b), b if depth == 0 else first, s - math.log(pb), depth + 1, (prefix << 1) | b)

    walk(state, 0, 0.0, 0, 0)
    return A, best, nodes


def certified(readout, state, h, hbar, mode, tol=1e-3):
    """Best-first by prefix probability. Returns A, bounds E, Z interval, nodes expanded, certified flag."""
    A = [0j, 0j]
    heap = []                                   # (-logP, tiebreak, state, first, depth, s)
    cnt = 0
    E = [0.0, 0.0]                              # bound on the unexpanded mass per first bit

    def bound(s, depth):
        return math.sqrt(2.0 ** (h - depth) * math.exp(-s))

    p0 = readout.p0(state)
    for b, pb in ((0, p0), (1, 1.0 - p0)):
        s = -math.log(pb)
        heapq.heappush(heap, (s, cnt, readout.advance(state, b), b, 1, s))
        cnt += 1
        E[b] += bound(s, 1)
    nodes = 1
    while heap:
        lo = [max(0.0, abs(A[b]) - E[b]) for b in (0, 1)]
        hi = [abs(A[b]) + E[b] for b in (0, 1)]
        if mode == "collapse" and (lo[0] > hi[1] or lo[1] > hi[0]):
            return A, E, nodes, True
        if mode == "sample":
            zlo = lo[0] ** 2 / (lo[0] ** 2 + hi[1] ** 2) if lo[0] + hi[1] > 0 else 0.0
            zhi = hi[0] ** 2 / (hi[0] ** 2 + lo[1] ** 2) if hi[0] + lo[1] > 0 else 1.0
            if zhi - zlo <= tol:
                return A, E, nodes, True
        s, _, st, first, depth, _ = heapq.heappop(heap)
        E[first] -= bound(s, depth)
        nodes += 1
        if depth == h:
            A[first] += math.exp(-0.5 * s) * complex(math.cos(s / hbar), math.sin(s / hbar))
            continue
        q0 = readout.p0(st)
        for b, pb in ((0, q0), (1, 1.0 - q0)):
            s2 = s - math.log(pb)
            heapq.heappush(heap, (s2, cnt, readout.advance(st, b), first, depth + 1, s2))
            cnt += 1
            E[first] += bound(s2, depth + 1)
    E = [max(0.0, e) for e in E]
    return A, E, nodes, True


def least_action(readout, state, h):
    """Dijkstra on S: the first complete path popped is the least-action path, exactly."""
    heap = [(0.0, 0, state, 0, 0)]
    cnt, nodes = 1, 0
    while heap:
        s, _, st, depth, prefix = heapq.heappop(heap)
        nodes += 1
        if depth == h:
            return s, prefix, nodes
        q0 = readout.p0(st)
        for b, pb in ((0, q0), (1, 1.0 - q0)):
            nxt = (s - math.log(pb), cnt, readout.advance(st, b), depth + 1, (prefix << 1) | b)
            heapq.heappush(heap, nxt)
            cnt += 1


def main():
    root = Path(__file__).resolve().parents[2]
    texts = [p.read_bytes() for p in sorted((root / "spec").glob("*.md"))]
    texts += [(root / n).read_bytes() for n in ("DESIGN.md", "RESULTS.md", "experiments/PROTOCOL.md", "AUDIT.md")]
    data = b"\n".join(texts)
    cut = int(len(data) * 0.9)
    train, test = data[:cut], data[cut:]
    law = Config(cells=1, context=1, consolidating=True, dt=1.0)
    m = NestedMachine(NestedConfig(law, depths=(8, 16, 24, 32, 48, 64), table_bits=18, prior=4.0))
    t = time.time()
    q = m.run(bytes_to_bits(train), learning=True)
    print(f"trained {len(train)} bytes, train bpb {-np.log2(q).sum() / len(train):.3f}, {time.time() - t:.1f}s")
    rng = np.random.default_rng(7)
    starts = rng.integers(64, len(test) - 64, size=int(sys.argv[1]) if len(sys.argv) > 1 else 60)
    rows = []
    for h in (8, 12):
        for hbar in (0.25, 0.5, 1.0):
            agree = 0
            nf = nc = ns = 0
            zerr = []
            la_nodes = []
            la_ok = 0
            for st0 in starts:
                m.run(bytes_to_bits(test[st0 - 16:st0]), learning=False)   # set the register to a real context
                ro = Readout(m)
                state = ro.state0
                A, best, nodes_f = full(ro, state, h, hbar)
                Zf = np.abs(np.array(A)) ** 2
                Zf = Zf / Zf.sum()
                Ac, _E, nodes_c, _ = certified(ro, state, h, hbar, "collapse")
                dec_f = int(Zf[1] > Zf[0])
                Zc = np.abs(np.array(Ac)) ** 2
                dec_c = int(Zc[1] > Zc[0])
                agree += dec_f == dec_c
                As, _Es, nodes_s, _ = certified(ro, state, h, hbar, "sample", tol=1e-3)
                Zs = np.abs(np.array(As)) ** 2
                Zs = Zs / Zs.sum() if Zs.sum() > 0 else np.array([.5, .5])
                zerr.append(abs(Zs[0] - Zf[0]))
                s_la, _p_la, n_la = least_action(ro, state, h)
                la_ok += abs(s_la - best[0]) < 1e-9
                la_nodes.append(n_la)
                nf += nodes_f
                nc += nodes_c
                ns += nodes_s
            n = len(starts)
            rows.append((h, hbar, agree, n, nf / n, nc / n, ns / n, max(zerr), la_ok, float(np.mean(la_nodes))))
            print(f"h={h:2d} hbar={hbar:4.2f}: collapse decisions equal {agree}/{n}; nodes full {nf / n:7.0f}, "
                  f"certified collapse {nc / n:7.1f}, certified sample(tol 1e-3) {ns / n:7.1f}, "
                  f"max |dZ| {max(zerr):.2e}; "
                  f"least action exact {la_ok}/{n} in {np.mean(la_nodes):.1f} nodes")


if __name__ == "__main__":
    main()
