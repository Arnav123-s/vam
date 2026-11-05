"""Population-scale density matrices (H1).

The owner's formulation: huge numbers of small density matrices -- one per node --
each with its basis sized by the Page limit (page.py), insulated by Markov blankets
(a node sees only events where its key is active), summarized by an overlying
"legend" layer: "a density matrix can become an overlying layer -- like a topological
map or a legend on a map."

Node: holds a density matrix rho over K basis states (joint of key-micro-state and
next bit), updated by conditioning on observed events.  Readout is the conditional
p(bit | micro) from rho.  A dephasing knob interpolates between the fully coherent
node (lambda=0) and the classical table (lambda=1).

Legend: one overlying density matrix over the slow feature, mixed with the node's
readout -- the map's legend summarizing the territory.
"""

import numpy as np

from .page import page_rank


class Node:
    """One small density matrix, blanket-insulated to its key.

    Phase 1: the node is diagonal (a classical table in density-matrix form).
    The matrix form is scaffolding -- Phase 3 (tensor assembly, H4) adds
    coherence and unitary dynamics.  Honesty note: on a classical source,
    coherence cannot help; Phase 1 tests the *population structure*, not coherence.
    """

    def __init__(self, n_micro, rng, dephase=1.0, prior=1.0):
        self.K = n_micro          # micro basis states (e.g. last-2-bits = 4)
        self.dim = n_micro * 2    # joint (micro, bit)
        self.counts = np.full((n_micro, 2), prior)  # blanket-local counts
        self.prior = prior
        self.n = 0
        self.dephase = dephase
        self._rank = 0
        self.rho = np.eye(self.dim, dtype=complex) / self.dim

    def _idx(self, micro, bit):
        return micro * 2 + bit

    def _sync_rho(self):
        """Rho's diagonal = the blanket-local joint distribution."""
        joint = self.counts / self.counts.sum()
        d = np.zeros(self.dim)
        for m in range(self.K):
            d[self._idx(m, 0)] = joint[m, 0]
            d[self._idx(m, 1)] = joint[m, 1]
        rho = np.diag(d).astype(complex)
        if self.dephase < 1.0:
            # Placeholder for coherent dynamics (Phase 3).  No source of
            # off-diagonals in Phase 1; the knob is API, not physics.
            pass
        self.rho = rho

    def observe(self, micro, bit):
        """Admit one event through the blanket; update the local matrix."""
        self.counts[micro, bit] += 1
        self.n += 1
        self._sync_rho()

    def predict(self, micro):
        """p(bit=1 | micro) from the node's density matrix diagonal."""
        i0, i1 = self._idx(micro, 0), self._idx(micro, 1)
        p0 = self.rho[i0, i0].real
        p1 = self.rho[i1, i1].real
        tot = p0 + p1
        if tot <= 0:
            return 0.5
        return p1 / tot

    def page_rank(self):
        """Current Page-selected rank of this node's basis."""
        self._rank, _, _ = page_rank(self.counts)
        return self._rank


class Legend:
    """The overlying layer: a density matrix summarizing the slow feature.

    May be FROZEN after a bootstrap phase (the "immutable bedrock", H3): a
    latent tracker that trains its own likelihood model on its own outputs
    collapses to a degenerate fixed point (logged 2026-11-04).  The likelihood
    model must be anchored independently of the tracker's decisions.
    """

    def __init__(self, n_slow, prior=1.0):
        self.n_slow = n_slow
        self.dim = n_slow * 2
        self.counts = np.full((n_slow, 2), prior)
        self.rho = np.eye(self.dim, dtype=complex) / self.dim
        self.frozen = False
        self._sync()

    def _idx(self, slow, bit):
        return slow * 2 + bit

    def _sync(self):
        joint = self.counts / self.counts.sum()
        d = np.zeros(self.dim)
        for s in range(self.n_slow):
            d[self._idx(s, 0)] = joint[s, 0]
            d[self._idx(s, 1)] = joint[s, 1]
        self.rho = np.diag(d).astype(complex)

    def observe(self, slow, bit):
        if self.frozen:
            return
        self.counts[slow, bit] += 1
        self._sync()

    def predict(self, slow):
        i0, i1 = self._idx(slow, 0), self._idx(slow, 1)
        p0 = self.rho[i0, i0].real
        p1 = self.rho[i1, i1].real
        tot = p0 + p1
        return 0.5 if tot <= 0 else p1 / tot


class Population:
    """Huge-numbers-ready population of small density matrices + legend.

    key_fn(bits) -> (node_id, slow_id, micro_id).  Each node is blanket-insulated:
    it observes only events carrying its key.
    """

    def __init__(self, n_nodes, n_slow, n_micro, key_fn, rng=None, dephase=1.0):
        self.rng = rng or np.random.default_rng(0)
        self.key_fn = key_fn
        self.nodes = {i: Node(n_micro, self.rng, dephase) for i in range(n_nodes)}
        self.legend = Legend(n_slow)
        self.n_nodes = n_nodes

    def step(self, bits, bit):
        """Observe one bit; return -log2 p(bit) BEFORE seeing it (prequential)."""
        node_id, slow_id, micro_id = self.key_fn(bits)
        node = self.nodes[node_id]
        p_node = node.predict(micro_id)
        p_leg = self.legend.predict(slow_id)
        # Mix node with legend by the node's maturity (young node trusts legend).
        w = node.n / (node.n + 8.0)
        p = w * p_node + (1 - w) * p_leg
        p = min(max(p, 1e-6), 1 - 1e-6)
        loss = -np.log2(p if bit else 1 - p)
        node.observe(micro_id, bit)
        self.legend.observe(slow_id, bit)
        return loss

    def ranks(self):
        return {i: n.page_rank() for i, n in self.nodes.items()}

    def drift_snapshot(self):
        """Per-node predictions on all micro states (for stability measurement)."""
        return {i: [n.predict(m) for m in range(n.K)] for i, n in self.nodes.items()}
