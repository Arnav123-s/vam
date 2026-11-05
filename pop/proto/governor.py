"""Governor as criticality (H2, Phase 4).

The owner's N-L6: "keep the engine operating permanently at the edge of
criticality" -- a global entropy monitor against capacity; "halts volume
expansion, purges micro-noise through boundary dissipative sinks, and radiates
completed invariants outward."

Operational version for a slot-limited population:
- ENTROPY MONITOR: tracks total system entropy S = sum of slot predictive
  entropies against capacity C = n_slots * log(outcomes).  Reported; volume
  expansion is halted by the slot cap itself.
- PURGE: when a new regime arrives and no slot is free, the stalest slot
  (fewest recent visits) is purged -- its micro-noise goes through the sink.
- RADIATE: before purging, a converged slot's table is exported as a FROZEN
  invariant to the outward library ("radiates completed invariants outward").
- RESTORE: when a regime returns, its radiated invariant is restored instantly
  instead of relearned.

What was extracted instead: per-level table occupancy, doorkeepers, verify
thresholds.  The outward radiation was never built.
"""

import numpy as np


class Slot:
    """One population slot: a small count table over (micro -> bit)."""

    def __init__(self, n_micro=4, prior=1.0):
        self.counts = np.full((n_micro, 2), prior)
        self.regime = None
        self.visits_recent = 0
        self.visits_total = 0
        self._old_pred = None

    def predict(self, micro):
        tot = self.counts[micro].sum()
        return self.counts[micro, 1] / tot

    def observe(self, micro, bit):
        self.counts[micro, bit] += 1
        self.visits_recent += 1
        self.visits_total += 1

    def entropy(self):
        tot = self.counts.sum()
        p = self.counts / tot
        with np.errstate(divide="ignore"):
            h = -(p * np.log2(np.maximum(p, 1e-12))).sum()
        return float(h)

    def snapshot(self):
        return np.array([self.predict(m) for m in range(self.counts.shape[0])])

    def drift_from(self, old):
        return float(np.abs(self.snapshot() - old).max())


class Governor:
    """Criticality governor over a fixed population of slots."""

    def __init__(self, n_slots=4, n_micro=4, radiate_visits=5000,
                 drift_eps=0.05, recent_window=5000):
        self.slots = [Slot(n_micro) for _ in range(n_slots)]
        self.invariants = {}  # regime -> frozen counts (radiated outward)
        self.n_micro = n_micro
        self.radiate_visits = radiate_visits
        self.drift_eps = drift_eps
        self.recent_window = recent_window
        self.log = []  # (t, action, regime, slot)
        self.entropy_trace = []

    def capacity(self):
        return len(self.slots) * np.log2(self.n_micro * 2)

    def total_entropy(self):
        return sum(s.entropy() for s in self.slots)

    def _stalest_slot(self):
        return min(range(len(self.slots)),
                   key=lambda i: self.slots[i].visits_recent)

    def on_regime(self, t, regime):
        """A new regime arrives. Returns slot index assigned to it."""
        # 1. Already resident?
        for i, s in enumerate(self.slots):
            if s.regime == regime:
                self.log.append((t, "resident", regime, i))
                return i
        # 2. Radiated invariant? Restore it.
        if regime in self.invariants:
            i = self._free_or_purge(t, regime)
            self.slots[i].counts = self.invariants[regime].copy()
            self.slots[i].regime = regime
            self.slots[i].visits_recent = 0
            self.log.append((t, "restore", regime, i))
            return i
        # 3. Fresh regime: take a free slot or purge the stalest.
        i = self._free_or_purge(t, regime)
        self.slots[i] = Slot(self.n_micro)
        self.slots[i].regime = regime
        self.log.append((t, "assign", regime, i))
        return i

    def _free_or_purge(self, t, regime):
        for i, s in enumerate(self.slots):
            if s.regime is None:
                return i
        # Halt volume expansion: no new slots. Purge the stalest through the sink.
        i = self._stalest_slot()
        victim = self.slots[i]
        # Radiate outward if it converged (completed invariant).
        if victim.visits_total >= self.radiate_visits:
            self.invariants[victim.regime] = victim.counts.copy()
            self.log.append((t, "radiate", victim.regime, i))
        self.log.append((t, "purge", victim.regime, i))
        return i

    def tick(self, t):
        """Decay recent-visit counts; record entropy vs capacity."""
        if t % self.recent_window == 0:
            for s in self.slots:
                s.visits_recent = 0
        self.entropy_trace.append((t, self.total_entropy(), self.capacity()))
