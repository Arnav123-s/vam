"""Nested context levels: step 14 of the master loop for the massless VAM-1 (decision D9).

T14 step 14 says: reuse or extend the hierarchy, using isometric nesting for
additional distinctions and declared coarse-graining for redundant ones. The
bit kernel (``bitkernel.py``) keeps one fixed partition of the past into
``2**k`` context sectors. This module keeps several partitions at once,
nested from coarse to fine:

* **Levels.** Level ``l`` distinguishes the last ``d_l`` bits
  (``d_0 < d_1 < ...``, at most 64, all held by the chain's right-movers).
  Every sector of level ``l + 1`` lies inside one sector of level ``l``, so
  the levels are coarse and fine readouts of one pointer configuration
  (T8.5, T13.1). Every level learns by the same activity-gated T12.1 law as
  VAM-1; in the massless einselected limit one sector per level is active.
* **Isometric nesting (T11.4).** A fine sector is created the first time its
  context occurs (``inherit=True``). Its relation is set so that its own
  prediction equals the coarser readout at that moment (the mixture of the
  levels above it, defined below), and its evidence starts at zero. Adding it
  therefore leaves the machine's prediction exactly as it was (tested);
  creating a distinction changes nothing the machine knew and only lets
  later evidence tell the two apart. Its activity count is the parent
  sector's, capped at ``prior``.
* **Readout: every supported level contributes (steps 8 and 13).** Each fine
  sector keeps ``ev``, the running log2-likelihood ratio between its own
  predictions and those of the coarser readout, over the visits it has seen.
  The readout runs from coarse to fine. At each level it mixes the coarser
  readout with the sector's own prediction, giving the sector the weight
  ``2**ev / (1 + 2**ev)``, which is the Bayesian posterior of "this
  distinction matters" against "it does not" from even prior odds
  (``readout="mixture"``). A distinction that does not change the predicted
  future loses weight and is coarse-grained away in effect (the canalization
  of step 13). One that does change it takes over, and while neither
  dominates both contribute (step 8). ``readout="mature"`` is the plain
  alternative: use the finest sector seen at least ``mature`` times.
* **Declared storage.** Levels with ``d_l <= table_bits`` are stored
  directly. Finer levels each use a fixed table of ``2**table_bits`` sectors,
  addressed by a 64-bit hash of the context with a 32-bit check tag. A new
  context that lands on an occupied slot replaces the occupant. That
  replacement is this realization's declared forgetting. It is counted in
  ``evictions``, and the state size never grows.

Layers of the nested fractal architecture (spec N, DESIGN section 10)
--------------------------------------------------------------------
Three options add the owner's layers 3, 5 and 6; their defaults leave the
machine exactly as above.

* **Coherent readout (N-L3, ``coherence = c``).** Each mixing step is a
  density matrix over two hypotheses, the coarser readout ``p`` and the
  sector's own prediction ``o``, with weights ``(1 - s, s)``:
  ``rho = (1 - c) diag(1 - s, s) + c |a><a|``, ``a = (sqrt(1 - s), sqrt(s))``.
  The bit is read with ``P_b = |phi_b><phi_b|``, ``phi_b = (sqrt(p_b),
  sqrt(o_b))``, and normalized over ``b``:
  ``T_b = (1 - c)((1 - s) p_b + s o_b) + c (sqrt((1 - s) p_b) + sqrt(s o_b))^2``.
  ``c = 0`` is the mixture; ``c = 1`` adds the hypotheses as amplitudes, so
  agreeing hypotheses reinforce. This is an explicit enlarged state (T13.1): a hypothesis label
  tensored with the purified hypotheses ``|h_x> = sqrt(x_0)|0> + sqrt(x_1)|1>``, in the state
  ``(1 - c) diag(1 - s, s) + c |a><a|`` on the label, read by the trace-nonincreasing element
  ``|+><+| (x) |b><b|`` (the label erased, the conditioning case of T10.1) and then normalized over
  ``b``. The amplitudes are real with zero relative phase by choice (a chosen realization, not derived).
  The result lies in ``[0, 1]`` (indeed between ``p`` and ``o``) by
  construction. Nesting the two-hypothesis densities level by level (a
  density matrix inside the next one) keeps isometric nesting exact for every
  ``c``: a new sector whose prediction equals the coarser readout scales
  ``T_0`` and ``T_1`` by the same factor (tested). One density over all
  levels at once would not.
* **Verified growth (N-L5, ``verify = v``).** A context gets a slot of a
  hashed level only at its ``v``-th sighting. A doorkeeper counts the
  sightings: ``2**door_bits`` buckets per hashed level in pairs (two-way
  sets), each holding a 16-bit check tag of a candidate and its count. A
  candidate seen again in its set counts up; a new one replaces the weaker of
  the pair (that is how the doorkeeper forgets); an admitted candidate leaves
  the doorkeeper. Until verified, the context takes no slot and evicts
  nobody: a one-off context never grows the memory.
* **Page governor (N-L6, ``governor=True``).** The stored information of a
  hashed level (occupied slots) is watched against its capacity. Below half
  (Page's turning point) the level grows freely. From half on: growth halts
  for unverified contexts (admission needs ``max(v, 2)`` sightings), and a
  sector that has proven itself (evidence ``ev > protect`` bits over the
  coarser readout) is protected from replacement. Each verified newcomer
  that knocks on a protected sector drains it through a sink, ``ev <- ev
  e^{-2 rate}``: a unit-rate sink has the eigenvalue ``Re lambda = -2``. Only
  a sector drained to ``ev <= protect`` can be replaced, so noise is purged
  and proven patterns stay.

``NestedReference`` is a plain implementation with exact contexts and no
tables, for tests. ``NestedMachine`` is the compiled one. The two agree exactly
until the first eviction (tested), and with ``hashed=True`` the reference
keeps the same tables, doorkeepers and governor and agrees throughout.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .bits import bytes_to_bits
from .linalg import ATOL
from .realizations.vam1 import Config

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
_C1, _C2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
W, N, EV = 0, 1, 2                      # columns of a sector row


@dataclass(frozen=True)
class NestedConfig:
    law: Config                         # eps, g, kappa, gamma, dt and consolidating are used; cells/context are not
    depths: tuple[int, ...] = (8, 16, 24, 32, 48, 64)
    table_bits: int = 22
    readout: str = "mixture"
    mature: int = 0
    inherit: bool = True
    prior: float = 0.0
    coherence: float = 0.0              # N-L3: 0 = mixture, 1 = amplitudes
    verify: int = 1                     # N-L5: sightings before a hashed context takes a slot
    door_bits: int = 0                  # doorkeeper buckets per hashed level (0: table_bits + 1)
    governor: bool = False              # N-L6: the Page governor on hashed levels
    protect: float = 1.0                # evidence (bits) that makes a sector a protected invariant
    sink_rate: float = 1.0              # sink rate: ev <- ev exp(-2 rate) per knock (Re lambda = -2 rate)

    def __post_init__(self) -> None:
        d = self.depths
        if not d or any(a >= b for a, b in itertools.pairwise(d)) or d[0] < 1 or d[-1] > 64:
            raise ValueError("depths must increase strictly inside 1..64")
        if not 1 <= self.table_bits <= 30 or d[0] > self.table_bits:
            raise ValueError("need 1 <= table_bits <= 30 and the coarsest level stored directly")
        if self.readout not in ("mixture", "mature") or self.mature < 0:
            raise ValueError("readout is 'mixture' or 'mature', and mature >= 0")
        if self.law.theta != 0.0:
            raise ValueError("nested levels are defined for the massless chain (theta = 0)")
        if self.prior < 0:
            raise ValueError("prior must be nonnegative")
        if not 0.0 <= self.coherence <= 1.0:
            raise ValueError("coherence lies in [0, 1]")
        if self.coherence > 0.0 and self.readout != "mixture":
            raise ValueError("coherence modifies the mixture readout")
        if not 1 <= self.verify <= 255:
            raise ValueError("verify counts sightings, 1..255")
        if self.door_bits != 0 and not 2 <= self.door_bits <= 30:
            raise ValueError("door_bits lies in 2..30 (0 for the default)")
        if self.sink_rate < 0:
            raise ValueError("sink rate must be nonnegative")

    @property
    def levels(self) -> int:
        return len(self.depths)

    def direct(self, level: int) -> bool:
        return self.depths[level] <= self.table_bits

    def size(self, level: int) -> int:
        return 1 << self.depths[level] if self.direct(level) else 1 << self.table_bits

    @property
    def doors(self) -> int:
        """Doorkeeper buckets per hashed level."""
        return 1 << (self.door_bits or self.table_bits + 1)

    @property
    def sink_factor(self) -> float:
        return math.exp(-2.0 * self.sink_rate)


def context_hash(ctx: int, level: int) -> int:
    """SplitMix64 finalizer of ``ctx XOR salt_level``; the same arithmetic as the compiled loop."""
    x = (ctx ^ (((level + 1) * GOLDEN) & MASK64)) & MASK64
    x = ((x ^ (x >> 30)) * _C1) & MASK64
    x = ((x ^ (x >> 27)) * _C2) & MASK64
    return x ^ (x >> 31)


DOOR_SALT = 0xD1B54A32D192ED03


def door_key(h: int, door_shift: int) -> tuple[int, int]:
    """Doorkeeper bucket (top bits) and 16-bit check tag (low bits, never 0) of a context with hash ``h``,
    from a second SplitMix64 round."""
    x = (h ^ DOOR_SALT) & MASK64
    x = ((x ^ (x >> 30)) * _C1) & MASK64
    x = ((x ^ (x >> 27)) * _C2) & MASK64
    x ^= x >> 31
    return x >> door_shift, (x & 0xFFFF) | 1


def coherent_mix(p0: float, own: float, share: float, c: float) -> float:
    """N-L3 readout of two hypotheses (coarser ``p0``, sector ``own``) with weights ``(1 - share, share)``."""
    if c == 0.0:
        return p0 + share * (own - p0)
    r, s = 1.0 - share, share
    t0 = (1.0 - c) * (r * p0 + s * own) + c * (math.sqrt(r * p0) + math.sqrt(s * own)) ** 2
    t1 = ((1.0 - c) * (r * (1.0 - p0) + s * (1.0 - own))
          + c * (math.sqrt(r * (1.0 - p0)) + math.sqrt(s * (1.0 - own))) ** 2)
    return t0 / (t0 + t1)


def _learn(cfg: Config, w: float, n: float, b: int) -> tuple[float, float]:
    target = cfg.g / cfg.kappa if b == 0 else -cfg.g / cfg.kappa
    gs = cfg.gamma + cfg.kappa * cfg.dt * n if cfg.consolidating else cfg.gamma
    return target + (w - target) * math.exp(-cfg.kappa * cfg.dt / gs), n + 1.0


def _p0(cfg: Config, w: float) -> float:
    return 0.5 * (1.0 + (1.0 - 2.0 * cfg.eps) * min(1.0, max(-1.0, w)))


def _q(p0: float, b: int) -> float:
    return p0 if b == 0 else 1.0 - p0


def _w_of(cfg: Config, p0: float) -> float:
    """The relation whose own prediction is ``p0`` (the inverse of :func:`_p0` on ``[eps, 1 - eps]``)."""
    return min(1.0, max(-1.0, (2.0 * p0 - 1.0) / (1.0 - 2.0 * cfg.eps)))


def _share(ev: float) -> float:
    """``2**ev / (1 + 2**ev)`` without overflow."""
    if ev >= 0.0:
        return 1.0 / (1.0 + 2.0 ** -ev)
    e = 2.0 ** ev
    return e / (1.0 + e)


class NestedReference:
    """Plain dictionaries; the semantics of :class:`NestedMachine` written independently, for tests.

    ``hashed=False`` keeps exact contexts (no forgetting) and counts doorkeeper sightings exactly.
    ``hashed=True`` keys the finer levels by the same hash slot and check tag as the compiled tables
    and uses the same doorkeeper buckets, so forgetting, admission and protection happen at the same
    places.
    """

    def __init__(self, cfg: NestedConfig, hashed: bool = False) -> None:
        self.cfg = cfg
        self.hashed = hashed
        self.reg = 0
        self.evictions = 0
        self.refused = 0                 # sightings turned away by a doorkeeper (N-L5)
        self.protected = 0               # knocks absorbed by a protected sector's sink (N-L6)
        self.tables: list[dict[int, list[float]]] = [{} for _ in cfg.depths]
        self.doors: list = [np.zeros(cfg.doors, dtype=np.int64) if hashed else {} for _ in cfg.depths]
        self._door_shift = 64 - int(math.log2(cfg.doors))

    def _key(self, level: int, ctx: int) -> tuple[int, int]:
        if not self.hashed or self.cfg.direct(level):
            return ctx, 0
        h = context_hash(ctx, level)
        return h >> (64 - self.cfg.table_bits), (h & 0xFFFFFFFF) | 1

    def _lookup(self) -> tuple[list, list]:
        keys, entries = [], []
        for lv, (table, d) in enumerate(zip(self.tables, self.cfg.depths, strict=True)):
            key, tag = self._key(lv, self.reg & ((1 << d) - 1))
            entry = table.get(key)
            if entry is not None and entry[3] != tag:
                entry = None
            keys.append((key, tag))
            entries.append(entry)
        return keys, entries

    def _entries(self) -> list:
        return self._lookup()[1]

    def _chain(self, entries) -> list[float]:
        """``p0`` of the readout that uses levels ``0..l`` only, for every ``l``."""
        cfg = self.cfg
        p0 = _p0(cfg.law, entries[0][W] if entries[0] is not None else 0.0)
        out = [p0]
        for entry in entries[1:]:
            if entry is not None and entry[N] >= cfg.mature:
                own = _p0(cfg.law, entry[W])
                p0 = own if cfg.readout == "mature" else coherent_mix(p0, own, _share(entry[EV]), cfg.coherence)
            out.append(p0)
        return out

    def predict(self, b: int) -> float:
        return _q(self._chain(self._entries())[-1], b)

    def occupancy(self, level: int) -> int:
        return len(self.tables[level])

    def governed(self, level: int) -> bool:
        """Has this hashed level reached its Page threshold (half of its capacity)?"""
        cfg = self.cfg
        return cfg.governor and not cfg.direct(level) and 2 * len(self.tables[level]) >= cfg.size(level)

    def _sighting(self, lv: int, ctx: int, need: int) -> int:
        """Count one sighting of an unstored context at its doorkeeper; returns the count. A candidate
        that reaches ``need`` leaves the doorkeeper."""
        door = self.doors[lv]
        if self.hashed:
            d, tag = door_key(context_hash(ctx, lv), self._door_shift)
            d0, d1 = d & ~1, d | 1
            e0, e1 = int(door[d0]), int(door[d1])
            if e0 >> 16 == tag:
                idx, count = d0, min(255, (e0 & 0xFFFF) + 1)
            elif e1 >> 16 == tag:
                idx, count = d1, min(255, (e1 & 0xFFFF) + 1)
            else:                                            # a new candidate replaces the weaker one
                idx, count = (d0 if (e0 & 0xFFFF) <= (e1 & 0xFFFF) else d1), 1
            door[idx] = 0 if count >= need else (tag << 16) | count
            return count
        count = min(255, door.get(ctx, 0) + 1)
        if count >= need:
            door.pop(ctx, None)
        else:
            door[ctx] = count
        return count

    def _admit(self, lv: int, ctx: int, key: int) -> bool:
        cfg = self.cfg
        if cfg.direct(lv):
            return True
        governed = self.governed(lv)
        need = max(cfg.verify, 2) if governed else cfg.verify
        if need > 1 and self._sighting(lv, ctx, need) < need:
            self.refused += 1
            return False
        occupant = self.tables[lv].get(key)
        if occupant is not None and governed and occupant[EV] > cfg.protect:
            occupant[EV] *= cfg.sink_factor                   # the knock drains it through the sink
            self.protected += 1
            return False
        return True

    def step(self, b: int, learning: bool = True) -> float:
        cfg = self.cfg
        keys, entries = self._lookup()
        chain = self._chain(entries)
        q = _q(chain[-1], b)
        if learning:
            parent_n = 0.0
            for lv, table in enumerate(self.tables):
                entry = entries[lv]
                key, tag = keys[lv]
                if entry is None:
                    if not self._admit(lv, self.reg & ((1 << cfg.depths[lv]) - 1), key):
                        continue
                    if key in table:
                        self.evictions += 1
                    if cfg.inherit and lv > 0:                    # own prediction = the coarser readout
                        inherited = (_w_of(cfg.law, chain[lv - 1]), min(parent_n, cfg.prior))
                    else:
                        inherited = (0.0, 0.0)
                    entry = table[key] = [*inherited, 0.0, tag]
                elif lv > 0:
                    own = _q(_p0(cfg.law, entry[W]), b)
                    entry[EV] += math.log2(own) - math.log2(_q(chain[lv - 1], b))
                parent_n = entry[N]                              # this level's count before the update
                entry[W], entry[N] = _learn(cfg.law, entry[W], entry[N], b)
        self.reg = ((self.reg << 1) | int(b)) & MASK64
        return q


CHUNK = 256    # bits per block: the first pass of a block finds and touches every row it will use


def _make_loop(jit):
    """The reading loop. Each block first computes every level's row and check tag from the register
    alone (they never depend on the tables) and touches those rows, so the cache misses of large tables
    overlap; the second pass then runs the event in stream order, giving the results of a plain loop."""

    def loop(bits, wn, tags, offsets, depths, hidx, reg, shift, mixture, mature, scale, gamma, gk, kdt,
             consolidating, learning, inherit, prior, out, stats, coherence, verify, governor, protect,
             sink_factor, door, door_shift, occ):
        L = depths.shape[0]
        size = bits.shape[0]
        tsize = np.int64(1) << np.int64(64 - shift)              # slots of a hashed level
        blk_rows = np.empty((CHUNK, L), dtype=np.int64)
        blk_tags = np.empty((CHUNK, L), dtype=np.uint32)
        blk_door = np.empty((CHUNK, L), dtype=np.int64)
        blk_dtag = np.empty((CHUNK, L), dtype=np.int64)
        fresh = np.empty(L, dtype=np.bool_)
        chain = np.empty(L)                                     # p0 of the readout using levels 0..l
        one = np.uint64(1)
        sink = 0.0
        door_on = verify > 1 or governor                        # the doorkeeper's hashes cost nothing when it is off
        for start in range(0, size, CHUNK):
            stop = min(size, start + CHUNK)
            r = reg
            for i in range(start, stop):
                for lv in range(L):
                    d = depths[lv]
                    ctx = r if d >= 64 else r & ((one << np.uint64(d)) - one)
                    h = hidx[lv]
                    if h < 0:
                        s = np.int64(ctx)
                        t = np.uint32(0)
                    else:
                        x = ctx ^ (np.uint64(lv + 1) * np.uint64(0x9E3779B97F4A7C15))
                        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
                        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
                        x = x ^ (x >> np.uint64(31))
                        s = np.int64(x >> shift)
                        t = np.uint32((x & np.uint64(0xFFFFFFFF)) | one)
                        sink += tags[h, s]
                        if door_on:
                            y = x ^ np.uint64(0xD1B54A32D192ED03)
                            y = (y ^ (y >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
                            y = (y ^ (y >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
                            y = y ^ (y >> np.uint64(31))
                            blk_door[i - start, lv] = np.int64(y >> door_shift)
                            blk_dtag[i - start, lv] = np.int64((y & np.uint64(0xFFFF)) | one)
                    blk_rows[i - start, lv] = offsets[lv] + s
                    blk_tags[i - start, lv] = t
                    sink += wn[offsets[lv] + s, 0]
                r = (r << one) | np.uint64(bits[i])
            for i in range(start, stop):
                b = int(bits[i])
                p0 = 0.0
                for lv in range(L):
                    row = blk_rows[i - start, lv]
                    h = hidx[lv]
                    if h < 0:
                        is_new = wn[row, 1] == 0.0
                    else:
                        is_new = tags[h, row - offsets[lv]] != blk_tags[i - start, lv]
                    fresh[lv] = is_new
                    if lv == 0 or (not is_new and wn[row, 1] >= mature):
                        w = wn[row, 0]
                        if w > 1.0:
                            w = 1.0
                        elif w < -1.0:
                            w = -1.0
                        own = 0.5 * (1.0 + scale * w)
                        if lv == 0 or not mixture:
                            p0 = own
                        else:
                            ev = wn[row, 2]
                            if ev >= 0.0:
                                share = 1.0 / (1.0 + 2.0 ** -ev)
                            else:
                                e = 2.0 ** ev
                                share = e / (1.0 + e)
                            if coherence == 0.0:
                                p0 = p0 + share * (own - p0)
                            else:
                                rr = 1.0 - share
                                a0 = math.sqrt(rr * p0) + math.sqrt(share * own)
                                a1 = math.sqrt(rr * (1.0 - p0)) + math.sqrt(share * (1.0 - own))
                                t0 = (1.0 - coherence) * (rr * p0 + share * own) + coherence * a0 * a0
                                t1 = (1.0 - coherence) * (rr * (1.0 - p0) + share * (1.0 - own)) + coherence * a1 * a1
                                p0 = t0 / (t0 + t1)
                    chain[lv] = p0
                out[i] = p0 if b == 0 else 1.0 - p0
                if learning:
                    target = gk if b == 0 else -gk
                    parent_n = 0.0
                    for lv in range(L):
                        row = blk_rows[i - start, lv]
                        if fresh[lv]:
                            h = hidx[lv]
                            if h >= 0:
                                slot = row - offsets[lv]
                                occupied = tags[h, slot] != 0
                                if door_on:                              # N-L5 and N-L6; off, the loop is the plain one
                                    governed = governor and 2 * occ[h] >= tsize
                                    need = verify
                                    if governed and need < 2:
                                        need = 2
                                    if need > 1:                            # N-L5: the doorkeeper counts sightings
                                        dd = blk_door[i - start, lv]
                                        dtag = blk_dtag[i - start, lv]
                                        d0 = dd & ~np.int64(1)
                                        d1 = d0 | np.int64(1)
                                        e0 = np.int64(door[h, d0])
                                        e1 = np.int64(door[h, d1])
                                        if (e0 >> 16) == dtag:
                                            idx = d0
                                            cnt = (e0 & 0xFFFF) + 1
                                        elif (e1 >> 16) == dtag:
                                            idx = d1
                                            cnt = (e1 & 0xFFFF) + 1
                                        else:                               # a new candidate replaces the weaker
                                            idx = d0 if (e0 & 0xFFFF) <= (e1 & 0xFFFF) else d1
                                            cnt = 1
                                        if cnt > 255:
                                            cnt = 255
                                        if cnt >= need:
                                            door[h, idx] = 0                # admitted: it leaves the doorkeeper
                                        else:
                                            door[h, idx] = np.uint32((dtag << 16) | cnt)
                                        if cnt < need:
                                            stats[3] += 1
                                            continue
                                    if governed and occupied and wn[row, 2] > protect:
                                        wn[row, 2] *= sink_factor            # N-L6: the knock drains the occupant
                                        stats[4] += 1
                                        continue
                                if occupied:
                                    stats[0] += 1               # an occupied slot is overwritten
                                    if stats[1] < 0:
                                        stats[1] = stats[2] + i
                                else:
                                    occ[h] += 1
                                tags[h, slot] = blk_tags[i - start, lv]
                            if inherit and lv > 0:              # own prediction = the coarser readout
                                w = (2.0 * chain[lv - 1] - 1.0) / scale
                                wn[row, 0] = 1.0 if w > 1.0 else (-1.0 if w < -1.0 else w)
                                wn[row, 1] = parent_n if parent_n < prior else prior
                            else:
                                wn[row, 0] = 0.0
                                wn[row, 1] = 0.0
                            wn[row, 2] = 0.0
                        elif lv > 0:
                            w = wn[row, 0]
                            if w > 1.0:
                                w = 1.0
                            elif w < -1.0:
                                w = -1.0
                            own = 0.5 * (1.0 + scale * w)
                            prev = chain[lv - 1]
                            if b == 1:
                                own = 1.0 - own
                                prev = 1.0 - prev
                            wn[row, 2] += math.log2(own / prev)
                        parent_n = wn[row, 1]
                        gs = gamma + kdt * wn[row, 1] if consolidating else gamma
                        wn[row, 0] = target + (wn[row, 0] - target) * math.exp(-kdt / gs)
                        wn[row, 1] += 1.0
                reg = (reg << one) | np.uint64(b)
        stats[2] += size
        if sink != sink:
            raise ValueError("a table entry is not a number")
        return reg

    return jit(loop) if jit is not None else loop


_compiled_loop = _make_loop(njit(cache=True)) if NUMBA_AVAILABLE else None


class NestedMachine:
    """Compiled nested-level VAM-1 (massless, einselected), with the interface of ``BitMachine``."""

    def __init__(self, cfg: NestedConfig) -> None:
        if not NUMBA_AVAILABLE:
            raise RuntimeError("NestedMachine needs numba; NestedReference has the same semantics")
        self.cfg = cfg
        sizes = [cfg.size(lv) for lv in range(cfg.levels)]
        self._sizes = np.array(sizes, dtype=np.int64)
        self._offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        self._wn = np.zeros((int(sum(sizes)), 3))
        hashed = [lv for lv in range(cfg.levels) if not cfg.direct(lv)]
        self._hidx = np.full(cfg.levels, -1, dtype=np.int64)
        self._hidx[hashed] = np.arange(len(hashed))
        self._tags = np.zeros((max(1, len(hashed)), 1 << cfg.table_bits if hashed else 1), dtype=np.uint32)
        self._depths = np.array(cfg.depths, dtype=np.int64)
        self._shift = np.uint64(64 - cfg.table_bits)
        self.reg = np.uint64(0)
        # evictions, first eviction (bit index), bits read, doorkeeper refusals, protected knocks
        self._stats = np.array([0, -1, 0, 0, 0], dtype=np.int64)
        used = (cfg.verify > 1 or cfg.governor) and hashed
        self._door = np.zeros((max(1, len(hashed)), cfg.doors) if used else (1, 1), dtype=np.uint32)
        self._door_shift = np.uint64(64 - int(math.log2(cfg.doors)) if used else 63)
        self._occ = np.zeros(max(1, len(hashed)), dtype=np.int64)

    @property
    def evictions(self) -> int:
        return int(self._stats[0])

    @property
    def first_eviction(self) -> int:
        """Index of the bit at which a table slot was first overwritten, or -1."""
        return int(self._stats[1])

    @property
    def refused(self) -> int:
        """Sightings a doorkeeper turned away (N-L5)."""
        return int(self._stats[3])

    @property
    def protected(self) -> int:
        """Knocks a protected sector absorbed through its sink (N-L6)."""
        return int(self._stats[4])

    def occupancy(self) -> dict[int, float]:
        """Fraction of the slots in use, per hashed level: the Page governor's measure (N-L6)."""
        cap = 1 << self.cfg.table_bits
        return {lv: float(self._occ[self._hidx[lv]]) / cap for lv in range(self.cfg.levels) if self._hidx[lv] >= 0}

    def stored(self, ctx: int, level: int) -> bool:
        """Does ``level`` hold a sector for the context ``ctx`` (its last ``depths[level]`` bits)?"""
        cfg = self.cfg
        ctx &= (1 << cfg.depths[level]) - 1
        if cfg.direct(level):
            return bool(self.level_table(level)[ctx, N] > 0.0)
        h = context_hash(ctx, level)
        return int(self._tags[self._hidx[level], h >> (64 - cfg.table_bits)]) == ((h & 0xFFFFFFFF) | 1)

    def stored_evidence(self) -> dict[int, float]:
        """Total positive evidence (bits over the coarser readout) held by each level's sectors."""
        out = {}
        for lv in range(1, self.cfg.levels):
            ev = self.level_table(lv)[:, EV]
            out[lv] = float(ev[ev > 0].sum())
        return out

    def level_table(self, level: int) -> np.ndarray:
        """``(w, n, ev)`` rows of one level (a view)."""
        o = self._offsets[level]
        return self._wn[o:o + self._sizes[level]]

    @property
    def w(self) -> np.ndarray:
        """Every sector weight of every level (a view)."""
        return self._wn[:, W]

    @property
    def n(self) -> np.ndarray:
        """Visit counts of the finest directly stored level (the visitation statistics of E2)."""
        direct = [lv for lv in range(self.cfg.levels) if self.cfg.direct(lv)]
        return self.level_table(direct[-1])[:, N]

    def copy(self) -> NestedMachine:
        other = NestedMachine.__new__(NestedMachine)
        other.__dict__.update(self.__dict__)
        other._wn, other._tags, other._stats = self._wn.copy(), self._tags.copy(), self._stats.copy()
        other._door, other._occ = self._door.copy(), self._occ.copy()
        return other

    def nbytes(self) -> int:
        return 8 + int(self._wn.nbytes + self._tags.nbytes + self._door.nbytes + self._occ.nbytes)

    def run(self, bits: np.ndarray, learning: bool = True) -> np.ndarray:
        bits = np.ascontiguousarray(bits, dtype=np.uint8)
        c = self.cfg.law
        out = np.empty(bits.shape[0])
        self.reg = _compiled_loop(bits, self._wn, self._tags, self._offsets, self._depths, self._hidx,
                                  np.uint64(self.reg), self._shift, self.cfg.readout == "mixture",
                                  float(self.cfg.mature), 1.0 - 2.0 * c.eps, c.gamma, c.g / c.kappa,
                                  c.kappa * c.dt, c.consolidating, learning, self.cfg.inherit,
                                  float(self.cfg.prior), out, self._stats, float(self.cfg.coherence),
                                  int(self.cfg.verify), bool(self.cfg.governor), float(self.cfg.protect),
                                  float(self.cfg.sink_factor), self._door, self._door_shift, self._occ)
        return out

    def step(self, b: int, learning: bool = True) -> float:
        return float(self.run(np.array([b], dtype=np.uint8), learning)[0])

    def bits_per_byte(self, data: bytes) -> float:
        return float(-np.log2(self.run(bytes_to_bits(data))).sum() / len(data))

    # -- frozen readout, for planning ----------------------------------------
    def _p0_at(self, reg: int) -> float:
        cfg = self.cfg
        p0 = 0.0
        for lv, d in enumerate(cfg.depths):
            ctx = reg & ((1 << d) - 1)
            if cfg.direct(lv):
                row = self._wn[self._offsets[lv] + ctx]
                present = row[N] > 0.0
            else:
                hsh = context_hash(ctx, lv)
                s = hsh >> (64 - cfg.table_bits)
                row = self._wn[self._offsets[lv] + s]
                present = int(self._tags[self._hidx[lv], s]) == ((hsh & 0xFFFFFFFF) | 1)
            if lv == 0 or (present and row[N] >= cfg.mature):
                own = _p0(cfg.law, float(row[W]))
                p0 = own if lv == 0 or cfg.readout == "mature" else coherent_mix(p0, own, _share(float(row[EV])),
                                                                                  cfg.coherence)
        return p0

    def predict(self, b: int) -> float:
        return _q(self._p0_at(int(self.reg)), b)

    def continuations(self, horizon: int) -> dict[tuple, float]:
        """Exact distribution over the next ``horizon`` emitted bits; plasticity off (as ``BitMachine``)."""
        cut = max(1e-10, ATOL)
        dist: dict[tuple, float] = {}

        def walk(reg: int, outs: tuple, prob: float) -> None:
            if len(outs) == horizon:
                dist[outs] = dist.get(outs, 0.0) + prob
                return
            p0 = self._p0_at(reg)
            for t, pt in ((0, p0), (1, 1.0 - p0)):
                if pt > cut:
                    walk(((reg << 1) | t) & MASK64, (*outs, t), prob * pt)

        walk(int(self.reg), (), 1.0)
        return dist

    def generate(self, n_bits: int, horizon: int = 8, select: str = "max",
                 rng: np.random.Generator | None = None) -> list[int]:
        """Macro/micro generation exactly as ``BitMachine.generate``; the tables never change."""
        if select not in ("max", "sample"):
            raise ValueError("select must be 'max' or 'sample'")
        if select == "sample" and rng is None:
            raise ValueError("sampling needs a random generator")
        emitted = []
        for _ in range(n_bits):
            dist = self.continuations(horizon)
            if select == "max":
                macro = max(dist, key=lambda k: (dist[k], tuple(map(str, k))))
            else:
                keys = sorted(dist, key=lambda k: tuple(map(str, k)))
                pv = np.array([dist[k] for k in keys])
                macro = keys[int(rng.choice(len(keys), p=pv / pv.sum()))]
            first = int(macro[0])
            self.reg = np.uint64(((int(self.reg) << 1) | first) & MASK64)
            emitted.append(first)
        return emitted
