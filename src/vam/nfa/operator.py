"""E6: the shared linear operator level O (DESIGN section 12, V2 and V6; PROTOCOL E6).

O is an observable operator model over bytes (the form of T2.2: a linear map per observed symbol on a state vector):
the state ``b`` (``k`` floats), operators ``B_x`` for every byte ``x``, a start state ``b_1`` and a normalizer
``b_inf``; the next-byte distribution is ``b_inf' B_x b`` normalized, and reading ``x`` sends ``b`` to
``B_x b / (b_inf' B_x b)``. It is learned forward only: :class:`MomentStore` keeps fixed-size sums of counts of the
stream as read, and :func:`compile_operator` rebuilds the operators from the counts alone (the spectral algorithm of
Hsu, Kakade and Zhang 2009 on count matrices, with the basis cut at the Page rank).

:class:`OperatorMachine` is E4's machine with one extra stage after level 0 (a two-hypothesis coherent mixture of
level 0's prediction and O's, with one evidence per partial-byte state); with ``mode="off"`` it is E4's machine bit
for bit. Nothing here scores, ranks or selects: the stage is the machine's own mixture and O's prediction is a
probability distribution.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ..nested import NUMBA_AVAILABLE
from . import page
from .hier import BLOCK_BITS, ST_REG, HierConfig, HierMachine, _fill_keys
from .oploop import make_loop_o

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

PINV_RCOND = 1e-6              # pseudo-inverse cutoff (relative); a module constant, recorded in every result
MIN_COUNTS = 1000              # positions needed before the first compile
ALPHABET = 256


@dataclass(frozen=True)
class OperatorConfig:
    """PROTOCOL E6.3: every constant is fixed there and none is tuned."""

    mode: str = "spectral"          # "spectral" (E4+O), "unigram" (the rank-1 arm), "off" (E4)
    L: int = 4                      # bytes of past (the N-L2 turnaround: depth of the largest Page rank, e3_t4.json)
    past_bits: int = 12             # past buckets: 2**past_bits
    compile_bytes: int = 1 << 20    # bytes between compiles (a smoke run uses 1 << 15)
    k_cap: int = 256
    k_min: int = 2
    k_force: int = 0                # the timing check only: use this rank whatever the Page rank
    floor: int = 256                # the count floor of the Page rank (e4_hier.FLOOR)
    margin: float = 0.1
    splits: int = 3                 # split-half copies for the Page edge
    split_block: int = 4096
    seed: int = 0
    eps_o: float = 2.0 ** -12       # byte floor; the bad-update threshold
    eps_bit: float = 2.0 ** -12     # the machine's epsilon: bit probabilities lie in [eps, 1 - eps]
    c3_bytes: tuple = ()            # E6 Amendment 1 (R-a): C3 over these byte values only; () keeps all 256 rows

    def __post_init__(self) -> None:
        if self.mode not in ("spectral", "unigram", "off"):
            raise ValueError("mode is 'spectral', 'unigram' or 'off'")
        if not 1 <= self.L <= 8 or not 1 <= self.past_bits <= 16:
            raise ValueError("L is 1..8 bytes, past_bits 1..16")
        if self.compile_bytes % 8 or self.compile_bytes < 8:
            raise ValueError("compile_bytes is a multiple of 8")


def _mix(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


class MomentStore:
    """Fixed-size sums of counts (T11.5; not a transcript).

    ``c2[f, p]``: positions with past bucket ``p`` followed by byte ``f``. ``c3[x, f, p]``: positions with past ``p``,
    byte ``x`` and then byte ``f``. ``c2h``: the same as ``c2`` for random halves of blocks of ``split_block`` positions
    (the split-half edge of the Page rank). The last ``L + 1`` bytes are kept to join one update to the next.
    """

    def __init__(self, cfg: OperatorConfig) -> None:
        self.cfg = cfg
        P = 1 << cfg.past_bits
        self.c2 = np.zeros((ALPHABET, P), dtype=np.int64)
        self.c2h = [np.zeros((ALPHABET, P), dtype=np.int64) for _ in range(cfg.splits)]
        self.cmap = None
        R = ALPHABET
        if cfg.c3_bytes:
            self.cmap = np.full(ALPHABET, -1, dtype=np.int64)
            self.cmap[list(cfg.c3_bytes)] = np.arange(len(cfg.c3_bytes))
            R = len(cfg.c3_bytes)
        self.c3 = np.zeros((R, R, P), dtype=np.uint32)
        self.tail = b""
        self.n = 0            # bytes seen
        self.n2 = 0           # positions counted in c2
        self.n3 = 0           # positions counted in c3
        self.max_cell = 0

    def bucket(self, buf: np.ndarray, js: np.ndarray) -> np.ndarray:
        """Past bucket of the ``L`` bytes before each position ``js`` of ``buf``."""
        v = np.zeros(len(js), dtype=np.uint64)
        for i in range(self.cfg.L):
            v |= buf[js - 1 - i].astype(np.uint64) << np.uint64(8 * i)
        if 8 * self.cfg.L <= self.cfg.past_bits:
            return v.astype(np.int64)
        h = v * np.uint64(0x9E3779B97F4A7C15)
        return (h >> np.uint64(64 - self.cfg.past_bits)).astype(np.int64)

    def update(self, data: bytes) -> None:
        cfg, P, L = self.cfg, 1 << self.cfg.past_bits, self.cfg.L
        if not data:
            return
        buf = np.frombuffer(self.tail + data, dtype=np.uint8)
        ntail = len(self.tail)
        g0 = self.n - ntail                                  # global index of buf[0]
        m = len(buf)
        end = self.n + len(data)
        # c2: global positions g in [max(n, L), end)
        lo2 = max(self.n, L) - g0
        if lo2 < m:
            js = np.arange(lo2, m)
            p = self.bucket(buf, js)
            f = buf[js].astype(np.int64)
            idx = f * P + p
            self.c2 += np.bincount(idx, minlength=ALPHABET * P).reshape(ALPHABET, P)
            g = js + g0
            blk = (g // cfg.split_block).astype(np.uint64)
            for s in range(cfg.splits):
                salt = np.uint64(((s + 1) * 0x9E3779B97F4A7C15 + cfg.seed) & 0xFFFFFFFFFFFFFFFF)
                sel = (_mix(blk ^ salt) & np.uint64(1)).astype(bool)
                self.c2h[s] += np.bincount(idx[sel], minlength=ALPHABET * P).reshape(ALPHABET, P)
            self.n2 += len(js)
        # c3: global positions g in [max(n - 1, L), end - 1)
        lo3 = max(self.n - 1, L) - g0
        if lo3 < m - 1:
            js = np.arange(lo3, m - 1)
            p = self.bucket(buf, js)
            x = buf[js].astype(np.int64)
            f = buf[js + 1].astype(np.int64)
            R = self.c3.shape[0]
            if self.cmap is not None:
                x, f = self.cmap[x], self.cmap[f]
                keep = (x >= 0) & (f >= 0)
                x, f, p = x[keep], f[keep], p[keep]
            idx = (x * R + f) * P + p
            uniq, cnt = np.unique(idx, return_counts=True)
            flat = self.c3.reshape(-1)
            new = flat[uniq].astype(np.uint64) + cnt.astype(np.uint64)
            if len(new):
                if int(new.max()) > 0xFFFFFFFF:
                    raise OverflowError("a C3 cell would overflow 32 bits")
                flat[uniq] = new.astype(np.uint32)
                self.max_cell = max(self.max_cell, int(new.max()))
            self.n3 += len(js)
        self.tail = bytes(buf[-(L + 1):])
        self.n = end

    def digest(self) -> str:
        h = hashlib.sha256()
        for a in (self.c2, *self.c2h, self.c3):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(self.tail + str((self.n, self.n2, self.n3)).encode())
        return h.hexdigest()

    def occupied_buckets(self) -> int:
        """Past buckets that hold at least one position (the distinct contexts per bucket is this over the distinct
        contexts, which the run script counts from the stream)."""
        return int((self.c2.sum(axis=0) > 0).sum())


class OperatorModel:
    """Compiled operators; immutable (the state lives in the machine)."""

    def __init__(self, Bs, W, b1, b_inf, U, pi, k: int, rank: int, info: dict) -> None:
        self.Bs, self.W, self.b1, self.b_inf, self.U, self.pi = Bs, W, b1, b_inf, U, pi
        self.k, self.rank, self.info = k, rank, info

    def reproject(self, b: np.ndarray, old: OperatorModel) -> np.ndarray:
        """Carry a state over a recompile: ``b <- U_new' (U_old')^+ b`` rescaled so that ``b_inf' b = 1``."""
        if old.U is None or self.U is None:
            return self.b1.copy()
        nb = self.U.T @ (np.linalg.pinv(old.U.T, rcond=PINV_RCOND) @ b)
        mass = float(self.b_inf @ nb)
        if not np.isfinite(mass) or mass <= 0.0:
            return self.b1.copy()
        return nb / mass


def page_rank_of(store: MomentStore, cfg: OperatorConfig) -> int:
    """The Page rank of the past-by-next-byte counts with E4's count-floor rule (``e4_hier.floor_rank``)."""
    c2 = store.c2
    r, c = c2.sum(1), c2.sum(0)
    kr, kc = np.flatnonzero(r >= cfg.floor), np.flatnonzero(c >= cfg.floor)
    if len(kr) < 2 or len(kc) < 2:
        return 0
    full = sparse.csr_matrix(c2[kr][:, kc].astype(float))
    halves = [sparse.csr_matrix(h[kr][:, kc].astype(float)) for h in store.c2h]
    halves = [h for h in halves if 0.0 < h.sum() < full.sum()]       # a half needs samples on both sides of the split
    return int(page.page_spectrum(full, halves or None, cfg.margin)["rank"])


def compile_operator(store: MomentStore, cfg: OperatorConfig) -> OperatorModel | None:
    """Rebuild O from the counts alone (nothing else is read). ``None`` before there are enough positions."""
    if store.n2 < MIN_COUNTS or store.n3 < MIN_COUNTS:
        return None
    t0 = time.perf_counter()
    c2 = store.c2.astype(float)
    n2 = float(c2.sum())
    pi = c2.sum(1) / n2
    if cfg.mode == "unigram":
        k = 1
        Bs = pi.reshape(ALPHABET, 1, 1).copy()
        W = pi.reshape(ALPHABET, 1).copy()
        return OperatorModel(Bs, W, np.ones(1), np.ones(1), None, pi, 1, 1, {"seconds": time.perf_counter() - t0})
    r_sum, c_sum = c2.sum(1), c2.sum(0)
    rank = page_rank_of(store, cfg)
    rows, cols = np.flatnonzero(r_sum > 0), np.flatnonzero(c_sum > 0)
    k = cfg.k_force or min(cfg.k_cap, max(cfg.k_min, rank))
    k = max(1, min(k, len(rows), len(cols)))
    C = c2[rows][:, cols] / np.sqrt(r_sum[rows])[:, None] / np.sqrt(c_sum[cols])[None, :]
    Uw = np.linalg.svd(C, full_matrices=False)[0][:, :k]
    U = np.zeros((ALPHABET, k))
    U[rows] = np.sqrt(r_sum[rows] / n2)[:, None] * Uw                 # D_r^(1/2) U_k: spans the column space of C2
    P2 = c2 / n2
    A = U.T @ P2                                                      # k x P
    Q = np.linalg.pinv(A, rcond=PINV_RCOND)                           # P x k
    n3 = float(store.n3)
    Bs = np.empty((ALPHABET, k, k))
    UT = np.ascontiguousarray(U.T)
    for x in range(ALPHABET):
        if store.cmap is None:
            P3x = store.c3[x].astype(float) / n3                      # f x P
        else:
            P3x = np.zeros((ALPHABET, c2.shape[1]))
            if store.cmap[x] >= 0:
                P3x[list(cfg.c3_bytes)] = store.c3[store.cmap[x]].astype(float) / n3
        Bs[x] = (UT @ P3x) @ Q
    b1 = U.T @ pi
    m = c2.sum(0) / n2
    b_inf = np.linalg.pinv(P2.T @ U, rcond=PINV_RCOND) @ m            # (P2' U)^+ m
    W = np.einsum("xij,i->xj", Bs, b_inf)                             # row x: b_inf' B_x
    info = {"k": k, "page_rank": rank, "rows": len(rows), "cols": len(cols), "samples": n2,
            "seconds": time.perf_counter() - t0}
    return OperatorModel(np.ascontiguousarray(Bs), np.ascontiguousarray(W), b1, b_inf, U, pi, k, rank, info)


def _make_o_pass(jit):
    def o_pass(data, b, Bs, W, b1, pi, eps_o, eps, extp, extu, ostat):
        """For each byte of ``data``: the probability of bit 0 at each of its 8 bit positions (given the bits of the
        byte before it), then the state update. ``ostat``: negative-mass share sum, bytes with a negative part, bytes,
        resets."""
        q = np.empty(256)
        cs = np.empty(257)
        for t in range(data.shape[0]):
            x = int(data[t])
            raw = np.dot(W, b)
            tot = raw.sum()
            invalid = not (tot > 1e-12)
            if invalid:
                for i in range(256):
                    q[i] = pi[i]
            else:
                neg = 0.0
                ab = 0.0
                for i in range(256):
                    v = raw[i] / tot
                    q[i] = v
                    ab += abs(v)
                    if v < 0.0:
                        neg += -v
                ostat[0] += neg / ab
                if neg > 0.0:
                    ostat[1] += 1.0
            ostat[2] += 1.0
            s = 0.0
            for i in range(256):
                if q[i] < eps_o:
                    q[i] = eps_o
                s += q[i]
            cs[0] = 0.0
            for i in range(256):
                cs[i + 1] = cs[i] + q[i] / s
            for j in range(8):
                sh = 8 - j
                lo = (x >> sh) << sh
                size = 1 << sh
                p0 = (cs[lo + size // 2] - cs[lo]) / (cs[lo + size] - cs[lo])
                if p0 < eps:
                    p0 = eps
                elif p0 > 1.0 - eps:
                    p0 = 1.0 - eps
                extp[8 * t + j] = p0
                extu[8 * t + j] = (1 << j) | (x >> sh)
            absum = 0.0
            for i in range(256):
                absum += abs(raw[i])
            mass = raw[x]
            if invalid or not (mass > eps_o * absum):
                for i in range(b.shape[0]):
                    b[i] = b1[i]
                ostat[3] += 1.0
            else:
                nb = np.dot(Bs[x], b)
                for i in range(b.shape[0]):
                    b[i] = nb[i] / mass

    return jit(o_pass) if jit is not None else o_pass


_o_pass = _make_o_pass(njit(cache=True)) if NUMBA_AVAILABLE else None
_loop_o = make_loop_o(njit(cache=True)) if NUMBA_AVAILABLE else None


def hier_config(spec: dict, law) -> HierConfig:
    """The :class:`HierConfig` that ``hier.build(spec, law)`` makes (same rules; tested equal)."""
    from ..nested import NestedConfig as NC
    from .hier import ALL_UPPER, S0, S1, W1, W2, W4
    base_keys = ("depths", "coherence", "verify", "governor", "protect", "sink_rate")
    extra = {k: spec[k] for k in base_keys if k in spec}
    depths = tuple(extra.pop("depths", (8, 16, 24, 32, 48, 64)))
    base = NC(law, depths=depths, table_bits=22, prior=spec["prior"], **extra)
    names = {"W": (W1, W2, W4), "S": (S0, S1), "WS": ALL_UPPER, "": ()}
    kw = {"upper": names[spec.get("hier_levels", "WS")]}
    for src, dst in (("hier_bits", "upper_bits"), ("hier_verify", "upper_verify"), ("hier_segment", "segment"),
                     ("hier_scope", "scope"), ("hier_raise", "raise_bar"), ("hier_control", "control")):
        if src in spec:
            kw[dst] = spec[src]
    return HierConfig(base, **kw)


class OperatorMachine(HierMachine):
    """E4's machine with the O stage. ``mode="off"``: E4 bit for bit (the same loop code, the stage skipped)."""

    def __init__(self, hcfg: HierConfig, ocfg: OperatorConfig) -> None:
        super().__init__(hcfg)
        self.ocfg = ocfg
        self._store = MomentStore(ocfg) if ocfg.mode != "off" else None
        self._model: OperatorModel | None = None
        self._b: np.ndarray | None = None
        self._since = 0
        self._ev_o = np.zeros(256)                 # evidence of the 255 mixing nodes (index = partial-byte state u)
        self._acc = np.zeros((256, 3))             # per node: bits of the stage input, of O, of the stage output
        self._ostat = np.zeros(4)
        self._top = 0                              # E7: 1 puts the stage after the top level (the final readout)
        self.compiles: list[dict] = []

    # -- state ----------------------------------------------------------------------------------
    def snapshot(self) -> tuple:
        return (self.state, self._stats.copy(), None if self._b is None else self._b.copy(), self._ostat.copy())

    def restore(self, snap: tuple) -> None:
        self.set_state(list(snap[0]))
        self._stats[:] = snap[1]
        self._b = None if snap[2] is None else snap[2].copy()
        self._ostat[:] = snap[3]

    def copy(self):
        other = type(self).__new__(type(self))
        other.__dict__.update(self.__dict__)
        for name in ("_wn", "_tags", "_stats", "_door", "_occ", "_st", "_ev_o", "_acc", "_ostat"):
            setattr(other, name, getattr(self, name).copy())
        other._b = None if self._b is None else self._b.copy()
        other.compiles = list(self.compiles)
        return other        # the count store and the compiled operators are shared: frozen reads write none

    def nbytes(self) -> int:
        return super().nbytes() + (0 if self._store is None else int(self._store.c3.nbytes + self._store.c2.nbytes))

    # -- O ------------------------------------------------------------------------------------------
    def table_hashes(self) -> dict:
        names = ("_wn", "_tags", "_door", "_occ")
        out = {n: hashlib.sha256(np.ascontiguousarray(getattr(self, n)).tobytes()).hexdigest() for n in names}
        if self._store is not None:
            out["store"] = self._store.digest()
        if self._model is not None:
            out["operators"] = hashlib.sha256(np.ascontiguousarray(self._model.Bs).tobytes()).hexdigest()
        return out

    def _compile(self) -> None:
        new = compile_operator(self._store, self.ocfg)
        if new is None:
            return
        self._b = new.b1.copy() if (self._model is None or self._b is None) else new.reproject(self._b, self._model)
        self._model = new
        self._since = 0
        self.compiles.append({"bytes": self._store.n, **new.info})

    def reenter(self, byte: int) -> bool:
        """Re-entry of a symbol the machine itself emitted (generation mode; DESIGN V6): the same operator update as a
        read, no learning. Returns whether the state changed. External bytes being read are never re-entered."""
        if self._model is None or self._b is None:
            return False
        m = self._model
        raw = float(m.W[byte] @ self._b)
        absum = float(np.abs(m.W @ self._b).sum())
        old = self._b
        if not raw > self.ocfg.eps_o * absum:
            self._b = m.b1.copy()
        else:
            self._b = (m.Bs[byte] @ old) / raw
        return not np.array_equal(old, self._b)

    # -- reading --------------------------------------------------------------------------------------
    def _stage(self, data: np.ndarray, extp: np.ndarray, extu: np.ndarray, dummy: tuple, learning: bool) -> tuple:
        """The per-bit probability and node of the stage for one block of bytes, and whether the stage is on. E6's stage
        is O; ``FamilyMachine`` (E7) overrides this and nothing else of ``run``."""
        if self.ocfg.mode != "off" and self._model is not None:
            mo = self._model
            _o_pass(data, self._b, mo.Bs, mo.W, mo.b1, mo.pi, self.ocfg.eps_o, self.ocfg.eps_bit, extp, extu,
                    self._ostat)
            return extp, extu, True
        return dummy[0], dummy[1], False

    def run(self, bits: np.ndarray, learning: bool = True) -> np.ndarray:
        bits = np.ascontiguousarray(bits, dtype=np.uint8)
        n = bits.shape[0]
        if n % 8:
            raise ValueError("O reads whole bytes")
        c = self.cfg.law
        out = np.empty(n)
        text, comm = self.hcfg.segment == "text", self.hcfg.control == "comm"
        st = self._st
        st[ST_REG] = np.uint64(self.reg)
        seg = min(BLOCK_BITS, self.ocfg.compile_bytes * 8)
        ext = np.empty((min(n, seg), max(1, self._Lu)), dtype=np.uint64)
        extp_all = np.empty(min(n, seg))
        extu_all = np.empty(min(n, seg), dtype=np.int64)
        dummy_p, dummy_u = np.zeros(1), np.zeros(1, dtype=np.int64)
        glob = self.hcfg.scope == "global"
        for a in range(0, n, seg):
            blk = bits[a:a + seg]
            m = blk.shape[0]
            if self._Lu:
                _fill_keys(blk, st, ext[:m], self._kinds, self._dups, text, comm)
            else:
                st[ST_REG] = np.uint64(self.reg)
            data = np.packbits(blk)
            ep, eu, o_on = self._stage(data, extp_all[:m], extu_all[:m], (dummy_p, dummy_u), learning)
            reg = _loop_o(blk, ext[:m], self._wn, self._tags, self._tagoff, self._offsets, self._depths, self._hidx,
                          self._extcol, np.uint64(self.reg), self._shifts, self.cfg.readout == "mixture",
                          float(self.cfg.mature), 1.0 - 2.0 * c.eps, c.gamma, c.g / c.kappa, c.kappa * c.dt,
                          c.consolidating, learning, self.cfg.inherit, float(self.cfg.prior), out[a:a + m],
                          self._stats, float(self.cfg.coherence), self._verify, bool(self.cfg.governor),
                          float(self.cfg.protect), float(self.cfg.sink_factor), self._door, self._dooroff,
                          self._door_shifts, self._occ, self._tsizes, self._tcap, glob, bool(self.hcfg.raise_bar),
                          self._door_on, ep, eu, self._ev_o, self._acc, o_on, self._top)
            self.reg = reg
            st[ST_REG] = np.uint64(reg)
            if learning and self._store is not None:
                self._store.update(data.tobytes())
                self._since += len(data)
                if self._since >= self.ocfg.compile_bytes:
                    self._compile()
        return out

    # -- measures -------------------------------------------------------------------------------------
    def negative_mass(self) -> dict:
        s = self._ostat
        nb = max(1.0, float(s[2]))
        return {"bytes": float(s[2]), "mean_negative_share": float(s[0]) / nb,
                "share_of_bytes_with_negative": float(s[1]) / nb, "resets": float(s[3])}

    def stage_regret(self) -> dict:
        """Per node: the loss of the stage output against the smaller of its input's and O's loss (bits)."""
        acc = self._acc[1:]
        excess = acc[:, 2] - np.minimum(acc[:, 0], acc[:, 1])
        return {"max_excess_bits": float(excess.max()), "total_excess_bits": float(excess.sum()),
                "nodes_used": int((acc.sum(axis=1) > 0).sum()), "bound_per_node": 2.0}


def build_operator(spec: dict, law, ocfg: OperatorConfig) -> OperatorMachine:
    return OperatorMachine(hier_config(spec, law), ocfg)


__all__ = [
    "MomentStore",
    "OperatorConfig",
    "OperatorMachine",
    "OperatorModel",
    "build_operator",
    "compile_operator",
    "hier_config",
    "page_rank_of",
]
