"""Word-class levels on text (E6b; experiments/PROTOCOL.md, E6b; DESIGN section 12, V2').

The class map is a compile-time fit to forward-only counts of E4's unit stream (T11.2 canalization: units whose
neighbors are alike share a class), and K1, K2, K4 are ordinary upper levels of E4's chain keyed by the classes of the
last 1, 2 and 4 units. ``KMachine`` is E4's machine with that pre-pass; with the K levels off it is E4 bit for bit.

* ``ClassStore``: 2^17 hashed rows, a majority-vote tag per row, 512 hashed neighbor counters per side (fixed size).
* ``compile_classes``: positive PMI of the 2^15 largest rows, 64 Gram eigenvectors, spherical frequency-weighted k-means
  with 256 classes, warm-started from the previous map so ids persist; reads the store and the previous map only.
* ``control_map``: a persistent random map with the real map's rows per class.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

from ..nested import NUMBA_AVAILABLE
from .hier import (
    BLOCK_BITS,
    GOLDEN,
    MASK64,
    S0,
    S1,
    SEED_C,
    ST_L3,
    ST_R0,
    ST_REG,
    W1,
    W2,
    W4,
    HierConfig,
    HierMachine,
    _closes_unit,
    _compiled_loop,
    advance,
    initial_state,
    mix64,
)
from .operator import hier_config

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

from .mixloop import make_loop_deep
from .wcprepass import SALT_K, make_prepass_k

K1, K2, K4 = 5, 6, 7
K_ORDER = (K1, W1, K2, W2, K4, W4, S0, S1)         # the declared order in the mixture (E6b.3)
K_AFTER = (W1, W2, W4, S0, S1, K1, K2, K4)         # the development check: the class levels after S1
UNK = 256                                           # the reserved class of a unit outside the map
NOMAP = 0xFFFF                                      # an unselected row in a class array
ST_Q0 = 11                                          # class ring: st[11:15]
ST_SIZE_K = 15

_fill_keys_k = make_prepass_k(njit(cache=True)) if NUMBA_AVAILABLE else None
_loop_deep = make_loop_deep(njit(cache=True)) if NUMBA_AVAILABLE else None


# ------------------------------------------------------------------------------------------------------------------
# the count store
# ------------------------------------------------------------------------------------------------------------------

def _make_store_update(jit):
    def store_update(codes, rbits, feats, tag, tcnt, counts):
        for i in range(1, codes.shape[0] - 1):
            code = codes[i]
            r = np.int64(code >> np.uint64(64 - rbits))
            if tag[r] == code:
                tcnt[r] += 1
            elif tcnt[r] == 0:
                tag[r] = code
                tcnt[r] = 1
            else:
                tcnt[r] -= 1
            counts[r, np.int64(codes[i + 1] % np.uint64(feats))] += 1
            counts[r, feats + np.int64(codes[i - 1] % np.uint64(feats))] += 1

    return jit(store_update) if jit is not None else store_update


_store_update = _make_store_update(njit(cache=True)) if NUMBA_AVAILABLE else _make_store_update(None)


@dataclass(frozen=True)
class WordClassConfig:
    """PROTOCOL E6b.3: every constant is fixed there and none is tuned."""

    mode: str = "real"              # "real" (clustered map), "random" (the control), "off" (E4: no K levels)
    row_bits: int = 17              # rows of the count store: 2**row_bits
    feats: int = 512                # hashed neighbor counters per side
    V: int = 1 << 15                # rows clustered at a compile (the V rule of E6b.8 may halve it)
    d: int = 64                     # eigenvectors kept
    C: int = 256                    # classes
    iters: int = 15                 # Lloyd iterations
    min_count: int = 4              # a row needs this tag count to be clustered
    compile_bytes: int = 1 << 20    # bytes between compiles (a smoke run uses 1 << 15)
    seed: int = 0
    order: str = "interleaved"      # "interleaved" (E6b.3) or "after" (K1, K2, K4 after S1; the development check only)
    fix_map: bool = False           # the map is compiled once and never again (the development check only)

    def __post_init__(self) -> None:
        if self.mode not in ("real", "random", "off"):
            raise ValueError("mode is 'real', 'random' or 'off'")
        if self.order not in ("interleaved", "after"):
            raise ValueError("order is 'interleaved' or 'after'")
        if self.compile_bytes % 8 or self.compile_bytes < 8:
            raise ValueError("compile_bytes is a multiple of 8")


class ClassStore:
    """Fixed-size sums of counts of the unit stream (T11.5; not a transcript)."""

    def __init__(self, cfg: WordClassConfig) -> None:
        self.cfg = cfg
        rows = 1 << cfg.row_bits
        self.tag = np.zeros(rows, dtype=np.uint64)
        self.tcnt = np.zeros(rows, dtype=np.int64)
        self.counts = np.zeros((rows, 2 * cfg.feats), dtype=np.uint32)
        self.carry = np.zeros(0, dtype=np.uint64)
        self.n_units = 0

    def update(self, codes: np.ndarray) -> None:
        """Add the units of a block (their codes in reading order); the last two codes are carried to the next call."""
        if len(codes) == 0:
            return
        seq = np.concatenate([self.carry, np.ascontiguousarray(codes, dtype=np.uint64)])
        if len(seq) >= 3:
            _store_update(seq, self.cfg.row_bits, self.cfg.feats, self.tag, self.tcnt, self.counts)
        self.carry = seq[-2:].copy()
        self.n_units += len(codes)

    def digest(self) -> str:
        h = hashlib.sha256()
        for a in (self.tag, self.tcnt, self.counts, self.carry):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(str(self.n_units).encode())
        return h.hexdigest()

    def nbytes(self) -> int:
        return int(self.tag.nbytes + self.tcnt.nbytes + self.counts.nbytes)


# ------------------------------------------------------------------------------------------------------------------
# the class map
# ------------------------------------------------------------------------------------------------------------------

@dataclass
class ClassMap:
    tag: np.ndarray                 # the row tags at the compile (a unit is in the map iff its row tag is its code)
    cls: np.ndarray                 # uint16 per row: a class id, or NOMAP for a row not clustered
    sel: np.ndarray                 # the rows clustered
    info: dict = field(default_factory=dict)

    def digest(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.tag).tobytes() + np.ascontiguousarray(self.cls).tobytes()
                              ).hexdigest()


def compile_classes(store: ClassStore, prev: ClassMap | None, cfg: WordClassConfig, index: int) -> ClassMap | None:
    """Rebuild the map from the counts alone (and the previous map, for the class ids). ``None`` while fewer than
    ``2 C`` rows qualify."""
    t0 = time.perf_counter()
    order = np.argsort(-store.tcnt, kind="stable")[:cfg.V]
    sel = order[store.tcnt[order] >= cfg.min_count]
    if len(sel) < 2 * cfg.C:
        return None
    P = store.counts[sel].astype(np.float64)
    tot, rs, cs = P.sum(), P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.log(P * tot / (rs * cs))
    P[~np.isfinite(P)] = 0.0
    np.maximum(P, 0.0, out=P)
    w, vec = np.linalg.eigh(P.T @ P)
    X = P @ vec[:, np.argsort(-w, kind="stable")[:cfg.d]]
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    wt = store.tcnt[sel].astype(np.float64)
    wt /= wt.sum()
    rng = np.random.default_rng(cfg.seed + index)
    cent = np.empty((cfg.C, X.shape[1]))
    if prev is None:
        cent[:] = X[rng.choice(len(sel), cfg.C, replace=False, p=wt)]
    else:
        pc = prev.cls[sel].astype(np.int64)
        for c in range(cfg.C):
            m = pc == c
            if m.any():
                v = (X[m] * wt[m, None]).sum(0)
                cent[c] = v / max(np.linalg.norm(v), 1e-12)
            else:
                cent[c] = X[rng.integers(len(sel))]
    for _ in range(cfg.iters):
        a = np.argmax(X @ cent.T, axis=1)
        for c in range(cfg.C):
            m = a == c
            if m.any():
                v = (X[m] * wt[m, None]).sum(0)
                cent[c] = v / max(np.linalg.norm(v), 1e-12)
    a = np.argmax(X @ cent.T, axis=1)
    cls = np.full(len(store.tag), NOMAP, dtype=np.uint16)
    cls[sel] = a
    rows_per = np.bincount(a, minlength=cfg.C)
    mass_per = np.bincount(a, weights=store.tcnt[sel].astype(np.float64), minlength=cfg.C)
    info = {"index": index, "seconds": time.perf_counter() - t0, "rows": len(sel),
            "classes_used": int((rows_per > 0).sum()),
            "rows_per_class": np.percentile(rows_per, [0, 25, 50, 75, 100]).tolist(),
            "mass_per_class": np.percentile(mass_per, [0, 25, 50, 75, 100]).tolist(),
            "mass_in_map_share": float(store.tcnt[sel].sum() / max(1, store.tcnt.sum()))}
    if prev is not None:
        both = prev.cls[sel] != NOMAP
        if both.any():
            info["churn"] = float((wt[both] * (prev.cls[sel][both] != a[both])).sum() / wt[both].sum())
    return ClassMap(store.tag.copy(), cls, sel, info)


def control_map(real: ClassMap, prev: ClassMap | None, cfg: WordClassConfig, index: int) -> ClassMap:
    """The random control: the same rows, the real map's rows per class, assignments that persist (E6b.3). A row that
    already had a control class keeps it; the newly selected rows take the real classes of the newly selected rows in a
    random order. Uses neither the factorization nor the k-means, only the class sizes."""
    rng = np.random.default_rng(cfg.seed + 7919 + index)
    sel = real.sel
    real_cls = real.cls[sel]
    cls = np.full(len(real.cls), NOMAP, dtype=np.uint16)
    if prev is None:
        cls[sel] = real_cls[rng.permutation(len(sel))]
    else:
        have = prev.cls[sel] != NOMAP
        cls[sel[have]] = prev.cls[sel[have]]
        new = np.flatnonzero(~have)
        cls[sel[new]] = real_cls[new][rng.permutation(len(new))]
    info = dict(real.info, control=True, real_churn=real.info.get("churn"))
    info.pop("churn", None)
    if prev is not None:                           # the control's own churn: rows in both maps whose class changed
        both = (prev.cls[sel] != NOMAP)
        info["churn"] = float((prev.cls[sel][both] != cls[sel][both]).mean()) if both.any() else 0.0
    return ClassMap(real.tag, cls, sel, info)


# ------------------------------------------------------------------------------------------------------------------
# units, keys: the pure Python writing, and offline helpers
# ------------------------------------------------------------------------------------------------------------------

def _make_unit_codes(jit):
    def unit_codes(data):
        n = data.shape[0]
        out = np.empty(n, dtype=np.uint64)
        k = 0
        c = np.uint64(0x243F6A8885A308D3)
        m30, m27, m31 = np.uint64(30), np.uint64(27), np.uint64(31)
        c1, c2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)
        golden = np.uint64(0x9E3779B97F4A7C15)
        for i in range(n):
            b = np.int64(data[i])
            c = c ^ ((np.uint64(b) + np.uint64(1)) * golden)
            c = (c ^ (c >> m30)) * c1
            c = (c ^ (c >> m27)) * c2
            c = c ^ (c >> m31)
            if not ((48 <= b <= 57) or (65 <= b <= 90) or (97 <= b <= 122) or b == 39):
                out[k] = c
                k += 1
                c = np.uint64(0x243F6A8885A308D3)
        return out[:k]

    return jit(unit_codes) if jit is not None else unit_codes


unit_codes = _make_unit_codes(njit(cache=True)) if NUMBA_AVAILABLE else _make_unit_codes(None)


def class_of(code: int, cmap: ClassMap | None, row_bits: int) -> int:
    if cmap is None:
        return UNK
    r = code >> (64 - row_bits)
    return int(cmap.cls[r]) if (int(cmap.tag[r]) == code and cmap.cls[r] < 256) else UNK


def advance_k(st: list[int], bit: int, cmap: ClassMap | None, row_bits: int, text: bool = True) -> list[int]:
    """One bit through the registers including the class ring (the pure Python writing of the pre-pass)."""
    new = advance(st[:ST_Q0], bit, text, False)
    out = [*new, *st[ST_Q0:ST_Q0 + 4]]
    byte_done = st[1] >= 128                          # the eighth bit of a byte: ST_U holds 1 and the leading bit
    if byte_done and _closes_unit(((st[1] << 1) | bit) & 255, new[ST_L3], text):       # a unit closed: new head
        out[ST_Q0 + 3], out[ST_Q0 + 2], out[ST_Q0 + 1] = st[ST_Q0 + 2], st[ST_Q0 + 1], st[ST_Q0]
        out[ST_Q0] = class_of(new[ST_R0], cmap, row_bits)
    return out


def level_key_k(st: list[int], level: int) -> int:
    """The key of a class level for the bit about to be read (the pure Python writing)."""
    u, c = st[1], st[2]
    n = (1, 2, 4)[level - K1]
    h = SEED_C
    for k in range(n):
        h = mix64(h ^ ((st[ST_Q0 + k] + 0x5851F42D4C957F2D) & MASK64))
    return mix64(h ^ mix64((c + 0x2FFD72DBD01ADFB7) & MASK64) ^ ((u * GOLDEN) & MASK64) ^ SALT_K[level])


def context_keys(data: bytes, kind: int) -> np.ndarray:
    """The key of upper level ``kind`` (W1 = 0, W2 = 1, ...) at the first bit of every byte of ``data``, read with the
    registers of E4 (no class map). Equal keys are equal contexts: the key at a byte start is a function of the last
    units and the running hash of the partial unit alone."""
    from ..bits import bytes_to_bits
    st = np.array(initial_state() + [UNK] * 4, dtype=np.uint64)
    kinds, dups = np.array([kind], dtype=np.int64), np.zeros(1, dtype=np.int64)
    rows = 1 << 4
    ctag, ccls = np.zeros(rows, dtype=np.uint64), np.full(rows, NOMAP, dtype=np.uint16)
    codes, ncodes = np.empty(BLOCK_BITS // 8 + 8, dtype=np.uint64), np.zeros(1, dtype=np.int64)
    out = np.empty(len(data), dtype=np.uint64)
    step = BLOCK_BITS // 8
    for a in range(0, len(data), step):
        bits = bytes_to_bits(data[a:a + step])
        ext = np.empty((bits.shape[0], 1), dtype=np.uint64)
        ncodes[0] = 0
        _fill_keys_k(bits, st, ext, kinds, dups, True, False, 4, ctag, ccls, codes, ncodes, UNK)
        out[a:a + step] = ext[::8, 0]
    return out


# ------------------------------------------------------------------------------------------------------------------
# the machine
# ------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class KHierConfig(HierConfig):
    """``HierConfig`` whose upper levels may include the class levels, in the declared order."""

    def __post_init__(self) -> None:
        if any(k not in (*range(5), K1, K2, K4) for k in self.upper) or len(set(self.upper)) != len(self.upper):
            raise ValueError("upper levels are a subset of W1, W2, W4, S0, S1, K1, K2, K4")
        if self.segment not in ("content", "text") or self.scope not in ("level", "global"):
            raise ValueError("segment is 'content' or 'text'; scope is 'level' or 'global'")
        if self.control not in ("none", "dup", "comm"):
            raise ValueError("control is 'none', 'dup' or 'comm'")
        if not 4 <= self.upper_bits <= 26:
            raise ValueError("upper_bits lies in 4..26")
        if self.upper_verify and not 1 <= self.upper_verify <= 255:
            raise ValueError("verify counts sightings, 1..255")


def khier_config(spec: dict, law, mode: str, order: str = "interleaved") -> KHierConfig:
    base = hier_config(spec, law)
    upper = base.upper if mode == "off" else (K_ORDER if order == "interleaved" else K_AFTER)
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(HierConfig)}
    fields["upper"] = upper
    return KHierConfig(**fields)


class KMachine(HierMachine):
    """E4's machine with the class ring in its pre-pass. ``mode="off"``: E4 bit for bit (no K levels, no store)."""

    def __init__(self, hcfg: KHierConfig, wcfg: WordClassConfig) -> None:
        super().__init__(hcfg)
        self.wcfg = wcfg
        self._st = np.array(initial_state() + [UNK] * 4, dtype=np.uint64)
        rows = 1 << wcfg.row_bits
        self._store = ClassStore(wcfg) if wcfg.mode != "off" else None
        self._ctag = np.zeros(rows, dtype=np.uint64)
        self._ccls = np.full(rows, NOMAP, dtype=np.uint16)
        self._real: ClassMap | None = None
        self._ctrl: ClassMap | None = None
        self._since = 0
        self._ncompiles = 0
        self.compiles: list[dict] = []

    # -- state ---------------------------------------------------------------------------------------------
    def snapshot(self) -> tuple:
        return (self.state, self._stats.copy())

    def restore(self, snap: tuple) -> None:
        self.set_state(list(snap[0]))
        self._stats[:] = snap[1]

    def copy(self) -> KMachine:
        other = type(self).__new__(type(self))
        other.__dict__.update(self.__dict__)
        for name in ("_wn", "_tags", "_stats", "_door", "_occ", "_st"):
            setattr(other, name, getattr(self, name).copy())
        other.compiles = list(self.compiles)
        return other                # the count store and the class map are shared: frozen reads write neither

    def nbytes(self) -> int:
        return super().nbytes() + (0 if self._store is None else self._store.nbytes())

    def table_hashes(self) -> dict:
        names = ("_wn", "_tags", "_door", "_occ")
        out = {n: hashlib.sha256(np.ascontiguousarray(getattr(self, n)).tobytes()).hexdigest() for n in names}
        if self._store is not None:
            out["store"] = self._store.digest()
        out["map"] = hashlib.sha256(self._ctag.tobytes() + self._ccls.tobytes()).hexdigest()
        return out

    # -- the map ------------------------------------------------------------------------------------------
    def _compile(self) -> None:
        if self.wcfg.fix_map and self._ncompiles >= 1:
            self._since = 0
            return
        real = compile_classes(self._store, self._real, self.wcfg, self._ncompiles)
        if real is None:
            return
        self._real = real
        active = real
        if self.wcfg.mode == "random":
            active = control_map(real, self._ctrl, self.wcfg, self._ncompiles)
            self._ctrl = active
        self._ctag, self._ccls = active.tag, active.cls
        self._ncompiles += 1
        self._since = 0
        self.compiles.append({"bytes": int(self._store.n_units), **active.info})

    @property
    def class_map(self) -> ClassMap | None:
        return self._ctrl if self.wcfg.mode == "random" else self._real

    # -- reading ------------------------------------------------------------------------------------------
    def run(self, bits: np.ndarray, learning: bool = True) -> np.ndarray:
        return self._run(bits, learning, None)

    def run_deep(self, bits: np.ndarray, learning: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """``run`` that also returns, per bit, the deepest level that contributes to the readout (E6b Amendment 1)."""
        deep = np.empty(np.asarray(bits).shape[0], dtype=np.int8)
        return self._run(bits, learning, deep), deep

    def _run(self, bits: np.ndarray, learning: bool, deep: np.ndarray | None) -> np.ndarray:
        bits = np.ascontiguousarray(bits, dtype=np.uint8)
        c = self.cfg.law
        n = bits.shape[0]
        out = np.empty(n)
        text, comm = self.hcfg.segment == "text", self.hcfg.control == "comm"
        st = self._st
        st[ST_REG] = np.uint64(self.reg)
        ext = np.empty((min(n, BLOCK_BITS), max(1, self._Lu)), dtype=np.uint64)
        codes = np.empty(BLOCK_BITS // 8 + 8, dtype=np.uint64)
        ncodes = np.zeros(1, dtype=np.int64)
        glob = self.hcfg.scope == "global"
        for a in range(0, n, BLOCK_BITS):
            blk = bits[a:a + BLOCK_BITS]
            m = blk.shape[0]
            ncodes[0] = 0
            _fill_keys_k(blk, st, ext[:m], self._kinds, self._dups, text, comm, self.wcfg.row_bits, self._ctag,
                         self._ccls, codes, ncodes, UNK)
            args = (blk, ext[:m], self._wn, self._tags, self._tagoff, self._offsets, self._depths,
                    self._hidx, self._extcol, np.uint64(self.reg), self._shifts,
                    self.cfg.readout == "mixture", float(self.cfg.mature), 1.0 - 2.0 * c.eps, c.gamma,
                    c.g / c.kappa, c.kappa * c.dt, c.consolidating, learning, self.cfg.inherit,
                    float(self.cfg.prior), out[a:a + m], self._stats, float(self.cfg.coherence),
                    self._verify, bool(self.cfg.governor), float(self.cfg.protect),
                    float(self.cfg.sink_factor), self._door, self._dooroff, self._door_shifts,
                    self._occ, self._tsizes, self._tcap, glob, bool(self.hcfg.raise_bar),
                    self._door_on)
            reg = _compiled_loop(*args) if deep is None else _loop_deep(*args, deep[a:a + m])
            self.reg = reg
            st[ST_REG] = np.uint64(reg)
            if learning and self._store is not None:
                self._store.update(codes[:ncodes[0]])
                self._since += m // 8
                if self._since >= self.wcfg.compile_bytes:
                    self._compile()
        return out


def build_kmachine(spec: dict, law, wcfg: WordClassConfig) -> KMachine:
    return KMachine(khier_config(spec, law, wcfg.mode, wcfg.order), wcfg)


__all__ = [
    "K1", "K2", "K4", "K_AFTER", "K_ORDER", "NOMAP", "UNK", "ClassMap", "ClassStore", "KHierConfig", "KMachine",
    "WordClassConfig", "advance_k", "build_kmachine", "class_of", "compile_classes", "context_keys", "control_map",
    "khier_config", "level_key_k", "unit_codes",
]
