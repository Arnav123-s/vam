"""N above one equation, on text (E4; DESIGN section 11, decision D11, a chosen realization).

The nested machine reads one 64-bit register: 8 bytes of context. E4 adds levels whose contexts come from registers
of registers, all present state, all updated by the stream alone:

* **units** close at content-defined boundaries (a fixed function of the last 3 bytes; the text rule, a tokenizer, is
  one labelled alternative); a closed unit is one 64-bit code; the **W ring** holds the last 4 codes;
* **sentences** fold unit codes and close at content-defined boundaries; the **S state** holds the running sentence
  code and the last 2 closed ones.

The five upper levels ``W1, W2, W4, S0, S1`` join the evidence mixture of :class:`vam.nested.NestedMachine` after its
byte levels (the order is declared, never selected). Their keys are functions of the stream alone, so they are computed
in a pre-pass (:func:`fill_keys`) per block; the compiled loop then uses them as it uses ``reg & mask`` for a
byte level.
The pre-pass and the pure Python :func:`advance` are two writings of one state machine (tested equal).

Options, all off by default: ``upper=()`` is exactly the nested machine (bit for bit, tested).
``scope="global"`` makes the governor one (Page point at half of the total slot capacity).
``raise_bar`` raises the admission bar of an unverified context by one past the Page point (the halt, DESIGN 11.3).
``control`` replaces the upper levels by an equal-cost control: ``"dup"`` duplicates byte levels with new salts (no
extra span); ``"comm"`` destroys order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..nested import (
    _C1,
    _C2,
    EV,
    GOLDEN,
    MASK64,
    NUMBA_AVAILABLE,
    NestedConfig,
    NestedMachine,
    W,
    _p0,
    _share,
    coherent_mix,
)
from ..nested import N as NCOL

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

#: upper levels, in mixture order: W1, W2, W4 (ring depths 1, 2, 4), S0 (the sentence so far), S1 (and the last one)
W1, W2, W4, S0, S1 = range(5)
LEVEL_NAMES = ("W1", "W2", "W4", "S0", "S1")
ALL_UPPER = (W1, W2, W4, S0, S1)
DUP_DEPTHS = (24, 32, 48, 64, 64)           # the control "dup": byte levels the upper levels duplicate

SEED_C, SEED_S = 0x243F6A8885A308D3, 0x13198A2E03707344     # fixed seeds of a unit code and a sentence code
SALT_B, SALT_S = 0xA4093822299F31D0, 0x082EFA98EC4E6C89     # boundary tests
SALT_L = (0x452821E638D01377, 0xBE5466CF34E90C6C, 0xC0AC29B7C97C50DD, 0x3F84D5B5B5470917, 0x9216D5D98979FB1B)
K_C, K_1, K_2, K_3 = 0x2FFD72DBD01ADFB7, 0xB8E1AFED6A267E96, 0xBA7C9045F12C7F99, 0x24A19947B3916CF7
DOOR_SALT = 0xD1B54A32D192ED03
SALT_L_ARR = np.array(SALT_L, dtype=np.uint64)
MOD_UNIT, MOD_SENT = 6, 16                  # a unit closes about every 6 bytes, a sentence about every 16 units
ARITY = 4                                   # unit codes in the W ring (N-L1: at most line_words = 8)
BLOCK_BITS = 1 << 20                        # bits per pre-pass block
# state layout: reg, u, c, ring0..3, s, p1, p2, last3 (last 3 bytes), last2 (last 2 bytes, for the text rule)
ST_REG, ST_U, ST_C, ST_R0, ST_S, ST_P1, ST_P2, ST_L3 = 0, 1, 2, 3, 7, 8, 9, 10
ST_SIZE = 11


def mix64(x: int) -> int:
    x &= MASK64
    x = ((x ^ (x >> 30)) * _C1) & MASK64
    x = ((x ^ (x >> 27)) * _C2) & MASK64
    return x ^ (x >> 31)


def initial_state() -> list[int]:
    st = [0] * ST_SIZE
    st[ST_U], st[ST_C], st[ST_S] = 1, SEED_C, SEED_S
    return st


def is_word_byte(byte: int) -> bool:
    return 48 <= byte <= 57 or 65 <= byte <= 90 or 97 <= byte <= 122 or byte == 39


def _closes_unit(byte: int, last3: int, text: bool) -> bool:
    if text:
        return not is_word_byte(byte)
    return mix64(last3 ^ SALT_B) % MOD_UNIT == 0


def _closes_sentence(code: int, last3: int, text: bool) -> bool:
    if text:                                    # . ? ! then a space or newline: the closing byte is a space or newline
        prev = (last3 >> 8) & 255
        return (last3 & 255) in (32, 10) and prev in (46, 63, 33)
    return mix64(code ^ SALT_S) % MOD_SENT == 0


def advance(st: list[int], bit: int, text: bool = False, comm: bool = False) -> list[int]:
    """One bit through the registers (the pure Python writing of the pre-pass)."""
    st = list(st)
    st[ST_REG] = ((st[ST_REG] << 1) | bit) & MASK64
    u = (st[ST_U] << 1) | bit
    if u < 256:
        st[ST_U] = u
        return st
    byte = u & 255
    st[ST_U] = 1
    st[ST_C] = mix64(st[ST_C] ^ (((byte + 1) * GOLDEN) & MASK64))
    st[ST_L3] = ((st[ST_L3] << 8) | byte) & 0xFFFFFF
    if _closes_unit(byte, st[ST_L3], text):
        code = st[ST_C]
        st[ST_R0 + 3], st[ST_R0 + 2], st[ST_R0 + 1], st[ST_R0] = st[ST_R0 + 2], st[ST_R0 + 1], st[ST_R0], code
        st[ST_S] = (st[ST_S] + code) & MASK64 if comm else mix64(st[ST_S] ^ code)
        st[ST_C] = SEED_C
        if _closes_sentence(code, st[ST_L3], text):
            st[ST_P2], st[ST_P1] = st[ST_P1], st[ST_S]
            st[ST_S] = SEED_S
    return st


def advance_byte(st: list[int], byte: int, text: bool = False, comm: bool = False) -> list[int]:
    """Eight bits of ``byte`` (most significant first) through the registers, from a byte boundary."""
    for k in range(7, -1, -1):
        st = advance(st, (byte >> k) & 1, text, comm)
    return st


def level_key(st: list[int], level: int, comm: bool = False, dup_depth: int = 0) -> int:
    """The key of an upper level for the bit about to be read (``u`` is the byte read so far)."""
    u, c = st[ST_U], st[ST_C]
    r = st[ST_R0:ST_R0 + 4]
    if dup_depth:
        reg = st[ST_REG]
        ctx = reg if dup_depth >= 64 else reg & ((1 << dup_depth) - 1)
        return mix64(ctx ^ SALT_L[level])
    if level in (W1, W2, W4):
        n = (1, 2, 4)[level]
        if comm:
            h = sum(r[:n]) & MASK64
        else:
            h = SEED_C
            for code in r[:n]:
                h = mix64(h ^ code)
    elif level == S0:
        h = st[ST_S]
    else:                                       # S1
        h = (st[ST_S] + st[ST_P1]) & MASK64 if comm else mix64(st[ST_S] ^ mix64((st[ST_P1] + K_3) & MASK64))
    return mix64(h ^ mix64((c + K_C) & MASK64) ^ ((u * GOLDEN) & MASK64) ^ SALT_L[level])


def _make_prepass(jit):
    def fill_keys(bits, st, ext, kinds, dups, text, comm):
        """Fill ``ext[i, j]`` with the key of upper level ``j`` for bit ``i`` and advance ``st`` through the block."""
        one = np.uint64(1)
        m30, m27, m31 = np.uint64(30), np.uint64(27), np.uint64(31)
        c1, c2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)
        golden = np.uint64(0x9E3779B97F4A7C15)
        for i in range(bits.shape[0]):
            u = st[1]
            c = st[2]
            for j in range(kinds.shape[0]):
                lvl = kinds[j]
                if dups[j] > 0:
                    d = dups[j]
                    ctx = st[0] if d >= 64 else st[0] & ((one << np.uint64(d)) - one)
                    h = ctx ^ np.uint64(SALT_L_ARR[lvl])
                    h = (h ^ (h >> m30)) * c1
                    h = (h ^ (h >> m27)) * c2
                    ext[i, j] = h ^ (h >> m31)
                    continue
                if lvl <= 2:
                    n = lvl + 1 if lvl < 2 else 4
                    if comm:
                        h = np.uint64(0)
                        for k in range(n):
                            h += st[3 + k]
                    else:
                        h = np.uint64(0x243F6A8885A308D3)
                        for k in range(n):
                            h = h ^ st[3 + k]
                            h = (h ^ (h >> m30)) * c1
                            h = (h ^ (h >> m27)) * c2
                            h = h ^ (h >> m31)
                elif lvl == 3:
                    h = st[7]
                else:
                    if comm:
                        h = st[7] + st[8]
                    else:
                        g = st[8] + np.uint64(0x24A19947B3916CF7)
                        g = (g ^ (g >> m30)) * c1
                        g = (g ^ (g >> m27)) * c2
                        g = g ^ (g >> m31)
                        h = st[7] ^ g
                        h = (h ^ (h >> m30)) * c1
                        h = (h ^ (h >> m27)) * c2
                        h = h ^ (h >> m31)
                q = c + np.uint64(0x2FFD72DBD01ADFB7)
                q = (q ^ (q >> m30)) * c1
                q = (q ^ (q >> m27)) * c2
                q = q ^ (q >> m31)
                x = h ^ q ^ (u * golden) ^ np.uint64(SALT_L_ARR[lvl])
                x = (x ^ (x >> m30)) * c1
                x = (x ^ (x >> m27)) * c2
                ext[i, j] = x ^ (x >> m31)
            # advance
            b = np.uint64(bits[i])
            st[0] = (st[0] << one) | b
            u = (u << one) | b
            if u < np.uint64(256):
                st[1] = u
                continue
            byte = u & np.uint64(255)
            st[1] = one
            cc = c ^ ((byte + one) * golden)
            cc = (cc ^ (cc >> m30)) * c1
            cc = (cc ^ (cc >> m27)) * c2
            cc = cc ^ (cc >> m31)
            st[10] = ((st[10] << np.uint64(8)) | byte) & np.uint64(0xFFFFFF)
            if text:
                bb = np.int64(byte)
                closes = not ((48 <= bb <= 57) or (65 <= bb <= 90) or (97 <= bb <= 122) or bb == 39)
            else:
                z = st[10] ^ np.uint64(0xA4093822299F31D0)
                z = (z ^ (z >> m30)) * c1
                z = (z ^ (z >> m27)) * c2
                z = z ^ (z >> m31)
                closes = z % np.uint64(6) == np.uint64(0)
            if not closes:
                st[2] = cc
                continue
            st[6] = st[5]
            st[5] = st[4]
            st[4] = st[3]
            st[3] = cc
            if comm:
                st[7] = st[7] + cc
            else:
                z = st[7] ^ cc
                z = (z ^ (z >> m30)) * c1
                z = (z ^ (z >> m27)) * c2
                st[7] = z ^ (z >> m31)
            st[2] = np.uint64(0x243F6A8885A308D3)
            if text:
                prev = (st[10] >> np.uint64(8)) & np.uint64(255)
                last = st[10] & np.uint64(255)
                sc = (last == np.uint64(32) or last == np.uint64(10)) and (
                    prev == np.uint64(46) or prev == np.uint64(63) or prev == np.uint64(33))
            else:
                z = cc ^ np.uint64(0x082EFA98EC4E6C89)
                z = (z ^ (z >> m30)) * c1
                z = (z ^ (z >> m27)) * c2
                z = z ^ (z >> m31)
                sc = z % np.uint64(16) == np.uint64(0)
            if sc:
                st[9] = st[8]
                st[8] = st[7]
                st[7] = np.uint64(0x13198A2E03707344)
    return jit(fill_keys) if jit is not None else fill_keys


_fill_keys = _make_prepass(njit(cache=True)) if NUMBA_AVAILABLE else None

CHUNK = 256


def _make_loop(jit):
    """:func:`vam.nested._make_loop` with per-level sources, sizes and verify, and the global governor. The first pass
    of a block finds and touches every row; the second runs the events in stream order."""

    def loop(bits, ext, wn, tags, tagoff, offsets, depths, hidx, extcol, reg, shifts, mixture, mature, scale, gamma,
             gk, kdt, consolidating, learning, inherit, prior, out, stats, coherence, verify, governor, protect,
             sink_factor, door, dooroff, door_shifts, occ, tsizes, tcap, glob, raise_bar, door_on):
        L = depths.shape[0]
        nh = tsizes.shape[0]
        size = bits.shape[0]
        blk_rows = np.empty((CHUNK, L), dtype=np.int64)
        blk_tags = np.empty((CHUNK, L), dtype=np.uint32)
        blk_door = np.empty((CHUNK, L), dtype=np.int64)
        blk_dtag = np.empty((CHUNK, L), dtype=np.int64)
        fresh = np.empty(L, dtype=np.bool_)
        chain = np.empty(L)
        one = np.uint64(1)
        sink = 0.0
        for start in range(0, size, CHUNK):
            stop = min(size, start + CHUNK)
            r = reg
            for i in range(start, stop):
                for lv in range(L):
                    h = hidx[lv]
                    if extcol[lv] >= 0:
                        x = ext[i, extcol[lv]]
                    else:
                        d = depths[lv]
                        ctx = r if d >= 64 else r & ((one << np.uint64(d)) - one)
                        if h < 0:
                            x = ctx
                        else:
                            x = ctx ^ (np.uint64(lv + 1) * np.uint64(0x9E3779B97F4A7C15))
                            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
                            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
                            x = x ^ (x >> np.uint64(31))
                    if h < 0:
                        s = np.int64(x)
                        t = np.uint32(0)
                    else:
                        s = np.int64(x >> shifts[h])
                        t = np.uint32((x & np.uint64(0xFFFFFFFF)) | one)
                        sink += tags[tagoff[h] + s]
                        if door_on:
                            y = x ^ np.uint64(0xD1B54A32D192ED03)
                            y = (y ^ (y >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
                            y = (y ^ (y >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
                            y = y ^ (y >> np.uint64(31))
                            blk_door[i - start, lv] = np.int64(y >> door_shifts[h])
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
                        is_new = tags[tagoff[h] + row - offsets[lv]] != blk_tags[i - start, lv]
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
                                occupied = tags[tagoff[h] + slot] != 0
                                if door_on:
                                    if glob:
                                        governed = governor and 2 * occ[nh] >= tcap
                                    else:
                                        governed = governor and 2 * occ[h] >= tsizes[h]
                                    need = verify[lv]
                                    if governed:
                                        if raise_bar:
                                            need += 1
                                        elif need < 2:
                                            need = 2
                                    if need > 1:
                                        dd = blk_door[i - start, lv]
                                        dtag = blk_dtag[i - start, lv]
                                        d0 = dooroff[h] + (dd & ~np.int64(1))
                                        d1 = d0 | np.int64(1)
                                        e0 = np.int64(door[d0])
                                        e1 = np.int64(door[d1])
                                        if (e0 >> 16) == dtag:
                                            idx = d0
                                            cnt = (e0 & 0xFFFF) + 1
                                        elif (e1 >> 16) == dtag:
                                            idx = d1
                                            cnt = (e1 & 0xFFFF) + 1
                                        else:
                                            idx = d0 if (e0 & 0xFFFF) <= (e1 & 0xFFFF) else d1
                                            cnt = 1
                                        if cnt > 255:
                                            cnt = 255
                                        if cnt >= need:
                                            door[idx] = 0
                                        else:
                                            door[idx] = np.uint32((dtag << 16) | cnt)
                                        if cnt < need:
                                            stats[3] += 1
                                            continue
                                    if governed and occupied and wn[row, 2] > protect:
                                        wn[row, 2] *= sink_factor
                                        stats[4] += 1
                                        continue
                                if occupied:
                                    stats[0] += 1
                                    if stats[1] < 0:
                                        stats[1] = stats[2] + i
                                else:
                                    occ[h] += 1
                                    occ[nh] += 1
                                tags[tagoff[h] + slot] = blk_tags[i - start, lv]
                            if inherit and lv > 0:
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


@dataclass(frozen=True)
class HierConfig:
    base: NestedConfig                       # the byte levels and every option of the nested machine (E3's chosen row)
    upper: tuple[int, ...] = ALL_UPPER       # which upper levels; () is exactly the nested machine
    upper_bits: int = 21                     # slots of an upper level: 2**upper_bits (DESIGN 11.4)
    upper_verify: int = 0                    # sightings an upper level needs (0: the base's verify)
    segment: str = "content"                 # "content" (modality-free) or "text" (a tokenizer; labelled)
    scope: str = "level"                     # the governor: "level" (E3's) or "global"
    raise_bar: bool = False                  # past the Page point an unverified context needs one sighting more
    control: str = "none"                    # "none", "dup" (equal cost, no extra span), "comm" (order destroyed)

    def __post_init__(self) -> None:
        if any(k not in ALL_UPPER for k in self.upper) or len(set(self.upper)) != len(self.upper):
            raise ValueError("upper levels are a subset of W1, W2, W4, S0, S1")
        if tuple(sorted(self.upper)) != self.upper:
            raise ValueError("upper levels keep their declared order")
        if self.segment not in ("content", "text") or self.scope not in ("level", "global"):
            raise ValueError("segment is 'content' or 'text'; scope is 'level' or 'global'")
        if self.control not in ("none", "dup", "comm"):
            raise ValueError("control is 'none', 'dup' or 'comm'")
        if not 4 <= self.upper_bits <= 26:
            raise ValueError("upper_bits lies in 4..26")
        if ARITY > 8:  # pragma: no cover - N-L1: a cache line holds 8 registers
            raise ValueError("arity exceeds the cache line")
        if self.upper_verify and not 1 <= self.upper_verify <= 255:
            raise ValueError("verify counts sightings, 1..255")

    @property
    def verify_upper(self) -> int:
        return self.upper_verify or self.base.verify


class HierMachine(NestedMachine):
    """The nested machine with upper levels (E4). With ``upper=()`` it equals :class:`NestedMachine` bit for bit."""

    def __init__(self, hcfg: HierConfig) -> None:
        if not NUMBA_AVAILABLE:
            raise RuntimeError("HierMachine needs numba")
        cfg = hcfg.base
        self.hcfg = hcfg
        self.cfg = cfg
        Lb, Lu = cfg.levels, len(hcfg.upper)
        self._Lb, self._Lu = Lb, Lu
        sizes = [cfg.size(lv) for lv in range(Lb)] + [1 << hcfg.upper_bits] * Lu
        self._sizes = np.array(sizes, dtype=np.int64)
        self._offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        self._wn = np.zeros((int(sum(sizes)), 3))
        hashed = [lv for lv in range(Lb) if not cfg.direct(lv)] + list(range(Lb, Lb + Lu))
        self._hashed = hashed
        self._hidx = np.full(Lb + Lu, -1, dtype=np.int64)
        self._hidx[hashed] = np.arange(len(hashed))
        tbits = [cfg.table_bits if lv < Lb else hcfg.upper_bits for lv in hashed]
        tsz = [1 << b for b in tbits]
        self._tsizes = np.array(tsz if tsz else [1], dtype=np.int64)
        self._tagoff = np.concatenate([[0], np.cumsum(tsz)[:-1]]).astype(np.int64) if tsz else np.zeros(1, np.int64)
        self._tags = np.zeros(max(1, int(sum(tsz))), dtype=np.uint32)
        self._shifts = np.array([64 - b for b in tbits] if tbits else [63], dtype=np.uint64)
        self._tcap = int(sum(tsz))
        dbits = [(cfg.door_bits or b + 1) if lv < Lb else b + 1 for lv, b in zip(hashed, tbits, strict=True)]
        self._depths = np.array([*cfg.depths, *([0] * Lu)], dtype=np.int64)
        self._extcol = np.array([-1] * Lb + list(range(Lu)), dtype=np.int64)
        self._verify = np.array([cfg.verify] * Lb + [hcfg.verify_upper] * Lu, dtype=np.int64)
        self._door_on = bool(max(self._verify) > 1 or cfg.governor)
        dsz = [1 << b for b in dbits]
        self._dooroff = np.concatenate([[0], np.cumsum(dsz)[:-1]]).astype(np.int64) if dsz else np.zeros(1, np.int64)
        self._door = np.zeros(max(1, int(sum(dsz))) if (self._door_on and hashed) else 1, dtype=np.uint32)
        self._door_shifts = np.array([64 - b for b in dbits] if dbits else [63], dtype=np.uint64)
        self._occ = np.zeros(max(1, len(hashed)) + 1, dtype=np.int64)    # per hashed level, then the total
        self._stats = np.array([0, -1, 0, 0, 0], dtype=np.int64)
        self._kinds = np.array(hcfg.upper, dtype=np.int64)
        self._dups = np.array([DUP_DEPTHS[k] if hcfg.control == "dup" else 0 for k in hcfg.upper], dtype=np.int64)
        self._st = np.array(initial_state(), dtype=np.uint64)
        self.reg = np.uint64(0)

    # -- state ------------------------------------------------------------------------
    @property
    def state(self) -> list[int]:
        st = [int(x) for x in self._st]
        st[ST_REG] = int(self.reg)
        return st

    def set_state(self, st: list[int]) -> None:
        self._st = np.array(st, dtype=np.uint64)
        self.reg = np.uint64(st[ST_REG])

    def state_bytes(self) -> int:
        return int(self._st.nbytes)

    def copy(self) -> HierMachine:
        other = HierMachine.__new__(HierMachine)
        other.__dict__.update(self.__dict__)
        for name in ("_wn", "_tags", "_stats", "_door", "_occ", "_st"):
            setattr(other, name, getattr(self, name).copy())
        return other

    def nbytes(self) -> int:
        return 8 + int(self._wn.nbytes + self._tags.nbytes + self._door.nbytes + self._occ.nbytes + self._st.nbytes)

    # -- measures -----------------------------------------------------------------------
    @property
    def levels(self) -> int:
        return self._Lb + self._Lu

    def occupancy(self) -> dict[int, float]:
        return {lv: float(self._occ[self._hidx[lv]]) / float(self._tsizes[self._hidx[lv]])
                for lv in range(self.levels) if self._hidx[lv] >= 0}

    def global_occupancy(self) -> float:
        return float(self._occ[-1]) / float(self._tcap) if self._tcap else 0.0

    def level_table(self, level: int) -> np.ndarray:
        o = self._offsets[level]
        return self._wn[o:o + self._sizes[level]]

    def stored_evidence(self) -> dict[int, float]:
        out = {}
        for lv in range(1, self.levels):
            ev = self.level_table(lv)[:, EV]
            out[lv] = float(ev[ev > 0].sum())
        return out

    # -- reading ---------------------------------------------------------------------------
    def run(self, bits: np.ndarray, learning: bool = True) -> np.ndarray:
        bits = np.ascontiguousarray(bits, dtype=np.uint8)
        c = self.cfg.law
        n = bits.shape[0]
        out = np.empty(n)
        text, comm = self.hcfg.segment == "text", self.hcfg.control == "comm"
        st = self._st
        st[ST_REG] = np.uint64(self.reg)
        ext = np.empty((min(n, BLOCK_BITS), max(1, self._Lu)), dtype=np.uint64)
        glob = self.hcfg.scope == "global"
        for a in range(0, n, BLOCK_BITS):
            blk = bits[a:a + BLOCK_BITS]
            m = blk.shape[0]
            if self._Lu:
                _fill_keys(blk, st, ext[:m], self._kinds, self._dups, text, comm)
            else:
                st[ST_REG] = np.uint64(self.reg)
            reg = _compiled_loop(blk, ext[:m], self._wn, self._tags, self._tagoff, self._offsets, self._depths,
                                 self._hidx, self._extcol, np.uint64(self.reg), self._shifts,
                                 self.cfg.readout == "mixture", float(self.cfg.mature), 1.0 - 2.0 * c.eps, c.gamma,
                                 c.g / c.kappa, c.kappa * c.dt, c.consolidating, learning, self.cfg.inherit,
                                 float(self.cfg.prior), out[a:a + m], self._stats, float(self.cfg.coherence),
                                 self._verify, bool(self.cfg.governor), float(self.cfg.protect),
                                 float(self.cfg.sink_factor), self._door, self._dooroff, self._door_shifts,
                                 self._occ, self._tsizes, self._tcap, glob, bool(self.hcfg.raise_bar),
                                 self._door_on)
            self.reg = reg
            st[ST_REG] = np.uint64(reg)
        return out

    # -- frozen readout, for planning ---------------------------------------------------------
    def _keys(self, st: list[int]) -> list[int]:
        comm = self.hcfg.control == "comm"
        return [level_key(st, k, comm, DUP_DEPTHS[k] if self.hcfg.control == "dup" else 0) for k in self.hcfg.upper]

    def p0_state(self, st: list[int]) -> float:
        cfg = self.cfg
        reg = st[ST_REG]
        keys = self._keys(st)
        p0 = 0.0
        for lv in range(self.levels):
            h = int(self._hidx[lv])
            if lv < self._Lb:
                ctx = reg & ((1 << cfg.depths[lv]) - 1)
                if h < 0:
                    row = self._wn[self._offsets[lv] + ctx]
                    present = row[NCOL] > 0.0
                else:
                    x = mix64(ctx ^ (((lv + 1) * GOLDEN) & MASK64))
            else:
                x = keys[lv - self._Lb]
            if h >= 0:
                slot = x >> int(self._shifts[h])
                row = self._wn[self._offsets[lv] + slot]
                present = int(self._tags[self._tagoff[h] + slot]) == ((x & 0xFFFFFFFF) | 1)
            if lv == 0 or (present and row[NCOL] >= cfg.mature):
                own = _p0(cfg.law, float(row[W]))
                p0 = own if lv == 0 or cfg.readout == "mature" else coherent_mix(p0, own, _share(float(row[EV])),
                                                                                  cfg.coherence)
        return p0

    def _p0_at(self, st) -> float:                               # the planner's hook
        return self.p0_state(st)

    def predict(self, b: int) -> float:
        p0 = self.p0_state(self.state)
        return p0 if b == 0 else 1.0 - p0

    def advance(self, st: list[int], bit: int) -> list[int]:
        return advance(st, bit, self.hcfg.segment == "text", self.hcfg.control == "comm")

    def continuations(self, horizon: int) -> dict[tuple, float]:
        from ..linalg import ATOL
        cut = max(1e-10, ATOL)
        dist: dict[tuple, float] = {}

        def walk(st: list[int], outs: tuple, prob: float) -> None:
            if len(outs) == horizon:
                dist[outs] = dist.get(outs, 0.0) + prob
                return
            p0 = self.p0_state(st)
            for t, pt in ((0, p0), (1, 1.0 - p0)):
                if pt > cut:
                    walk(self.advance(st, t), (*outs, t), prob * pt)

        walk(self.state, (), 1.0)
        return dist

    def generate(self, n_bits: int, horizon: int = 8, select: str = "max",
                 rng: np.random.Generator | None = None) -> list[int]:
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
            self.set_state(self.advance(self.state, int(macro[0])))
            emitted.append(int(macro[0]))
        return emitted


def label_entropy(share: float, c: float) -> float:
    """Entropy (bits) of the hypothesis-label state ``(1 - c) diag(1 - s, s) + c |a><a|`` of one cut: at most 1 bit
    (bond 2, so the area law is vacuous on text, DESIGN 11.3)."""
    a = np.array([math.sqrt(1.0 - share), math.sqrt(share)])
    rho = (1.0 - c) * np.diag([1.0 - share, share]) + c * np.outer(a, a)
    ev = np.clip(np.linalg.eigvalsh(rho), 0.0, 1.0)
    ev = ev[ev > 1e-300]
    return float(-(ev * np.log2(ev)).sum())


def cut_entropies(m: HierMachine, st: list[int]) -> list[float]:
    """The label entropy of every cut whose level holds a sector at state ``st``."""
    cfg = m.cfg
    reg, keys, out = st[ST_REG], m._keys(st), []
    for lv in range(1, m.levels):
        h = int(m._hidx[lv])
        if lv < m._Lb:
            ctx = reg & ((1 << cfg.depths[lv]) - 1)
            x = ctx if h < 0 else mix64(ctx ^ (((lv + 1) * GOLDEN) & MASK64))
        else:
            x = keys[lv - m._Lb]
        if h < 0:
            row, present = m._wn[m._offsets[lv] + x], m._wn[m._offsets[lv] + x][NCOL] > 0.0
        else:
            slot = x >> int(m._shifts[h])
            row = m._wn[m._offsets[lv] + slot]
            present = int(m._tags[m._tagoff[h] + slot]) == ((x & 0xFFFFFFFF) | 1)
        if present and row[NCOL] >= cfg.mature:
            out.append(label_entropy(_share(float(row[EV])), cfg.coherence))
    return out


def build(spec: dict, law) -> HierMachine:
    """The machine of an E4 spec: the E3 keys of ``spec`` for the base, plus the E4 keys (``hier_*``)."""
    from ..nested import NestedConfig as NC
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
    return HierMachine(HierConfig(base, **kw))
