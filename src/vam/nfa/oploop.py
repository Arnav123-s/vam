"""The compiled reading loop of E6: a copy of ``hier._make_loop`` with the O stage (generated once from ``hier.py``;
``tests/test_operator.py`` checks that removing the E6 lines gives ``hier._make_loop`` back)."""

from __future__ import annotations

import math

import numpy as np

CHUNK = 256


def make_loop_o(jit):
    """A copy of :func:`vam.nfa.hier._make_loop` with the O stage of E6 (PROTOCOL E6.4) after level 0. Lines between
    ``# E6 begin`` and ``# E6 end`` and the extra arguments are the only differences (a test compares the two sources);
    with ``o_on`` false it computes exactly what the E4 loop computes."""

    def loop(bits, ext, wn, tags, tagoff, offsets, depths, hidx, extcol, reg, shifts, mixture, mature, scale, gamma,
             gk, kdt, consolidating, learning, inherit, prior, out, stats, coherence, verify, governor, protect,
             sink_factor, door, dooroff, door_shifts, occ, tsizes, tcap, glob, raise_bar, door_on,
             extp, extu, ev_o, acc, o_on, o_top):  # E6 signature
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
                    if lv == 0 and o_on and o_top == 0:  # E6 begin: the O stage after level 0 (PROTOCOL E6.4)
                        po = extp[i]
                        if po == po:
                            un = extu[i]
                            evo = ev_o[un]
                            if evo >= 0.0:
                                shr = 1.0 / (1.0 + 2.0 ** -evo)
                            else:
                                ee = 2.0 ** evo
                                shr = ee / (1.0 + ee)
                            p_in = p0
                            if coherence == 0.0:
                                p0 = p_in + shr * (po - p_in)
                            else:
                                rr = 1.0 - shr
                                a0 = math.sqrt(rr * p_in) + math.sqrt(shr * po)
                                a1 = math.sqrt(rr * (1.0 - p_in)) + math.sqrt(shr * (1.0 - po))
                                t0 = (1.0 - coherence) * (rr * p_in + shr * po) + coherence * a0 * a0
                                t1 = (1.0 - coherence) * (rr * (1.0 - p_in) + shr * (1.0 - po)) + coherence * a1 * a1
                                p0 = t0 / (t0 + t1)
                            if learning:
                                qi = p_in if b == 0 else 1.0 - p_in
                                qo = po if b == 0 else 1.0 - po
                                qs = p0 if b == 0 else 1.0 - p0
                                ev_o[un] += math.log2(qo) - math.log2(qi)
                                acc[un, 0] += -math.log2(qi)
                                acc[un, 1] += -math.log2(qo)
                                acc[un, 2] += -math.log2(qs)
                    # E6 end
                    chain[lv] = p0
                if o_on and o_top == 1:  # E6 begin: the same stage after the top level (E7)
                    po = extp[i]
                    if po == po:
                        un = extu[i]
                        evo = ev_o[un]
                        if evo >= 0.0:
                            shr = 1.0 / (1.0 + 2.0 ** -evo)
                        else:
                            ee = 2.0 ** evo
                            shr = ee / (1.0 + ee)
                        p_in = p0
                        if coherence == 0.0:
                            p0 = p_in + shr * (po - p_in)
                        else:
                            rr = 1.0 - shr
                            a0 = math.sqrt(rr * p_in) + math.sqrt(shr * po)
                            a1 = math.sqrt(rr * (1.0 - p_in)) + math.sqrt(shr * (1.0 - po))
                            t0 = (1.0 - coherence) * (rr * p_in + shr * po) + coherence * a0 * a0
                            t1 = (1.0 - coherence) * (rr * (1.0 - p_in) + shr * (1.0 - po)) + coherence * a1 * a1
                            p0 = t0 / (t0 + t1)
                        if learning:
                            qi = p_in if b == 0 else 1.0 - p_in
                            qo = po if b == 0 else 1.0 - po
                            qs = p0 if b == 0 else 1.0 - p0
                            ev_o[un] += math.log2(qo) - math.log2(qi)
                            acc[un, 0] += -math.log2(qi)
                            acc[un, 1] += -math.log2(qo)
                            acc[un, 2] += -math.log2(qs)
                # E6 end
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
