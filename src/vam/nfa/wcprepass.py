"""The pre-pass of E6b: a copy of ``hier._make_prepass`` with the class ring and the class levels K1, K2, K4.

Generated once from ``hier.py``; ``tests/test_wordclass.py`` checks that removing the E6b lines gives
``hier._make_prepass`` back. The class ring lives in ``st[11:15]``; the unit codes of a block are written to ``codes``
for the count store."""

from __future__ import annotations

import numpy as np

from .hier import SALT_L, SALT_L_ARR

#: salts of the levels: W1 to S1 as in hier, then K1, K2, K4
SALT_K = (*SALT_L, 0x7F4A7C15B0D6C3A9, 0x1B873593CC9E2D51, 0xE6546B64AC4C9B0D)
SALT_K_ARR = np.array(SALT_K, dtype=np.uint64)
assert (SALT_K_ARR[:5] == SALT_L_ARR).all()


def make_prepass_k(jit):
    def fill_keys(bits, st, ext, kinds, dups, text, comm,
                  rbits, ctag, ccls, codes, ncodes, cunk):  # E6b signature
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
                if lvl >= 5:  # E6b begin: a class level (K1, K2, K4)
                    nk = 1 if lvl == 5 else (2 if lvl == 6 else 4)
                    h = np.uint64(0x243F6A8885A308D3)
                    for k in range(nk):
                        h = h ^ (st[11 + k] + np.uint64(0x5851F42D4C957F2D))
                        h = (h ^ (h >> m30)) * c1
                        h = (h ^ (h >> m27)) * c2
                        h = h ^ (h >> m31)
                    q = c + np.uint64(0x2FFD72DBD01ADFB7)
                    q = (q ^ (q >> m30)) * c1
                    q = (q ^ (q >> m27)) * c2
                    q = q ^ (q >> m31)
                    x = h ^ q ^ (u * golden) ^ np.uint64(SALT_K_ARR[lvl])
                    x = (x ^ (x >> m30)) * c1
                    x = (x ^ (x >> m27)) * c2
                    ext[i, j] = x ^ (x >> m31)
                    continue
                # E6b end
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
            # E6b begin: the class ring and the unit code stream
            st[14] = st[13]
            st[13] = st[12]
            st[12] = st[11]
            rw = np.int64(cc >> np.uint64(64 - rbits))
            if ctag[rw] == cc and ccls[rw] < 256:
                st[11] = np.uint64(ccls[rw])
            else:
                st[11] = np.uint64(cunk)
            codes[ncodes[0]] = cc
            ncodes[0] += 1
            # E6b end
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
