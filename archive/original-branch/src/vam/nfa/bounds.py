"""Layer 1 of N: the Bekenstein bound and the machine's register (N-L1, and the size caps of N-L3, N-L5).

N-L1 says the Bekenstein bound gives the largest amount of information one equation can be built
from, "keeping in mind the capabilities of a general CPU register" (64-bit words, 64-byte lines).
This module computes both sides exactly and takes the smaller one.

Physical side
-------------
* Bekenstein (1981): a system of radius ``R`` and total energy ``E`` holds at most
  ``I = 2 pi R E / (hbar c ln 2)`` bits.
* Bekenstein-Hawking: a horizon of area ``A`` holds ``S = A c^3 / (4 G hbar)`` nats
  ``= A / (4 l_P^2 ln 2)`` bits; it is the largest entropy any region of that boundary can hold,
  and it is what layer 6's global governor measures the machine against.

Machine side
------------
The register word (``word_bits``, 64 here), the cache line (64 bytes, so 8 words), the data
caches (L1d, L2, L3) and the float64 mantissa (52 stored bits) are read from ``/sys`` and NumPy.

The binding limit
-----------------
For any physical register the Bekenstein number is astronomically large (a 1-micrometre silicon
sphere: about 1.6e22 bits, :func:`silicon_bekenstein_bits`). So the cap on one equation's label is
``min(Bekenstein, word_bits) = 64`` bits: one register. That is why the nested text machine's
deepest context is 64 bits (``NestedConfig`` refuses deeper ones, and :func:`check_depths` states
the reason). The caches cap the sizes of layers 3 and 5: the largest density matrix that fits in a
cache (:func:`max_density_dim`) and the largest MPO bond whose two-site working tensor fits
(:func:`max_mpo_bond`).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HBAR = 1.054571817e-34        # J s (exact by the 2019 SI definition of h)
C = 299_792_458.0             # m / s (exact)
G = 6.67430e-11               # m^3 / (kg s^2), CODATA 2018
K_B = 1.380649e-23            # J / K (exact)
PLANCK_LENGTH = math.sqrt(HBAR * G / C ** 3)
SILICON_DENSITY = 2329.0      # kg / m^3

CACHE_ROOT = "/sys/devices/system/cpu/cpu0/cache"


def bekenstein_bits(radius_m: float, energy_j: float) -> float:
    """Bekenstein's bound ``2 pi R E / (hbar c ln 2)`` in bits."""
    if radius_m < 0 or energy_j < 0:
        raise ValueError("radius and energy must be nonnegative")
    return 2.0 * math.pi * radius_m * energy_j / (HBAR * C * math.log(2.0))


def bekenstein_hawking_bits(area_m2: float) -> float:
    """Bekenstein-Hawking entropy ``A c^3 / (4 G hbar)`` of a horizon of area ``A``, in bits."""
    if area_m2 < 0:
        raise ValueError("area must be nonnegative")
    return area_m2 * C ** 3 / (4.0 * G * HBAR * math.log(2.0))


def schwarzschild_radius(mass_kg: float) -> float:
    return 2.0 * G * mass_kg / C ** 2


def silicon_bekenstein_bits(radius_m: float = 0.5e-6) -> float:
    """Bekenstein bound of a silicon sphere of the given radius, with its rest energy ``m c^2``.

    The default (1 micrometre across) is larger than a register file entry, so the true bound of
    a register is smaller than this number, but still about 20 orders above 64 bits.
    """
    mass = SILICON_DENSITY * 4.0 / 3.0 * math.pi * radius_m ** 3
    return bekenstein_bits(radius_m, mass * C ** 2)


@dataclass(frozen=True)
class Hardware:
    """The register and caches of the machine the code runs on (bytes unless stated)."""

    word_bits: int = 64
    line_bytes: int = 64
    l1d_bytes: int = 48 * 1024
    l2_bytes: int = 2048 * 1024
    l3_bytes: int = 260 * 1024 * 1024
    mantissa_bits: int = 52
    detected: bool = False

    @staticmethod
    def detect(root: str = CACHE_ROOT) -> Hardware:
        """Read the register width from the C pointer size, the mantissa from NumPy and the caches
        from ``/sys`` (Linux). Missing entries keep the defaults above, and ``detected`` says
        whether the cache entries were read."""
        word = struct.calcsize("P") * 8
        mant = int(np.finfo(np.float64).nmant)
        found: dict[str, int] = {}
        base = Path(root)
        if base.is_dir():
            for idx in sorted(base.glob("index*")):
                try:
                    level = int((idx / "level").read_text().strip())
                    kind = (idx / "type").read_text().strip()
                    size = _parse_size((idx / "size").read_text().strip())
                    line = int((idx / "coherency_line_size").read_text().strip())
                except (OSError, ValueError):
                    continue
                if kind == "Instruction":
                    continue
                found.setdefault("line", line)
                found[{1: "l1d", 2: "l2", 3: "l3"}.get(level, f"l{level}")] = size
        dflt = Hardware()
        return Hardware(word_bits=word, line_bytes=found.get("line", dflt.line_bytes),
                        l1d_bytes=found.get("l1d", dflt.l1d_bytes), l2_bytes=found.get("l2", dflt.l2_bytes),
                        l3_bytes=found.get("l3", dflt.l3_bytes), mantissa_bits=mant,
                        detected={"l1d", "l2"} <= found.keys())

    @property
    def line_bits(self) -> int:
        return 8 * self.line_bytes

    @property
    def line_words(self) -> int:
        """Words in one cache line: the parameters one equation can load in a single memory access."""
        return self.line_bits // self.word_bits

    def cache_bytes(self, level: int) -> int:
        return {1: self.l1d_bytes, 2: self.l2_bytes, 3: self.l3_bytes}[level]


def _parse_size(text: str) -> int:
    text = text.strip().upper()
    for suffix, scale in (("K", 1024), ("M", 1024 ** 2), ("G", 1024 ** 3)):
        if text.endswith(suffix):
            return int(text[:-1]) * scale
    return int(text)


def register_cap(hw: Hardware | None = None, physical_bits: float | None = None) -> int:
    """Bits one equation's label may hold: ``min(Bekenstein bound of the register, word_bits)``."""
    hw = hw or Hardware.detect()
    phys = silicon_bekenstein_bits() if physical_bits is None else physical_bits
    return int(min(hw.word_bits, math.floor(phys)))


def check_depths(depths, hw: Hardware | None = None) -> None:
    """Raise if any context depth exceeds the register cap (N-L1)."""
    cap = register_cap(hw)
    bad = [d for d in depths if d > cap]
    if bad:
        raise ValueError(f"context depths {bad} exceed the register cap of {cap} bits (N-L1)")


def resolution(hw: Hardware | None = None) -> float:
    """Relative resolution of one register-held parameter, ``2**-mantissa_bits``: below it a change
    to a parameter is lost (numerical saturation)."""
    hw = hw or Hardware.detect()
    return 2.0 ** -hw.mantissa_bits


def max_density_dim(hw: Hardware | None = None, level: int = 1, itemsize: int = 16) -> int:
    """Largest ``d`` whose ``d x d`` density matrix (complex128 by default) fits in the cache (N-L3)."""
    hw = hw or Hardware.detect()
    return math.isqrt(hw.cache_bytes(level) // itemsize)


def max_density_qubits(hw: Hardware | None = None, level: int = 1, itemsize: int = 16) -> int:
    """Largest number of qubits whose full density matrix fits in the cache."""
    d = max_density_dim(hw, level, itemsize)
    return math.floor(math.log2(d)) if d >= 1 else 0


def max_mpo_bond(hw: Hardware | None = None, local_dim: int = 4, level: int = 2, itemsize: int = 16) -> int:
    """Largest bond ``D`` whose two-site tensor ``D x local_dim**2 x D`` fits in the cache (N-L5).

    The two-site tensor is the input of every truncating SVD of the MPO engine, its largest
    working object. ``local_dim = 4`` is one qubit of ``vec(rho)``.
    """
    hw = hw or Hardware.detect()
    return math.isqrt(hw.cache_bytes(level) // (itemsize * local_dim * local_dim))


def summary(hw: Hardware | None = None) -> dict:
    """Every number of this module for one machine, for the record."""
    hw = hw or Hardware.detect()
    return {
        "hardware": hw.__dict__ | {"line_words": hw.line_words},
        "silicon_um_sphere_bekenstein_bits": silicon_bekenstein_bits(),
        "register_cap_bits": register_cap(hw),
        "parameter_resolution": resolution(hw),
        "max_density_dim": {lv: max_density_dim(hw, lv) for lv in (1, 2, 3)},
        "max_density_qubits": {lv: max_density_qubits(hw, lv) for lv in (1, 2, 3)},
        "max_mpo_bond": {lv: max_mpo_bond(hw, level=lv) for lv in (1, 2, 3)},
    }
