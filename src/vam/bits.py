"""The universal bit substrate (F-Part 10, T2.1).

Source statement: the lowest layer is pure bits; text, images and sound all
arrive as bits through the same interaction algebra, with no tokenizer,
patcher or spectrogram encoder. This module only turns bytes into a bit
stream (most significant bit first) and back; it never segments, labels or
buffers a stream beyond the chunk being read.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

SENSORY = "sensory"
ACTIVE = "active"


def bytes_to_bits(data: bytes) -> np.ndarray:
    """MSB-first bits of ``data`` as a ``uint8`` array of 0/1."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: Iterable[int]) -> bytes:
    arr = np.asarray(list(bits), dtype=np.uint8)
    if arr.size % 8:
        raise ValueError("bit count is not a multiple of 8")
    return np.packbits(arr).tobytes()


def iter_bits(data: bytes | bytearray | memoryview) -> Iterator[int]:
    for b in bytes_to_bits(bytes(data)):
        yield int(b)


def iter_file_bits(path: str | Path, chunk_bytes: int = 1 << 16) -> Iterator[int]:
    """Stream a file's bits without holding the file in memory."""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                return
            for b in bytes_to_bits(chunk):
                yield int(b)
