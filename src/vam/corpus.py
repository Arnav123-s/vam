"""Real-text corpora for experiment E2 (experiments/PROTOCOL.md, E2).

The machine reads raw bytes as bits (``vam.bits``), so this module never
tokenizes or segments what the machine sees. It only fixes, before any run:

* cleaning: UTF-8, line endings made ``\\n``, Project Gutenberg's licence
  header and footer removed (the book text itself is untouched);
* splits: by position inside each book (train, validation, test), cut at a
  line end; held-out books are never trained on;
* controls: the same bytes rearranged (sentences shuffled within a book, or
  words shuffled within a sentence). A rearrangement keeps every byte, so the
  control has exactly the same length and character counts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_START = re.compile(r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*$", re.MULTILINE | re.IGNORECASE)
_END = re.compile(r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*$", re.MULTILINE | re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])(\s+)")
_SPACE = re.compile(r"(\s+)")

SPLIT = (0.8, 0.1, 0.1)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_text(raw: bytes) -> bytes:
    """Book text between the Gutenberg markers (when present), as UTF-8 with ``\\n`` line ends."""
    text = raw.decode("utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start, end = _START.search(text), _END.search(text)
    if start and end and start.end() < end.start():
        text = text[start.end():end.start()]
    return text.strip("\n").encode("utf-8") + b"\n"


def split_book(text: bytes, fractions: tuple[float, float, float] = SPLIT) -> tuple[bytes, bytes, bytes]:
    """Train, validation and test parts by position; each cut moves forward to the next line end."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to one")

    def cut(frac: float) -> int:
        i = int(len(text) * frac)
        j = text.find(b"\n", i)
        return len(text) if j < 0 else j + 1

    a = cut(fractions[0])
    b = max(a, cut(fractions[0] + fractions[1]))
    return text[:a], text[a:b], text[b:]


def _shuffle_pieces(pieces: list[str], rng: np.random.Generator) -> list[str]:
    """Permute the even-indexed pieces (the content), keeping odd-indexed separators in place."""
    content = pieces[0::2]
    order = rng.permutation(len(content))
    out = list(pieces)
    out[0::2] = [content[i] for i in order]
    return out


def shuffle_sentences(text: bytes, rng: np.random.Generator) -> bytes:
    """Sentences in random order; the whitespace between them stays where it was."""
    pieces = _SENTENCE_END.split(text.decode("utf-8"))
    return "".join(_shuffle_pieces(pieces, rng)).encode("utf-8")


def shuffle_words(text: bytes, rng: np.random.Generator) -> bytes:
    """Words shuffled inside each sentence: the same words, the arrangement destroyed.

    The sentence's closing punctuation stays at its end, so sentence boundaries do not move.
    """
    pieces = _SENTENCE_END.split(text.decode("utf-8"))
    for i in range(0, len(pieces), 2):
        body = pieces[i].rstrip(".!?")
        close = pieces[i][len(body):]
        pieces[i] = "".join(_shuffle_pieces(_SPACE.split(body), rng)) + close
    return "".join(pieces).encode("utf-8")


@dataclass(frozen=True)
class Book:
    id: str
    title: str
    stage: str               # "A" structured (textbooks), "B" unstructured
    role: str                # "train" (split 80/10/10) or "heldout" (never trained on)
    order: int               # reading order inside its stage
    file: str                # cleaned text, relative to the manifest's directory
    sha256: str              # of the cleaned text
    bytes: int
    license: str
    source: str


@dataclass(frozen=True)
class Corpus:
    root: Path
    books: tuple[Book, ...]

    @staticmethod
    def load(manifest: str | Path) -> Corpus:
        manifest = Path(manifest)
        data = json.loads(manifest.read_text())
        books = tuple(sorted((Book(**b) for b in data["books"]), key=lambda b: (b.stage, b.order)))
        return Corpus(manifest.parent, books)

    def text(self, book: Book, verify: bool = True) -> bytes:
        data = (self.root / book.file).read_bytes()
        if verify and sha256(data) != book.sha256:
            raise ValueError(f"{book.file} does not match its manifest hash")
        return data

    def stream(self, stage: str, part: str, control=None, rng: np.random.Generator | None = None) -> bytes:
        """Concatenated ``part`` ("train", "validation", "test" or "heldout") of one stage, in reading order.

        ``control`` (``shuffle_sentences`` or ``shuffle_words``) rearranges each book's part separately.
        """
        index = {"train": 0, "validation": 1, "test": 2}
        chunks = []
        for book in self.books:
            if book.stage != stage:
                continue
            if part == "heldout":
                if book.role != "heldout":
                    continue
                chunk = self.text(book)
            else:
                if book.role != "train":
                    continue
                chunk = split_book(self.text(book))[index[part]]
            if control is not None:
                chunk = control(chunk, rng)
            chunks.append(chunk)
        return b"".join(chunks)


def write_manifest(path: str | Path, books: list[Book], note: str = "") -> None:
    Path(path).write_text(json.dumps({"note": note, "books": [asdict(b) for b in books]}, indent=1) + "\n")
