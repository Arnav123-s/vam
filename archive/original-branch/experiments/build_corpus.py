"""Turn a book selection into the hashed E2 corpus (experiments/PROTOCOL.md, E2 data).

    python experiments/build_corpus.py SOURCE OUT

SOURCE is a folder or zip holding ``selection.json`` and the files it lists:
the output of ``fetch_gutenberg.py``, or of ``select_local.py`` for books the
owner already has. OUT gets one cleaned UTF-8 file per book and
``MANIFEST.json``. Rules, fixed in advance:

* the reading order is the selection order (stage A first);
* a book whose bytes are not valid UTF-8 is left out;
* a book that repeats an earlier one, in either stage, is left out, so no
  edition of a training book can reach the unseen books. A repeat is a book
  whose cleaned text has the same SHA-256 as an earlier one, or whose title
  and authors, lowercased with punctuation and spacing removed, match an
  earlier one's;
* every tenth remaining book of each stage (the 10th, 20th, ...) is held out
  and never trained on.

Every book left out is listed in the manifest's note with its reason.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

from _common import ROOT  # noqa: F401  (puts src on the path)

from vam.corpus import Book, clean_text, sha256, write_manifest

LICENSE = "Public domain in the USA (Project Gutenberg)"     # unless the selection names one per book


def identity(entry: dict) -> str:
    """Title and authors, lowercased, with everything but letters and digits removed."""
    return re.sub(r"[\W_]+", "", f"{entry['title']}|{entry.get('authors', '')}".lower())


def read_source(source: Path):
    if source.suffix == ".zip":
        zf = zipfile.ZipFile(source)
        return json.loads(zf.read("selection.json")), zf.read
    return json.loads((source / "selection.json").read_text()), lambda name: (source / name).read_bytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    selection, read = read_source(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    books, skipped, repeats, position = [], [], [], {"A": 0, "B": 0}
    seen_text, seen_identity = {}, {}
    for entry in selection["books"]:
        raw = read(entry["file"])
        if sha256(raw) != entry["sha256"]:
            raise ValueError(f"{entry['file']} does not match the selection hash")
        try:
            text = clean_text(raw)
        except UnicodeDecodeError:
            skipped.append(entry["id"])
            continue
        earlier = seen_text.get(sha256(text)) or seen_identity.get(identity(entry))
        if earlier is not None:
            repeats.append(f"{entry['id']} (repeats {earlier})")
            continue
        seen_text[sha256(text)] = seen_identity[identity(entry)] = entry["id"]
        stage = entry["stage"]
        position[stage] += 1
        name = f"{stage}/{entry.get('prefix', 'pg')}{entry['id']}.txt"
        (args.out / stage).mkdir(exist_ok=True)
        (args.out / name).write_bytes(text)
        books.append(Book(id=entry["id"], title=entry["title"], stage=stage,
                          role="heldout" if position[stage] % 10 == 0 else "train",
                          order=position[stage], file=name, sha256=sha256(text), bytes=len(text),
                          license=entry.get("license", LICENSE), source=entry.get("url") or entry["source"]))
    origin = (f"catalog sha256 {selection['catalog_sha256']}" if "catalog_sha256" in selection
              else selection.get("origin", "no catalog"))
    note = (f"Built from selection.json ({origin}). "
            f"Left out as not UTF-8: {', '.join(skipped) or 'none'}. "
            f"Left out as repeats of an earlier book: {', '.join(repeats) or 'none'}.")
    write_manifest(args.out / "MANIFEST.json", books, note)
    for stage in ("A", "B"):
        sel = [b for b in books if b.stage == stage]
        print(f"stage {stage}: {len(sel)} books, {sum(b.bytes for b in sel) / 2**20:.1f} MiB, "
              f"{sum(b.role == 'heldout' for b in sel)} held out")


if __name__ == "__main__":
    main()
