"""Download the books listed in a selection.json and check each against its recorded SHA-256.

    python experiments/fetch_selection.py SELECTION OUT

SELECTION is the ``selection.json`` of ``fetch_pg19.py`` (E2 data, PROTOCOL.md). OUT gets a copy of
``selection.json`` and ``files/<id>.txt`` for every book, in the layout ``build_corpus.py`` reads:

    python experiments/build_corpus.py OUT experiments/data/pg19

Each book is fetched from ``<source>/<pg19_split>/<id>.txt``, the object ``fetch_pg19.py`` read. The
``url`` field of a book is its catalogue page, not its text, so it is recorded but not fetched.

Standard library only. The run stops on the first download that fails after retries, and on the first
raw file whose SHA-256 differs from the selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

AGENT = "vam-research-corpus/1.0 (one-time download of public-domain texts for a study)"


def get(url: str, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            if attempt == tries - 1:
                raise RuntimeError(f"{url}: {err}") from err
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("selection", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    selection = json.loads(args.selection.read_text())
    source = selection["source"].rstrip("/")
    (args.out / "files").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.selection, args.out / "selection.json")
    for n, book in enumerate(selection["books"], 1):
        url = f"{source}/{book['pg19_split']}/{book['id']}.txt"
        raw = get(url)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != book["sha256"]:
            sys.exit(f"SHA-256 mismatch for {book['id']} ({url}): got {digest}, recorded {book['sha256']}")
        (args.out / book["file"]).write_bytes(raw)
        print(f"{n}/{len(selection['books'])} {book['id']} {len(raw)} bytes ok", flush=True)
    print(f"all {len(selection['books'])} books match their recorded SHA-256")


if __name__ == "__main__":
    main()
