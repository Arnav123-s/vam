"""Select and download the E2 corpus from Project Gutenberg (experiments/PROTOCOL.md, E2 data).

Standard library only, so it runs on any machine with Python 3.9+ and web access:

    python fetch_gutenberg.py --out D:/ai/vam-data

The selection is fixed by rule, not by hand:

* Stage A, structured: English texts whose Library of Congress subjects
  include ``Textbooks`` (the cataloguers' own label for a textbook),
  excluding books for learning a foreign language. Reading order: by
  Library of Congress class, then by ebook number.
* Stage B, unstructured: English fiction from English and American
  literature (classes PR and PS) that is not a textbook, in a seeded
  random order.
* Each stage stops when its budget of downloaded bytes is reached.

Requests are spaced two seconds apart by default, the wait Project Gutenberg's
robot-access policy asks of automated downloads.

The script writes ``selection.json`` (every chosen book with its catalog
fields, URL, size and SHA-256), the raw files, and one zip of everything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
TEXT_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
AGENT = "vam-research-corpus/1.0 (one-time download of public-domain texts for a study)"
SEED = 20261018
MIB = 1 << 20


def fetch(url: str, tries: int = 4) -> bytes | None:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2 ** (attempt + 1))
    return None


def subjects(row: dict) -> list[str]:
    return [s.strip() for s in row["Subjects"].split(";") if s.strip()]


def locc(row: dict) -> str:
    return row["LoCC"].split(";")[0].strip()


def is_textbook(row: dict) -> bool:
    subs = subjects(row)
    return any("Textbooks" in s for s in subs) and not any("foreign speakers" in s for s in subs)


def is_fiction(row: dict) -> bool:
    return (locc(row)[:2] in ("PR", "PS") and any("Fiction" in s for s in subjects(row))
            and not any("Textbooks" in s for s in subjects(row)))


def candidates(rows: list[dict]) -> dict[str, list[dict]]:
    english = [r for r in rows if r["Type"] == "Text" and r["Language"].strip() == "en"]
    a = sorted((r for r in english if is_textbook(r)), key=lambda r: (locc(r), int(r["Text#"])))
    b = sorted((r for r in english if is_fiction(r)), key=lambda r: int(r["Text#"]))
    random.Random(SEED).shuffle(b)
    return {"A": a, "B": b}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--budget-mib", type=float, default=64.0, help="downloaded bytes per stage")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")       # titles print on any console encoding
    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    catalog = fetch(CATALOG)
    if catalog is None:
        sys.exit("could not download the Gutenberg catalog")
    (args.out / "pg_catalog.csv").write_bytes(catalog)
    rows = list(csv.DictReader(io.StringIO(catalog.decode("utf-8"))))
    chosen = []
    for stage, pool in candidates(rows).items():
        total = 0
        for row in pool:
            if total >= args.budget_mib * MIB:
                break
            path = raw_dir / f"pg{row['Text#']}.txt"
            data = path.read_bytes() if path.exists() else fetch(TEXT_URL.format(id=row["Text#"]))
            if not path.exists():
                time.sleep(args.delay)
            if data is None:
                print(f"skip {row['Text#']} (no plain-text file)", flush=True)
                continue
            path.write_bytes(data)
            total += len(data)
            chosen.append({"id": row["Text#"], "stage": stage, "title": row["Title"], "authors": row["Authors"],
                           "subjects": row["Subjects"], "locc": row["LoCC"], "issued": row["Issued"],
                           "url": TEXT_URL.format(id=row["Text#"]), "file": f"raw/{path.name}",
                           "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            print(f"{stage} {total / MIB:6.1f} MiB  {row['Text#']}  {row['Title'][:60]}", flush=True)
    sel = {"rule": __doc__, "seed": SEED, "budget_mib": args.budget_mib, "catalog_sha256":
           hashlib.sha256(catalog).hexdigest(), "books": chosen}
    (args.out / "selection.json").write_text(json.dumps(sel, indent=1))
    with zipfile.ZipFile(args.out / "vam-gutenberg.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.out / "selection.json", "selection.json")
        zf.write(args.out / "pg_catalog.csv", "pg_catalog.csv")
        for book in chosen:
            zf.write(args.out / book["file"], book["file"])
    print(f"done: {len(chosen)} books, {args.out / 'vam-gutenberg.zip'}")


if __name__ == "__main__":
    main()
