"""Select and download the E2 corpus from PG-19 (experiments/PROTOCOL.md, E2 data, revision of 2026-10-18).

    python experiments/fetch_pg19.py --out DIR

PG-19 (Rae et al., 2019) is DeepMind's fixed snapshot of 28,752 Project
Gutenberg books published before 1919, served from the public bucket
``storage.googleapis.com/deepmind-gutenberg`` (dataset Apache-2.0, books in
the public domain). This machine can reach that bucket but not gutenberg.org,
so this replaces ``fetch_gutenberg.py`` as the E2 source. The selection is
fixed by rule, not by hand:

* Stage A, structured: books whose title names an instructional form
  (``FORM``: text-book, elements of, elementary, primer, introduction to,
  manual, handbook, lessons, treatise, principles of, ...) or a school subject
  (``SUBJECT``: arithmetic, algebra, geometry, physics, chemistry, botany,
  grammar, ...), and whose text is didactic: at least ``DIDACTIC_MIN`` (10)
  words per 10,000 among exercise, lesson, fig., example, problem,
  definition, theorem, proposition, experiment (``didactic``). A title alone
  misleads (*Cobb's Anatomy* is a humour book); the threshold sits above the
  95th percentile of 150 random PG-19 books (8.2) measured before any E2 run.
  Books for learning a language other than English are left out
  (``FOREIGN``), because they mix languages. Reading order: titles naming a
  subject first, then the others, each by ebook number.
* Stage B, unstructured: every book whose title matches neither rule, in a
  seeded random order.
* A book is skipped unless it reads as English: at least 12% of its words are
  among twenty common English function words (``english``).
* Each stage stops when its budget of downloaded bytes is reached.

Standard library only. Writes ``selection.json`` (every chosen book with its
title, URL, size and SHA-256), the raw files under ``files/``, and a copy of
the metadata table, all in the format ``build_corpus.py`` reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "https://storage.googleapis.com/deepmind-gutenberg"
LISTING = "https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o"
SPLITS = ("train", "validation", "test")
SEED = 20261018
LICENSE = "Public domain in the USA (Project Gutenberg, via PG-19, dataset Apache-2.0)"

FORM = re.compile(r"\b(text-?books?|text book|elements of|elementary|first (book|lessons|principles)|primer|"
                  r"introduction to|introductory|manual of|handbook|lessons (in|on)|course (in|of)|treatise on|"
                  r"principles of|outlines of|rudiments|exercises in)\b", re.IGNORECASE)
SUBJECT = re.compile(r"\b(arithmetic|algebra|geometry|trigonometry|calculus|mathematics|physics|chemistry|botany|"
                     r"zoology|physiology|astronomy|geology|biology|mechanics|grammar|rhetoric|logic|economy|anatomy|"
                     r"hygiene|electricity|magnetism|optics|mineralogy|surveying|navigation|bookkeeping|"
                     r"composition)\b", re.IGNORECASE)
FOREIGN = re.compile(r"\b(greek|latin|french|german|spanish|italian|portuguese|esperanto|malay|bisaya|bohemian|"
                     r"gaelic|welsh|irish|hebrew|sanskrit|chinese|japanese|arabic|russian|new zealand language|"
                     r"maori|hawaiian|tagalog|volap.k)\b", re.IGNORECASE)
DIDACTIC = re.compile(r"\b(exercises?|lesson|fig\.|figs\.|example|problems?|definition|theorem|proposition|"
                      r"experiment|review questions|questions for review)\b", re.IGNORECASE)
DIDACTIC_MIN = 10.0
FUNCTION_WORDS = frozenset("the of and to a in that is it was he for with as his on be at by i".split())
ENGLISH_SHARE = 0.12


def get(url: str, attempts: int = 4) -> bytes:
    for k in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except OSError:
            if k == attempts - 1:
                raise
            time.sleep(2 ** (k + 1))
    raise AssertionError("unreachable")


def english(raw: bytes) -> bool:
    words = re.findall(r"[a-z]+", raw[:2_000_000].decode("utf-8", "replace").lower())
    return bool(words) and sum(w in FUNCTION_WORDS for w in words) >= ENGLISH_SHARE * len(words)


def didactic(raw: bytes) -> float:
    """Didactic markers per 10,000 words (first 3 MB)."""
    text = raw[:3_000_000].decode("utf-8", "replace")
    return 1e4 * len(DIDACTIC.findall(text)) / max(1, len(text.split()))


def split_of() -> dict[str, str]:
    """Ebook id -> the PG-19 split folder that holds it (from the bucket listing)."""
    out, token = {}, None
    while True:
        query = {"maxResults": "1000", "fields": "items(name),nextPageToken"}
        if token:
            query["pageToken"] = token
        page = json.loads(get(f"{LISTING}?{urllib.parse.urlencode(query)}"))
        for item in page.get("items", []):
            folder, _, name = item["name"].partition("/")
            if folder in SPLITS and name.endswith(".txt"):
                out[name[:-4]] = folder
        token = page.get("nextPageToken")
        if not token:
            return out


def stage_a_key(row: list[str]) -> tuple[int, int]:
    return (0 if SUBJECT.search(row[1]) else 1, int(row[0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--budget-mib", type=float, default=64.0, help="downloaded bytes per stage")
    args = ap.parse_args()
    out = args.out
    (out / "files").mkdir(parents=True, exist_ok=True)
    meta = get(f"{BUCKET}/metadata.csv")
    (out / "metadata.csv").write_bytes(meta)
    rows = [r for r in (line.split(",") for line in meta.decode("utf-8").splitlines()) if len(r) >= 4]
    rows = [[r[0], ",".join(r[1:-2]), r[-2], r[-1]] for r in rows]          # titles may contain commas
    where = split_of()
    rows = [r for r in rows if r[0] in where]
    titled = {r[0] for r in rows if FORM.search(r[1]) or SUBJECT.search(r[1])}
    a = sorted((r for r in rows if r[0] in titled and not FOREIGN.search(r[1])), key=stage_a_key)
    b = sorted((r for r in rows if r[0] not in titled), key=lambda r: int(r[0]))
    random.Random(SEED).shuffle(b)
    budget = args.budget_mib * 2**20
    chosen, skipped, not_didactic = [], [], []
    for stage, order in (("A", a), ("B", b)):
        used = 0
        for ebook, title, year, url in order:
            if used >= budget:
                break
            raw = get(f"{BUCKET}/{where[ebook]}/{ebook}.txt")
            if not english(raw):
                skipped.append(ebook)
                continue
            if stage == "A" and didactic(raw) < DIDACTIC_MIN:
                not_didactic.append(ebook)
                continue
            name = f"files/{ebook}.txt"
            (out / name).write_bytes(raw)
            used += len(raw)
            chosen.append({"id": ebook, "title": title, "authors": "", "year": year, "stage": stage,
                           "pg19_split": where[ebook], "file": name, "bytes": len(raw),
                           "sha256": hashlib.sha256(raw).hexdigest(), "url": url, "license": LICENSE})
            print(f"{stage} {ebook} {len(raw) / 1024:.0f} KiB {title[:60]}", flush=True)
        print(f"stage {stage}: {sum(c['stage'] == stage for c in chosen)} books, {used / 2**20:.1f} MiB",
              flush=True)
    selection = {"source": BUCKET, "catalog_sha256": hashlib.sha256(meta).hexdigest(),
                 "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "stage_a_candidates": len(a), "skipped_not_english": skipped,
                 "stage_a_not_didactic": not_didactic, "books": chosen}
    (out / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    print(f"done: {len(chosen)} books; skipped as not English: {len(skipped)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="backslashreplace")
    main()
