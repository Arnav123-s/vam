"""Select the E2 corpus from text the owner already has (experiments/PROTOCOL.md, E2 data).

    python experiments/select_local.py RULES ROOT

ROOT is a folder of plain-text files copied from the owner's computer. RULES
is a JSON file, committed before any E2 run, that fixes the selection:

    {"origin": "...", "license": "...",
     "stages": [{"stage": "A", "sources": ["data_physics/*.txt"], "order": "path", "budget_mib": 64},
                {"stage": "B", "sources": ["pile_raw/**/*.txt"], "order": "seeded", "budget_mib": 64}],
     "exclude": ["quarantine_contaminated/"]}

Files under an excluded folder are never read. Within a stage, files
matching the sources are read in the stated order:
``path`` (the source patterns in turn, each sorted by path) or ``seeded`` (one
permutation with the protocol seed). A file is taken while the stage's budget
lasts; a file that an earlier stage's sources match is never read by a later
stage. A file is skipped when it is not UTF-8 or when fewer than 95% of its letters
are ASCII (the stage is English). Nothing is picked by hand. The script writes
``selection.json`` in the format ``build_corpus.py`` reads; the file name
stands in for the title, so the same file under two folders is caught as a
repeat there.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _common import ROOT as _ROOT  # noqa: F401  (puts src on the path)

from vam.corpus import sha256

SEED = 20261018
ENGLISH = 0.95


def english(raw: bytes) -> bool:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return False
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) >= ENGLISH * len(letters)


def candidates(root: Path, stage: dict, exclude: list[str], seen: set[Path]) -> list[Path]:
    """Files of one stage in reading order; ``seen`` (files any earlier stage considered) are passed over."""
    files = []
    for pattern in stage["sources"]:
        for path in sorted(root.glob(pattern)):
            rel = path.relative_to(root).as_posix()
            if path.is_file() and path not in seen and not any(rel.startswith(x) for x in exclude):
                seen.add(path)
                files.append(path)
    if stage.get("order", "path") == "seeded":
        random.Random(stage.get("seed", SEED)).shuffle(files)
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rules", type=Path)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    rules = json.loads(args.rules.read_text())
    books, skipped, seen = [], [], set()
    for stage in rules["stages"]:
        budget, used = stage["budget_mib"] * 2**20, 0
        for path in candidates(args.root, stage, rules.get("exclude", []), seen):
            if used >= budget:
                break
            raw = path.read_bytes()
            rel = path.relative_to(args.root).as_posix()
            if not english(raw):
                skipped.append(rel)
                continue
            used += len(raw)
            books.append({"id": f"{len(books) + 1:05d}", "prefix": "local", "title": path.stem,
                          "authors": "", "stage": stage["stage"], "file": rel, "sha256": sha256(raw),
                          "bytes": len(raw), "license": rules["license"], "source": f"{rules['origin']}: {rel}"})
        print(f"stage {stage['stage']}: {sum(b['stage'] == stage['stage'] for b in books)} files, "
              f"{used / 2**20:.1f} MiB")
    out = {"origin": rules["origin"], "rules_sha256": sha256(args.rules.read_bytes()),
           "skipped_not_english": skipped, "books": books}
    (args.root / "selection.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"skipped as not English or not UTF-8: {len(skipped)}")


if __name__ == "__main__":
    main()
