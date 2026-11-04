"""Rebuild the E5 stage files from their public sources and verify every one against data/e5/sources.json.

    python experiments/e5_restore.py --data DIR            # about 280 files, 130 MB; needs curl and network
    python experiments/e5_restore.py --data DIR --check    # only verify what is already in DIR

The stage texts are not in git. ``sources.json`` names every file (stage, ebook id, bytes, SHA-256); nothing is
selected here. Each file is fetched (PG-19 bucket first, then Project Gutenberg as text, then as HTML converted
to text), cleaned with ``vam.corpus.clean_text`` and cut by the rule that made it (S1 blocks, the last 3% of each
stage-5 work, the end of the C prose), and kept only if its SHA-256 equals the recorded one. A file that cannot be
reproduced is listed and the exit status is 1. Gutenberg answers 406 unless gzip is accepted, so curl is used with
``--compressed``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

from _common import ROOT

from vam.corpus import clean_text

DATA = ROOT / "experiments" / "data" / "e5"
PG19 = "https://storage.googleapis.com/deepmind-gutenberg"
#: stage-5 works: file stem -> (ebook id, source order); "pg19" means the PG-19 bucket first, "html" a converted page
S5 = {"principia": ("76404", "html"), "opticks": ("33504", "txt"), "relativity": ("30155", "txt"),
      "notebooks": ("5000", "txt"), "euclid_casey": ("21076", "html"), "republic": ("1497", "pg19"),
      "charmides": ("1580", "pg19"), "laches": ("1584", "pg19"), "euthydemus": ("1598", "pg19"),
      "cratylus": ("1616", "pg19"), "ion": ("1635", "pg19"), "meno": ("1643", "pg19"), "apology": ("1656", "pg19"),
      "crito": ("1657", "pg19"), "phaedo": ("1658", "pg19"), "gorgias": ("1672", "pg19"),
      "parmenides": ("1687", "pg19"), "philebus": ("1744", "pg19"),
      "einstein_sidelights": ("7333", "txt"), "einstein_meaning": ("36276", "html"),
      "einstein_fundamental": ("69572", "html"), "einstein_principle": ("66944", "txt"),
      "leonardo_thoughts": ("29904", "txt"), "xen_memorabilia": ("1177", "txt"), "xen_apology": ("1171", "txt")}
WEBSTER = "29765"
BLOCK = 65536

_BLOCKS = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "table", "blockquote", "pre", "section",
           "hr", "dd", "dt", "center"}


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self.skip += 1
        if tag in _BLOCKS:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self.skip = max(0, self.skip - 1)
        if tag in _BLOCKS:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_to_text(raw: bytes) -> bytes:
    p = _Text()
    p.feed(raw.decode("utf-8", "replace"))
    t = "".join(p.out).replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" ?\n ?", "\n", t)
    t = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", t)
    return t.strip("\n").encode("utf-8") + b"\n"


def curl(url: str) -> bytes | None:
    for _ in range(2):
        r = subprocess.run(["curl", "-s", "-f", "--compressed", "--max-time", "180", url], capture_output=True)
        if r.returncode == 0:
            return r.stdout
        if r.returncode == 22:                       # HTTP error (404): no retry
            return None
    return None


def candidates(ebook: str, order: tuple[str, ...], clean_html: bool = True):
    """Cleaned texts of one ebook from each source in turn. A converted HTML page goes through ``clean_text`` as well
    when ``clean_html`` (stages 2 to 4 and C did; the stage-5 works and the dictionary did not)."""
    for kind in order:
        if kind == "pg19":
            for split in ("train", "validation", "test"):
                raw = curl(f"{PG19}/{split}/{ebook}.txt")
                if raw is not None:
                    yield clean_text(raw)
                    break
        elif kind == "txt":
            raw = curl(f"https://www.gutenberg.org/cache/epub/{ebook}/pg{ebook}.txt")
            if raw is not None:
                yield clean_text(raw)
        else:
            raw = curl(f"https://www.gutenberg.org/cache/epub/{ebook}/pg{ebook}-images.html")
            if raw is not None:
                t = html_to_text(raw).decode("utf-8")
                m1, m2 = re.search(r"\*\*\* START OF.*?\*\*\*", t), re.search(r"\*\*\* END OF.*?\*\*\*", t)
                if m1 and m2:
                    t = t[m1.end():m2.start()]
                page = t.strip("\n").encode("utf-8") + b"\n"
                yield clean_text(page) if clean_html else page


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def derive(file: str) -> tuple[str, tuple[str, ...], callable, bool]:
    """(source key, fetch order, function from the cleaned text to this file's bytes)."""
    if file in ("S1_train.txt", "S1_heldout.txt"):
        def cut(c: bytes) -> bytes:
            blocks, i = [], 0
            while i < len(c):
                j = min(len(c), i + BLOCK)
                k = c.find(b"\n", j)
                j = len(c) if k < 0 else k + 1
                blocks.append(c[i:j])
                i = j
            held = file == "S1_heldout.txt"
            return b"".join(b for n, b in enumerate(blocks) if (n % 20 == 19) == held)
        return WEBSTER, ("txt", "pg19"), cut, True
    m = re.fullmatch(r"S55([ab])_(\w+?)(_ho)?\.txt", file)
    if m:
        ebook, order = S5[m.group(2)]

        def cut5(c: bytes) -> bytes:
            cutpoint = int(len(c) * 0.97)
            k = c.find(b"\n", cutpoint)
            cutpoint = len(c) if k < 0 else k + 1
            return c[cutpoint:] if m.group(3) else c[:cutpoint]
        return ebook, (order, *(o for o in ("pg19", "txt", "html") if o != order)), cut5, True
    m = re.fullmatch(r"(S2p|S3|S4|C)_(\d+)\.txt", file)
    if m:
        return m.group(2), ("pg19", "txt", "html"), lambda c: c, False
    raise SystemExit(f"e5_restore: no rule for {file}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--sources", type=Path, default=DATA / "sources.json")
    ap.add_argument("--check", action="store_true", help="verify only")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    sources = json.loads(args.sources.read_text())
    want = {rec["file"]: rec for g in ("T", "C", "heldout") for files in sources[g].values() for rec in files}
    args.data.mkdir(parents=True, exist_ok=True)
    todo = {f: r for f, r in want.items()
            if not (args.data / f).exists() or sha((args.data / f).read_bytes()) != r["sha256"]}
    print(len(want), "files named,", len(todo), "to build", flush=True)
    if args.check or not todo:
        sys.exit(1 if todo else 0)
    by_source: dict[str, list[str]] = {}
    plan = {}
    for f in todo:
        key, order, cut, raw_html = derive(f)
        plan[f] = (key, order, cut, raw_html)
        by_source.setdefault(key, []).append(f)

    def one(key: str) -> list[tuple[str, bool]]:
        files = by_source[key]
        order = plan[files[0]][1]
        done: dict[str, bool] = {f: False for f in files}
        for text in candidates(key, order, clean_html=not plan[files[0]][3]):
            for f in files:
                if done[f]:
                    continue
                piece = plan[f][2](text)
                if f.startswith("C_"):
                    piece = piece[:want[f]["bytes"]]          # the last C book was cut at a line end
                if sha(piece) == want[f]["sha256"]:
                    (args.data / f).write_bytes(piece)
                    done[f] = True
            if all(done.values()):
                break
        return list(done.items())

    bad = []
    with ThreadPoolExecutor(args.threads) as ex:
        for res in ex.map(one, list(by_source)):
            bad += [f for f, ok in res if not ok]
    print("rebuilt", len(todo) - len(bad), "files;", len(bad), "could not be reproduced", flush=True)
    for f in bad:
        print("  not reproduced:", f)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
