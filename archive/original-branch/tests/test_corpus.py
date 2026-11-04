"""Corpus handling for experiment E2: cleaning, splits, controls and the manifest."""

import collections

import numpy as np
import pytest

from vam.corpus import Book, Corpus, clean_text, sha256, shuffle_sentences, shuffle_words, split_book, write_manifest

RAW = (b"The Project Gutenberg eBook of Algebra\r\n\r\n"
       b"*** START OF THE PROJECT GUTENBERG EBOOK ALGEBRA ***\r\n"
       b"Chapter 1. Numbers.\r\nA number is a count. Two plus two is four!\r\nIs it? Yes.\r\n"
       b"*** END OF THE PROJECT GUTENBERG EBOOK ALGEBRA ***\r\nLicense text.\r\n")


def test_cleaning_keeps_only_the_book_text():
    text = clean_text(RAW)
    assert text == b"Chapter 1. Numbers.\nA number is a count. Two plus two is four!\nIs it? Yes.\n"
    assert clean_text(b"plain text without markers\r\n") == b"plain text without markers\n"
    assert clean_text("café ≤ x\n".encode()) == "café ≤ x\n".encode()


def test_cleaning_rejects_bytes_that_are_not_utf8():
    with pytest.raises(UnicodeDecodeError):
        clean_text(b"\xff\xfe broken")


def test_split_by_position_at_line_ends():
    text = b"".join(b"line %03d\n" % i for i in range(100))
    train, val, test = split_book(text)
    assert train + val + test == text
    assert train.endswith(b"\n") and val.endswith(b"\n")
    assert abs(len(train) / len(text) - 0.8) < 0.02 and abs(len(val) / len(text) - 0.1) < 0.02


def test_controls_rearrange_without_changing_a_byte(rng):
    text = clean_text(RAW) * 20
    for control in (shuffle_sentences, shuffle_words):
        out = control(text, rng)
        assert len(out) == len(text)
        assert collections.Counter(out) == collections.Counter(text)
        assert out != text


def test_word_shuffle_stays_inside_each_sentence(rng):
    text = b"alpha beta gamma. delta epsilon zeta. eta theta iota."
    out = shuffle_words(text, rng).decode()
    sentences = [set(s.split()) for s in out.split(". ")]
    assert sentences[0] == {"alpha", "beta", "gamma"} and sentences[1] == {"delta", "epsilon", "zeta"}


def test_manifest_streams_and_hashes(tmp_path, rng):
    books = []
    for i, (stage, role) in enumerate([("A", "train"), ("A", "train"), ("A", "heldout"), ("B", "train")]):
        text = b"".join(b"%s book %d line %d.\n" % (stage.encode(), i, j) for j in range(50))
        (tmp_path / f"{i}.txt").write_bytes(text)
        books.append(Book(id=str(i), title=f"t{i}", stage=stage, role=role, order=-i, file=f"{i}.txt",
                          sha256=sha256(text), bytes=len(text), license="public domain", source="test"))
    write_manifest(tmp_path / "MANIFEST.json", books)
    corpus = Corpus.load(tmp_path / "MANIFEST.json")
    a_train = corpus.stream("A", "train")
    assert a_train.startswith(b"A book 1 ")                      # reading order follows ``order``
    assert b"book 2" not in a_train and corpus.stream("A", "heldout").startswith(b"A book 2 ")
    parts = [corpus.stream("A", p) for p in ("train", "validation", "test")]
    assert sum(map(len, parts)) == books[0].bytes + books[1].bytes
    shuffled = corpus.stream("A", "train", control=shuffle_sentences, rng=np.random.default_rng(1))
    assert len(shuffled) == len(a_train) and shuffled != a_train
    (tmp_path / "0.txt").write_bytes(b"tampered\n")
    with pytest.raises(ValueError):
        corpus.stream("A", "train")
