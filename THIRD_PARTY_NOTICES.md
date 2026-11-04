# Third-party notices and data rights

The Apache-2.0 grant in LICENSE covers original VAM contributions only. It does
not assert ownership of external texts or expand their permitted uses.

## PG-19 and Project Gutenberg

The experiments use PG-19 and Project Gutenberg books. Full books and training
streams are not included in this repository. The preserved selection manifests
identify titles, source URLs, sizes and hashes. Generated reports and saved
experiment outputs may contain short source-derived snippets; E5 question
metadata contains quotation evidence and source locators.

- [PG-19 project](https://github.com/google-deepmind/pg19) and its
  [Apache-2.0 license](https://github.com/google-deepmind/pg19/blob/master/LICENSE).
- [Project Gutenberg license and rights guidance](https://www.gutenberg.org/policy/license.html).
  Project Gutenberg distinguishes book text from its trademark and distribution
  terms. Its U.S. status statements are not a guarantee of rights elsewhere.
- Original book text, quotations and corpus-derived snippets are not being
  relicensed by the VAM maintainers. No claim of worldwide public-domain status
  is made. Check the applicable source notice and local law before reuse.

### E5 quotation sources

`experiments/data/e5/questions.json` preserves 140 questions and 172 quotation
records spanning the following 13 works. On 2026-11-04, the linked Project
Gutenberg catalog entry for each edition labeled it public domain in the USA.
This is a record of the catalog designation, not a legal determination for every
jurisdiction. Question and result hashes are preserved unchanged.

| Work key | Source edition |
|---|---|
| `principia` | [Isaac Newton, Principia; Andrew Motte translation, N. W. Chittenden edition (1846)](https://www.gutenberg.org/ebooks/76404) |
| `opticks` | [Isaac Newton, Opticks](https://www.gutenberg.org/ebooks/33504) |
| `relativity` | [Albert Einstein, Relativity; Robert W. Lawson translation, revised edition (1924)](https://www.gutenberg.org/ebooks/30155) |
| `euclid` | [Euclid and John Casey, The First Six Books of the Elements of Euclid](https://www.gutenberg.org/ebooks/21076) |
| `notebooks` | [Leonardo da Vinci, Notebooks; Jean Paul Richter translation (1888)](https://www.gutenberg.org/ebooks/5000) |
| `republic` | [Plato, The Republic; Benjamin Jowett translation](https://www.gutenberg.org/ebooks/1497) |
| `apology` | [Plato, Apology; Benjamin Jowett translation](https://www.gutenberg.org/ebooks/1656) |
| `meno` | [Plato, Meno; Benjamin Jowett translation](https://www.gutenberg.org/ebooks/1643) |
| `phaedo` | [Plato, Phaedo; Benjamin Jowett translation](https://www.gutenberg.org/ebooks/1658) |
| `philebus` | [Plato, Philebus; Benjamin Jowett translation](https://www.gutenberg.org/ebooks/1744) |
| `charmides` | [Plato, Charmides; Benjamin Jowett translation](https://www.gutenberg.org/ebooks/1580) |
| `gorgias` | [Plato, Gorgias; Benjamin Jowett translation](https://www.gutenberg.org/ebooks/1672) |
| `euthydemus` | [Plato, Euthydemus; Benjamin Jowett translation](https://www.gutenberg.org/ebooks/1598) |

The underlying corpus restoration rules and additional E5 teaching works are
recorded in `experiments/e5_restore.py` and `experiments/data/e5/sources.json`.
Those additional full texts are downloaded separately, not redistributed here.

## Software dependencies

NumPy, SciPy, SymPy, Numba, pytest, Hypothesis, Ruff and their dependencies are
installed separately through the package manager. They are not vendored in this
repository and remain under their respective upstream licenses. The declared
dependencies and optional extras are in `pyproject.toml`.

## Scientific references

The specifications and design cite the original scientific papers used for
context. Citations acknowledge those sources; they do not imply endorsement or
transfer any rights in their publications. See the reference sections of the
specifications for authors, titles and source links.
