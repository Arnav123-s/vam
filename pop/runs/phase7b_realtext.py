"""Phase 7b: real text -- the mountain.

Pride and Prejudice (Gutenberg), cleaned to lowercase a-z + space (27 chars).
The hierarchical bet: words are the slow scale (observable via spaces -- the
bootstrap is free), chars are the fast scale.

Population (word-level hierarchical):
  - Vocabulary: words from training (min count 3).
  - Word bigram: p(next_word | prev_word).
  - Spelling: deterministic per word (like 7a).
  - Prediction: track chars since last space -> word prefix. If it matches a
    unique vocab word, next char is deterministic. If ambiguous, marginalize.
    At a boundary (prev char was space), use bigram to predict next word's
    first char. Back off to char 4-gram for OOV/unknown.
  - This is H1 (population) + H3 (words as stable slow units) + the 7a
    bootstrap lesson (spaces ground the word discovery).

Baseline: flat order-5 char n-gram with stupid backoff.

Metric: bpb on held-out 5k chars. Generation: 200 chars from each, eyeball
for salad vs English.

Protocol (written before the run): train 30k chars, test 5k. Expectation:
population < flat5 because word bigrams reach further than 5 chars. If the
population loses, the hierarchy doesn't pay on real text.
"""

import json
import re
import time

import numpy as np


def load_corpus(path="/tmp/pride.txt", n_train=30000, n_test=5000):
    with open(path) as f:
        txt = f.read()
    # strip Gutenberg header/footer
    s = txt.find("*** START OF")
    e = txt.find("*** END OF")
    txt = txt[s:e] if s > 0 else txt
    txt = txt.lower()
    txt = re.sub(r"[^a-z ]+", " ", txt)
    txt = re.sub(r" +", " ", txt)
    # find a clean split (train then test, contiguous)
    return txt[:n_train], txt[n_train:n_train + n_test]


class WordPopulation:
    def __init__(self, min_count=3):
        self.vocab = {}      # word -> idx
        self.words = []      # idx -> word
        self.spell = {}      # word -> chars
        self.bigram = {}     # prev_idx -> {next_idx: count}
        self.unigram = {}
        self.char4 = {}      # backoff char 4-gram
        self.min_count = min_count

    def train(self, text):
        words = text.split()
        # vocab
        from collections import Counter
        c = Counter(words)
        for w, n in c.items():
            if n >= self.min_count and w not in self.vocab:
                self.vocab[w] = len(self.words)
                self.words.append(w)
        V = len(self.words)
        # bigrams
        ids = [self.vocab[w] for w in words if w in self.vocab]
        for a, b in zip(ids[:-1], ids[1:]):
            self.bigram.setdefault(a, Counter())[b] += 1
            self.unigram[a] = self.unigram.get(a, 0) + 1
        self.unigram[ids[-1]] = self.unigram.get(ids[-1], 0) + 1
        # char 4-gram backoff (on raw text)
        for i in range(len(text) - 4):
            ctx = text[i:i+4]
            ch = text[i+4]
            self.char4.setdefault(ctx, Counter())[ch] += 1
        self._V = V

    def _char_backoff(self, hist):
        ctx = hist[-4:] if len(hist) >= 4 else hist
        while ctx:
            if ctx in self.char4:
                v = self.char4[ctx]
                tot = sum(v.values())
                return {ch: n / tot for ch, n in v.items()}
            ctx = ctx[1:]
        return None

    ALPHA = "abcdefghijklmnopqrstuvwxyz "

    def _smooth(self, dist, eps=0.02):
        """Full 27-char support. Never None, never zero."""
        out = {ch: eps / 27 for ch in self.ALPHA}
        if dist:
            s = sum(dist.values())
            if s > 0:
                for ch, p in dist.items():
                    if ch in out:
                        out[ch] += (1 - eps) * p / s
        tot = sum(out.values())
        return {ch: p / tot for ch, p in out.items()}

    def predict_dist(self, hist):
        """p(next_char | hist). hist = chars so far (we predict next)."""
        # current word prefix = chars since last space
        last_space = hist.rfind(" ")
        prefix = hist[last_space + 1:]
        if prefix == "" :
            # at word boundary: use bigram
            prev_words = hist[:last_space].split() if last_space > 0 else []
            prev = prev_words[-1] if prev_words else None
            dist = {}
            if prev in self.vocab and self.vocab[prev] in self.bigram:
                bg = self.bigram[self.vocab[prev]]
                tot = sum(bg.values())
                for ni, cnt in bg.items():
                    w = self.words[ni]
                    ch = w[0]
                    dist[ch] = dist.get(ch, 0) + cnt / tot
            else:
                # unigram fallback
                tot = sum(self.unigram.values())
                for ni, cnt in self.unigram.items():
                    w = self.words[ni]
                    ch = w[0]
                    dist[ch] = dist.get(ch, 0) + cnt / tot
            # smooth with char backoff
            cb = self._char_backoff(hist)
            if cb:
                for ch, p in cb.items():
                    dist[ch] = dist.get(ch, 0) * 0.7 + p * 0.3
            return self._smooth(dist)
        else:
            # mid-word: find vocab words with this prefix
            cands = [w for w in self.words if w.startswith(prefix) and len(w) > len(prefix)]
            if cands:
                # weight by unigram
                tot = sum(self.unigram.get(self.vocab[w], 1) for w in cands)
                dist = {}
                for w in cands:
                    ch = w[len(prefix)]
                    dist[ch] = dist.get(ch, 0) + self.unigram.get(self.vocab[w], 1) / tot
                return self._smooth(dist)
            # OOV: char backoff
            return self._smooth(self._char_backoff(hist))

    def bpb(self, text):
        lps = []
        hist = ""
        for ch in text:
            d = self.predict_dist(hist)
            p = d.get(ch, 1e-6) if d else 1e-6
            lps.append(-np.log2(max(p, 1e-6)))
            hist = (hist + ch)[-50:]  # bounded history
        return float(np.mean(lps))

    def generate(self, n, seed=0):
        rng = np.random.default_rng(seed)
        hist, out = "", ""
        for _ in range(n):
            d = self.predict_dist(hist)
            if not d:
                ch = " "
            else:
                chs, ps = zip(*d.items())
                ps = np.array(ps) / sum(ps)
                ch = chs[rng.choice(len(chs), p=ps)]
            out += ch
            hist = (hist + ch)[-50:]
        return out


def flat_bpb(train, test, order=5):
    from collections import Counter
    counts = {}
    for i in range(len(train) - order):
        ctx = train[i:i+order]
        counts.setdefault(ctx, Counter())[train[i+order]] += 1
    lps = []
    for i in range(len(test) - order):
        ctx = test[i:i+order]
        ch = test[i+order]
        # stupid backoff
        c = ctx
        while c and c not in counts:
            c = c[1:]
        v = counts.get(c, Counter())
        tot = sum(v.values())
        p = v.get(ch, 0.5) / (tot + 0.5 * 27) if tot else 1/27
        lps.append(-np.log2(max(p, 1e-9)))
    return float(np.mean(lps))


def main():
    t0 = time.time()
    train, test = load_corpus()
    print(f"train {len(train)} chars, test {len(test)} chars")
    pop = WordPopulation()
    pop.train(train)
    print(f"vocab {len(pop.words)} words")
    pb = pop.bpb(test)
    f5 = flat_bpb(train, test, 5)
    out = {
        "protocol": "phase7b_real_text",
        "corpus": "Pride and Prejudice (Gutenberg), a-z+space",
        "population_bpb": round(pb, 4),
        "flat5_bpb": round(f5, 4),
        "vocab_size": len(pop.words),
    }
    gen = pop.generate(200, seed=1)
    out["pop_generation"] = gen
    print(f"population bpb: {pb:.4f}")
    print(f"flat5 bpb:      {f5:.4f}")
    print(f"--- generation ---\n{gen}")
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase7b.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase7b.json")


if __name__ == "__main__":
    main()
