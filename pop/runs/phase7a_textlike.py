"""Phase 7a: text-like hierarchical source -- the population meets language.

A tiny language with TRUE multi-scale structure (words -> chars), designed so
a flat char n-gram struggles but a hierarchical population shines.

Language:
  Alphabet: {0,1,2,3} (think a,b,c,d). 4 words x 3 chars, NO separators:
    W0 = 0,1,0   W1 = 1,2,3   W2 = 2,0,1   W3 = 3,3,3
  Word grammar: Markov chain over words (structured transitions).
  Stream: words concatenated. The model must find boundaries AND learn
  the word grammar from the char stream.

Population (HMM-style, hierarchical):
  State: (word_id, pos_in_word) = 12 states. Exact forward Bayes inference
  (perceive). p(char | word, pos) tables + p(next_word | word) transitions
  (learn). Predict: marginalize over states.

Baselines:
  flat4: order-4 char n-gram. flat6: order-6 char n-gram.
  (A 4-char window spans a word boundary; the word grammar is learnable but
  needs many 4-grams. Order-6 sees two full words.)

Metric: bpb on held-out stream. Expectation: population << flat4, and
population <= flat6 (it factorizes what flat6 memorizes). Generation: sample
30 chars from each; population should emit valid words in grammatical order,
flat4 may produce invalid words (salad).

Protocol (written before the run): seed 7, 20k train / 5k test chars.
"""

import json
import time

import numpy as np


WORDS = np.array([[0, 1, 0], [1, 2, 3], [2, 0, 1], [3, 3, 3]])
NW, WL, NC = 4, 3, 4
# word transition matrix (rows sum to 1)
TRANS = np.array([
    [0.1, 0.6, 0.2, 0.1],
    [0.3, 0.1, 0.5, 0.1],
    [0.2, 0.2, 0.1, 0.5],
    [0.4, 0.3, 0.2, 0.1],
])


def gen_stream(n_words, seed=0):
    rng = np.random.default_rng(seed)
    words = []
    w = rng.integers(NW)
    for _ in range(n_words):
        words.append(w)
        w = rng.choice(NW, p=TRANS[w])
    chars = np.concatenate([WORDS[w] for w in words])
    return chars, np.array(words)


class HierPopulation:
    """12-state (word,pos) HMM. Exact forward inference; learns emissions
    p(char|word,pos) and transitions p(next_word|word).

    Bootstrap (Phase 2 lesson): unsupervised discovery from a symmetric start
    fails. We exploit observable structure: streams start at a word boundary,
    words are 3 chars, emissions are deterministic. Group the first 300
    3-char windows by identity -> 4 word clusters -> initialize emit/trans.
    Then run unsupervised."""

    def __init__(self, bootstrap_words=300):
        self.emit = np.ones((NW, WL, NC))
        self.trans = np.ones((NW, NW))
        self.alpha = np.ones((NW, WL)) / (NW * WL)
        self._boot = bootstrap_words * WL
        self._buf = []
        self._booted = False

    def _do_bootstrap(self):
        chars = np.array(self._buf[:self._boot])
        nw = self._boot // WL
        # cluster 3-char windows by identity
        cmap = {}
        wseq = []
        for i in range(nw):
            w = tuple(chars[i*WL:(i+1)*WL])
            if w not in cmap:
                cmap[w] = len(cmap)
            wseq.append(cmap[w])
        # initialize emissions (deterministic: strong)
        self.emit = np.full((NW, WL, NC), 0.01)
        for w, wtup in enumerate(cmap):
            for p, c in enumerate(wtup):
                self.emit[w, p, c] = 10.0
        # initialize transitions from cluster sequence
        self.trans = np.ones((NW, NW))
        for a, b in zip(wseq[:-1], wseq[1:]):
            self.trans[a, b] += 1
        # reset alpha to uniform; start fresh after bootstrap
        self.alpha = np.ones((NW, WL)) / (NW * WL)
        self._booted = True

    def _trans_state(self, w, p):
        """Distribution over next (word,pos) from (w,p)."""
        out = np.zeros((NW, WL))
        if p < WL - 1:
            out[w, p + 1] = 1.0
        else:
            out[:, 0] = self.trans[w] / self.trans[w].sum()
        return out

    def predict(self):
        """p(next_char): propagate alpha one step, marginalize emissions."""
        # predict next state distribution
        pred = np.zeros((NW, WL))
        for w in range(NW):
            for p in range(WL):
                pred += self.alpha[w, p] * self._trans_state(w, p)
        # marginalize emissions
        pc = np.zeros(NC)
        for w in range(NW):
            for p in range(WL):
                e = self.emit[w, p] / self.emit[w, p].sum()
                pc += pred[w, p] * e
        return np.clip(pc, 1e-6, 1), pred

    def update(self, char):
        if not self._booted:
            self._buf.append(char)
            if len(self._buf) >= self._boot:
                self._do_bootstrap()
            return np.ones(NC) / NC
        pc, pred = self.predict()
        # Bayes: posterior ∝ pred * emission(char)
        post = np.zeros((NW, WL))
        for w in range(NW):
            for p in range(WL):
                e = self.emit[w, p, char] / self.emit[w, p].sum()
                post[w, p] = pred[w, p] * e
        post /= post.sum()
        # learn: SOFT posterior-weighted counts (online EM, stable).
        # hard MAP from a misaligned start never recovers.
        for w in range(NW):
            for p in range(WL):
                self.emit[w, p, char] += post[w, p]
        # word transitions: expected crossings (from pos WL-1)
        if post[:, 0].sum() > 0.5:  # likely at a boundary
            w_to = np.arange(NW)
            # distribute by posterior at pos 0 (to) weighted by pre-image
            for a in range(NW):
                for b in range(NW):
                    self.trans[a, b] += self.alpha[a, WL-1] * post[b, 0]
        self.alpha = post
        return pc


def ngram_bpb(train, test, order):
    counts = {}
    for i in range(len(train) - order):
        ctx = tuple(train[i:i + order])
        c = int(train[i + order])
        counts.setdefault(ctx, np.ones(NC))[c] += 1
    lps = []
    for i in range(len(test) - order):
        ctx = tuple(test[i:i + order])
        c = int(test[i + order])
        v = counts.get(ctx, np.ones(NC))
        p = v[c] / v.sum()
        lps.append(-np.log2(max(p, 1e-6)))
    return float(np.mean(lps))


def main():
    t0 = time.time()
    train, _ = gen_stream(7000, seed=7)   # 21k chars
    test, _ = gen_stream(2000, seed=99)   # 6k chars

    pop = HierPopulation()
    lps_pop = []
    for t, ch in enumerate(train):
        if t > 0:
            pc = pop.predict()[0]
            # (we predict before update; record on test only)
        pop.update(int(ch))
    for ch in test:
        pc = pop.predict()[0]
        lps_pop.append(-np.log2(pc[int(ch)]))
        pop.update(int(ch))

    out = {
        "protocol": "phase7a_text_like",
        "population_bpb": round(float(np.mean(lps_pop)), 4),
        "flat4_bpb": round(ngram_bpb(train, test, 4), 4),
        "flat6_bpb": round(ngram_bpb(train, test, 6), 4),
    }
    # generation: sample 30 chars from each model, check word validity
    rng = np.random.default_rng(0)
    # population generation
    pop2 = HierPopulation()
    for ch in train:
        pop2.update(int(ch))
    gen_chars = []
    for _ in range(30):
        pc = pop2.predict()[0]
        ch = int(rng.choice(NC, p=pc / pc.sum()))
        gen_chars.append(ch)
        pop2.update(ch)
    gen_words = [tuple(gen_chars[i:i+3]) for i in range(0, 30, 3)]
    valid = sum(1 for w in gen_words if any(np.array_equal(w, WORDS[i]) for i in range(NW)))
    out["pop_gen_valid_words"] = f"{valid}/10"
    out["pop_gen_sample"] = "".join("abcd"[c] for c in gen_chars)
    for k, v in out.items():
        if k != "protocol":
            print(f"{k:18s} {v}")
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open("runs/results_phase7a.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved runs/results_phase7a.json")


if __name__ == "__main__":
    main()
