"""Fairer TF-IDF baselines, run from the repo root:  python extra/baseline_fair.py
Fixes two mismatches with the neural models: (1) the baselines saw the whole response while the CNN/BiLSTM saw
only the first 400 tokens; (2) the word baseline used sklearn's default tokenizer, which drops punctuation, markdown
and newlines. Here every model can see exactly the tokens the CNN sees (regex \\w+|[^\\w\\s]|\\n, case kept).
C is chosen on the validation split from {1, 4, 16} for the headline configs; the rest use C=4.
Writes results/extra_baseline.csv (val/test scores per config and C) and results/extra_baseline_preds.csv."""
import re, sys, time, random
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

TOK = re.compile(r"\w+|[^\w\s]|\n")
d = pd.read_csv("data/dataset.csv")
tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))


def head(s, n):
    ms = list(TOK.finditer(s))
    return s[:ms[n - 1].end()] if len(ms) >= n else s


def both400(p, r):  # first 100 prompt tokens + [SEP] + response, 400 tokens in all, as in models.py
    pt = len(TOK.findall(p)[:100])
    return head(p, 100) + " [SEP] " + head(r, 400 - pt - 1)


def drop_first(s, n=10):
    ms = list(TOK.finditer(s)); return s[ms[n].start():] if len(ms) > n else ""


def drop_last(s, n=10):
    ms = list(TOK.finditer(s)); return s[:ms[-n - 1].end()] if len(ms) > n else ""


def last(s, n=10):
    ms = list(TOK.finditer(s)); return s[ms[-n].start():] if len(ms) >= n else s


_rng = np.random.default_rng(0)
def shuffled(s):
    t = TOK.findall(s); return " ".join(_rng.permutation(t)) if t else s


CHAR = dict(analyzer="char", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True, lowercase=False)
WTOK = dict(token_pattern=r"\w+|[^\w\s]|\n", lowercase=False, ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
UNI = dict(WTOK, ngram_range=(1, 1))
R = lambda f: (lambda x: x.response.map(f))                      # response transform
B = lambda x: pd.Series([both400(p, r) for p, r in zip(x.prompt, x.response)], index=x.index)

SWEEP = [1, 4, 16]
CONFIGS = [  # name, vectorizer kwargs, text function, C values
    ("wordtok_full",  WTOK, R(lambda s: s),            SWEEP),
    ("wordtok_400",   WTOK, R(lambda s: head(s, 400)), SWEEP),
    ("char_400",      CHAR, R(lambda s: head(s, 400)), SWEEP),
    ("char_full",     CHAR, R(lambda s: s),            SWEEP),
    ("wordtok_both_400", WTOK, B, [4]),
    ("char_both_400",    CHAR, B, [4]),
    # order / position probes on the word-token model, full length
    ("wordtok_unigram_full", UNI, R(lambda s: s),      [4]),
    ("wordtok_shuffled_full", WTOK, R(shuffled),       [4]),
    ("wordtok_first10_full", WTOK, R(lambda s: head(s, 10)), [4]),
    ("wordtok_last10_full",  WTOK, R(last),            [4]),
    ("wordtok_dropfirst10_full", WTOK, R(drop_first),  [4]),
    ("wordtok_droplast10_full",  WTOK, R(drop_last),   [4]),
]


def run(i):  # index, so the worker pickles an int, not a lambda
    name, vk, tf, Cs = CONFIGS[i]
    t0 = time.time()
    vec = TfidfVectorizer(**vk)
    Xtr = vec.fit_transform(tf(tr)); Xva, Xte = vec.transform(tf(va)), vec.transform(tf(te))
    rows, preds = [], []
    for C in Cs:
        clf = LogisticRegression(max_iter=2000, C=C).fit(Xtr, tr.family)
        for sp, X, y in (("val", Xva, va.family), ("test", Xte, te.family)):
            p = clf.predict(X)
            rows.append(dict(config=name, C=C, split=sp, acc=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro")))
            if sp == "test":
                preds.append(pd.DataFrame(dict(config=name, C=C, row=np.arange(len(y)), true=y.values, pred=p)))
        print(f"{name} C={C} done {time.time() - t0:.0f}s  val={rows[-2]['acc']:.3f} test={rows[-1]['acc']:.3f}", flush=True)
    return pd.DataFrame(rows), pd.concat(preds)


if __name__ == "__main__":
    only = sys.argv[1:]
    idx = [i for i, c in enumerate(CONFIGS) if not only or c[0] in only]
    with ProcessPoolExecutor(2) as ex:
        out = list(ex.map(run, idx))
    pd.concat(o[0] for o in out).to_csv("results/extra_baseline.csv", index=False)
    pd.concat(o[1] for o in out).to_csv("results/extra_baseline_preds.csv", index=False)
