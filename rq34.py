"""RQ3 + RQ4 neural experiments. Needs models.py + dataset.csv next to it.
  python rq34.py rq3   cross-task: train without one task, test in-domain vs on the unseen task (CNN + RNN, 3 seeds)
  python rq34.py rq4   ablations: retrain CNN on perturbed/truncated responses (3 seeds)
Appends to results/neural.csv and results/neural_preds.csv (the two parts can run in separate sessions)."""
import os, re, sys, collections
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
from models import CNN, RNN, TOK, SEEDS, encode, predict, train

d = pd.read_csv("data/dataset.csv" if os.path.exists("data/dataset.csv") else "dataset.csv")
FAM = sorted(d.family.unique())
y = torch.tensor(d.family.map(FAM.index).values)
SPLIT = {s: (d.split == s).values for s in ["train", "val", "test"]}
os.makedirs("results", exist_ok=True)


def run(exp, seqs, train_mask, val_mask, tests, models):
    cnt = collections.Counter(w for s, t in zip(seqs, train_mask) if t for w in s)
    vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common(50_000)) if c >= 2}
    X = encode(seqs, vocab)
    for name, M in models:
        for seed in SEEDS:
            print(f"{exp} | {name} | seed={seed}")
            m = train(M, len(vocab) + 2, len(FAM), X[train_mask], y[train_mask], X[val_mask], y[val_mask], seed)
            res, prd = [], []
            for tname, mask in tests.items():
                p, t = predict(m, X[mask]), y[mask].numpy()
                res.append(dict(exp=exp, model=name, seed=seed, test=tname,
                                acc=accuracy_score(t, p), macro_f1=f1_score(t, p, average="macro")))
                prd.append(pd.DataFrame(dict(exp=exp, model=name, seed=seed, test=tname,
                                             true=[FAM[i] for i in t], pred=[FAM[i] for i in p])))
                print(f"  {tname}: acc={res[-1]['acc']:.3f}")
            for df, f in [(pd.DataFrame(res), "results/neural.csv"), (pd.concat(prd), "results/neural_preds.csv")]:
                df.to_csv(f, mode="a", header=not os.path.exists(f), index=False)  # save after every run


def rq3():
    seqs = [TOK.findall(r) for r in d.response]
    for task in ["coding", "creative", "math"]:
        out = (d.task == task).values
        run(f"rq3-holdout-{task}", seqs, SPLIT["train"] & ~out, SPLIT["val"] & ~out,
            {"in_domain": SPLIT["test"] & ~out, "held_out_task": out}, [("cnn", CNN), ("rnn", RNN)])


def rq4():
    rng = np.random.default_rng(0)
    shuffle = lambda t: list(rng.permutation(t)) if t else t
    perturb = {  # each removes one kind of information; the accuracy drop shows how much the model relied on it
        "original":     lambda s: TOK.findall(s),
        "lowercase":    lambda s: TOK.findall(s.lower()),
        "no_newlines":  lambda s: TOK.findall(" ".join(s.split())),           # removes layout, keeps words and punctuation
        "words_only":   lambda s: re.findall(r"\w+", s),                      # removes punctuation, markdown and layout
        "shuffled":     lambda s: shuffle(TOK.findall(s)),                    # removes word order, keeps the bag of tokens
        **{f"first_{n}": (lambda n: lambda s: TOK.findall(s)[:n])(n) for n in [10, 25, 50, 100, 200]},
    }
    for name, f in perturb.items():
        run(f"rq4-{name}", [f(r) for r in d.response], SPLIT["train"], SPLIT["val"],
            {"test": SPLIT["test"]}, [("cnn", CNN)])


if __name__ == "__main__":
    for part in sys.argv[1:] or ["rq3", "rq4"]:
        {"rq3": rq3, "rq4": rq4}[part]()
    r = pd.read_csv("results/neural.csv")
    print(r.groupby(["exp", "model", "test"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(3).to_string())
