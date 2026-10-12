"""TF-IDF + logistic regression baseline.
RQ1/RQ2: input setting {response, prompt, both} x features {word 1-2gram, char 2-5gram}.
RQ3: char model trained with one task held out, tested in-domain and on the held-out task.
Writes results/baseline.csv and results/baseline_preds.csv."""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

d = pd.read_csv("data/dataset.csv")
d["both"] = d.prompt + " [SEP] " + d.response
os.makedirs("results", exist_ok=True)
VEC = {"word": lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True),
       "char": lambda: TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, max_features=300_000,
                                       sublinear_tf=True, lowercase=False)}
rows, preds = [], []


def fit_eval(exp, feat, col, train, tests):
    vec = VEC[feat]()
    clf = LogisticRegression(max_iter=2000, C=4).fit(vec.fit_transform(train[col]), train.family)
    for tname, te in tests.items():
        p = clf.predict(vec.transform(te[col]))
        rows.append(dict(exp=exp, model=f"tfidf-{feat}", input=col, test=tname,
                         acc=accuracy_score(te.family, p), macro_f1=f1_score(te.family, p, average="macro")))
        preds.append(pd.DataFrame(dict(exp=exp, model=f"tfidf-{feat}", input=col, test=tname, true=te.family, pred=p)))
        print(rows[-1])


tr, te = d[d.split == "train"], d[d.split == "test"]
for col in ["response", "prompt", "both"]:
    for feat in VEC:
        fit_eval("rq12", feat, col, tr, {"test": te})

for task in ["coding", "creative", "math"]:  # RQ3: train without `task`, then test on every row of it
    out = d.task == task
    fit_eval(f"rq3-holdout-{task}", "char", "response", d[~out & (d.split == "train")],
             {"in_domain": d[~out & (d.split == "test")], "held_out_task": d[out]})

pd.DataFrame(rows).to_csv("results/baseline.csv", index=False)
pd.concat(preds).to_csv("results/baseline_preds.csv", index=False)
