"""OPTIONAL follow-up neural runs (GPU; about 40 min on a Colab T4). Run from the repo root:  python extra/neural_extra.py
Not used for the numbers in the report, which rely only on saved predictions and CPU baselines.
  lc_*      learning curve: CNN trained on 25/50/75% of the training prompts (does more data still help?)
  pos_*     targeted position ablations: CNN without the first 10 tokens, or with tokens shuffled only inside each line
            (keeps where each line sits, destroys the phrases), or with whole lines shuffled (keeps phrases, destroys position)
  ctl_original  a fresh control run of the unmodified setup on the same machine, to compare against
Appends to results/neural.csv and results/neural_preds.csv like rq34.py."""
import os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from rq34 import d, SPLIT, run, TOK  # noqa: E402  (loads the dataset and split masks)
from models import CNN  # noqa: E402

rng = np.random.default_rng(0)


def shuffle_within_lines(s):
    return [t for line in s.split("\n") for t in (list(rng.permutation(TOK.findall(line))) + ["\n"])]


def shuffle_lines(s):
    lines = s.split("\n")
    return [t for line in rng.permutation(lines) for t in TOK.findall(line) + ["\n"]]


if __name__ == "__main__":
    base = [TOK.findall(r) for r in d.response]
    run("ctl_original", base, SPLIT["train"], SPLIT["val"], {"test": SPLIT["test"]}, [("cnn", CNN)])
    prompts = np.random.default_rng(42).permutation(d[SPLIT["train"]].prompt.unique())
    for frac in [0.25, 0.5, 0.75]:
        keep = d.prompt.isin(set(prompts[: int(frac * len(prompts))])).values & SPLIT["train"]
        run(f"lc_{int(frac * 100)}", base, keep, SPLIT["val"], {"test": SPLIT["test"]}, [("cnn", CNN)])
    for name, f in {"pos_drop_first10": lambda s: TOK.findall(s)[10:],
                    "pos_shuffle_within_lines": shuffle_within_lines,
                    "pos_shuffle_lines": shuffle_lines}.items():
        run(name, [f(r) for r in d.response], SPLIT["train"], SPLIT["val"], {"test": SPLIT["test"]}, [("cnn", CNN)])
