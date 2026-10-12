"""TextCNN + BiLSTM-attention for RQ1/RQ2.  Run: python models.py   (Colab GPU: ~15-25 min for the full grid)

Grid: model {cnn, rnn} x input {response, prompt, both} x 3 seeds.
Writes results.csv (one row per run) and preds/<model>_<input>_<seed>.csv (test predictions, for confusion matrices / RQ4).
"""
import os, re, collections
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAXLEN, BATCH, EPOCHS, PATIENCE, LR = 400, 64, 15, 3, 1e-3
SEEDS = [0, 1, 2]
TOK = re.compile(r"\w+|[^\w\s]|\n")  # words, each punctuation/markdown char, and newlines; case is kept so formatting stays visible


def tokens(r, col):
    if col == "both":
        return TOK.findall(r.prompt)[:100] + ["[SEP]"] + TOK.findall(r.response)
    return TOK.findall(getattr(r, col))


class CNN(nn.Module):  # Kim (2014): parallel convs of width 3/4/5, max-over-time pooling, dropout, linear
    def __init__(s, V, C, E=128, F=128):
        super().__init__()
        s.emb = nn.Embedding(V, E, padding_idx=0)
        s.convs = nn.ModuleList(nn.Conv1d(E, F, k) for k in (3, 4, 5))
        s.drop, s.out = nn.Dropout(0.5), nn.Linear(3 * F, C)

    def forward(s, x):
        e = s.emb(x).transpose(1, 2)
        return s.out(s.drop(torch.cat([c(e).relu().amax(2) for c in s.convs], 1)))


class RNN(nn.Module):  # BiLSTM, additive attention pooling (weights can be inspected for RQ4), linear
    def __init__(s, V, C, E=128, H=128):
        super().__init__()
        s.emb = nn.Embedding(V, E, padding_idx=0)
        s.lstm = nn.LSTM(E, H, batch_first=True, bidirectional=True)
        s.att, s.drop, s.out = nn.Linear(2 * H, 1), nn.Dropout(0.5), nn.Linear(2 * H, C)

    def forward(s, x):
        L = (x != 0).sum(1).clamp(min=1).cpu()
        h = nn.utils.rnn.pack_padded_sequence(s.emb(x), L, batch_first=True, enforce_sorted=False)
        h, _ = nn.utils.rnn.pad_packed_sequence(s.lstm(h)[0], batch_first=True, total_length=x.size(1))
        a = s.att(h).squeeze(2).masked_fill(x == 0, -1e9).softmax(1)
        return s.out(s.drop((a.unsqueeze(2) * h).sum(1)))


def encode(seqs, vocab):
    X = np.zeros((len(seqs), MAXLEN), dtype=np.int64)
    for i, t in enumerate(seqs):
        ids = [vocab.get(w, 1) for w in t[:MAXLEN]]  # 0 = pad, 1 = unknown
        X[i, :len(ids)] = ids
    return torch.tensor(X)


@torch.no_grad()
def predict(m, X):
    m.eval()
    return torch.cat([m(X[i:i + 256].to(DEV)).argmax(1).cpu() for i in range(0, len(X), 256)]).numpy()


def train(M, V, C, Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    m = M(V, C).to(DEV)
    opt = torch.optim.Adam(m.parameters(), LR)
    best, bad, state = -1, 0, None
    for ep in range(EPOCHS):
        m.train()
        for idx in torch.randperm(len(Xtr)).split(BATCH):
            loss = nn.functional.cross_entropy(m(Xtr[idx].to(DEV)), ytr[idx].to(DEV))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5)
            opt.step()
        f = f1_score(yva, predict(m, Xva), average="macro")
        print(f"    epoch {ep + 1}: val macroF1={f:.3f}")
        if f > best:
            best, bad, state = f, 0, {k: v.clone() for k, v in m.state_dict().items()}
        elif (bad := bad + 1) == PATIENCE:
            break
    m.load_state_dict(state)  # early stopping: restore the epoch with the best validation F1
    return m


if __name__ == "__main__":
    d = pd.read_csv("data/dataset.csv" if os.path.exists("data/dataset.csv") else "dataset.csv")
    FAM = sorted(d.family.unique())
    y = torch.tensor(d.family.map(FAM.index).values)
    os.makedirs("preds", exist_ok=True)
    results = []
    for col in ["response", "prompt", "both"]:
        seqs = [tokens(r, col) for r in d.itertuples()]
        tr = (d.split == "train").values
        cnt = collections.Counter(w for s, t in zip(seqs, tr) if t for w in s)
        vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common(50_000)) if c >= 2}
        X = encode(seqs, vocab)
        S = {s: (d.split == s).values for s in ["train", "val", "test"]}
        for name, M in [("cnn", CNN), ("rnn", RNN)]:
            for seed in SEEDS:
                print(f"{name} | input={col} | seed={seed}")
                m = train(M, len(vocab) + 2, len(FAM), X[S["train"]], y[S["train"]], X[S["val"]], y[S["val"]], seed)
                p, t = predict(m, X[S["test"]]), y[S["test"]].numpy()
                results.append(dict(model=name, input=col, seed=seed,
                                    acc=accuracy_score(t, p), macro_f1=f1_score(t, p, average="macro")))
                print(f"  TEST acc={results[-1]['acc']:.3f} macroF1={results[-1]['macro_f1']:.3f}")
                pd.DataFrame({"true": [FAM[i] for i in t], "pred": [FAM[i] for i in p]}).to_csv(
                    f"preds/{name}_{col}_{seed}.csv", index=False)
                pd.DataFrame(results).to_csv("results.csv", index=False)
    r = pd.DataFrame(results)
    print(r.groupby(["model", "input"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(3))
