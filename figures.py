"""Report figures + tables from results/. Run after baseline.py, features.py and the Colab runs (results.csv, preds/).
Neural RQ3/RQ4 figures are drawn only once results/neural.csv exists."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e4e3df"
COL = {"TF-IDF char": "#2a78d6", "CNN": "#eb6834", "RNN": "#1baf7a"}  # validated categorical slots 1-3
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "axes.axisbelow": True})
os.makedirs("figures", exist_ok=True)
FAM = ["Claude", "GPT", "Gemini", "Llama", "Mistral"]
save = lambda f: (plt.tight_layout(), plt.savefig(f"figures/{f}", dpi=200), plt.close())

base = pd.read_csv("results/baseline.csv")
nn = pd.read_csv("results.csv")  # written by models.py


def bars(ax, groups, series, chance=0.2):
    """groups: x labels; series: {name: (means, stds)}."""
    w = 0.8 / len(series)
    for i, (name, (m, s)) in enumerate(series.items()):
        x = np.arange(len(groups)) + (i - (len(series) - 1) / 2) * w
        ax.bar(x, m, w - 0.03, yerr=s, color=COL[name], label=name, capsize=2, error_kw={"ecolor": MUTED, "lw": 1})
        for xi, mi in zip(x, m):
            ax.text(xi, mi + 0.015, f"{mi:.0%}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(chance, color=MUTED, ls="--", lw=1, label="chance (20%)")
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.legend(frameon=False, ncol=len(series) + 1, loc="upper right", fontsize=9)


# ---- Table 1 + Fig 1: RQ1/RQ2 accuracy by input setting ----
inputs = {"response": "Output only", "prompt": "Input only", "both": "Input + output"}
agg = nn.groupby(["model", "input"])[["acc", "macro_f1"]].agg(["mean", "std"])
t1 = []
for col, lab in inputs.items():
    for f in ["word", "char"]:
        r = base[(base.exp == "rq12") & (base.model == f"tfidf-{f}") & (base.input == col)].iloc[0]
        t1.append((lab, f"TF-IDF {f} + LogReg", f"{r.acc:.3f}", f"{r.macro_f1:.3f}"))
    for m in ["cnn", "rnn"]:
        a = agg.loc[(m, col)]
        t1.append((lab, {"cnn": "TextCNN", "rnn": "BiLSTM+attention"}[m],
                   f"{a[('acc', 'mean')]:.3f} ± {a[('acc', 'std')]:.3f}", f"{a[('macro_f1', 'mean')]:.3f} ± {a[('macro_f1', 'std')]:.3f}"))
pd.DataFrame(t1, columns=["Input", "Model", "Accuracy", "Macro-F1"]).to_csv("figures/table1_rq1_rq2.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 3.6))
ch = base[(base.exp == "rq12") & (base.model == "tfidf-char")].set_index("input").acc
bars(ax, list(inputs.values()), {
    "TF-IDF char": ([ch[c] for c in inputs], [0] * 3),
    "CNN": ([agg.loc[("cnn", c), ("acc", "mean")] for c in inputs], [agg.loc[("cnn", c), ("acc", "std")] for c in inputs]),
    "RNN": ([agg.loc[("rnn", c), ("acc", "mean")] for c in inputs], [agg.loc[("rnn", c), ("acc", "std")] for c in inputs])})
ax.set_ylabel("Test accuracy (5 families)")
ax.set_title("What the model sees vs. how well it identifies the LLM", loc="left", fontsize=11)
save("fig1_rq2_inputs.png")

# ---- Fig 2: RQ1 confusion matrices (row-normalised, 3 seeds pooled) ----
preds = {"TF-IDF char": pd.read_csv("results/baseline_preds.csv").query("exp=='rq12' and model=='tfidf-char' and input=='response'"),
         "CNN": pd.concat(pd.read_csv(f"preds/cnn_response_{s}.csv") for s in range(3)),
         "RNN": pd.concat(pd.read_csv(f"preds/rnn_response_{s}.csv") for s in range(3))}
fig, axs = plt.subplots(1, 3, figsize=(11, 3.7))
for ax, (name, p) in zip(axs, preds.items()):
    cm = confusion_matrix(p.true, p.pred, labels=FAM, normalize="true")
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > .5 else INK)
    ax.set_xticks(range(5), FAM, rotation=30); ax.set_yticks(range(5), FAM); ax.grid(False)
    ax.set_xlabel("Predicted"); ax.set_title(name, fontsize=10)
axs[0].set_ylabel("True family")
save("fig2_rq1_confusion.png")

# ---- Fig 3: RQ4 stylistic features by family ----
fb = pd.read_csv("results/features_by_family.csv", index_col=0)
show = {"words": "Response length (words)", "bold": "**Bold** spans per reply", "opens_affirm": "Opens with 'Sure'/'Certainly'…",
        "closing_offer": "Ends with 'let me know'/'hope this helps'", "words_per_sentence": "Words per sentence", "commas": "Commas per 100 words"}
fig, axs = plt.subplots(2, 3, figsize=(10, 5))
for ax, (k, lab) in zip(axs.flat, show.items()):
    v = fb.loc[k, FAM]
    ax.bar(FAM, v, color=COL["TF-IDF char"], width=0.6)
    for i, x in enumerate(v):
        ax.text(i, x, f"{x:.0%}" if k in ("opens_affirm", "closing_offer") else f"{x:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(lab, fontsize=9, loc="left"); ax.tick_params(axis="x", labelsize=8)
    if k in ("opens_affirm", "closing_offer"):
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
save("fig3_rq4_style_features.png")

# ---- Fig 4/5: neural RQ3 + RQ4 (after `python rq34.py` on Colab) ----
if os.path.exists("results/neural.csv"):
    r = pd.read_csv("results/neural.csv")
    tasks = ["coding", "creative", "math"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    g = r[r.exp.str.startswith("rq3")].groupby(["exp", "model", "test"]).acc.agg(["mean", "std"])
    # x = held-out task; bars = CNN/RNN tested in-domain (solid) vs on the unseen task (hatched)
    w, x0 = 0.2, np.arange(len(tasks))
    for k, (m, name, test) in enumerate([(m, n, t) for m, n in [("cnn", "CNN"), ("rnn", "RNN")] for t in ["in_domain", "held_out_task"]]):
        mu = [g.loc[(f"rq3-holdout-{t}", m, test), "mean"] for t in tasks]
        sd = [g.loc[(f"rq3-holdout-{t}", m, test), "std"] for t in tasks]
        x = x0 + (k - 1.5) * w
        ax.bar(x, mu, w - 0.03, yerr=sd, color=COL[name], alpha=1 if test == "in_domain" else 0.45,
               hatch=None if test == "in_domain" else "//", edgecolor="white", capsize=2, error_kw={"ecolor": MUTED, "lw": 1},
               label=f"{name}, {'in-domain' if test == 'in_domain' else 'unseen task'}")
        for xi, mi in zip(x, mu):
            ax.text(xi, mi + 0.015, f"{mi:.0%}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.2, color=MUTED, ls="--", lw=1, label="chance (20%)")
    ax.set_xticks(x0, [f"held out: {t}" for t in tasks]); ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right"); ax.set_ylabel("Test accuracy")
    save("fig4_rq3_cross_task.png")

    a = r[r.exp.str.startswith("rq4")].assign(v=lambda x: x.exp.str[4:]).groupby("v").acc.agg(["mean", "std"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    abl = ["original", "lowercase", "no_newlines", "words_only", "shuffled"]
    names = {"original": "original", "lowercase": "lowercased", "no_newlines": "line breaks removed",
             "words_only": "punctuation + markdown removed", "shuffled": "word order shuffled"}
    ax1.barh([names[k] for k in abl[::-1]], a.loc[abl[::-1], "mean"], xerr=a.loc[abl[::-1], "std"], color=COL["CNN"], height=0.6,
             error_kw={"ecolor": MUTED, "lw": 1})
    for i, (v, sd) in enumerate(a.loc[abl[::-1], ["mean", "std"]].values):
        ax1.text(v + sd + 0.015, i, f"{v:.1%}", va="center", fontsize=8)
    ax1.set_xlim(0, 1); ax1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax1.grid(axis="x", color=GRID); ax1.grid(axis="y", visible=False); ax1.set_title("CNN retrained on ablated text", fontsize=10, loc="left")
    n = [10, 25, 50, 100, 200, 400]
    mu = [a.loc[f"first_{k}", "mean"] if k < 400 else a.loc["original", "mean"] for k in n]
    ax2.plot(n, mu, color=COL["CNN"], lw=2, marker="o", ms=5)
    for k, v in zip(n, mu):
        ax2.annotate(f"{v:.0%}", (k, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax2.set_xscale("log"); ax2.set_xticks(n, n); ax2.minorticks_off(); ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax2.set_xlabel("Tokens of the response seen"); ax2.set_title("Accuracy vs. response length", fontsize=10, loc="left")
    save("fig5_rq4_ablations.png")
print(sorted(os.listdir("figures")))
