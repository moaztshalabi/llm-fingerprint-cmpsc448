"""Figures for the revised report, drawn from results/extra_stats.json and the saved predictions.
Run from the repo root after extra/stats.py:   python extra/figures_extra.py
Same look as figures.py; error bars are now 95% bootstrap intervals over test prompts, not seed std."""
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e4e3df"
COL = {"TF-IDF char": "#2a78d6", "CNN": "#eb6834", "RNN": "#1baf7a"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "axes.axisbelow": True})
FAM = ["Claude", "GPT", "Gemini", "Llama", "Mistral"]
S = json.load(open("results/extra_stats.json"))
pct = matplotlib.ticker.PercentFormatter(100)


def save(f):
    plt.tight_layout(); plt.savefig(f"figures/{f}", dpi=200); plt.close()


def err(key):  # [mean, lo, hi] -> mean, (mean-lo, hi-mean)
    m, lo, hi = S[key]
    return m, (m - lo, hi - m)


# ---- fig6: every RQ1 model with a 95% interval; same-input comparisons sit next to each other ----
rows = [("Word TF-IDF, sklearn default tokens", "word_response", COL["TF-IDF char"], "full response"),
        ("Word TF-IDF, CNN tokens", "wordtok_full_C4", COL["TF-IDF char"], "full response"),
        ("Char TF-IDF", "char_response", COL["TF-IDF char"], "full response"),
        ("Word TF-IDF, CNN tokens", "wordtok_400_C4", COL["TF-IDF char"], "first 400 tokens"),
        ("Char TF-IDF", "char_400_C4", COL["TF-IDF char"], "first 400 tokens"),
        ("TextCNN", "cnn_response", COL["CNN"], "first 400 tokens"),
        ("BiLSTM + attention", "rnn_response", COL["RNN"], "first 400 tokens")][::-1]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for i, (lab, k, c, inp) in enumerate(rows):
    m, e = err(k)
    full = inp == "full response"
    ax.errorbar(m, i, xerr=[[e[0]], [e[1]]], fmt="o", color=c, mfc="white" if full else c, mew=1.6, ms=7, capsize=3, lw=1.4)
    ax.text(S[k][2] + 0.4, i, f"{m:.1f}%", va="center", fontsize=8.5, color=INK)
ax.set_yticks(range(len(rows)), [f"{lab}  ({inp})" for lab, _, _, inp in rows], fontsize=8.5)
ax.set_xlim(55, 72); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100, decimals=0))
ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
ax.set_xlabel("Test accuracy, 95% interval (hollow = sees the full response)")
ax.set_title("RQ1 output-only accuracy, matched inputs", loc="left", fontsize=10.5)
save("fig6_rq1_fair_comparison.png")

# ---- fig2b: confusion matrices (3 seeds pooled for neural) + per-family recall with intervals ----
bp = pd.read_csv("results/baseline_preds.csv")
preds = {"TF-IDF char": bp.query("exp=='rq12' and model=='tfidf-char' and input=='response'"),
         "CNN": pd.concat(pd.read_csv(f"preds/cnn_response_{s}.csv") for s in range(3)),
         "RNN": pd.concat(pd.read_csv(f"preds/rnn_response_{s}.csv") for s in range(3))}
fig, axs = plt.subplots(2, 2, figsize=(9, 8))
for ax, (name, p) in zip(axs.flat, preds.items()):
    cm = confusion_matrix(p.true, p.pred, labels=FAM, normalize="true")
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center", fontsize=10, color="white" if cm[i, j] > .5 else INK)
    ax.set_xticks(range(5), FAM, rotation=30); ax.set_yticks(range(5), FAM); ax.grid(False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True family"); ax.set_title(name if name != "RNN" else "BiLSTM", fontsize=11, loc="left")
ax = axs[1, 1]
for j, (name, key, c) in enumerate([("TF-IDF char", "char_response", COL["TF-IDF char"]), ("CNN", "cnn_response", COL["CNN"]),
                                    ("BiLSTM", "rnn_response", COL["RNN"])]):
    for i, f in enumerate(FAM):
        m, e = err(f"recall {key} {f}")
        ax.errorbar(i + (j - 1) * 0.22, m, yerr=[[e[0]], [e[1]]], fmt="o", color=c, ms=5, capsize=2, lw=1.2, label=name if i == 0 else None)
ax.set_xticks(range(5), FAM, rotation=30); ax.set_ylim(30, 100); ax.yaxis.set_major_formatter(pct)
ax.axhline(20, color=MUTED, ls="--", lw=1)
ax.set_title("Recall per family, 95% interval", fontsize=11, loc="left"); ax.legend(frameon=False, fontsize=9, loc="upper right")
save("fig2b_rq1_confusion_recall.png")

# ---- fig1b: RQ2 bars with 95% intervals ----
inputs = {"response": "Output only", "prompt": "Input only", "both": "Input + output"}
fig, ax = plt.subplots(figsize=(7, 3.6))
w = 0.8 / 3
for i, (name, pre) in enumerate([("TF-IDF char", "char"), ("CNN", "cnn"), ("RNN", "rnn")]):
    x = np.arange(3) + (i - 1) * w
    ms, es = zip(*[err(f"{pre}_{c}") for c in inputs])
    ax.bar(x, ms, w - 0.03, yerr=np.array(es).T, color=COL[name], label=name if name != "RNN" else "BiLSTM", capsize=2,
           error_kw={"ecolor": MUTED, "lw": 1})
    for xi, mi, ei in zip(x, ms, es):
        ax.text(xi, mi + ei[1] + 1.2, f"{mi:.0f}%", ha="center", va="bottom", fontsize=8)
ax.axhline(20, color=MUTED, ls="--", lw=1, label="chance (20%)")
ax.set_xticks(range(3), list(inputs.values())); ax.set_ylim(0, 100); ax.yaxis.set_major_formatter(pct)
ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9); ax.set_ylabel("Test accuracy (5 families)")
ax.set_title("RQ2: what the model sees vs. how well it identifies the LLM (95% intervals)", loc="left", fontsize=10.5)
save("fig1b_rq2_inputs_ci.png")

# ---- fig5b: CNN ablations and length curve with 95% intervals ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
abl = ["original", "lowercase", "no_newlines", "words_only", "shuffled"][::-1]
names = {"original": "original", "lowercase": "lowercased", "no_newlines": "line breaks removed",
         "words_only": "punctuation + markdown removed", "shuffled": "word order shuffled"}
for i, k in enumerate(abl):
    m, e = err(f"rq4 acc {k}")
    ax1.barh(i, m, height=0.6, color=COL["CNN"], xerr=[[e[0]], [e[1]]], error_kw={"ecolor": MUTED, "lw": 1})
    ax1.text(S[f"rq4 acc {k}"][2] + 1.2, i, f"{m:.1f}%", va="center", fontsize=8)
ax1.set_yticks(range(len(abl)), [names[k] for k in abl])
ax1.set_xlim(0, 100); ax1.xaxis.set_major_formatter(pct)
ax1.grid(axis="x", color=GRID); ax1.grid(axis="y", visible=False); ax1.set_title("CNN retrained on ablated text", fontsize=10, loc="left")
n = [10, 25, 50, 100, 200, 400]
vals = [S[f"rq4 acc first_{k}"] if k < 400 else S["rq4 acc original"] for k in n]
mu, lo, hi = map(np.array, zip(*vals))
ax2.fill_between(n, lo, hi, color=COL["CNN"], alpha=0.18, lw=0)
ax2.plot(n, mu, color=COL["CNN"], lw=2, marker="o", ms=5)
for k, v in zip(n, mu):
    ax2.annotate(f"{v:.0f}%", (k, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax2.set_xscale("log"); ax2.set_xticks(n, n); ax2.minorticks_off(); ax2.set_ylim(0, 100)
ax2.yaxis.set_major_formatter(pct)
ax2.set_xlabel("Tokens seen (CNN retrained at each length)"); ax2.set_title("Accuracy vs. response length", fontsize=10, loc="left")
save("fig5b_rq4_ablations_ci.png")
print("ok")
