"""Uncertainty and control analyses built from saved predictions (no retraining). Run from the repo root after
baseline.py, models.py, rq34.py and extra/baseline_fair.py:   python extra/stats.py
Everything uses a bootstrap that resamples test PROMPTS (a prompt can have several responses in the test set),
2000 resamples, 95% percentile intervals. For the neural models a row's correctness is averaged over the 3 seeds.
Prints a report and writes results/extra_stats.json."""
import json, re
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

B, RNG = 2000, np.random.default_rng(0)
FAM = ["Claude", "GPT", "Gemini", "Llama", "Mistral"]
TOK = re.compile(r"\w+|[^\w\s]|\n")
d = pd.read_csv("data/dataset.csv")
te = d[d.split == "test"].reset_index(drop=True)
OUT = {}


def q(df, **kw):  # equality filter
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).values
    return df[m]


def boot(vals, groups):
    """Mean of `vals` with a cluster bootstrap over `groups`. Returns (mean, lo, hi)."""
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    _, g = np.unique(groups, return_inverse=True)
    S, N = np.bincount(g, vals), np.bincount(g)
    idx = RNG.integers(0, len(S), (B, len(S)))
    bs = S[idx].sum(1) / N[idx].sum(1)
    return vals.mean(), *np.percentile(bs, [2.5, 97.5])


def fmt(t, pct=True):
    m, lo, hi = t
    return f"{100*m:.1f} [{100*lo:.1f}, {100*hi:.1f}]" if pct else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def report(key, t):
    OUT[key] = [100 * float(x) for x in t]
    print(f"  {key:<48} {fmt(t)}")


# ---------------- correctness vectors on the test rows ----------------
corr = {}
for m in ["cnn", "rnn"]:
    for inp in ["response", "prompt", "both"]:
        runs = [pd.read_csv(f"preds/{m}_{inp}_{s}.csv") for s in range(3)]
        assert all((r.true.values == te.family.values).all() for r in runs)
        corr[f"{m}_{inp}"] = np.mean([(r.true == r.pred).values for r in runs], 0)
        corr[f"{m}_{inp}_runs"] = [(r.true == r.pred).values for r in runs]
bp = pd.read_csv("results/baseline_preds.csv")
for f in ["word", "char"]:
    for inp in ["response", "prompt", "both"]:
        x = q(bp, exp="rq12", model=f"tfidf-{f}", input=inp)
        assert (x.true.values == te.family.values).all()
        corr[f"{f}_{inp}"] = (x.true == x.pred).values.astype(float)
xb = pd.read_csv("results/extra_baseline.csv")
xp = pd.read_csv("results/extra_baseline_preds.csv")
chosen = {}
for cfg, g in xb.groupby("config"):
    v = g[g.split == "val"]
    C = v.loc[v.macro_f1.idxmax(), "C"]  # C chosen on validation macro-F1
    chosen[cfg] = int(C)
    x = xp[(xp.config == cfg) & (xp.C == C)].sort_values("row")
    assert (x.true.values == te.family.values).all()
    corr[cfg] = (x.true == x.pred).values.astype(float)
    x4 = xp[(xp.config == cfg) & (xp.C == 4)].sort_values("row")
    corr[cfg + "_C4"] = (x4.true == x4.pred).values.astype(float)
OUT["chosen_C"] = chosen
grp = te.prompt.values

print("\n=== RQ1/RQ2 accuracy, 95% CI (prompt-cluster bootstrap) ===")
for k in ["word_response", "char_response", "cnn_response", "rnn_response", "word_prompt", "char_prompt", "cnn_prompt",
          "rnn_prompt", "word_both", "char_both", "cnn_both", "rnn_both"] + sorted(chosen) + [c + "_C4" for c in sorted(chosen)]:
    report(k, boot(corr[k], grp))
print("  chosen C on validation:", chosen)

print("\n=== Paired differences (A minus B, points) ===")
pairs = [("cnn_response", "rnn_response"), ("char_response", "cnn_response"), ("char_response", "rnn_response"),
         ("char_response", "word_response"), ("wordtok_full", "word_response"), ("wordtok_full", "char_full"),
         ("char_400", "cnn_response"), ("wordtok_400", "cnn_response"), ("char_full", "char_400"),
         ("wordtok_full", "wordtok_400"), ("cnn_response", "cnn_both"), ("rnn_response", "rnn_both"),
         ("char_response", "char_both"), ("char_400", "char_both_400"), ("wordtok_400", "wordtok_both_400"),
         ("wordtok_full", "wordtok_unigram_full"), ("wordtok_full", "wordtok_shuffled_full"),
         ("wordtok_full", "wordtok_dropfirst10_full"), ("wordtok_full", "wordtok_droplast10_full"),
         ("wordtok_first10_full", "wordtok_last10_full")]
for a, b in pairs:
    report(f"{a} - {b}", boot(corr[a] - corr[b], grp))

print("\n=== McNemar exact test, CNN seed s vs char baseline / vs BiLSTM seed s ===")
mc = {}
for s in range(3):
    c, r, ch = corr["cnn_response_runs"][s], corr["rnn_response_runs"][s], corr["char_response"].astype(bool)
    for name, a, b in [("char_vs_cnn", ch, c), ("cnn_vs_rnn", c, r)]:
        n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue
        mc[f"{name}_seed{s}"] = dict(a_only=n10, b_only=n01, p=p)
        print(f"  {name} seed {s}: A right/B wrong {n10}, B right/A wrong {n01}, p = {p:.3g}")
OUT["mcnemar"] = mc

print("\n=== Seed / run variance of the CNN (output only) ===")
nn = pd.read_csv("results.csv").query("model=='cnn' and input=='response'").acc.values
r4 = pd.read_csv("results/neural.csv").query("exp=='rq4-original'").acc.values
both = np.r_[nn, r4]
print(f"  models.py runs {np.round(nn*100,1)}  rq34 'original' runs (same config, same seeds) {np.round(r4*100,1)}")
print(f"  pooled 6 runs: mean {100*both.mean():.1f}, sd {100*both.std(ddof=1):.1f}; rq34 alone sd {100*r4.std(ddof=1):.1f}")
OUT["cnn_runs"] = dict(models_py=list(nn), rq34=list(r4), pooled_mean=both.mean(), pooled_sd=both.std(ddof=1))

print("\n=== Input-only vs chance (binomial test against 20%, rows treated as independent) ===")
for k in ["word_prompt", "char_prompt", "cnn_prompt", "rnn_prompt"]:
    k_right = int(round(corr[k].sum())) if not k.startswith(("cnn", "rnn")) else None
    if k_right is not None:
        p = binomtest(k_right, len(te), 0.2, alternative="greater").pvalue
        print(f"  {k}: {k_right}/{len(te)} right, one-sided p = {p:.3g}")
        OUT[f"{k}_binom_p"] = p
    else:
        ps = [binomtest(int(r.sum()), len(te), 0.2, alternative="greater").pvalue for r in corr[k + "_runs"]]
        print(f"  {k}: per-seed one-sided p = {[round(x, 3) for x in ps]}")
        OUT[f"{k}_binom_p"] = ps

print("\n=== Per-family recall, 95% CI ===")
for k in ["char_response", "cnn_response", "rnn_response", "wordtok_full"]:
    for f in FAM:
        m = (te.family == f).values
        report(f"recall {k} {f}", boot(corr[k][m], grp[m]))

print("\n=== Recall by model version within family (test rows) ===")
ver = {}
for f in FAM:
    for mv, g in te[te.family == f].groupby("model"):
        m = (te.model == mv).values
        ver[mv] = dict(n=int(m.sum()), cnn=corr["cnn_response"][m].mean(), char=corr["char_response"][m].mean())
        print(f"  {f:<8} {mv:<28} n={m.sum():>3}  CNN {100*corr['cnn_response'][m].mean():5.1f}  char {100*corr['char_response'][m].mean():5.1f}")
OUT["by_version"] = ver
gpt4 = te.model.str.startswith("gpt-4").values
gpt = (te.family == "GPT").values
report("cnn recall gpt-4 rows", boot(corr["cnn_response"][gpt & gpt4], grp[gpt & gpt4]))
report("cnn recall gpt-3.5 rows", boot(corr["cnn_response"][gpt & ~gpt4], grp[gpt & ~gpt4]))
report("cnn gpt-4 minus gpt-3.5 (unpaired)", (corr["cnn_response"][gpt & gpt4].mean() - corr["cnn_response"][gpt & ~gpt4].mean(), np.nan, np.nan))

print("\n=== Truncation: accuracy on responses longer vs shorter than 400 tokens ===")
ntok = te.response.map(lambda s: len(TOK.findall(s))).values
long_ = ntok > 400
print(f"  long test rows: {long_.sum()} of {len(te)} ({100*long_.mean():.1f}%)")
for k in ["cnn_response", "char_full", "char_400", "wordtok_full", "wordtok_400"]:
    report(f"{k} on long rows", boot(corr[k][long_], grp[long_]))
    report(f"{k} on short rows", boot(corr[k][~long_], grp[~long_]))
report("char_full - char_400 on long rows", boot((corr["char_full"] - corr["char_400"])[long_], grp[long_]))
report("wordtok_full - wordtok_400 on long rows", boot((corr["wordtok_full"] - corr["wordtok_400"])[long_], grp[long_]))
OUT["share_over_400_all"] = float((d.response.map(lambda s: len(TOK.findall(s))) > 400).mean())

# ---------------- RQ3: split the drop into "harder task" and "never saw the task" ----------------
print("\n=== RQ3 decomposition on the held-out task's TEST-split rows ===")
npd = pd.read_csv("results/neural_preds.csv")
rq3 = {}
pooled = {k: [] for k in ["cnn", "rnn", "char"]}
pooled_g = []
for t in ["coding", "creative", "math"]:
    sub = d[d.task == t]
    in_test = (sub.split == "test").values            # which held-out rows are also test rows
    tmask = (te.task == t).values                     # same rows, in test order
    assert (sub[in_test].family.values == te[tmask].family.values).all()
    row = {"n_test_rows": int(tmask.sum())}
    for m in ["cnn", "rnn", "char"]:
        if m == "char":
            ho = q(bp, exp=f"rq3-holdout-{t}", test="held_out_task")
            hold = (ho.true == ho.pred).values[in_test].astype(float)
            ido = q(bp, exp=f"rq3-holdout-{t}", test="in_domain")
            hold_other = (ido.true == ido.pred).values.mean()
            full = corr["char_response"]
        else:
            hs = [q(npd, exp=f"rq3-holdout-{t}", model=m, seed=s, test="held_out_task") for s in range(3)]
            hold = np.mean([(h.true == h.pred).values[in_test] for h in hs], 0)
            hold_other = q(npd, exp=f"rq3-holdout-{t}", model=m, test="in_domain").pipe(lambda x: (x.true == x.pred).mean())
            full = corr[f"{m}_response"]
        full_task, full_other = full[tmask], full[~tmask]
        row[m] = dict(full_other=full_other.mean(), full_task=full_task.mean(), hold_task_testrows=hold.mean(),
                      hold_other=hold_other,
                      difficulty=full_other.mean() - full_task.mean(),
                      unseen_cost=boot(full_task - hold, grp[tmask]))
        pooled[m].append(full_task - hold)
        print(f"  {t:<8} {m}: full model other-task test {100*full_other.mean():.1f}, same task test rows {100*full_task.mean():.1f} "
              f"(difficulty gap {100*(full_other.mean()-full_task.mean()):.1f}); held-out model on those rows {100*hold.mean():.1f} "
              f"-> cost of never seeing the task {fmt(row[m]['unseen_cost'])}")
    pooled_g.append(grp[tmask])
    rq3[t] = row
for m in pooled:
    t = boot(np.concatenate(pooled[m]), np.concatenate(pooled_g))
    report(f"RQ3 pooled unseen-task cost {m} (418 rows)", t)
OUT["rq3"] = json.loads(json.dumps(rq3, default=float))

print("\n=== RQ3 per-family table as in the report (CNN, pooled tasks and seeds) + CIs ===")
for f in FAM:
    i_d = q(npd[npd.exp.str.startswith("rq3")], model="cnn", test="in_domain", true=f)
    h = q(npd[npd.exp.str.startswith("rq3")], model="cnn", test="held_out_task", true=f)
    print(f"  {f:<8} in-domain {100*(i_d.true==i_d.pred).mean():.1f}  unseen {100*(h.true==h.pred).mean():.1f}  drop {100*((i_d.true==i_d.pred).mean()-(h.true==h.pred).mean()):.1f}")
# per-family "cost of never seeing the task" on test rows, CNN, pooled over tasks
tm = te.task.isin(["coding", "creative", "math"]).values
cost = np.full(len(te), np.nan)
for t, c in zip(["coding", "creative", "math"], pooled["cnn"]):
    cost[(te.task == t).values] = c
for f in FAM:
    m = tm & (te.family == f).values
    report(f"CNN unseen-task cost on test rows, {f} (n={m.sum()})", boot(cost[m], grp[m]))

print("\n=== RQ4 ablations (CNN retrained, paired over test rows, seeds averaged) ===")
abl = {}
for e in ["original", "lowercase", "no_newlines", "words_only", "shuffled", "first_10", "first_25", "first_50", "first_100", "first_200"]:
    rs = [q(npd, exp=f"rq4-{e}", seed=s) for s in range(3)]
    assert all((r.true.values == te.family.values).all() for r in rs)
    abl[e] = np.mean([(r.true == r.pred).values for r in rs], 0)
for e in abl:
    report(f"rq4 acc {e}", boot(abl[e], grp))
    if e != "original":
        report(f"rq4 original - {e}", boot(abl["original"] - abl[e], grp))
report("rq4 first_200 vs original (original - first_200)", boot(abl["original"] - abl["first_200"], grp))

# ---------------- leakage / data checks ----------------
print("\n=== Near-duplicate prompts across splits ===")
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
P = vec.fit_transform(d.prompt)
trm = (d.split == "train").values
tem = (d.split == "test").values
sims = np.asarray((P[tem] @ P[trm].T).max(1).todense()).ravel()
for th in [0.95, 0.9, 0.8]:
    m = sims >= th
    print(f"  test rows with a train prompt at cosine >= {th}: {m.sum()} ({100*m.mean():.1f}%)")
    OUT[f"neardup_{th}"] = int(m.sum())
m = sims >= 0.9
report("cnn acc on near-dup test rows (>=0.9)", boot(corr["cnn_response"][m], grp[m]))
report("cnn acc on other test rows", boot(corr["cnn_response"][~m], grp[~m]))
report("char acc on near-dup test rows (>=0.9)", boot(corr["char_response"][m], grp[m]))
report("char acc on other test rows", boot(corr["char_response"][~m], grp[~m]))

print("\n=== Prompt language (heuristic: non-Latin script share, else English function-word share) ===")
EN = set("the a an and or of to in on for with is are was were be it this that what how why can you i do does my your "
         "me please write give make if from as at by about which who when where should would could will not have has".split())


def lang(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return "other"
    if sum(not c.isascii() for c in letters) / len(letters) > 0.3:
        return "non-Latin script"
    w = re.findall(r"[a-z]+", s.lower())
    return "English" if not w or sum(x in EN for x in w) / len(w) >= 0.08 or len(w) < 4 else "other Latin-script"


d["lang"] = d.prompt.map(lang)
te["lang"] = te.prompt.map(lang)
print(d.lang.value_counts(normalize=True).round(3).to_string())
print(pd.crosstab(d.task, d.lang, normalize="columns").round(3).to_string())
OUT["lang_share"] = d.lang.value_counts(normalize=True).to_dict()
en = (te.lang == "English").values
for k in ["cnn_response", "char_response"]:
    report(f"{k} English prompts (n={en.sum()})", boot(corr[k][en], grp[en]))
    report(f"{k} non-English prompts (n={(~en).sum()})", boot(corr[k][~en], grp[~en]))

print("\n=== Task-rule spot-check sample (40 per label, seed 0) -> extra/task_spotcheck.csv ===")
import os
if not os.path.exists("extra/task_spotcheck.csv"):  # never overwrite the hand-labelled copy
    sample = d.drop_duplicates("prompt").groupby("task").sample(40, random_state=0)[["task", "prompt"]]
    sample["prompt"] = sample.prompt.str.slice(0, 300).str.replace("\n", " ")
    sample.to_csv("extra/task_spotcheck.csv", index=False)

# ---------------- RQ4 handcrafted features: which ones does the forest actually use? ----------------
print("\n=== Handcrafted features: permutation importance on test, and log-scaled logistic regression ===")
import importlib.util, sys, io, contextlib
spec = importlib.util.spec_from_file_location("features_mod", "features.py")
src = open("features.py").read().split("F = pd.DataFrame")[0]  # reuse feats() without re-running the whole script
ns = {}
exec(compile(src, "features.py", "exec"), ns)
F = pd.DataFrame([ns["feats"](s) for s in d.response]).astype(float)
trn, tst = (d.split == "train").values, (d.split == "test").values
rf = RandomForestClassifier(500, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(F[trn], d.family[trn])
print(f"  forest test acc {100*accuracy_score(d.family[tst], rf.predict(F[tst])):.1f}")
pi = permutation_importance(rf, F[tst], d.family[tst], n_repeats=10, random_state=0, n_jobs=-1)
imp = pd.DataFrame(dict(perm_mean=pi.importances_mean, perm_sd=pi.importances_std, impurity=rf.feature_importances_), index=F.columns)
imp = imp.sort_values("perm_mean", ascending=False)
imp["perm_rank"] = range(1, len(imp) + 1)
imp["impurity_rank"] = imp.impurity.rank(ascending=False).astype(int)
print((imp.assign(perm_mean=lambda x: 100 * x.perm_mean, perm_sd=lambda x: 100 * x.perm_sd)).round(2).to_string())
imp.to_csv("results/extra_feature_importance.csv")
log_lr = make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), LogisticRegression(max_iter=3000))
acc_log = accuracy_score(d.family[tst], log_lr.fit(F[trn], d.family[trn]).predict(F[tst]))
print(f"  logistic regression on log1p(features), standardized: {100*acc_log:.1f}")
OUT["logreg_log1p_acc"] = acc_log
OUT["perm_top10"] = imp.head(10)[["perm_mean", "perm_sd", "impurity_rank"]].round(4).to_dict("index")

print("\n=== Top word n-grams per family, CNN tokens (punctuation, markdown, newlines kept; case kept) ===")
v2 = TfidfVectorizer(token_pattern=r"\w+|[^\w\s]|\n", lowercase=False, ngram_range=(1, 2), min_df=5, sublinear_tf=True)
lr2 = LogisticRegression(max_iter=2000, C=4).fit(v2.fit_transform(d.response[trn]), d.family[trn])
voc = np.array([t.replace("\n", "\\n") for t in v2.get_feature_names_out()])
top = pd.DataFrame({f: voc[np.argsort(-lr2.coef_[i])[:15]] for i, f in enumerate(lr2.classes_)})
top.to_csv("results/extra_top_ngrams.csv", index=False)
print(top.to_string())

json.dump(OUT, open("results/extra_stats.json", "w"), indent=1, default=float)

# ---------------- extras: C=4 paired comparisons, a proper chance level, top features by family ----------------
print("\n=== Paired differences with the baselines at C=4 (the setting the original baselines used) ===")
for a, b in [("char_400_C4", "char_both_400_C4"), ("wordtok_400_C4", "wordtok_both_400_C4"), ("wordtok_full_C4", "char_response"), ("wordtok_full_C4", "word_response"), ("char_400_C4", "cnn_response"),
             ("wordtok_400_C4", "cnn_response"), ("char_response", "char_400_C4"), ("wordtok_full_C4", "wordtok_400_C4"),
             ("char_400_C4", "rnn_response")]:
    report(f"{a} - {b}", boot(corr[a] - corr[b], grp))

print("\n=== Prompt-only: permutation test (labels shuffled against predictions, 5000 times) ===")
print(f"  largest family share in test = {te.family.value_counts(normalize=True).max():.4f}")
y = te.family.values
for k in ["word_prompt", "char_prompt"]:
    pred = q(bp, exp="rq12", model=f"tfidf-{k.split('_')[0]}", input="prompt").pred.values
    obs = (pred == y).mean()
    null = np.array([(pred == RNG.permutation(y)).mean() for _ in range(5000)])
    p = (1 + (null >= obs).sum()) / 5001
    print(f"  {k}: observed {100*obs:.1f}, null mean {100*null.mean():.1f}, p = {p:.3f}")
    OUT[f"{k}_perm_p"] = p
for m in ["cnn", "rnn"]:
    for s in range(3):
        pred = pd.read_csv(f"preds/{m}_prompt_{s}.csv").pred.values
        obs = (pred == y).mean()
        null = np.array([(pred == RNG.permutation(y)).mean() for _ in range(5000)])
        p = (1 + (null >= obs).sum()) / 5001
        print(f"  {m}_prompt seed {s}: observed {100*obs:.1f}, null mean {100*null.mean():.1f}, p = {p:.3f}")
        OUT[f"{m}_prompt_{s}_perm_p"] = p

print("\n=== Family means of the top permutation-importance features ===")
print(F.groupby(d.family).mean().T.loc[imp.index[:10], FAM].round(3).to_string())
json.dump(OUT, open("results/extra_stats.json", "w"), indent=1, default=float)

# order / position probes, all at C=4 (appended last so earlier random draws stay the same)
print("\n=== Order / position probes on the word-token model, all at C=4 ===")
for a, b in [("wordtok_full_C4", "wordtok_unigram_full_C4"), ("wordtok_full_C4", "wordtok_shuffled_full_C4"),
             ("wordtok_full_C4", "wordtok_dropfirst10_full_C4"), ("wordtok_full_C4", "wordtok_droplast10_full_C4")]:
    report(f"{a} - {b}", boot(corr[a] - corr[b], grp))
json.dump(OUT, open("results/extra_stats.json", "w"), indent=1, default=float)
