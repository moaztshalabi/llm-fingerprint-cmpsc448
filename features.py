"""RQ4: which interpretable signals distinguish the families?
1) 25 handcrafted style features fed to logistic regression and a random forest (how far do style stats alone get?)
2) per-family means of those features (what each family does differently)
3) top word n-grams per family from a TF-IDF logistic regression (vocabulary fingerprints)
Writes results/features_*.csv."""
import re
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

d = pd.read_csv("data/dataset.csv")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
M = re.M


def feats(s):
    w = s.split()
    n = max(len(w), 1)
    per100 = lambda k: 100 * k / n
    sents = [x for x in re.split(r"[.!?]+\s", s) if x.strip()]
    lines = [l for l in s.split("\n") if l.strip()]
    low = s.lower()
    return {
        # length & structure
        "words": len(w), "chars_per_word": len(s) / n, "words_per_sentence": n / max(len(sents), 1),
        "lines": len(lines), "words_per_line": n / max(len(lines), 1),
        "type_token_ratio_first100": len({x.lower() for x in w[:100]}) / max(min(n, 100), 1),
        # markdown / formatting
        "bullet_lines_frac": sum(bool(re.match(r"\s*([-*•]|\d+[.)])\s", l)) for l in lines) / max(len(lines), 1),
        "headers": len(re.findall(r"^#{1,6}\s", s, M)), "bold": s.count("**") // 2,
        "code_blocks": s.count("```") // 2, "colon_line_ends": len(re.findall(r":\s*$", s, M)),
        # punctuation (per 100 words)
        "commas": per100(s.count(",")), "exclaims": per100(s.count("!")), "questions": per100(s.count("?")),
        "parens": per100(s.count("(")), "semicolons": per100(s.count(";")), "dashes": per100(len(re.findall(r" [-–—] ", s))),
        "has_emoji": bool(EMOJI.search(s)),  # a yes/no flag because a few emoji-only replies make a per-100-word rate explode
        # voice / stock phrases
        "i_rate": per100(len(re.findall(r"\bI\b", s))), "you_rate": per100(len(re.findall(r"\byou\b", low))),
        "opens_affirm": bool(re.match(r"\s*(sure|certainly|of course|absolutely|great question)", low)),
        "opens_i": bool(re.match(r"\s*I\b", s)),
        "hedges": per100(len(re.findall(r"however|it'?s important to|keep in mind|ultimately|generally|may |might ", low))),
        "apology_refusal": bool(re.search(r"i apologi[sz]e|i'?m sorry|i cannot|i can'?t|as an ai|language model", low)),
        "closing_offer": bool(re.search(r"(let me know|hope this helps|feel free)[^\n]*$", low.strip())),
    }


F = pd.DataFrame([feats(s) for s in d.response]).astype(float)
tr, te = (d.split == "train").values, (d.split == "test").values
out = []
for name, clf in [("handcrafted-logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
                  ("handcrafted-forest", RandomForestClassifier(500, min_samples_leaf=3, random_state=0, n_jobs=-1))]:
    p = clf.fit(F[tr], d.family[tr]).predict(F[te])
    out.append(dict(model=name, acc=accuracy_score(d.family[te], p), macro_f1=f1_score(d.family[te], p, average="macro")))
    print(out[-1])
pd.DataFrame(out).to_csv("results/features_scores.csv", index=False)

imp = pd.Series(clf.feature_importances_, F.columns).sort_values(ascending=False)
means = F.groupby(d.family).mean().T.loc[imp.index]
means.insert(0, "forest_importance", imp)
means.round(3).to_csv("results/features_by_family.csv")
print(means.round(2).to_string())

# vocabulary: highest-weight word 1- and 2-grams per family
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
lr = LogisticRegression(max_iter=2000, C=4).fit(vec.fit_transform(d.response[tr]), d.family[tr])
vocab = np.array(vec.get_feature_names_out())
top = pd.DataFrame({f: vocab[np.argsort(-lr.coef_[i])[:15]] for i, f in enumerate(lr.classes_)})
top.to_csv("results/features_top_ngrams.csv", index=False)
print(top.to_string())
