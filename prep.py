"""Build (LLM_name, LLM_input, LLM_output) dataset from Chatbot Arena 55k.

Output: data/dataset.csv with columns
  family, model, prompt, response, task, self_id, split
"""
import json, re
import numpy as np, pandas as pd

SRC = "train.csv"  # arena-human-preference-55k, see README
SEED = 42

# First-party models only. Community fine-tunes (vicuna, zephyr, ...) are left out because their family is unclear.
FAMILY = {
    "GPT":     r"^gpt-(3\.5|4)",
    "Claude":  r"^claude-",
    "Gemini":  r"^gemini-pro",
    "Llama":   r"^llama-2-\d+b-chat$",
    "Mistral": r"^(mistral-(7b-instruct|medium)|mixtral-8x7b-instruct)",
}

# Self-identification: the response names its own model or maker. Kept as a flag for the leakage check.
SELF_ID = re.compile(r"\b(?:openai|chatgpt|gpt-?[34]|anthropic|claude|google|gemini|bard|palm|"
                     r"meta\b|llama|mistral)\b", re.I)

TASKS = [  # first match wins; used for RQ3 cross-task splits
    ("coding",   r"```|\b(code|python|javascript|java|c\+\+|sql|function|script|program|bug|regex|html|css)\b"),
    ("math",     r"\b(solve|calculate|equation|integral|derivative|probability|sum of|how many|math)\b|\d+\s*[\+\-\*/\^]\s*\d+"),
    ("creative", r"\b(write|poem|story|essay|song|lyrics|haiku|joke|rap|limerick|fiction)\b"),
]


def family(m):
    return next((f for f, p in FAMILY.items() if re.search(p, m)), None)


def first_turn(s):
    try:
        v = json.loads(s)
        return v[0] if v and isinstance(v[0], str) else None
    except Exception:
        return None


def task(p):
    return next((t for t, pat in TASKS if re.search(pat, p, re.I)), "qa/other")


raw = pd.read_csv(SRC)
# Each Arena row has one prompt answered by two models, so it becomes two (model, prompt, response) rows.
rows = pd.concat([
    raw[["id", f"model_{s}", "prompt", f"response_{s}"]]
       .set_axis(["conv_id", "model", "prompt", "response"], axis=1)
    for s in "ab"])
n_raw = len(rows)

rows["family"] = rows.model.map(family)
rows = rows.dropna(subset=["family"])
rows["prompt"] = rows.prompt.map(first_turn)      # first turn only, so each output answers exactly one input
rows["response"] = rows.response.map(first_turn)
rows = rows.dropna(subset=["prompt", "response"])
rows = rows[rows.response.str.strip().str.len() >= 10]
rows = rows.drop_duplicates(subset=["model", "prompt", "response"])

rows["task"] = rows.prompt.map(task)
rows["self_id"] = rows.response.str.contains(SELF_ID)

# Balance by downsampling every family to the size of the smallest.
cap = rows.family.value_counts().min()
rows = rows.groupby("family").sample(cap, random_state=SEED)

# Split by prompt, not by row, so the same prompt never lands in both train and test.
prompts = np.random.default_rng(SEED).permutation(rows.prompt.unique())
n = len(prompts)
split_of = {p: ("train" if i < .8 * n else "val" if i < .9 * n else "test") for i, p in enumerate(prompts)}
rows["split"] = rows.prompt.map(split_of)

rows = rows[["family", "model", "prompt", "response", "task", "self_id", "split"]].reset_index(drop=True)
rows.to_csv("data/dataset.csv", index=False)

# ---- stats for the report ----
print(f"raw arena responses: {n_raw}  ->  final: {len(rows)}  ({cap} per family)\n")
print(pd.crosstab(rows.family, rows.split, margins=True), "\n")
print(pd.crosstab(rows.family, rows.task, margins=True), "\n")
print("models per family:\n", rows.groupby("family").model.unique().to_string(), "\n")
print("self-id rate by family:\n", rows.groupby("family").self_id.mean().round(3).to_string(), "\n")
rows["resp_words"] = rows.response.str.split().str.len()
print("response length (words):\n", rows.groupby("family").resp_words.describe()[["mean", "50%", "max"]].round(0))

# sanity: no prompt leaks across splits
assert rows.groupby("prompt").split.nunique().max() == 1
