# Identifying the LLM behind the text

CMPSC 448 midterm project: can a CNN or RNN tell which LLM family (GPT, Claude, Gemini, Llama, Mistral) wrote a response?

Report: [Identifying LLM fingerprints CMPSCI 448 MIDTERM.pdf](Identifying%20LLM%20fingerprints%20CMPSCI%20448%20MIDTERM.pdf) (Moaz Shalabi)

Data: [Chatbot Arena human-preference 55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k), reduced to `data/dataset.csv` (13,410 responses, 2,682 per family, split by prompt).

| Step | Command | Needs | Output |
| --- | --- | --- | --- |
| 1 | `python prep.py` | `train.csv` from the link above | `data/dataset.csv` |
| 2 | `python baseline.py` | CPU | `results/baseline*.csv`: TF-IDF baselines (RQ1 to RQ3) |
| 3 | `python features.py` | CPU | `results/features_*.csv`: style features and top n-grams (RQ4) |
| 4 | `python models.py` | GPU | `results.csv` and `preds/`: TextCNN and BiLSTM (RQ1, RQ2) |
| 5 | `python rq34.py` | GPU | `results/neural*.csv`: cross-task and ablation runs (RQ3, RQ4) |
| 6 | `python figures.py` | CPU | `figures/` |
| 7 | `python extra/baseline_fair.py` | CPU | `results/extra_baseline*.csv`: baselines on the CNN's tokens and first 400 tokens |
| 8 | `python extra/stats.py` | CPU | `results/extra_stats.json` and more: bootstrap intervals, paired tests, RQ3 controls, feature importance |
| 9 | `python extra/figures_extra.py` | CPU | revised figures (`fig1b`, `fig2b`, `fig5b`, `fig6`) |
| opt. | `python extra/neural_extra.py` | GPU | learning curve and position ablations (not used in the report) |

Requirements: `pandas numpy scikit-learn torch matplotlib`. Results (95% intervals over test prompts): TextCNN 63.5% [61.3, 65.8], BiLSTM 60.8% [58.7, 63.0], char TF-IDF 66.1% [63.7, 68.7] on full responses and 64.6% [62.1, 67.1] on the same 400 tokens the neural models see (chance 20%).
