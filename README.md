# Audience-to-Campaign Fit Advisor for Agentic Retail Segmentation

**AAI-590 Capstone Project — Group 5**
Matt Hashemi, Marwah Faraj, Rebecca Cloe
Master of Science in Applied Artificial Intelligence, University of San Diego

![Agentic Retail Campaign Intelligence](assets/agentic_retail_campaign_hero.png)

Capstone prototype for advising whether an AI-generated retail audience fits a marketer's campaign objective and which activation style should be used.

Agentic segmentation systems can translate a marketer's natural-language request into an audience segment. This project focuses on the decision that comes after: once the audience exists, does it fit the campaign objective, which activation style should be used, and why? The prototype analyzes household purchase journeys inside a segment and recommends one of six activation styles, with a plain-language rationale.

**Research question:** Can customer purchase sequences be used to evaluate audience-to-campaign fit and recommend activation styles for AI-generated retail segments in a way that is accurate, explainable, and useful to marketers?

## Activation Style Classes

The model predicts one of six activation styles:

- New customer onboarding
- Win-back reminder
- Price-led coupon
- Cross-sell bundle
- Loyalty reward
- Seasonal spotlight

## Architecture

![Audience-to-Campaign Fit Advisor Architecture](assets/audience_campaign_fit_architecture.png)

```text
Complete Journey retail transactions
        ↓
Data cleaning and EDA                  (notebook 01)
        ↓
Weekly journey sequences + future-window labels   (notebook 02)
        ↓
Baselines (KNN, Random Forest) + BiLSTM w/ attention   (notebook 03)
        ↓
Hyperparameter optimization + error analysis      (notebook 04)
        ↓
Marketer-facing Streamlit app with explanations   (app.py)
```

## Dataset

The primary dataset is *The Complete Journey* from 84.51°: one year of household-level grocery transactions (purchase week, product category, quantity, sales value, discounts, coupon use). It is accessed through the data files of the open-source `completejourney` R package:

- https://github.com/bradleyboehmke/completejourney
- https://bradleyboehmke.github.io/completejourney/

This project uses the published transaction sample (75,000 transactions, ~2,400 households, 53 weeks) joined with the full product table. Data files are downloaded automatically on the first run of notebook 01 and are not committed to the repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Reproduce the Pipeline

The analysis lives in four notebooks that must run **in order**, because each one reads the outputs of the previous one. In VS Code or Jupyter, open each notebook and use Restart + Run All:

| Order | Notebook | What it does | Outputs |
| --- | --- | --- | --- |
| 1 | `notebooks/01_data_cleaning_eda.ipynb` | Load data, quality checks, EDA, outlier decisions | `data/processed/transactions_clean.parquet` |
| 2 | `notebooks/02_features_and_labels.ipynb` | Weekly journey sequences, future-window weak labels | `sequences_x.npy`, `labels_y.npy`, `sample_index.parquet` |
| 3 | `notebooks/03_baselines_and_lstm.ipynb` | Grouped split, baselines, LSTM training with validation curves | model + split artifacts in `artifacts/` |
| 4 | `notebooks/04_optimization_and_evaluation.ipynb` | Grid search, final model, confusion matrix, error analysis | figures and tables in `reports/` |

Important: if anything in `src/` or the label rules changes, re-run the chain from notebook 02 onward. Notebooks 03 and 04 read saved arrays, not live code.

## Leakage-Safe Design

Three design decisions distinguish the final pipeline from an earlier prototype:

1. **Future-window labels.** Activation-style labels are decided by household behavior in the 4 weeks *after* each input sequence, never by signals inside the sequence itself. Samples with no clear future signal are abstained (excluded), which is standard weak-supervision practice.
2. **Grouped splits.** Train/validation/test splits are made by household, so overlapping windows from one household can never appear on both sides of a split.
3. **Train-only scaling.** The feature scaler is fit on the training split only.

## Results (held-out test set)

| Model | Accuracy | Macro-F1 |
| --- | --- | --- |
| **BiLSTM + attention (tuned)** | **0.44** | **0.50** |
| Random Forest | 0.40 | 0.44 |
| KNN | 0.37 | 0.40 |

With six classes, the majority-class baseline sits near 0.29 accuracy. The sequence model outperforms both tabular baselines on both metrics. Full per-class metrics, confusion matrices, and error analysis are in notebook 04 and `reports/`.

## Run the Demo App

```bash
streamlit run app.py
```

Example marketer prompts:

```text
Create me a segment of game day heavy shoppers for a seasonal campaign
Find lapsed snack buyers for a win-back coupon campaign
Audience of loyal dairy shoppers for a weekend bundle campaign
```

### Optional local LLM explanations

The app can use Mistral through Ollama (free, local, no API key) to turn model outputs into marketer-friendly text. A deterministic fallback explanation is used when Ollama is not running.

```bash
brew install ollama
ollama pull mistral
ollama serve
```

Then enable **Use local Ollama Mistral if running** in the Streamlit sidebar.

## Repository Guide

| Path | Purpose |
| --- | --- |
| `notebooks/01–04` | The full analysis pipeline (see table above) |
| `src/campaign_strategist/config.py` | Paths, campaign classes, category aliases |
| `src/campaign_strategist/data.py` | Dataset download, normalization, synthetic fallback for tests |
| `src/campaign_strategist/features.py` | Weekly journey frame and sequence dataset builder |
| `src/campaign_strategist/labels.py` | Future-window weak supervision rules and thresholds |
| `src/campaign_strategist/model.py` | PyTorch BiLSTM with attention pooling |
| `src/campaign_strategist/training.py` | Grouped splits, train-only scaling, training loop, metrics |
| `src/campaign_strategist/baselines.py` | KNN and Random Forest baselines |
| `src/campaign_strategist/strategy.py` | Campaign-fit recommendation and Ollama explanation layer |
| `app.py` | Streamlit demo application |
| `reports/figures/`, `reports/tables/` | Generated figures and metric tables used in the report |
| `docs/` | Capstone requirement mapping and drafts |
| `assets/` | Diagrams and illustrations for README, report, and slides |

## Capstone Requirement Coverage

| Requirement | Where |
| --- | --- |
| Data cleaning | Notebook 01, `src/.../data.py` |
| Exploratory data analysis | Notebook 01, `reports/figures/eda_*.png` |
| Feature engineering / pipeline design | Notebook 02, `src/.../features.py`, `src/.../labels.py` |
| Deep learning model (from scratch) | Notebook 03, `src/.../model.py` |
| Baseline comparison | Notebooks 03–04, `src/.../baselines.py` |
| Model optimization | Notebook 04, `reports/tables/optimization_results.csv` |
| Model/pipeline analysis | Notebook 04 (confusion matrix, error analysis) |
| Deployment prototype | `app.py` |

## Academic Notes

Labels are weak-supervision labels derived from interpretable future-window rules, not real campaign response data; this keeps the project feasible in seven weeks while still requiring deep learning training from scratch. Known limitations are discussed in the final report, including the abstention rate of the labeling rules, the near-deterministic onboarding class, and the use of the published transaction sample rather than the full dataset. A production extension would replace weak labels with actual campaign response outcomes.
