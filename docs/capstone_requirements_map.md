# Capstone Requirements Map

This document maps the capstone requirements to concrete files in this repository.

## Applied AI Problem

**Requirement:** Research, propose, implement, test, evaluate, and report on an applied machine learning problem.

**Coverage:** This project addresses audience-to-campaign fit advising for agentic retail segmentation. The system evaluates whether a generated audience fits a known campaign objective and recommends an activation style.

Relevant files:

- `README.md`
- `app.py`
- `src/campaign_strategist/`
- `reports/drafts/introduction_draft.md`

## Dataset Selection and Cleaning

**Requirement:** Identify and cleanse a dataset.

**Coverage:** The project loads public `completejourney` retail data when available and includes a synthetic retail simulator for reproducible development. Cleaning standardizes transaction fields, category values, sales, discount, coupon, and weekly customer journey columns.

Relevant files:

- `src/campaign_strategist/data.py`
- `scripts/01_data_cleaning_eda.py`
- `reports/data_cleaning_eda_summary.md`
- `reports/figures/eda_*.png`

## Exploratory Data Analysis

**Requirement:** Perform data cleaning and exploratory data analysis.

**Coverage:** The EDA script creates summary tables and visualizations for category sales, weekly sales, active households, coupon usage, and model label distribution.

Relevant files:

- `scripts/01_data_cleaning_eda.py`
- `reports/figures/eda_top_categories.png`
- `reports/figures/eda_weekly_sales.png`
- `reports/figures/eda_weekly_active_households.png`
- `reports/figures/eda_coupon_rate_by_category.png`
- `reports/figures/eda_activation_label_distribution.png`

## Deep Learning Model Training

**Requirement:** Include at least one neural network/deep learning model and model training.

**Coverage:** The project trains a PyTorch LSTM from scratch on weekly customer journey sequences.

Relevant files:

- `src/campaign_strategist/model.py`
- `src/campaign_strategist/train.py`
- `scripts/02_model_training_evaluation.py`
- `artifacts/journey_lstm.pt` generated after training

## Traditional Baseline Comparison

**Requirement:** Not mandatory, but strengthens evaluation and complexity.

**Coverage:** The project compares the LSTM against KNN classifier and random forest baselines using flattened sequence features.

Relevant files:

- `src/campaign_strategist/baselines.py`
- `scripts/02_model_training_evaluation.py`
- `reports/tables/model_comparison.csv`
- `reports/figures/model_comparison.png`

## Model Optimization

**Requirement:** Include model optimization work.

**Coverage:** The optimization script compares sequence length, hidden size, and learning rate settings using macro-F1.

Relevant files:

- `scripts/03_model_optimization.py`
- `reports/tables/optimization_results.csv`
- `reports/tables/best_hyperparameters.json`
- `reports/figures/optimization_macro_f1.png`

## Model/Pipeline Analysis

**Requirement:** Include model or pipeline analysis and discussion.

**Coverage:** The pipeline analysis script generates representative inputs and outputs for marketer requests, audience profiles, activation styles, confidence, messages, and risk notes.

Relevant files:

- `scripts/04_pipeline_analysis.py`
- `reports/pipeline_analysis_summary.md`
- `reports/tables/pipeline_demo_cases.csv`

## Deployable System / Portfolio Artifact

**Requirement:** Project should be portfolio-ready and may include a complete ML system.

**Coverage:** The Streamlit app demonstrates the workflow end-to-end, including local Ollama + Mistral explanation when available.

Relevant files:

- `app.py`
- `src/campaign_strategist/strategy.py`
- `assets/agentic_retail_campaign_hero.png`
- `assets/audience_campaign_fit_architecture.png`

## Final Presentation Visualizations

**Requirement:** Individual projects require at least 3 visualizations.

**Coverage:** The project generates more than 3 report-ready visualizations in `reports/figures/`.

Suggested presentation visuals:

- `assets/audience_campaign_fit_architecture.png`
- `reports/figures/eda_top_categories.png`
- `reports/figures/model_comparison.png`
- `reports/figures/model_confusion_matrix.png`
- `reports/figures/optimization_macro_f1.png`
