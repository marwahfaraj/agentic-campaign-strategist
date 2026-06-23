# Results Section Outline

This file should be updated after running the analysis scripts.

## Data Summary Results

Use outputs from:

- `reports/data_cleaning_eda_summary.md`
- `reports/figures/eda_top_categories.png`
- `reports/figures/eda_weekly_sales.png`
- `reports/figures/eda_weekly_active_households.png`
- `reports/figures/eda_coupon_rate_by_category.png`

Key points to discuss:

- Number of households, transactions, weeks, and categories.
- Category concentration and major purchase patterns.
- Coupon usage and discount behavior.
- Whether the dataset supports a customer journey modeling task.

## Model Evaluation Results

Use outputs from:

- `reports/model_training_evaluation_summary.md`
- `reports/tables/model_comparison.csv`
- `reports/figures/model_training_loss.png`
- `reports/figures/model_confusion_matrix.png`
- `reports/figures/model_comparison.png`

Key points to discuss:

- LSTM performance by accuracy and macro-F1.
- Comparison to KNN classifier and random forest baselines.
- Which activation styles are easiest or hardest to classify.
- Whether temporal sequence modeling adds useful signal.

## Optimization Results

Use outputs from:

- `reports/model_optimization_summary.md`
- `reports/tables/optimization_results.csv`
- `reports/figures/optimization_macro_f1.png`

Key points to discuss:

- Best sequence length, hidden size, and learning rate.
- Tradeoff between complexity and performance.
- Why macro-F1 was chosen as the primary optimization metric.

## Pipeline Demonstration Results

Use outputs from:

- `reports/pipeline_analysis_summary.md`
- `reports/tables/pipeline_demo_cases.csv`

Key points to discuss:

- Representative marketer inputs.
- Simulated SegmentAI audience creation.
- Audience profile differences.
- Activation style recommendations.
- Marketer-friendly explanation and validation notes.

## Limitations

- Current activation labels are weak-supervision labels, not real campaign response labels.
- Public or synthetic data does not fully represent enterprise-scale retail behavior.
- The LLM explanation layer improves readability but should not be treated as the source of truth.
- Future work should evaluate the advisor using approved campaign response, incrementality, and policy data.
