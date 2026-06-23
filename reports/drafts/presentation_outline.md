# Presentation Outline

Recommended length for an individual project: 5-7 minutes.

## Slide 1: Problem and Motivation

- Marketers often know the campaign goal but need help validating whether the generated audience fits that goal.
- Agentic segmentation can create audiences; this project focuses on post-segmentation campaign-fit guidance.

Visual:

- `assets/agentic_retail_campaign_hero.png`

## Slide 2: Project Workflow

- Marketer campaign objective and audience request.
- Simulated SegmentAI audience creation.
- Customer journey sequence model.
- Audience-to-campaign fit guidance.
- Optional local LLM explanation.

Visual:

- `assets/audience_campaign_fit_architecture.png`

## Slide 3: Data and EDA

- Public `completejourney` or synthetic fallback.
- Weekly household purchase journey features.
- Coupon, category, activity, and sales patterns.

Visual options:

- `reports/figures/eda_top_categories.png`
- `reports/figures/eda_weekly_sales.png`
- `reports/figures/eda_coupon_rate_by_category.png`

## Slide 4: Modeling Approach

- PyTorch LSTM trained from scratch.
- Baselines: KNN classifier and random forest.
- Activation labels from interpretable weak-supervision rules.

Visual:

- `reports/figures/model_comparison.png`

## Slide 5: Evaluation and Optimization

- Accuracy and macro-F1.
- Confusion matrix.
- Hyperparameter search over sequence length, hidden size, and learning rate.

Visual options:

- `reports/figures/model_confusion_matrix.png`
- `reports/figures/optimization_macro_f1.png`

## Slide 6: Demo and Business Interpretation

- Show Streamlit app with a representative request.
- Explain audience profile, activation style, confidence, message, and validation notes.

Visual:

- Live app or screenshot.

## Slide 7: Limitations and Future Work

- Weak labels should be replaced with approved campaign response labels.
- Evaluate incrementality, margin, fatigue, privacy, and policy constraints.
- Extend from prototype to production-grade API or agent tool.
