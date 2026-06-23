# Methods Section Outline

## Data Pipeline

The data pipeline loads public `completejourney` retail transactions when available and falls back to a synthetic retail simulator for reproducible demos. Transaction data is standardized into household, week, category, quantity, sales, discount, and coupon fields.

## Feature Engineering

The modeling pipeline converts transactions into weekly household sequences. Each sequence includes activity, sales, quantity, trip, discount, coupon, and category-share features. The default sequence length is 12 weeks.

## Weak-Supervision Labeling

Because public campaign response labels are limited, activation-style labels are generated using interpretable customer journey rules. These labels represent campaign-fit styles such as win-back, loyalty reward, seasonal spotlight, and price-led coupon.

## Deep Learning Model

The primary model is a PyTorch LSTM trained from scratch. The model reads weekly customer journey sequences and predicts the activation style that best fits the audience's behavior.

## Baseline Models

The project compares the LSTM with KNN classifier and random forest baselines. Baselines use flattened sequence summary features, including means, recent-week features, and simple trends.

## Optimization

The optimization experiment compares sequence length, hidden size, and learning rate configurations. Macro-F1 is the primary optimization metric because activation labels may be imbalanced.

## Application Layer

The Streamlit app simulates the post-segmentation workflow. A marketer enters a campaign-oriented audience request, the app creates a simulated audience, the LSTM predicts an activation style, and a local Ollama + Mistral explanation layer optionally generates marketer-friendly rationale.
