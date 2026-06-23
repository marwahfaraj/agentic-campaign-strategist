# Introduction Draft

## Project Title

Audience-to-Campaign Fit Advisor for Agentic Retail Segmentation

## Problem Statement

Retail marketers often begin with a campaign objective and ask an audience segmentation system to identify customers who match that objective. Agentic segmentation systems can help translate natural language audience requests into segment definitions and audience IDs. However, once the audience is created, marketers still need support understanding whether the audience fits the campaign objective, which activation style is most appropriate, and why the recommendation is supported by customer behavior.

This capstone project proposes an audience-to-campaign fit advisor that evaluates customer journey signals within an AI-generated audience and recommends an activation style such as a price-led coupon, win-back reminder, cross-sell bundle, loyalty reward, seasonal spotlight, or new customer onboarding.

## Research Question

Can customer purchase sequences be used to evaluate audience-to-campaign fit and recommend activation styles for AI-generated retail segments in a way that is accurate, explainable, and useful to marketers?

## Project Goal

The goal is to build a working machine learning prototype that:

- Ingests public or synthetic retail transaction data.
- Cleans and transforms customer purchase history into weekly journey sequences.
- Trains a deep learning sequence model from scratch.
- Compares the deep learning model against traditional machine learning baselines.
- Evaluates model performance and optimization choices.
- Demonstrates audience-to-campaign fit recommendations in a Streamlit application.
- Uses a local LLM explanation layer to convert model signals into marketer-friendly guidance.

## Dataset

The primary dataset is the public `completejourney` grocery retail dataset from 84.51°. The project also includes a synthetic retail data simulator to support reproducible demos and development when public data downloads are unavailable. No proprietary company data, customer data, internal dashboards, or internal system details are used.

## Expected Contribution

The contribution of this project is not another generic recommendation engine. Instead, it focuses on the post-segmentation decision point in agentic retail marketing: after an audience is created, the system evaluates campaign fit, recommends an activation style, and provides explainable guidance for marketers.
