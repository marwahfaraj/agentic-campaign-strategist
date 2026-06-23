# Data Cleaning and EDA Summary

## Data Source

- Source used: `synthetic retail simulator`
- Rows after normalization: `133,310`
- Households: `900`
- Weeks: `53`
- Product categories: `10`

## Cleaning Steps

- Standardized column names from public/synthetic transaction sources.
- Normalized `household_id`, `week`, `product_category`, `quantity`, `sales_value`, discount, and coupon fields.
- Replaced invalid numeric values with safe defaults.
- Created `discount_value` and `used_coupon` fields.
- Saved cleaned transactions to `data/processed/clean_transactions.parquet`.

## EDA Outputs

- `reports/figures/eda_top_categories.png`
- `reports/figures/eda_weekly_sales.png`
- `reports/figures/eda_weekly_active_households.png`
- `reports/figures/eda_coupon_rate_by_category.png`
- `reports/figures/eda_activation_label_distribution.png`

## Modeling Dataset

The model converts cleaned transactions into weekly household sequences. Each sequence represents the most recent 12 weeks of behavior and receives a weak-supervision activation label for campaign-fit modeling.

- Training sequences created: `34,200`
- Label distribution: `{'price_led_coupon': 22962, 'cross_sell_bundle': 4059, 'seasonal_spotlight': 2699, 'new_customer_onboarding': 2258, 'win_back_reminder': 1607, 'loyalty_reward': 615}`
