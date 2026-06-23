from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.campaign_strategist.config import PROCESSED_DATA_DIR
from src.campaign_strategist.data import load_retail_transactions
from src.campaign_strategist.features import build_feature_bundle
from src.campaign_strategist.viz import (
    BLUE,
    DARK,
    GRAY,
    LIGHT_BLUE,
    ORANGE,
    add_subtitle,
    apply_report_style,
    clean_label,
    despine,
    format_count,
    percent_formatter,
    save_figure,
)


REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"


def save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    values = series.sort_values(ascending=True)
    labels = [clean_label(label) for label in values.index]
    colors = [LIGHT_BLUE] * len(values)
    colors[-1] = BLUE

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels, values.values, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_title(title, loc="left", pad=24)
    add_subtitle(ax, "Ranked view with the leading category highlighted")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E7ECF3", linewidth=1)
    ax.set_axisbelow(True)
    despine(ax)
    for bar, value in zip(bars, values.values):
        ax.text(
            bar.get_width() + values.max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            format_count(float(value)) if value > 1 else f"{value:.1%}",
            va="center",
            ha="left",
            color=GRAY,
            fontsize=9,
        )
    ax.set_xlim(0, values.max() * 1.18)
    save_figure(fig, path)


def run_eda(synthetic_only: bool = False) -> dict[str, object]:
    apply_report_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    transactions, source_name = load_retail_transactions(synthetic_only=synthetic_only)
    transactions.to_parquet(PROCESSED_DATA_DIR / "clean_transactions.parquet", index=False)

    missing = transactions.isna().sum().to_dict()
    summary = {
        "data_source": source_name,
        "rows": int(len(transactions)),
        "households": int(transactions["household_id"].nunique()),
        "weeks": int(transactions["week"].nunique()),
        "categories": int(transactions["product_category"].nunique()),
        "sales_total": float(transactions["sales_value"].sum()),
        "avg_transaction_value": float(transactions["sales_value"].mean()),
        "coupon_usage_rate": float(transactions["used_coupon"].mean()),
        "missing_values": {key: int(value) for key, value in missing.items()},
    }

    top_categories = (
        transactions.groupby("product_category")["sales_value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    weekly_sales = transactions.groupby("week")["sales_value"].sum()
    weekly_active_households = transactions.groupby("week")["household_id"].nunique()
    coupon_by_category = (
        transactions.groupby("product_category")["used_coupon"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    save_bar(
        top_categories,
        "Top Product Categories by Sales",
        "Product category",
        "Sales value",
        FIGURE_DIR / "eda_top_categories.png",
    )

    fig, ax = plt.subplots(figsize=(11, 5.8))
    rolling_sales = weekly_sales.rolling(4, min_periods=1).mean()
    ax.fill_between(weekly_sales.index, weekly_sales.values, color=LIGHT_BLUE, alpha=0.65)
    ax.plot(weekly_sales.index, weekly_sales.values, color=BLUE, linewidth=1.6, alpha=0.45, label="Weekly sales")
    ax.plot(rolling_sales.index, rolling_sales.values, color=DARK, linewidth=2.6, label="4-week moving average")
    ax.scatter([weekly_sales.idxmax()], [weekly_sales.max()], color=ORANGE, s=70, zorder=3)
    ax.annotate(
        f"Peak: {format_count(float(weekly_sales.max()))}",
        xy=(weekly_sales.idxmax(), weekly_sales.max()),
        xytext=(weekly_sales.idxmax() - 8, weekly_sales.max() * 1.05),
        arrowprops={"arrowstyle": "->", "color": GRAY},
        color=GRAY,
    )
    ax.set_title("Weekly Sales Trend", loc="left", pad=24)
    add_subtitle(ax, "Transaction value over time with smoothed demand signal")
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales value")
    ax.grid(axis="y", color="#E7ECF3", linewidth=1)
    ax.legend(loc="upper left")
    despine(ax)
    save_figure(fig, FIGURE_DIR / "eda_weekly_sales.png")

    fig, ax = plt.subplots(figsize=(11, 5.8))
    rolling_active = weekly_active_households.rolling(4, min_periods=1).mean()
    ax.fill_between(weekly_active_households.index, weekly_active_households.values, color=LIGHT_BLUE, alpha=0.65)
    ax.plot(
        weekly_active_households.index,
        weekly_active_households.values,
        color=BLUE,
        linewidth=1.6,
        alpha=0.45,
        label="Weekly active households",
    )
    ax.plot(rolling_active.index, rolling_active.values, color=DARK, linewidth=2.6, label="4-week moving average")
    ax.scatter(
        [weekly_active_households.idxmax()],
        [weekly_active_households.max()],
        color=ORANGE,
        s=70,
        zorder=3,
    )
    ax.annotate(
        f"Peak: {weekly_active_households.max():,.0f}",
        xy=(weekly_active_households.idxmax(), weekly_active_households.max()),
        xytext=(weekly_active_households.idxmax() - 8, weekly_active_households.max() * 1.03),
        arrowprops={"arrowstyle": "->", "color": GRAY},
        color=GRAY,
    )
    ax.set_title("Weekly Active Households", loc="left", pad=24)
    add_subtitle(ax, "Customer activity coverage across the shopping journey")
    ax.set_xlabel("Week")
    ax.set_ylabel("Active households")
    ax.grid(axis="y", color="#E7ECF3", linewidth=1)
    ax.legend(loc="upper left")
    despine(ax)
    save_figure(fig, FIGURE_DIR / "eda_weekly_active_households.png")

    save_bar(
        coupon_by_category,
        "Coupon Usage Rate by Category",
        "Product category",
        "Coupon usage rate",
        FIGURE_DIR / "eda_coupon_rate_by_category.png",
    )

    bundle = build_feature_bundle(transactions)
    label_counts = bundle.sample_index["label"].value_counts().sort_values(ascending=False)
    save_bar(
        label_counts,
        "Weak-Supervision Activation Labels",
        "Activation style",
        "Training samples",
        FIGURE_DIR / "eda_activation_label_distribution.png",
    )

    top_categories.reset_index(name="sales_value").to_csv(TABLE_DIR / "top_categories.csv", index=False)
    coupon_by_category.reset_index(name="coupon_usage_rate").to_csv(
        TABLE_DIR / "coupon_rate_by_category.csv",
        index=False,
    )

    summary["training_samples_after_sequence_build"] = int(len(bundle.x))
    summary["activation_label_counts"] = {key: int(value) for key, value in label_counts.items()}
    (TABLE_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown = f"""# Data Cleaning and EDA Summary

## Data Source

- Source used: `{source_name}`
- Rows after normalization: `{summary['rows']:,}`
- Households: `{summary['households']:,}`
- Weeks: `{summary['weeks']:,}`
- Product categories: `{summary['categories']:,}`

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

- Training sequences created: `{summary['training_samples_after_sequence_build']:,}`
- Label distribution: `{summary['activation_label_counts']}`
"""
    (REPORT_DIR / "data_cleaning_eda_summary.md").write_text(markdown, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data cleaning and EDA for the capstone project.")
    parser.add_argument("--synthetic-only", action="store_true", help="Use synthetic demo data only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_eda(synthetic_only=args.synthetic_only)
    print("EDA complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
