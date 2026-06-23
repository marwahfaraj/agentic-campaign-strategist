from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import CAMPAIGN_CLASSES, CATEGORY_ALIASES, OCCASION_CATEGORY_BUNDLES


@dataclass
class FeatureBundle:
    x: np.ndarray
    y: np.ndarray
    sample_index: pd.DataFrame
    feature_names: list[str]
    categories: list[str]
    scaler: StandardScaler


def build_feature_bundle(
    transactions: pd.DataFrame,
    sequence_length: int = 12,
    prediction_horizon: int = 4,
    top_n_categories: int = 8,
) -> FeatureBundle:
    data = transactions.copy()
    data["product_category"] = data["product_category"].fillna("unknown").astype(str).str.lower()

    top_categories = (
        data.groupby("product_category")["sales_value"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_categories)
        .index.tolist()
    )
    data["modeled_category"] = np.where(
        data["product_category"].isin(top_categories), data["product_category"], "other"
    )
    modeled_categories = top_categories + ["other"]

    weekly = _build_weekly_customer_frame(data, modeled_categories)
    feature_names = _feature_names(modeled_categories)
    samples, labels, sample_index = _build_sliding_window_samples(
        weekly=weekly,
        feature_names=feature_names,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
    )

    if len(samples) == 0:
        raise RuntimeError("No training samples were created. Check transaction coverage and sequence_length.")

    x = np.asarray(samples, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    scaler = StandardScaler()
    flat = x.reshape(-1, x.shape[-1])
    scaled = scaler.fit_transform(flat).reshape(x.shape).astype(np.float32)

    return FeatureBundle(
        x=scaled,
        y=y,
        sample_index=sample_index,
        feature_names=feature_names,
        categories=modeled_categories,
        scaler=scaler,
    )


def transform_segment_sequences(bundle: FeatureBundle, sample_rows: pd.DataFrame) -> np.ndarray:
    idx = sample_rows.index.to_numpy()
    return bundle.x[idx]


def detect_requested_category(query: str, categories: list[str]) -> str | None:
    detected = detect_requested_categories(query, categories)
    return detected[0] if detected else None


def detect_requested_categories(query: str, categories: list[str]) -> list[str]:
    query_lower = query.lower()
    category_lookup: dict[str, str] = {}
    for category in categories:
        category_lookup[category] = category
    for category, aliases in CATEGORY_ALIASES.items():
        category_lookup[category] = category
        for alias in aliases:
            category_lookup[alias] = category
    detected: list[str] = []
    for occasion, bundle_categories in OCCASION_CATEGORY_BUNDLES.items():
        if occasion in query_lower:
            detected.extend(bundle_categories)
    for alias, category in category_lookup.items():
        if alias in query_lower and category not in detected:
            detected.append(category)
    return list(dict.fromkeys(detected))


def parse_segment_request(query: str, categories: list[str]) -> dict[str, object]:
    query_lower = query.lower()
    detected_categories = detect_requested_categories(query_lower, categories)
    recent_terms = ["recent", "recently", "last", "current"]
    return {
        "categories": detected_categories,
        "recent_window": 8 if any(term in query_lower for term in recent_terms) else 16,
        "lapsed": any(term in query_lower for term in ["lapsed", "inactive", "win back", "win-back"]),
        "coupon_sensitive": any(term in query_lower for term in ["coupon", "discount", "price", "deal", "savings"]),
        "loyal_or_heavy": any(
            term in query_lower for term in ["loyal", "frequent", "high value", "heavy", "power shopper"]
        ),
        "new_customer": any(term in query_lower for term in ["new customer", "new shopper", "first time"]),
        "seasonal": any(term in query_lower for term in ["seasonal", "holiday", "game", "weekend", "party"]),
    }


def filter_segment_samples(
    query: str,
    transactions: pd.DataFrame,
    bundle: FeatureBundle,
    max_rows: int = 500,
) -> pd.DataFrame:
    """Simulate an upstream SegmentAI audience result using simple interpretable filters."""
    query_lower = query.lower()
    latest_week = int(transactions["week"].max())
    parsed = parse_segment_request(query_lower, bundle.categories)
    requested_categories = parsed["categories"]
    sample_index = bundle.sample_index.copy()

    candidate_households = set(sample_index["household_id"].astype(str))

    if requested_categories:
        recent_window = int(parsed["recent_window"])
        category_buyers = transactions[
            (transactions["product_category"].isin(requested_categories))
            & (transactions["week"] >= latest_week - recent_window + 1)
        ]["household_id"].astype(str)
        candidate_households &= set(category_buyers)

    if parsed["lapsed"]:
        recent_active = set(
            transactions[transactions["week"] >= latest_week - 6]["household_id"].astype(str)
        )
        historical_active = set(
            transactions[transactions["week"] < latest_week - 6]["household_id"].astype(str)
        )
        candidate_households &= historical_active - recent_active

    if parsed["coupon_sensitive"]:
        coupon_users = set(
            transactions[transactions["used_coupon"] > 0]["household_id"].astype(str)
        )
        candidate_households &= coupon_users

    if parsed["loyal_or_heavy"]:
        trips = transactions.groupby("household_id")["week"].nunique()
        loyal_households = set(trips[trips >= trips.quantile(0.7)].index.astype(str))
        candidate_households &= loyal_households

    if parsed["new_customer"]:
        first_week = transactions.groupby("household_id")["week"].min()
        new_households = set(first_week[first_week >= latest_week - 20].index.astype(str))
        candidate_households &= new_households

    if parsed["seasonal"]:
        seasonal_weeks = list(range(20, 28)) + list(range(45, 53))
        seasonal_households = set(
            transactions[transactions["week"].isin(seasonal_weeks)]["household_id"].astype(str)
        )
        candidate_households &= seasonal_households

    filtered = sample_index[sample_index["household_id"].astype(str).isin(candidate_households)]
    if filtered.empty:
        # Keep arbitrary prompts demoable, but make different unknown prompts select
        # different fallback audiences so the UI still shows the prompt matters.
        fallback_seed = int(hash(query_lower) % 10_000)
        filtered = sample_index.sample(min(max_rows, len(sample_index)), random_state=fallback_seed)
    else:
        # Use the latest available sequence for each household so the audience
        # profile represents households, not repeated historical windows.
        filtered = (
            filtered.sort_values(["household_id", "end_week"])
            .groupby("household_id", as_index=False)
            .tail(1)
            .sort_values("end_week", ascending=False)
            .head(max_rows)
        )
    filtered.attrs["segment_parse"] = parsed
    return filtered


def profile_segment(transactions: pd.DataFrame, households: list[str]) -> dict[str, object]:
    segment_data = transactions[transactions["household_id"].astype(str).isin(households)]
    if segment_data.empty:
        return {
            "households": 0,
            "avg_weekly_spend": 0.0,
            "coupon_rate": 0.0,
            "top_categories": [],
            "recent_activity_rate": 0.0,
        }

    latest_week = int(transactions["week"].max())
    weekly_spend = segment_data.groupby(["household_id", "week"])["sales_value"].sum()
    top_categories = (
        segment_data.groupby("product_category")["sales_value"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    recent_households = segment_data[segment_data["week"] >= latest_week - 6]["household_id"].nunique()
    return {
        "households": int(segment_data["household_id"].nunique()),
        "avg_weekly_spend": float(weekly_spend.mean()) if len(weekly_spend) else 0.0,
        "coupon_rate": float(segment_data["used_coupon"].mean()) if len(segment_data) else 0.0,
        "top_categories": top_categories,
        "recent_activity_rate": float(recent_households / max(segment_data["household_id"].nunique(), 1)),
    }


def _build_weekly_customer_frame(data: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    base = (
        data.groupby(["household_id", "week"])
        .agg(
            sales_value=("sales_value", "sum"),
            quantity=("quantity", "sum"),
            discount_value=("discount_value", "sum"),
            used_coupon=("used_coupon", "max"),
            trips=("week", "size"),
        )
        .reset_index()
    )
    pivot = (
        data.pivot_table(
            index=["household_id", "week"],
            columns="modeled_category",
            values="sales_value",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=categories, fill_value=0.0)
        .reset_index()
    )
    weekly = base.merge(pivot, on=["household_id", "week"], how="left")

    households = weekly["household_id"].unique()
    weeks = range(int(weekly["week"].min()), int(weekly["week"].max()) + 1)
    full_index = pd.MultiIndex.from_product([households, weeks], names=["household_id", "week"])
    weekly = weekly.set_index(["household_id", "week"]).reindex(full_index, fill_value=0.0).reset_index()

    weekly["is_active"] = (weekly["sales_value"] > 0).astype(float)
    weekly["discount_rate"] = weekly["discount_value"] / (weekly["sales_value"] + weekly["discount_value"] + 1e-6)
    weekly["coupon_rate"] = weekly["used_coupon"].clip(0, 1)
    weekly["log_sales"] = np.log1p(weekly["sales_value"])
    weekly["log_quantity"] = np.log1p(weekly["quantity"])
    weekly["log_trips"] = np.log1p(weekly["trips"])
    total_category_sales = weekly[categories].sum(axis=1).replace(0, np.nan)
    for category in categories:
        weekly[f"share_{category}"] = (weekly[category] / total_category_sales).fillna(0.0)
    return weekly


def _feature_names(categories: list[str]) -> list[str]:
    return [
        "is_active",
        "log_sales",
        "log_quantity",
        "log_trips",
        "discount_rate",
        "coupon_rate",
    ] + [f"share_{category}" for category in categories]


def _build_sliding_window_samples(
    weekly: pd.DataFrame,
    feature_names: list[str],
    sequence_length: int,
    prediction_horizon: int,
) -> tuple[list[np.ndarray], list[int], pd.DataFrame]:
    samples: list[np.ndarray] = []
    labels: list[int] = []
    index_rows: list[dict[str, object]] = []

    for household_id, group in weekly.groupby("household_id", sort=False):
        group = group.sort_values("week").reset_index(drop=True)
        if len(group) < sequence_length + prediction_horizon:
            continue
        first_active_week = int(group.loc[group["is_active"] > 0, "week"].min()) if group["is_active"].sum() else 999

        for end_pos in range(sequence_length - 1, len(group) - prediction_horizon):
            window = group.iloc[end_pos - sequence_length + 1 : end_pos + 1]
            future = group.iloc[end_pos + 1 : end_pos + 1 + prediction_horizon]
            label_name = _weak_label_campaign(window, future, first_active_week)
            samples.append(window[feature_names].to_numpy(dtype=np.float32))
            labels.append(CAMPAIGN_CLASSES.index(label_name))
            index_rows.append(
                {
                    "household_id": household_id,
                    "end_week": int(window["week"].max()),
                    "label": label_name,
                    "recent_spend": float(window["sales_value"].tail(4).sum()),
                    "recent_coupon_rate": float(window["coupon_rate"].tail(8).mean()),
                    "recent_active_weeks": int(window["is_active"].tail(8).sum()),
                }
            )

    return samples, labels, pd.DataFrame(index_rows)


def _weak_label_campaign(window: pd.DataFrame, future: pd.DataFrame, first_active_week: int) -> str:
    end_week = int(window["week"].max())
    recent = window.tail(4)
    prior = window.head(max(len(window) - 4, 1))
    recent_active = float(recent["is_active"].sum())
    prior_active = float(prior["is_active"].sum())
    recent_spend = float(recent["sales_value"].sum())
    avg_spend = float(window["sales_value"].mean())
    discount_rate = float(window["discount_rate"].tail(8).mean())
    coupon_rate = float(window["coupon_rate"].tail(8).mean())
    future_activity = float(future["is_active"].sum())

    if end_week - first_active_week <= 8:
        return "new_customer_onboarding"
    if prior_active >= 3 and recent_active <= 1 and future_activity <= 1:
        return "win_back_reminder"
    if coupon_rate >= 0.22 or discount_rate >= 0.18:
        return "price_led_coupon"
    if end_week in list(range(20, 28)) + list(range(45, 53)):
        return "seasonal_spotlight"
    if recent_active >= 3 and recent_spend >= max(avg_spend * 5, 30):
        return "loyalty_reward"
    return "cross_sell_bundle"
