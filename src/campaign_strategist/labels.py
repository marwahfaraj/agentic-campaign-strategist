"""Future-window weak supervision labels for activation styles.

Design principle (leakage control): every label is decided by what the
household does in the FUTURE window (the ``prediction_horizon`` weeks after
an input sequence), never by the behavior inside the input window itself.
The only exceptions are household tenure (used for onboarding, a property of
the customer rather than a behavioral signal in the window) and past activity
(used only as a precondition for win-back, so that "win back" is reserved for
households that were real customers before going quiet).

Samples with no clear activation signal return ``None`` (abstain) and are
excluded from training, which is standard weak-supervision practice.
"""

from __future__ import annotations

import pandas as pd

# Tuned thresholds: prefer cleaner, more separable classes over maximizing sample count.
DEFAULT_LABEL_CONFIG: dict[str, float | int | set[int]] = {
    "onboarding_max_tenure_weeks": 6,
    "win_back_min_window_active": 4,
    # Prefer coupon redemptions; plain discount must be clearly strong.
    "price_led_min_discount_rate": 0.28,
    # Holiday period only: weekly sales run ~1.08x the annual mean in weeks 46-53,
    # while weeks 20-27 (inherited from the synthetic simulator) show no lift (0.99x).
    "seasonal_weeks": set(range(46, 54)),
    "seasonal_min_spend_lift": 1.30,
    "cross_sell_min_new_spend_share": 0.22,
    "loyalty_min_future_active": 3,
    "loyalty_min_spend_lift": 1.05,
}


def label_future_window(
    window: pd.DataFrame,
    future: pd.DataFrame,
    first_active_week: int,
    category_columns: list[str],
    config: dict | None = None,
) -> str | None:
    """Assign an activation-style label from future behavior, or None to abstain."""
    cfg = {**DEFAULT_LABEL_CONFIG, **(config or {})}

    end_week = int(window["week"].max())
    tenure = end_week - int(first_active_week)
    window_active = float(window["is_active"].sum())
    future_active = float(future["is_active"].sum())

    # 1. New customer onboarding
    if tenure <= cfg["onboarding_max_tenure_weeks"] and window_active >= 1:
        return "new_customer_onboarding"

    # 2. Win-back reminder: established customer going silent
    if window_active >= cfg["win_back_min_window_active"] and future_active == 0:
        return "win_back_reminder"

    if future_active == 0:
        return None

    window_spend = _active_mean_spend(window)
    future_spend = _active_mean_spend(future)
    future_discount = _active_mean(future, "discount_rate")
    future_coupon = float(future["coupon_rate"].max())

    # 3. Price-led coupon: clear future promotion response
    if future_coupon >= 1 or future_discount >= cfg["price_led_min_discount_rate"]:
        return "price_led_coupon"

    # 4. Seasonal spotlight: elevated spend in seasonal weeks
    in_season = any(int(wk) in cfg["seasonal_weeks"] for wk in future["week"])
    if (
        in_season
        and window_spend > 0
        and future_spend > window_spend * cfg["seasonal_min_spend_lift"]
    ):
        return "seasonal_spotlight"

    # 5. Cross-sell bundle: meaningful adoption of a new category
    bought_before = window[category_columns].sum(axis=0) > 0
    future_cat_spend = future[category_columns].sum(axis=0)
    bought_future = future_cat_spend > 0
    new_cats = (~bought_before) & bought_future
    if bool(new_cats.any()):
        new_spend = float(future_cat_spend.loc[new_cats].sum())
        total_future_spend = float(future["sales_value"].sum())
        if total_future_spend > 0 and (new_spend / total_future_spend) >= cfg["cross_sell_min_new_spend_share"]:
            return "cross_sell_bundle"

    # 6. Loyalty reward: sustained, slightly elevated future activity
    if (
        future_active >= cfg["loyalty_min_future_active"]
        and window_spend > 0
        and future_spend >= window_spend * cfg["loyalty_min_spend_lift"]
    ):
        return "loyalty_reward"

    return None  # abstain


def _active_mean_spend(frame: pd.DataFrame) -> float:
    return _active_mean(frame, "sales_value")


def _active_mean(frame: pd.DataFrame, column: str) -> float:
    active = frame.loc[frame["is_active"] > 0, column]
    if len(active) == 0:
        return 0.0
    return float(active.mean())
