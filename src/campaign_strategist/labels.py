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

# Draft thresholds - owned/tuned by the labeling workstream (ticket #7).
DEFAULT_LABEL_CONFIG: dict[str, float | int | set[int]] = {
    # onboarding: household first became active recently
    "onboarding_max_tenure_weeks": 8,
    # win-back: was active in the input window, silent in the future window
    "win_back_min_window_active": 3,
    # price-led: any coupon redemption, or a high average discount share, in the future
    "price_led_min_discount_rate": 0.15,
    # seasonal: future overlaps seasonal weeks AND spend clearly above the household norm
    "seasonal_weeks": set(range(46, 54)),
    "seasonal_min_spend_lift": 1.2,
    # loyalty: sustained future activity at or above the household's usual spend
    "loyalty_min_future_active": 2,
}


def label_future_window(
    window: pd.DataFrame,
    future: pd.DataFrame,
    first_active_week: int,
    category_columns: list[str],
    config: dict | None = None,
) -> str | None:
    """Assign an activation-style label from future behavior, or None to abstain.

    Parameters
    ----------
    window : weekly rows of the input sequence (past, fed to the model).
    future : weekly rows of the prediction horizon (never fed to the model).
    first_active_week : first week this household was ever active.
    category_columns : weekly spend columns used for new-category detection.
    """
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

    # No future activity and not a win-back case: no observable signal
    if future_active == 0:
        return None

    # 3. Price-led coupon: responds to price incentives in the future
    future_coupon = float(future["coupon_rate"].max())
    future_discount = float(future["discount_rate"].mean())
    if future_coupon >= 1 or future_discount >= cfg["price_led_min_discount_rate"]:
        return "price_led_coupon"

    # 4. Seasonal spotlight: elevated spend during seasonal weeks
    window_spend = float(window["sales_value"].mean())
    future_spend = float(future["sales_value"].mean())
    in_season = any(int(wk) in cfg["seasonal_weeks"] for wk in future["week"])
    if in_season and future_spend > window_spend * cfg["seasonal_min_spend_lift"]:
        return "seasonal_spotlight"

    # 5. Cross-sell bundle: adopts a category not bought in the input window
    bought_before = window[category_columns].sum(axis=0) > 0
    bought_future = future[category_columns].sum(axis=0) > 0
    if bool((~bought_before & bought_future).any()):
        return "cross_sell_bundle"

    # 6. Loyalty reward: sustained activity at/above the household norm
    if future_active >= cfg["loyalty_min_future_active"] and future_spend >= window_spend:
        return "loyalty_reward"

    return None  # abstain
