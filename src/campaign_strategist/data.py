from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import COMPLETEJOURNEY_URLS, RAW_DATA_DIR


PUBLIC_SOURCE_NAME = "completejourney public sample"
SYNTHETIC_SOURCE_NAME = "synthetic retail simulator"


def _download_file(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def _read_rda(url_key: str) -> pd.DataFrame:
    try:
        import pyreadr
    except ImportError as exc:
        raise RuntimeError("pyreadr is required to parse completejourney R data files.") from exc

    url = COMPLETEJOURNEY_URLS[url_key]
    local_path = RAW_DATA_DIR / Path(url).name
    _download_file(url, local_path)
    result = pyreadr.read_r(local_path)
    if not result:
        raise RuntimeError(f"No data frame found in {local_path}")
    return next(iter(result.values()))


def load_completejourney_sample() -> tuple[pd.DataFrame, str]:
    """Load the public completejourney sample and normalize column names."""
    transactions = _read_rda("transactions_sample")
    products = _read_rda("products")

    transactions.columns = [str(col).lower() for col in transactions.columns]
    products.columns = [str(col).lower() for col in products.columns]

    if "product_id" not in transactions.columns or "product_id" not in products.columns:
        raise RuntimeError("completejourney data did not contain the expected product_id column.")

    product_cols = [
        col
        for col in ["product_id", "department", "product_category", "product_type", "brand"]
        if col in products.columns
    ]
    data = transactions.merge(products[product_cols], on="product_id", how="left")

    if "product_category" not in data.columns:
        data["product_category"] = data.get("department", "unknown")

    data = _normalize_transaction_columns(data)
    return data, PUBLIC_SOURCE_NAME


def load_retail_transactions(synthetic_only: bool = False) -> tuple[pd.DataFrame, str]:
    if synthetic_only:
        return generate_synthetic_retail_transactions(), SYNTHETIC_SOURCE_NAME

    try:
        return load_completejourney_sample()
    except Exception:
        return generate_synthetic_retail_transactions(), SYNTHETIC_SOURCE_NAME


def _normalize_transaction_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    defaults = {
        "household_id": "unknown",
        "week": 1,
        "quantity": 1.0,
        "sales_value": 0.0,
        "retail_disc": 0.0,
        "coupon_disc": 0.0,
        "coupon_match_disc": 0.0,
        "product_category": "unknown",
    }
    for col, default in defaults.items():
        if col not in normalized.columns:
            normalized[col] = default

    normalized["household_id"] = normalized["household_id"].astype(str)
    normalized["week"] = pd.to_numeric(normalized["week"], errors="coerce").fillna(1).astype(int)
    normalized["quantity"] = pd.to_numeric(normalized["quantity"], errors="coerce").fillna(0.0)
    normalized["sales_value"] = pd.to_numeric(normalized["sales_value"], errors="coerce").fillna(0.0)
    normalized["retail_disc"] = pd.to_numeric(normalized["retail_disc"], errors="coerce").fillna(0.0)
    normalized["coupon_disc"] = pd.to_numeric(normalized["coupon_disc"], errors="coerce").fillna(0.0)
    normalized["coupon_match_disc"] = pd.to_numeric(
        normalized["coupon_match_disc"], errors="coerce"
    ).fillna(0.0)
    normalized["product_category"] = (
        normalized["product_category"].fillna("unknown").astype(str).str.lower().str.strip()
    )
    normalized["discount_value"] = (
        normalized["retail_disc"] + normalized["coupon_disc"] + normalized["coupon_match_disc"]
    ).clip(lower=0.0)
    normalized["used_coupon"] = (normalized["coupon_disc"] + normalized["coupon_match_disc"] > 0).astype(int)
    return normalized[
        [
            "household_id",
            "week",
            "product_category",
            "quantity",
            "sales_value",
            "discount_value",
            "used_coupon",
        ]
    ]


def stable_household_id(prefix: str, index: int) -> str:
    digest = hashlib.sha1(f"{prefix}-{index}".encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


def generate_synthetic_retail_transactions(
    n_households: int = 900,
    n_weeks: int = 53,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate realistic-enough retail journeys for demos and tests."""
    rng = np.random.default_rng(random_state)
    categories = np.array(
        [
            "beverages",
            "snacks",
            "dairy",
            "frozen",
            "bakery",
            "meat",
            "produce",
            "personal care",
            "household",
            "breakfast",
        ]
    )
    archetypes = [
        "new",
        "lapsed",
        "coupon_sensitive",
        "loyal",
        "seasonal",
        "explorer",
    ]
    records: list[dict[str, object]] = []

    for idx in range(n_households):
        household_id = stable_household_id("hh", idx)
        archetype = rng.choice(archetypes, p=[0.14, 0.14, 0.22, 0.22, 0.12, 0.16])
        primary = rng.choice(categories)
        secondary = rng.choice(categories[categories != primary])
        start_week = int(rng.integers(1, 34)) if archetype == "new" else 1
        base_trip_rate = {
            "new": 0.35,
            "lapsed": 0.55,
            "coupon_sensitive": 0.72,
            "loyal": 0.85,
            "seasonal": 0.55,
            "explorer": 0.65,
        }[archetype]

        for week in range(start_week, n_weeks + 1):
            trip_rate = base_trip_rate
            if archetype == "lapsed" and week > 34:
                trip_rate *= 0.28
            if archetype == "seasonal" and week in list(range(20, 28)) + list(range(45, 53)):
                trip_rate *= 1.65
            if rng.random() > min(trip_rate, 0.98):
                continue

            basket_items = int(rng.integers(2, 8))
            for _ in range(basket_items):
                if rng.random() < 0.55:
                    category = primary
                elif rng.random() < 0.35:
                    category = secondary
                else:
                    category = rng.choice(categories)

                quantity = float(rng.integers(1, 4))
                base_price = float(rng.uniform(2.0, 18.0))
                sales_value = round(quantity * base_price * rng.uniform(0.75, 1.15), 2)
                coupon_probability = 0.42 if archetype == "coupon_sensitive" else 0.12
                used_coupon = int(rng.random() < coupon_probability)
                discount_value = round(sales_value * rng.uniform(0.08, 0.32), 2) if used_coupon else 0.0

                records.append(
                    {
                        "household_id": household_id,
                        "week": week,
                        "product_category": str(category),
                        "quantity": quantity,
                        "sales_value": sales_value,
                        "discount_value": discount_value,
                        "used_coupon": used_coupon,
                    }
                )

    return pd.DataFrame.from_records(records)
