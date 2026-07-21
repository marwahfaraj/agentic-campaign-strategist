"""End-to-end artifact preparation for the Streamlit app.

Mirrors notebooks 01-03 (clean -> sequences/labels -> train) so the app can
bootstrap itself on first launch when artifacts are missing. Uses the public
Complete Journey data with a synthetic fallback.
"""

from __future__ import annotations

import json

import joblib
import numpy as np

from .config import ARTIFACT_DIR, PROCESSED_DATA_DIR
from .data import load_retail_transactions
from .features import build_sequence_dataset
from .model import save_model
from .training import grouped_train_val_test_split, predict_labels, scale_splits, train_lstm, classification_metrics

EXCLUDED_CATEGORIES = ["coupon/misc items", "fuel"]

MODEL_PATH = ARTIFACT_DIR / "journey_lstm.pt"
SCALER_PATH = ARTIFACT_DIR / "sequence_scaler.joblib"
MODEL_CONFIG_PATH = ARTIFACT_DIR / "model_config.json"
TRANSACTIONS_PATH = PROCESSED_DATA_DIR / "transactions_clean.parquet"
SEQUENCES_PATH = PROCESSED_DATA_DIR / "sequences_x.npy"
LABELS_PATH = PROCESSED_DATA_DIR / "labels_y.npy"
SAMPLE_INDEX_PATH = PROCESSED_DATA_DIR / "sample_index.parquet"

APP_ARTIFACTS = [MODEL_PATH, SCALER_PATH, MODEL_CONFIG_PATH, TRANSACTIONS_PATH, SEQUENCES_PATH, SAMPLE_INDEX_PATH]


def app_artifacts_ready() -> bool:
    return all(path.exists() for path in APP_ARTIFACTS)


def prepare_app_artifacts(
    synthetic_only: bool = False,
    epochs: int = 12,
    hidden_size: int = 128,
    num_layers: int = 2,
    learning_rate: float = 8e-4,
) -> dict[str, object]:
    """Build cleaned data, sequences, and a trained model for the app."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    transactions, source = load_retail_transactions(synthetic_only=synthetic_only)
    transactions = transactions.drop_duplicates()
    transactions = transactions[
        ~transactions["product_category"].isin(EXCLUDED_CATEGORIES)
    ].reset_index(drop=True)
    transactions.to_parquet(TRANSACTIONS_PATH, index=False)

    dataset = build_sequence_dataset(
        transactions,
        sequence_length=12,
        prediction_horizon=4,
        top_n_categories=8,
        min_window_active_weeks=2,
    )
    np.save(SEQUENCES_PATH, dataset.x)
    np.save(LABELS_PATH, dataset.y)
    dataset.sample_index.to_parquet(SAMPLE_INDEX_PATH, index=False)
    (PROCESSED_DATA_DIR / "feature_names.txt").write_text("\n".join(dataset.feature_names))

    split = grouped_train_val_test_split(
        dataset.x, dataset.y, dataset.sample_index["household_id"].to_numpy(), random_state=7
    )
    scaled = scale_splits(dataset.x, dataset.y, split)

    model, _history = train_lstm(
        scaled.x_train,
        scaled.y_train,
        scaled.x_val,
        scaled.y_val,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        epochs=epochs,
        patience=4,
        batch_size=128,
        random_state=7,
    )
    test_metrics = classification_metrics(scaled.y_test, predict_labels(model, scaled.x_test))

    save_model(model, str(MODEL_PATH))
    joblib.dump(scaled.scaler, SCALER_PATH)

    config = {
        "n_features": int(dataset.x.shape[-1]),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "sequence_length": 12,
        "data_source": source,
        "n_samples": int(len(dataset.y)),
        "accuracy": float(test_metrics["accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
    }
    MODEL_CONFIG_PATH.write_text(json.dumps(config, indent=2))
    return config
