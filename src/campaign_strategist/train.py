from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from .config import ARTIFACT_DIR, PROCESSED_DATA_DIR
from .data import load_retail_transactions
from .features import build_feature_bundle
from .model import save_model, train_journey_model


def train_pipeline(
    synthetic_only: bool = False,
    epochs: int = 6,
    sequence_length: int = 12,
    prediction_horizon: int = 4,
    learning_rate: float = 1e-3,
    hidden_size: int = 48,
) -> dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    transactions, source_name = load_retail_transactions(synthetic_only=synthetic_only)
    bundle = build_feature_bundle(
        transactions,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
    )
    result = train_journey_model(
        bundle.x,
        bundle.y,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
    )

    model_path = ARTIFACT_DIR / "journey_lstm.pt"
    bundle_path = ARTIFACT_DIR / "feature_bundle.joblib"
    metrics_path = ARTIFACT_DIR / "metrics.json"
    transactions_path = PROCESSED_DATA_DIR / "app_transactions.parquet"

    save_model(result.model, str(model_path))
    joblib.dump(bundle, bundle_path)
    transactions.to_parquet(transactions_path, index=False)

    metrics = {
        **result.metrics,
        "data_source": source_name,
        "n_transactions": int(len(transactions)),
        "n_training_samples": int(len(bundle.x)),
        "n_features": int(bundle.x.shape[-1]),
        "sequence_length": sequence_length,
        "prediction_horizon": prediction_horizon,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "categories": bundle.categories,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the customer journey activation-style model.")
    parser.add_argument("--synthetic-only", action="store_true", help="Skip public data and train on synthetic data.")
    parser.add_argument("--epochs", type=int, default=6, help="Number of PyTorch training epochs.")
    parser.add_argument("--sequence-length", type=int, default=12, help="Weeks in each customer journey sequence.")
    parser.add_argument("--prediction-horizon", type=int, default=4, help="Future weeks used for weak labeling.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="PyTorch AdamW learning rate.")
    parser.add_argument("--hidden-size", type=int, default=48, help="LSTM hidden size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_pipeline(
        synthetic_only=args.synthetic_only,
        epochs=args.epochs,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
    )
    summary = pd.DataFrame(metrics["history"])
    print("Training complete")
    print(f"Data source: {metrics['data_source']}")
    print(f"Samples: {metrics['n_training_samples']:,}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
