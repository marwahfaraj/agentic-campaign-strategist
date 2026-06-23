from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CAMPAIGN_CLASSES


@dataclass
class BaselineResult:
    name: str
    metrics: dict[str, object]


def flatten_sequences(x: np.ndarray) -> np.ndarray:
    """Convert weekly sequences into fixed-width tabular features for baseline models."""
    means = x.mean(axis=1)
    recent = x[:, -1, :]
    trend = x[:, -1, :] - x[:, 0, :]
    return np.concatenate([means, recent, trend], axis=1)


def train_baselines(x: np.ndarray, y: np.ndarray, random_state: int = 7) -> list[BaselineResult]:
    features = flatten_sequences(x)
    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    models = {
        "knn_classifier": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=15, weights="distance")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
    }

    results: list[BaselineResult] = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        present_labels = sorted(np.unique(np.concatenate([y_test, predictions])).tolist())
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=present_labels,
                target_names=[CAMPAIGN_CLASSES[idx] for idx in present_labels],
                zero_division=0,
                output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=list(range(len(CAMPAIGN_CLASSES))),
            ).tolist(),
            "test_size": int(len(y_test)),
            "train_size": int(len(y_train)),
        }
        results.append(BaselineResult(name=name, metrics=metrics))
    return results
