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
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.campaign_strategist.baselines import train_baselines
from src.campaign_strategist.config import ARTIFACT_DIR, CAMPAIGN_CLASSES
from src.campaign_strategist.train import train_pipeline
from src.campaign_strategist.viz import (
    BLUE,
    DARK,
    GRAY,
    GREEN,
    LIGHT_BLUE,
    ORANGE,
    add_subtitle,
    apply_report_style,
    clean_label,
    despine,
    percent_formatter,
    save_figure,
)


REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"


def plot_training_loss(history: list[dict[str, float]], path: Path) -> None:
    frame = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(frame["epoch"], frame["loss"], marker="o", color=BLUE, linewidth=2.8, markersize=7)
    ax.fill_between(frame["epoch"], frame["loss"], frame["loss"].min() * 0.98, color=LIGHT_BLUE, alpha=0.55)
    best = frame.loc[frame["loss"].idxmin()]
    ax.scatter(best["epoch"], best["loss"], color=ORANGE, s=90, zorder=3)
    ax.annotate(
        f"Lowest loss: {best['loss']:.3f}",
        xy=(best["epoch"], best["loss"]),
        xytext=(best["epoch"], best["loss"] * 1.08),
        arrowprops={"arrowstyle": "->", "color": GRAY},
        color=GRAY,
        ha="center",
    )
    ax.set_title("LSTM Training Loss", loc="left", pad=24)
    add_subtitle(ax, "Lower loss indicates the sequence model is learning activation-style patterns")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(axis="y", color="#E7ECF3", linewidth=1)
    despine(ax)
    save_figure(fig, path)


def plot_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    array = np.asarray(matrix)
    row_totals = array.sum(axis=1, keepdims=True)
    normalized = np.divide(array, row_totals, out=np.zeros_like(array, dtype=float), where=row_totals != 0)

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=max(0.01, normalized.max()))
    ax.set_title("LSTM Confusion Matrix", loc="left", pad=28)
    add_subtitle(ax, "Cell labels show normalized row percentage with raw count in parentheses")
    ax.set_xlabel("Predicted activation style")
    ax.set_ylabel("True activation style")
    ax.set_xticks(range(len(CAMPAIGN_CLASSES)))
    ax.set_xticklabels([clean_label(label) for label in CAMPAIGN_CLASSES], rotation=35, ha="right")
    ax.set_yticks(range(len(CAMPAIGN_CLASSES)))
    ax.set_yticklabels([clean_label(label) for label in CAMPAIGN_CLASSES])
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            value = normalized[row, col]
            color = "white" if value > normalized.max() * 0.55 else DARK
            label = f"{value:.0%}\n({array[row, col]:,})" if array[row, col] else "-"
            ax.text(col, row, label, ha="center", va="center", fontsize=8.5, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.yaxis.set_major_formatter(percent_formatter)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_figure(fig, path)


def plot_model_comparison(results: pd.DataFrame, path: Path) -> None:
    plot_data = results.copy()
    plot_data["model_label"] = plot_data["model"].map(clean_label)
    plot_data = plot_data.sort_values("macro_f1")
    y_positions = np.arange(len(plot_data))
    height = 0.34

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y_positions + height / 2, plot_data["accuracy"], height=height, color=LIGHT_BLUE, label="Accuracy")
    ax.barh(y_positions - height / 2, plot_data["macro_f1"], height=height, color=BLUE, label="Macro-F1")
    for ypos, (_, row) in zip(y_positions, plot_data.iterrows()):
        ax.text(row["accuracy"] + 0.015, ypos + height / 2, f"{row['accuracy']:.1%}", va="center", color=GRAY)
        ax.text(row["macro_f1"] + 0.015, ypos - height / 2, f"{row['macro_f1']:.1%}", va="center", color=GRAY)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["model_label"])
    ax.set_xlim(0, min(1, max(plot_data["accuracy"].max(), plot_data["macro_f1"].max()) + 0.18))
    ax.xaxis.set_major_formatter(percent_formatter)
    ax.set_title("Model Comparison", loc="left", pad=24)
    add_subtitle(ax, "Sequence model compared with traditional ML baselines")
    ax.set_xlabel("Evaluation score")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E7ECF3", linewidth=1)
    ax.legend(loc="lower right")
    despine(ax)
    save_figure(fig, path)


def run_training_evaluation(
    synthetic_only: bool = False,
    epochs: int = 4,
    sequence_length: int = 12,
    hidden_size: int = 48,
    learning_rate: float = 1e-3,
) -> dict[str, object]:
    apply_report_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    lstm_metrics = train_pipeline(
        synthetic_only=synthetic_only,
        epochs=epochs,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
    )
    bundle = joblib.load(ARTIFACT_DIR / "feature_bundle.joblib")
    baseline_results = train_baselines(bundle.x, bundle.y)

    comparison_rows = [
        {
            "model": "lstm_sequence_model",
            "accuracy": lstm_metrics["accuracy"],
            "macro_f1": lstm_metrics["macro_f1"],
            "notes": "Deep learning model trained from scratch on weekly customer journey sequences.",
        }
    ]
    for result in baseline_results:
        comparison_rows.append(
            {
                "model": result.name,
                "accuracy": result.metrics["accuracy"],
                "macro_f1": result.metrics["macro_f1"],
                "notes": "Traditional ML baseline using flattened sequence summary features.",
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("macro_f1", ascending=False)
    comparison.to_csv(TABLE_DIR / "model_comparison.csv", index=False)
    (TABLE_DIR / "lstm_metrics.json").write_text(json.dumps(lstm_metrics, indent=2), encoding="utf-8")
    (TABLE_DIR / "baseline_metrics.json").write_text(
        json.dumps({result.name: result.metrics for result in baseline_results}, indent=2),
        encoding="utf-8",
    )

    plot_training_loss(lstm_metrics["history"], FIGURE_DIR / "model_training_loss.png")
    plot_confusion_matrix(lstm_metrics["confusion_matrix"], FIGURE_DIR / "model_confusion_matrix.png")
    plot_model_comparison(comparison, FIGURE_DIR / "model_comparison.png")

    comparison_markdown = comparison[["model", "accuracy", "macro_f1"]].to_string(index=False)
    markdown = f"""# Model Training and Evaluation Summary

## Models Compared

- **LSTM sequence model:** Required deep learning model trained from scratch using PyTorch.
- **KNN classifier baseline:** Traditional ML baseline on flattened sequence features.
- **Random forest baseline:** Traditional ML baseline on flattened sequence features.

## Results

```text
{comparison_markdown}
```

## Evaluation Artifacts

- `reports/tables/model_comparison.csv`
- `reports/tables/lstm_metrics.json`
- `reports/tables/baseline_metrics.json`
- `reports/figures/model_training_loss.png`
- `reports/figures/model_confusion_matrix.png`
- `reports/figures/model_comparison.png`

## Interpretation

The baseline models estimate how much signal can be captured from fixed customer summaries. The LSTM uses the temporal order of weekly behavior, which better matches the customer journey framing of the project. Final interpretation should consider both predictive performance and usefulness of the model outputs in the Streamlit decision-support workflow.
"""
    (REPORT_DIR / "model_training_evaluation_summary.md").write_text(markdown, encoding="utf-8")

    return {
        "lstm": {
            "accuracy": lstm_metrics["accuracy"],
            "macro_f1": lstm_metrics["macro_f1"],
        },
        "baselines": {
            result.name: {
                "accuracy": result.metrics["accuracy"],
                "macro_f1": result.metrics["macro_f1"],
            }
            for result in baseline_results
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM and baseline models.")
    parser.add_argument("--synthetic-only", action="store_true", help="Use synthetic demo data only.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of LSTM training epochs.")
    parser.add_argument("--sequence-length", type=int, default=12, help="Weeks in each sequence.")
    parser.add_argument("--hidden-size", type=int, default=48, help="LSTM hidden size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="LSTM learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_training_evaluation(
        synthetic_only=args.synthetic_only,
        epochs=args.epochs,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
    )
    print("Model training and evaluation complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
