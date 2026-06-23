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

from src.campaign_strategist.data import load_retail_transactions
from src.campaign_strategist.features import build_feature_bundle
from src.campaign_strategist.model import train_journey_model
from src.campaign_strategist.viz import (
    GRAY,
    GREEN,
    LIGHT_BLUE,
    add_subtitle,
    apply_report_style,
    despine,
    percent_formatter,
    save_figure,
)


REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"


def run_optimization(synthetic_only: bool = False, epochs: int = 2) -> pd.DataFrame:
    apply_report_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    transactions, source_name = load_retail_transactions(synthetic_only=synthetic_only)
    configs = [
        {"sequence_length": 8, "hidden_size": 32, "learning_rate": 1e-3},
        {"sequence_length": 12, "hidden_size": 48, "learning_rate": 1e-3},
        {"sequence_length": 12, "hidden_size": 64, "learning_rate": 7e-4},
        {"sequence_length": 16, "hidden_size": 48, "learning_rate": 7e-4},
    ]

    rows: list[dict[str, object]] = []
    for idx, config in enumerate(configs, start=1):
        bundle = build_feature_bundle(transactions, sequence_length=config["sequence_length"])
        result = train_journey_model(
            bundle.x,
            bundle.y,
            epochs=epochs,
            hidden_size=config["hidden_size"],
            learning_rate=config["learning_rate"],
        )
        rows.append(
            {
                "run": idx,
                "data_source": source_name,
                "sequence_length": config["sequence_length"],
                "hidden_size": config["hidden_size"],
                "learning_rate": config["learning_rate"],
                "epochs": epochs,
                "accuracy": result.metrics["accuracy"],
                "macro_f1": result.metrics["macro_f1"],
                "training_samples": len(bundle.x),
            }
        )

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    results.to_csv(TABLE_DIR / "optimization_results.csv", index=False)
    best = results.iloc[0].to_dict()
    (TABLE_DIR / "best_hyperparameters.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    plot_results = results.sort_values("macro_f1", ascending=True).copy()
    labels = [
        f"Seq {row.sequence_length} | Hidden {row.hidden_size} | LR {row.learning_rate:g}"
        for row in plot_results.itertuples()
    ]
    colors = [LIGHT_BLUE] * len(plot_results)
    colors[-1] = GREEN
    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars = ax.barh(labels, plot_results["macro_f1"], color=colors, edgecolor="white", linewidth=1.2)
    for bar, value in zip(bars, plot_results["macro_f1"]):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", color=GRAY)
    ax.set_title("LSTM Hyperparameter Search", loc="left", pad=24)
    add_subtitle(ax, "Macro-F1 leaderboard; best configuration highlighted in orange")
    ax.set_xlabel("Macro-F1")
    ax.set_ylabel("")
    ax.set_xlim(0, min(1, plot_results["macro_f1"].max() + 0.18))
    ax.xaxis.set_major_formatter(percent_formatter)
    ax.grid(axis="x", color="#E7ECF3", linewidth=1)
    despine(ax)
    save_figure(fig, FIGURE_DIR / "optimization_macro_f1.png")

    markdown = f"""# Model Optimization Summary

The optimization experiment compares a small set of LSTM configurations. The search is intentionally limited because the capstone timeline is seven weeks and the project also includes an end-to-end application layer.

## Best Configuration

```json
{json.dumps(best, indent=2)}
```

## Output Artifacts

- `reports/tables/optimization_results.csv`
- `reports/tables/best_hyperparameters.json`
- `reports/figures/optimization_macro_f1.png`

## Interpretation

Macro-F1 is used as the primary optimization metric because the activation-style labels can be imbalanced. The best configuration can be used for the final training run and discussed in the experimental methods section.
"""
    (REPORT_DIR / "model_optimization_summary.md").write_text(markdown, encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small hyperparameter search for the LSTM model.")
    parser.add_argument("--synthetic-only", action="store_true", help="Use synthetic demo data only.")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per optimization run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_optimization(synthetic_only=args.synthetic_only, epochs=args.epochs)
    print("Optimization complete")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
