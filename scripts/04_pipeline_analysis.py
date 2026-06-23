from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.campaign_strategist.config import ARTIFACT_DIR, PROCESSED_DATA_DIR
from src.campaign_strategist.features import filter_segment_samples, profile_segment, transform_segment_sequences
from src.campaign_strategist.model import load_model, predict_probabilities
from src.campaign_strategist.strategy import recommend_campaign


REPORT_DIR = Path("reports")
TABLE_DIR = REPORT_DIR / "tables"


DEMO_CASES = [
    {
        "query": "Create me a segment of game day heavy shoppers for a seasonal campaign",
        "objective": "Promote seasonal demand",
    },
    {
        "query": "Find lapsed snack buyers for a win-back coupon campaign",
        "objective": "Win back lapsed shoppers",
    },
    {
        "query": "Audience of loyal dairy shoppers for a weekend bundle campaign",
        "objective": "Grow basket size",
    },
    {
        "query": "Create a segment of new shoppers for a personal care onboarding campaign",
        "objective": "Increase repeat purchase",
    },
]


def run_pipeline_analysis() -> list[dict[str, object]]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(ARTIFACT_DIR / "feature_bundle.joblib")
    model = load_model(str(ARTIFACT_DIR / "journey_lstm.pt"), n_features=len(bundle.feature_names))
    transactions = pd.read_parquet(PROCESSED_DATA_DIR / "app_transactions.parquet")

    rows: list[dict[str, object]] = []
    for case in DEMO_CASES:
        segment_rows = filter_segment_samples(case["query"], transactions, bundle)
        households = segment_rows["household_id"].astype(str).unique().tolist()
        segment_profile = profile_segment(transactions, households)
        probabilities = predict_probabilities(model, transform_segment_sequences(bundle, segment_rows))
        recommendation = recommend_campaign(
            probabilities=probabilities,
            segment_profile=segment_profile,
            marketer_query=case["query"],
            objective=case["objective"],
            use_llm=False,
        )
        rows.append(
            {
                "query": case["query"],
                "objective": case["objective"],
                "parsed_logic": segment_rows.attrs.get("segment_parse", {}),
                "audience_households": segment_profile["households"],
                "top_categories": segment_profile["top_categories"][:3],
                "activation_style": recommendation.strategy_label,
                "confidence": recommendation.confidence,
                "suggested_message": recommendation.suggested_message,
                "risk_notes": recommendation.risk_notes,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_DIR / "pipeline_demo_cases.csv", index=False)
    (TABLE_DIR / "pipeline_demo_cases.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    markdown_rows = "\n".join(
        f"- **Input:** {row['query']}  \n"
        f"  **Objective:** {row['objective']}  \n"
        f"  **Audience size:** {row['audience_households']} households  \n"
        f"  **Activation style:** {row['activation_style']} ({row['confidence']:.1%} confidence)  \n"
        f"  **Suggested message:** {row['suggested_message']}"
        for row in rows
    )
    markdown = f"""# Pipeline Analysis and Representative Outputs

This section demonstrates the end-to-end workflow: audience request, simulated SegmentAI audience creation, customer journey scoring, campaign-fit recommendation, and marketer guidance.

{markdown_rows}

## Output Artifacts

- `reports/tables/pipeline_demo_cases.csv`
- `reports/tables/pipeline_demo_cases.json`

## Discussion

These examples are useful for the final presentation because they show representative model inputs and outputs. In a production setting, the simulated audience creation layer would be replaced by an approved segmentation system and the weak labels would be replaced by validated campaign response or incrementality labels.
"""
    (REPORT_DIR / "pipeline_analysis_summary.md").write_text(markdown, encoding="utf-8")
    return rows


def main() -> None:
    rows = run_pipeline_analysis()
    print("Pipeline analysis complete")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
