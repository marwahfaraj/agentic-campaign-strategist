from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.campaign_strategist.config import ARTIFACT_DIR, CAMPAIGN_LABELS, PROCESSED_DATA_DIR
from src.campaign_strategist.features import filter_segment_samples, profile_segment, parse_segment_request, transform_segment_sequences
from src.campaign_strategist.model import load_model, predict_probabilities
from src.campaign_strategist.strategy import recommend_campaign
from src.campaign_strategist.train import train_pipeline


st.set_page_config(
    page_title="Audience-to-Campaign Fit Advisor",
    page_icon=":bar_chart:",
    layout="wide",
)


MODEL_PATH = ARTIFACT_DIR / "journey_lstm.pt"
BUNDLE_PATH = ARTIFACT_DIR / "feature_bundle.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
TRANSACTIONS_PATH = PROCESSED_DATA_DIR / "app_transactions.parquet"


@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not all(path.exists() for path in [MODEL_PATH, BUNDLE_PATH, TRANSACTIONS_PATH]):
        train_pipeline(synthetic_only=True, epochs=3)

    bundle = joblib.load(BUNDLE_PATH)
    model = load_model(str(MODEL_PATH), n_features=len(bundle.feature_names))
    transactions = pd.read_parquet(TRANSACTIONS_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    return model, bundle, transactions, metrics


def probability_chart(probability_table: list[dict[str, object]]):
    chart_data = pd.DataFrame(probability_table)
    fig = px.bar(
        chart_data,
        x="probability",
        y="strategy",
        orientation="h",
        text=chart_data["probability"].map(lambda value: f"{value:.0%}"),
        color="probability",
        color_continuous_scale="Blues",
        range_x=[0, max(0.05, float(chart_data["probability"].max()) * 1.15)],
    )
    fig.update_layout(
        height=340,
        yaxis_title="",
        xaxis_title="Predicted probability",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


st.title("Audience-to-Campaign Fit Advisor")
st.caption("A post-segmentation advisor for agentic retail audience creation.")

with st.spinner("Loading model and retail journey data..."):
    model, bundle, transactions, metrics = load_artifacts()

with st.sidebar:
    st.header("Model")
    st.metric("Training Samples", f"{metrics.get('n_training_samples', 0):,}")
    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    st.write(f"**Data source:** {metrics.get('data_source', 'unknown')}")
    st.write(f"**Sequence length:** {metrics.get('sequence_length', 12)} weeks")

    st.divider()
    st.header("Explanation Layer")
    use_llm = st.toggle(
        "Use local Ollama Mistral if running",
        value=False,
        help="Free local LLM path. Run `ollama serve` and `ollama pull mistral`; otherwise the app uses a deterministic local explanation.",
    )

    st.divider()
    if st.button("Retrain quick synthetic demo model"):
        with st.spinner("Retraining demo model..."):
            train_pipeline(synthetic_only=True, epochs=3)
            st.cache_resource.clear()
            st.rerun()


default_query = "Create me a segment of households who bought beverages recently and used coupons before"
example_queries = [
    default_query,
    "Create me a segment of game day heavy shoppers for a seasonal campaign",
    "Find lapsed snack buyers for a win-back coupon campaign",
    "Audience of loyal dairy shoppers for a weekend bundle campaign",
    "Create a segment of new shoppers for a personal care onboarding campaign",
]

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("1. Simulated SegmentAI Audience")
    selected_example = st.selectbox("Try an example request", example_queries)
    marketer_query = st.text_area(
        "Natural language audience request",
        value=selected_example,
        height=110,
        help="This simulates the segment returned by an upstream agentic segmentation system.",
    )
    st.caption("Tip: after typing, click outside the text box or press Cmd+Enter so Streamlit reruns the app.")
    objective = st.selectbox(
        "Known campaign objective",
        [
            "Increase repeat purchase",
            "Grow basket size",
            "Win back lapsed shoppers",
            "Promote seasonal demand",
            "Protect loyal customers",
        ],
        help="In the real workflow, the marketer usually already has this goal in mind before requesting an audience.",
    )

    segment_rows = filter_segment_samples(marketer_query, transactions, bundle)
    parsed_request = segment_rows.attrs.get("segment_parse", parse_segment_request(marketer_query, bundle.categories))
    households = segment_rows["household_id"].astype(str).unique().tolist()
    segment_profile = profile_segment(transactions, households)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Audience Households", f"{segment_profile['households']:,}")
    m2.metric("Avg Weekly Spend", f"${segment_profile['avg_weekly_spend']:,.2f}")
    m3.metric("Coupon Usage", f"{segment_profile['coupon_rate']:.1%}")
    m4.metric("Recent Activity", f"{segment_profile['recent_activity_rate']:.1%}")

    st.write("**Top category signals:** " + ", ".join(segment_profile["top_categories"][:5]))
    with st.expander("Parsed audience logic", expanded=True):
        categories = parsed_request.get("categories") or ["none detected"]
        st.write(f"**Category/occasion signals:** {', '.join(categories)}")
        st.write(f"**Lookback window:** last {parsed_request.get('recent_window', 16)} weeks")
        active_filters = [
            label
            for label, active in [
                ("lapsed", parsed_request.get("lapsed")),
                ("coupon/price sensitive", parsed_request.get("coupon_sensitive")),
                ("loyal or heavy shopper", parsed_request.get("loyal_or_heavy")),
                ("new customer", parsed_request.get("new_customer")),
                ("seasonal/occasion based", parsed_request.get("seasonal")),
            ]
            if active
        ]
        st.write(f"**Journey filters:** {', '.join(active_filters) if active_filters else 'none'}")

with right:
    st.subheader("2. Campaign Fit Recommendation")
    segment_x = transform_segment_sequences(bundle, segment_rows)
    probabilities = predict_probabilities(model, segment_x)
    recommendation = recommend_campaign(
        probabilities=probabilities,
        segment_profile=segment_profile,
        marketer_query=marketer_query,
        objective=objective,
        use_llm=use_llm,
    )

    st.success(f"{recommendation.strategy_label} ({recommendation.confidence:.0%} confidence)")
    st.write(recommendation.explanation)

    st.write("**Suggested activation message**")
    st.info(recommendation.suggested_message)

    st.write("**Risk and validation notes**")
    for note in recommendation.risk_notes:
        st.write(f"- {note}")

st.divider()

chart_col, table_col = st.columns([1.1, 0.9], gap="large")
with chart_col:
    st.subheader("Model Strategy Distribution")
    st.plotly_chart(probability_chart(recommendation.probability_table), use_container_width=True)

with table_col:
    st.subheader("How This Maps To SegmentAI")
    st.markdown(
        """
        **SegmentAI-like system:** creates the audience from a marketer's campaign request.

        **This capstone layer:** checks whether the audience fits the campaign objective, recommends the best activation style, and explains why.

        **Production extension:** replace the synthetic/public data and weak labels with approved campaign response data, incrementality metrics, and policy guardrails.
        """
    )
    display_table = pd.DataFrame(recommendation.probability_table)
    display_table["probability"] = display_table["probability"].map(lambda value: f"{value:.1%}")
    st.dataframe(display_table, hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "Demo only. This project uses public or synthetic data and does not include proprietary company data, customer data, or internal system details."
)
