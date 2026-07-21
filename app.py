from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.campaign_strategist.config import CAMPAIGN_LABELS
from src.campaign_strategist.features import filter_segment_rows, parse_segment_request, profile_segment
from src.campaign_strategist.model import load_model, predict_probabilities
from src.campaign_strategist.pipeline import (
    MODEL_CONFIG_PATH,
    MODEL_PATH,
    SAMPLE_INDEX_PATH,
    SCALER_PATH,
    SEQUENCES_PATH,
    TRANSACTIONS_PATH,
    app_artifacts_ready,
    prepare_app_artifacts,
)
from src.campaign_strategist.strategy import ollama_status, recommend_campaign, warm_up_ollama
from src.campaign_strategist.training import transform_sequences

APP_DIR = Path(__file__).resolve().parent
HERO_IMAGE = APP_DIR / "assets" / "agentic_retail_campaign_hero.png"
ARCHITECTURE_IMAGE = APP_DIR / "assets" / "audience_campaign_fit_architecture.png"
FIGURES_DIR = APP_DIR / "reports" / "figures"

ACCENT = "#2563eb"
MUTED = "#94a3b8"

# Which activation styles serve each campaign objective (used for the fit score).
OBJECTIVE_FIT = {
    "Increase repeat purchase": ["new_customer_onboarding", "loyalty_reward", "price_led_coupon"],
    "Grow basket size": ["cross_sell_bundle", "price_led_coupon"],
    "Win back lapsed shoppers": ["win_back_reminder", "price_led_coupon"],
    "Promote seasonal demand": ["seasonal_spotlight", "cross_sell_bundle"],
    "Protect loyal customers": ["loyalty_reward", "cross_sell_bundle"],
}

EXAMPLE_QUERIES = [
    "Create me a segment of households who bought beverages recently and used coupons before",
    "Create me a segment of game day heavy shoppers for a seasonal campaign",
    "Find lapsed snack buyers for a win-back coupon campaign",
    "Audience of loyal dairy shoppers for a weekend bundle campaign",
    "Create a segment of new shoppers for a personal care onboarding campaign",
]

st.set_page_config(
    page_title="Audience-to-Campaign Fit Advisor",
    page_icon="🎯",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not app_artifacts_ready():
        prepare_app_artifacts(synthetic_only=False, epochs=8)

    config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    model = load_model(
        str(MODEL_PATH),
        n_features=int(config["n_features"]),
        hidden_size=int(config.get("hidden_size", 128)),
        num_layers=int(config.get("num_layers", 2)),
    )
    scaler = joblib.load(SCALER_PATH)
    transactions = pd.read_parquet(TRANSACTIONS_PATH)
    sequences = np.load(SEQUENCES_PATH)
    sample_index = pd.read_parquet(SAMPLE_INDEX_PATH)
    return model, scaler, transactions, sequences, sample_index, config


@st.cache_data(show_spinner=False)
def cached_recommendation(mean_probs: tuple, profile_json: str, query: str, objective: str, use_llm: bool):
    """Cache so Streamlit reruns (every widget click) don't re-call the LLM."""
    probabilities = np.asarray(mean_probs, dtype=np.float32).reshape(1, -1)
    return recommend_campaign(
        probabilities=probabilities,
        segment_profile=json.loads(profile_json),
        marketer_query=query,
        objective=objective,
        use_llm=use_llm,
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def probability_chart(probability_table: list[dict[str, object]], recommended_label: str):
    chart_data = pd.DataFrame(probability_table).sort_values("probability")
    colors = [ACCENT if s == recommended_label else "#cbd5e1" for s in chart_data["strategy"]]
    fig = go.Figure(
        go.Bar(
            x=chart_data["probability"],
            y=chart_data["strategy"],
            orientation="h",
            marker_color=colors,
            text=chart_data["probability"].map(lambda v: f"{v:.0%}"),
            textposition="outside",
        )
    )
    fig.update_layout(
        height=300,
        template="plotly_white",
        xaxis=dict(title="", tickformat=".0%", range=[0, max(0.1, chart_data["probability"].max()) * 1.25]),
        yaxis_title="",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def fit_gauge(fit_score: float, verdict: str):
    color = "#16a34a" if fit_score >= 0.5 else ("#d97706" if fit_score >= 0.3 else "#dc2626")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fit_score * 100,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": f"<b>{verdict}</b>", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": color, "thickness": 0.35},
                "steps": [
                    {"range": [0, 30], "color": "#fee2e2"},
                    {"range": [30, 50], "color": "#fef3c7"},
                    {"range": [50, 100], "color": "#dcfce7"},
                ],
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=25, r=25, t=45, b=5))
    return fig


def activity_trend_chart(transactions: pd.DataFrame, households: list[str]):
    segment = transactions[transactions["household_id"].astype(str).isin(households)]
    weekly = (
        segment.groupby("week")
        .agg(active_households=("household_id", "nunique"), spend=("sales_value", "sum"))
        .reset_index()
    )
    fig = px.area(weekly, x="week", y="active_households", template="plotly_white")
    fig.update_traces(line_color=ACCENT, fillcolor="rgba(37, 99, 235, 0.15)")
    fig.update_layout(
        height=260,
        xaxis_title="Week",
        yaxis_title="Active households",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def category_mix_chart(transactions: pd.DataFrame, households: list[str]):
    segment = transactions[transactions["household_id"].astype(str).isin(households)]
    mix = (
        segment.groupby("product_category")["sales_value"]
        .sum()
        .sort_values(ascending=False)
        .head(7)
        .reset_index()
    )
    fig = px.pie(
        mix,
        names="product_category",
        values="sales_value",
        hole=0.55,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig.update_traces(textinfo="percent+label", textfont_size=11)
    fig.update_layout(height=260, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
with st.spinner("Loading model and retail journey data (first launch trains a model, ~2 min)..."):
    model, scaler, transactions, sequences, sample_index, config = load_artifacts()

categories = sorted(transactions["product_category"].astype(str).str.lower().unique().tolist())


# ---------------------------------------------------------------------------
# Sidebar: model card + explanation engine
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Model Card")
    st.metric("Test Macro-F1", f"{config.get('macro_f1', 0):.2f}")
    st.metric("Test Accuracy", f"{config.get('accuracy', 0):.1%}")
    st.metric("Labeled Journeys", f"{config.get('n_samples', 0):,}")
    st.caption(
        f"Bidirectional LSTM over {config.get('sequence_length', 12)}-week household journeys. "
        f"Data: {config.get('data_source', 'unknown')}. Leakage-safe: future-window labels, "
        "household-grouped splits, train-only scaling."
    )

    st.divider()
    st.markdown("### 💬 Explanation Engine")
    status = ollama_status()
    if status["running"] and status["model_ready"]:
        st.success(f"Ollama online · {status['model']}")
    elif status["running"]:
        st.warning(f"Ollama online, but `{status['model']}` is not pulled")
    else:
        st.info("Ollama offline — using built-in explanations")
    use_llm = st.toggle(
        "Generate explanations with local Mistral",
        value=bool(status["running"] and status["model_ready"]),
        help="Uses the free local Ollama server. If unavailable, the app falls back to a deterministic explanation.",
    )
    if use_llm and status["running"] and not st.session_state.get("ollama_warmed"):
        warm_up_ollama()
        st.session_state["ollama_warmed"] = True

    st.divider()
    if st.button("🔄 Rebuild artifacts (retrain)"):
        with st.spinner("Rebuilding data and retraining the model..."):
            prepare_app_artifacts(synthetic_only=False, epochs=8)
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
header_text, header_image = st.columns([1.6, 1], gap="large", vertical_alignment="center")
with header_text:
    st.title("Audience-to-Campaign Fit Advisor")
    st.markdown(
        "An agentic segmentation system can *build* any audience a marketer asks for — but is that audience "
        "actually **right for the campaign**? This advisor scores the audience against six activation styles, "
        "checks the fit with your objective, and explains the verdict in marketer language."
    )
with header_image:
    if HERO_IMAGE.exists():
        st.image(str(HERO_IMAGE), width="stretch")

tab_advisor, tab_how, tab_performance = st.tabs(["🎯 Campaign Advisor", "⚙️ How It Works", "📈 Model Performance"])


# ===========================================================================
# TAB 1 — the main advisor flow
# ===========================================================================
with tab_advisor:
    # -- Step 1: the ask -----------------------------------------------------
    st.markdown("#### Step 1 · Describe the audience and your campaign goal")
    ask_left, ask_right = st.columns([1.4, 1], gap="large")
    with ask_left:
        selected_example = st.selectbox("Pick an example, or write your own below", EXAMPLE_QUERIES)
        marketer_query = st.text_area(
            "Audience request (what you'd ask the segmentation agent)",
            value=selected_example,
            height=90,
        )
    with ask_right:
        objective = st.selectbox("Campaign objective", list(OBJECTIVE_FIT.keys()))
        st.caption(
            "The advisor checks whether the audience's *predicted behavior* actually serves this goal — "
            "that's the gap it closes after segmentation."
        )

    # Simulate the upstream SegmentAI audience and score it.
    segment_rows = filter_segment_rows(marketer_query, transactions, sample_index, categories)
    parsed_request = segment_rows.attrs.get("segment_parse", parse_segment_request(marketer_query, categories))
    households = segment_rows["household_id"].astype(str).unique().tolist()
    segment_profile = profile_segment(transactions, households)

    segment_x = transform_sequences(sequences[segment_rows.index.to_numpy()], scaler)
    probabilities = predict_probabilities(model, segment_x)
    mean_probs = probabilities.mean(axis=0)

    spinner_text = "Asking local Mistral for an explanation..." if use_llm else "Scoring the audience..."
    with st.spinner(spinner_text):
        recommendation = cached_recommendation(
            tuple(float(p) for p in mean_probs),
            json.dumps(segment_profile),
            marketer_query,
            objective,
            use_llm,
        )

    st.divider()

    # -- Step 2: audience snapshot -------------------------------------------
    st.markdown("#### Step 2 · Meet the audience the agent found")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Households", f"{segment_profile['households']:,}")
    m2.metric("Avg Weekly Spend", f"${segment_profile['avg_weekly_spend']:,.2f}")
    m3.metric("Coupon Usage", f"{segment_profile['coupon_rate']:.1%}")
    m4.metric("Recently Active", f"{segment_profile['recent_activity_rate']:.1%}")

    snap_left, snap_right = st.columns(2, gap="large")
    with snap_left:
        st.markdown("**Weekly engagement of this audience**")
        st.plotly_chart(activity_trend_chart(transactions, households), width="stretch")
    with snap_right:
        st.markdown("**Where their money goes**")
        st.plotly_chart(category_mix_chart(transactions, households), width="stretch")

    with st.expander("🔍 How the request was interpreted"):
        detected = parsed_request.get("categories") or ["none detected"]
        active_filters = [
            label
            for label, active in [
                ("lapsed shoppers", parsed_request.get("lapsed")),
                ("coupon / price sensitive", parsed_request.get("coupon_sensitive")),
                ("loyal or heavy shoppers", parsed_request.get("loyal_or_heavy")),
                ("new customers", parsed_request.get("new_customer")),
                ("seasonal / occasion based", parsed_request.get("seasonal")),
            ]
            if active
        ]
        st.write(f"**Category signals:** {', '.join(detected)}")
        st.write(f"**Lookback window:** last {parsed_request.get('recent_window', 16)} weeks")
        st.write(f"**Behavior filters:** {', '.join(active_filters) if active_filters else 'none'}")

    st.divider()

    # -- Step 3: the verdict ---------------------------------------------------
    st.markdown("#### Step 3 · The advisor's verdict")

    aligned = OBJECTIVE_FIT[objective]
    strategy_keys = {CAMPAIGN_LABELS[k]: k for k in CAMPAIGN_LABELS}
    fit_score = float(
        sum(
            row["probability"]
            for row in recommendation.probability_table
            if strategy_keys.get(str(row["strategy"])) in aligned
        )
    )
    if fit_score >= 0.5:
        verdict = "Strong fit"
    elif fit_score >= 0.3:
        verdict = "Moderate fit"
    else:
        verdict = "Weak fit"

    v_left, v_mid, v_right = st.columns([0.9, 1.1, 1.2], gap="large")
    with v_left:
        st.markdown(f"**Does this audience fit “{objective}”?**")
        st.plotly_chart(fit_gauge(fit_score, verdict), width="stretch")
        st.caption("Share of the audience's predicted behavior that serves your objective.")
    with v_mid:
        st.markdown("**Recommended activation style**")
        st.success(f"**{recommendation.strategy_label}**\n\nModel confidence: {recommendation.confidence:.0%}")
        st.markdown("**How every style scored**")
        st.plotly_chart(
            probability_chart(recommendation.probability_table, recommendation.strategy_label),
            width="stretch",
        )
    with v_right:
        st.markdown("**Why (in marketer language)**")
        if recommendation.explanation_source == "ollama":
            st.caption("✨ Generated live by local Mistral (Ollama)")
        elif use_llm:
            st.caption("⚠️ Ollama was unavailable or timed out — showing the built-in explanation.")
        st.info(recommendation.explanation)

        st.markdown("**Ready-to-use campaign message**")
        st.markdown(f"> 💬 *{recommendation.suggested_message}*")

        st.markdown("**Before you launch, validate:**")
        for note in recommendation.risk_notes:
            st.markdown(f"- {note}")

    st.divider()
    st.caption(
        "Demo only. Built on public (Complete Journey) or synthetic data — no proprietary company, "
        "customer, or internal system data."
    )


# ===========================================================================
# TAB 2 — how it works
# ===========================================================================
with tab_how:
    st.markdown("### The idea in one sentence")
    st.markdown(
        "> Agentic segmentation tools answer **“who matches my description?”** — this project answers the "
        "next question: **“is that audience actually right for my campaign, and how should I activate it?”**"
    )
    if ARCHITECTURE_IMAGE.exists():
        st.image(str(ARCHITECTURE_IMAGE), width="stretch")

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("#### 1 · Simulated SegmentAI")
        st.markdown(
            "A natural-language request is parsed into interpretable filters (categories, recency, "
            "coupon sensitivity, loyalty) and applied to ~2,000 households of real retail transaction "
            "data — standing in for an upstream agentic segmentation system."
        )
    with c2:
        st.markdown("#### 2 · Deep learning fit scorer")
        st.markdown(
            "Each household's last 12 weeks of shopping become a behavior sequence. A bidirectional "
            "LSTM with attention predicts which of six activation styles each household is most likely "
            "to respond to over the next 4 weeks."
        )
    with c3:
        st.markdown("#### 3 · Advisor layer")
        st.markdown(
            "Predictions are aggregated into a fit score against the marketer's objective, and a local "
            "LLM (Ollama Mistral) turns the numbers into a plain-language recommendation with risks "
            "to validate."
        )

    with st.expander("🎤 30-second demo script (for presenting this app)"):
        st.markdown(
            """
1. *"Marketers can now ask an AI agent for any audience — but nobody checks if the audience fits the campaign goal. That's the gap we close."*
2. Pick an example request in **Step 1** and choose an objective that matches (e.g. *lapsed snack buyers* + *Win back lapsed shoppers*).
3. Point at **Step 2**: *"The system profiles the audience it found — size, spend, coupon behavior, category mix."*
4. Point at **Step 3**: *"Our LSTM scores every household's journey, the gauge shows the fit with my objective, and a local LLM explains the recommendation in marketing language."*
5. Change the objective to a mismatched one and show the fit gauge drop — *"same audience, wrong campaign — and the advisor catches it."*
            """
        )

    st.markdown("#### Why the evaluation is honest")
    st.markdown(
        "- **Future-window labels** — the training label comes only from what the household does in the *next* "
        "4 weeks, never from the input window.\n"
        "- **Household-grouped splits** — no household appears in both train and test.\n"
        "- **Train-only scaling** — normalization statistics never see the test set.\n"
    )


# ===========================================================================
# TAB 3 — model performance
# ===========================================================================
with tab_performance:
    st.markdown("### Evaluation results")
    p1, p2, p3 = st.columns(3)
    p1.metric("Test Macro-F1", f"{config.get('macro_f1', 0):.2f}", help="Primary metric — classes are imbalanced.")
    p2.metric("Test Accuracy", f"{config.get('accuracy', 0):.1%}")
    p3.metric("Chance level (6 classes)", "16.7%")

    figures = [
        ("model_comparison.png", "LSTM vs. KNN and Random Forest baselines"),
        ("model_confusion_matrix.png", "Where the model confuses activation styles"),
        ("optimization_macro_f1.png", "Hyperparameter search over macro-F1"),
        ("model_training_loss.png", "Training and validation loss"),
        ("eda_activation_label_distribution.png", "Weak-label class distribution"),
        ("eda_weekly_active_households.png", "Weekly active households in the data"),
    ]
    available = [(FIGURES_DIR / name, caption) for name, caption in figures if (FIGURES_DIR / name).exists()]
    if available:
        for row_start in range(0, len(available), 2):
            cols = st.columns(2, gap="large")
            for col, (path, caption) in zip(cols, available[row_start : row_start + 2]):
                with col:
                    st.image(str(path), caption=caption, width="stretch")
    else:
        st.info("Run notebooks 03 and 04 to generate evaluation figures (reports/figures/).")
