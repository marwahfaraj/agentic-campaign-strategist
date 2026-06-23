from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import requests

from .config import CAMPAIGN_CLASSES, CAMPAIGN_DESCRIPTIONS, CAMPAIGN_LABELS


@dataclass
class CampaignRecommendation:
    strategy_key: str
    strategy_label: str
    confidence: float
    explanation: str
    suggested_message: str
    risk_notes: list[str]
    probability_table: list[dict[str, object]]


def recommend_campaign(
    probabilities: np.ndarray,
    segment_profile: dict[str, object],
    marketer_query: str,
    objective: str,
    use_llm: bool = False,
) -> CampaignRecommendation:
    mean_probs = probabilities.mean(axis=0)
    top_idx = int(np.argmax(mean_probs))
    strategy_key = CAMPAIGN_CLASSES[top_idx]
    probability_table = [
        {
            "strategy": CAMPAIGN_LABELS[key],
            "probability": float(mean_probs[idx]),
        }
        for idx, key in enumerate(CAMPAIGN_CLASSES)
    ]
    probability_table = sorted(probability_table, key=lambda row: row["probability"], reverse=True)

    explanation = _generate_explanation(
        strategy_key=strategy_key,
        segment_profile=segment_profile,
        marketer_query=marketer_query,
        objective=objective,
        use_llm=use_llm,
    )
    return CampaignRecommendation(
        strategy_key=strategy_key,
        strategy_label=CAMPAIGN_LABELS[strategy_key],
        confidence=float(mean_probs[top_idx]),
        explanation=explanation,
        suggested_message=_suggested_message(strategy_key, segment_profile),
        risk_notes=_risk_notes(segment_profile, float(mean_probs[top_idx])),
        probability_table=probability_table,
    )


def _generate_explanation(
    strategy_key: str,
    segment_profile: dict[str, object],
    marketer_query: str,
    objective: str,
    use_llm: bool,
) -> str:
    if use_llm:
        generated = _try_ollama_explanation(strategy_key, segment_profile, marketer_query, objective)
        if generated:
            return generated

    top_categories = ", ".join(segment_profile.get("top_categories", [])[:3]) or "mixed categories"
    coupon_rate = segment_profile.get("coupon_rate", 0.0)
    recent_activity_rate = segment_profile.get("recent_activity_rate", 0.0)
    avg_weekly_spend = segment_profile.get("avg_weekly_spend", 0.0)
    base = CAMPAIGN_DESCRIPTIONS[strategy_key]

    return (
        f"{base} The segment's strongest category signals are {top_categories}. "
        f"Average weekly spend is about ${avg_weekly_spend:,.2f}, coupon usage is {coupon_rate:.1%}, "
        f"and {recent_activity_rate:.1%} of households show recent activity. "
        f"For the objective '{objective}', this strategy balances model confidence with marketer-readable actionability."
    )


def _try_ollama_explanation(
    strategy_key: str,
    segment_profile: dict[str, object],
    marketer_query: str,
    objective: str,
) -> str | None:
    """Use a free local Ollama model when available.

    The app remains fully functional without Ollama because this function
    returns None and the caller falls back to deterministic explanations.
    """
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    ollama_model = os.getenv("OLLAMA_MODEL", "mistral")
    prompt = f"""
You are a retail marketing data science assistant. Explain whether the audience fits the marketer's campaign objective and which activation style should be used.

Marketer query: {marketer_query}
Objective: {objective}
Recommended activation style: {CAMPAIGN_LABELS[strategy_key]}
Segment profile: {segment_profile}

Write 3 concise sentences for a marketer. Include whether the audience fits the objective, what signal supports the activation style, and what to validate before launch. Do not mention that you are an AI model.
"""
    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.25,
                    "num_predict": 180,
                },
            },
            timeout=45,
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        return text or None
    except (requests.RequestException, ValueError):
        return None


def _suggested_message(strategy_key: str, segment_profile: dict[str, object]) -> str:
    top_category = (segment_profile.get("top_categories") or ["your favorites"])[0]
    templates = {
        "new_customer_onboarding": f"Welcome back: discover weekly value on {top_category} and more.",
        "win_back_reminder": f"We saved something for you: fresh deals on {top_category} this week.",
        "price_led_coupon": f"Limited-time savings: unlock a personalized offer on {top_category}.",
        "cross_sell_bundle": f"Complete the basket: pair {top_category} with a deal picked for your next trip.",
        "loyalty_reward": f"Thanks for shopping with us: enjoy a reward on your favorite {top_category} items.",
        "seasonal_spotlight": f"Seasonal picks are here: stock up on {top_category} for the moment.",
    }
    return templates[strategy_key]


def _risk_notes(segment_profile: dict[str, object], confidence: float) -> list[str]:
    notes: list[str] = []
    household_count = int(segment_profile.get("households", 0))
    coupon_rate = float(segment_profile.get("coupon_rate", 0.0))

    if household_count < 50:
        notes.append("Audience is small; validate minimum reach and privacy thresholds before activation.")
    if confidence < 0.45:
        notes.append("Model confidence is modest; consider A/B testing against a simple baseline campaign.")
    if coupon_rate > 0.45:
        notes.append("High coupon sensitivity may reduce margin; pair response lift with profitability checks.")
    if not notes:
        notes.append("No major demo risk flags. In production, validate incrementality, policy, and fatigue limits.")
    return notes
