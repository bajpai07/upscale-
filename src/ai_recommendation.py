import os
import logging
from typing import Dict, Any

logger = logging.getLogger("ai_recommendation")

# Model chain. Both entries are verified reachable on the current API; the first
# is what the Gemini API itself recommends, the second is a faster stand-in used
# when the primary is overloaded (503) or retired. Override with GEMINI_MODEL.
# Note: floating aliases such as `gemini-flash-latest` are deliberately avoided —
# they return 503 under load and shift behaviour without notice.
DEFAULT_MODEL_CHAIN = ("gemini-3.6-flash", "gemini-3.5-flash")

# Hard ceiling on a single generation call, in milliseconds. Without this a hung
# request blocks the Streamlit worker indefinitely.
REQUEST_TIMEOUT_MS = 30_000

def generate_deterministic_fallback(
    upsell_probability: float,
    guardrail_triggered: bool,
    guardrail_reason: str,
    final_eligible: bool,
    campaign_recommendation: str,
    top_shap_drivers: list
) -> Dict[str, str]:
    """
    Deterministic template fallback generator when Gemini API is unconfigured or unavailable.
    """
    prob_pct = upsell_probability * 100.0

    if guardrail_triggered:
        explanation = (
            f"Responsible AI Guardrail Override Active: This customer exhibits elevated support friction "
            f"({guardrail_reason}). Even though their underlying usage score ({prob_pct:.1f}%) suggests potential product alignment, "
            f"pitching an upsell at this stage risks escalating churn friction. The customer must be routed directly to Customer Success for retention support."
        )
        outreach_opener = (
            "DO NOT PITCH UPSELL. Recommended Customer Success Opener: 'Hello, I noticed you recently reached out to our support team. "
            "I wanted to check in personally and ensure all your account issues have been fully resolved.'"
        )
    elif final_eligible:
        pos_drivers = [d['feature'] for d in top_shap_drivers if d.get('shap_value', 0) > 0]
        pos_str = ", ".join(pos_drivers[:2]) if pos_drivers else "high overall network engagement"
        
        explanation = (
            f"The customer demonstrates elevated upsell potential ({prob_pct:.1f}% opportunity score), "
            f"primarily driven by {pos_str}. Account tenure and billing cycle usage patterns confirm strong product adoption with zero active support friction."
        )
        outreach_opener = (
            f"Sales Outreach Opener: 'Hello, based on your heavy reliance on {pos_str} over your recent billing cycle, "
            f"we've identified an exclusive plan upgrade that offers enhanced capacity and premium feature rates.'"
        )
    else:
        neg_drivers = [d['feature'] for d in top_shap_drivers if d.get('shap_value', 0) < 0]
        neg_str = ", ".join(neg_drivers[:2]) if neg_drivers else "lower overall billing volume"
        
        explanation = (
            f"The customer currently demonstrates low upsell signal ({prob_pct:.1f}% opportunity score), "
            f"limited primarily by {neg_str}. Standard service tier maintenance is recommended to maintain baseline satisfaction."
        )
        outreach_opener = (
            "Routine Maintenance Opener: 'Hello, we are conducting a routine account check-in to confirm your current service tier "
            "continues to meet your monthly communication needs.'"
        )

    return {
        "rep_explanation": explanation,
        "outreach_opener": outreach_opener,
        "source": "Deterministic Template Fallback",
        # `degraded` distinguishes "no key configured, template by design" from
        # "the API was meant to answer and failed", so the UI can warn on the
        # second case instead of silently serving generic copy.
        "degraded": False,
        "error": None,
    }

def generate_ai_recommendation(customer_summary: Dict[str, Any]) -> Dict[str, str]:
    """
    Downstream GenAI layer (Gemini API) for producing sales rep explanations & outreach openers.
    Consumes ONLY pre-computed structured facts. Never overrides probability or guardrails.
    """
    prob = float(customer_summary.get("upsell_probability", 0.0))
    guardrail_triggered = bool(customer_summary.get("guardrail_triggered", False))
    guardrail_reason = str(customer_summary.get("guardrail_reason", ""))
    final_eligible = bool(customer_summary.get("final_upsell_eligible", False))
    rec = str(customer_summary.get("campaign_recommendation", ""))
    drivers = customer_summary.get("top_shap_drivers", [])

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        return generate_deterministic_fallback(prob, guardrail_triggered, guardrail_reason, final_eligible, rec, drivers)

    drivers_text = "\n".join([
        f"- {d.get('feature')}: Value={d.get('feature_value')}, SHAP Impact={d.get('shap_value'):+.3f} ({d.get('impact_direction')})"
        for d in drivers
    ])

    prompt = f"""
You are an executive Telecom Sales Intelligence Assistant.
Analyze the following PRE-COMPUTED customer structured facts:

- Opportunity Score: {prob * 100:.1f}%
- Guardrail Triggered: {guardrail_triggered}
- Guardrail Reason: {guardrail_reason}
- Final Upsell Eligible: {final_eligible}
- System Recommendation: {rec}
- Top SHAP Feature Drivers:
{drivers_text}

STRICT CONSTRAINTS:
1. NEVER calculate or alter the score or eligibility.
2. NEVER invent fake product names, prices, or discounts.
3. Use likelihood language ("elevated potential", "low signal"), NEVER certainty ("will upgrade").
4. If Guardrail Triggered is True, state clearly that NO UPSELL OUTREACH SHOULD HAPPEN and explain why.

FORMAT YOUR RESPONSE EXACTLY AS TWO SECTIONS:
[REP_EXPLANATION]
(2-3 sentence plain-English summary for sales rep)

[OUTREACH_OPENER]
(1-2 sentence contextual call opener referencing actual usage)
"""

    override = os.environ.get("GEMINI_MODEL")
    model_chain = (override,) if override else DEFAULT_MODEL_CHAIN

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
    except Exception as e:
        logger.warning("Gemini client could not be constructed (%s). Using template fallback.", e)
        result = generate_deterministic_fallback(
            prob, guardrail_triggered, guardrail_reason, final_eligible, rec, drivers
        )
        result.update(degraded=True, error=f"Gemini SDK unavailable: {e}")
        return result

    last_error = None
    for model_name in model_chain:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            # `.text` is None when the model returns no candidate at all — a safety
            # block, or the token budget spent entirely on reasoning. Treat that as
            # a failure for this model and move on to the next.
            text = (response.text or "").strip()
            if not text:
                raise ValueError("model returned an empty response")
        except Exception as e:
            last_error = f"{model_name}: {e}"
            logger.warning("Gemini call failed on %s (%s).", model_name, e)
            continue

        source_name = f"Gemini API ({model_name})"
        if "[REP_EXPLANATION]" in text and "[OUTREACH_OPENER]" in text:
            parts = text.split("[OUTREACH_OPENER]")
            explanation = parts[0].replace("[REP_EXPLANATION]", "").strip()
            opener = parts[1].strip()
        else:
            # Model answered but ignored the section markers; keep the prose
            # rather than throwing away a valid generation.
            explanation = text
            opener = "Contextual call opener generated by AI assistant."

        return {
            "rep_explanation": explanation,
            "outreach_opener": opener,
            "source": source_name,
            "degraded": False,
            "error": None,
        }

    logger.warning("All Gemini models failed (%s). Degrading to template fallback.", last_error)
    result = generate_deterministic_fallback(
        prob, guardrail_triggered, guardrail_reason, final_eligible, rec, drivers
    )
    result.update(degraded=True, error=last_error)
    return result
