"""
Explanation Service for AutoRisk AI.

Generates objective underwriting-style risk summaries and recommended actions.
Uses a deterministic template engine by default, and optionally enhances explanations
via an LLM if API credentials are provided. Guarantees that the LLM cannot alter
the ML model's prediction or score.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.app.config import settings

logger = logging.getLogger("autorisk.explainer")


def derive_evidence_factors(features: Dict[str, Any]) -> List[str]:
    """Identifies concrete underwriting risk factors present in the input features."""
    factors = []

    # 1. Fault attribution
    fault = features.get("Fault")
    if fault == "Policy Holder":
        factors.append("Policyholder is designated at-fault in the incident description.")
    elif fault == "Third Party":
        factors.append("Third-party attributed at-fault.")

    # 2. Prior claims history
    past_claims = features.get("PastNumberOfClaims")
    if past_claims in ["more than 4", "2 to 4"]:
        factors.append(f"Elevated claim frequency: policyholder has '{past_claims}' prior claims on record.")
    elif past_claims == "1":
        factors.append("Single prior claim noted in historical profile.")

    # 3. Driver rating tier
    driver_rating = features.get("DriverRating")
    if driver_rating in [3, 4, 3.0, 4.0]:
        factors.append(f"High risk driver tier: policy rating is {int(driver_rating)} (scale 1-4).")

    # 4. Vehicle value & category
    price = features.get("VehiclePrice")
    category = features.get("VehicleCategory")
    if price in ["more than 69000", "40000 to 59000"]:
        factors.append(f"High-tier vehicle valuation: recorded value bracket '{price}'.")
    if category == "Sport":
        factors.append("High-performance / Sport vehicle category.")

    # 5. Deductible
    deductible = features.get("Deductible")
    if deductible is not None:
        try:
            if float(deductible) >= 700:
                factors.append(f"High deductible policy (${int(float(deductible))}).")
        except (ValueError, TypeError):
            pass

    # 6. Supplements and repairs
    supplements = features.get("NumberOfSuppliments")
    if supplements in ["more than 5", "3 to 5"]:
        factors.append(f"Multiple repair cost supplements filed: '{supplements}'.")

    # 7. Address changes
    addr_change = features.get("AddressChange_Claim")
    if addr_change and addr_change not in ["no change", "none"]:
        factors.append(f"Recent policyholder address change recorded: '{addr_change}'.")

    # 8. Police report & witnesses
    if features.get("PoliceReportFiled") == "No":
        factors.append("No formal police accident report filed on record.")
    if features.get("WitnessPresent") == "No":
        factors.append("No independent witnesses documented for incident.")

    if not factors:
        factors.append("Standard policy profile; no notable adverse risk factors identified.")

    return factors


def generate_deterministic_explanation(
    prediction: Dict[str, Any],
    features: Dict[str, Any]
) -> Tuple[List[str], str, List[str]]:
    """
    Builds a reliable, deterministic underwriting explanation using domain templates.

    Returns:
        Tuple of (top_risk_factors, underwriting_summary, recommended_review_actions)
    """
    risk_tier = prediction.get("risk_category", "LOW")
    score = prediction.get("fraud_score", 0.0)
    indicative_score = prediction.get("indicative_score_0_100", 0.0)
    make = features.get("Make", "Vehicle")
    policy_type = features.get("PolicyType", features.get("BasePolicy", "Standard Policy"))

    risk_factors = derive_evidence_factors(features)

    # Recommendations based on risk tier
    if risk_tier == "HIGH":
        summary = (
            f"The statistical model assigned an elevated review priority score of {indicative_score}/100 "
            f"(uncalibrated fraud-class score: {score:.3f}) for this {make} claim under {policy_type}. "
            f"Key risk indicators include {risk_factors[0].lower() if risk_factors else 'historical profile factors'}. "
            "This claim warrants secondary verification by the Special Investigation Unit (SIU) before settlement."
        )
        actions = [
            "Request itemized repair estimate and shop invoices for physical damage validation.",
            "Verify proof of loss and confirm incident timeline against police / tow records.",
            "Review policyholder historical claims records across industry databases.",
            "Assign to Special Investigation Unit (SIU) desk for documentation audit."
        ]
    elif risk_tier == "MEDIUM":
        summary = (
            f"The model assigned a moderate indicative risk score of {indicative_score}/100 "
            f"(score: {score:.3f}) for this {make} claim. While not exhibiting critical indicators, "
            "minor discrepancies or profile attributes suggest standard desk verification is appropriate."
        )
        actions = [
            "Confirm that repair estimate matches typical part replacement costs for this vehicle make.",
            "Validate policy deductible and payment history prior to settlement authorization.",
            "Cross-reference driver rating and license status."
        ]
    else:
        summary = (
            f"The model scored this {make} claim in the standard risk band with an indicative score of "
            f"{indicative_score}/100 (score: {score:.3f}). Profile indicators align with standard historical "
            "loss patterns without notable statistical fraud flags."
        )
        actions = [
            "Proceed with regular claim adjudication and appraisal workflow.",
            "Standard automated deductible check and settlement processing."
        ]

    return risk_factors, summary, actions


def try_llm_explanation(
    prediction: Dict[str, Any],
    features: Dict[str, Any],
    base_factors: List[str]
) -> Optional[Tuple[str, List[str]]]:
    """
    Attempts to generate an LLM explanation using Gemini or OpenAI if configured.
    Returns (summary, actions) on success, or None on failure/fallback.
    """
    prompt = (
        "You are an insurance underwriting AI assistant. Summarize the following claim risk review:\n"
        f"- Vehicle/Policy: {features.get('Make', 'Unknown')} ({features.get('PolicyType', 'Unknown')})\n"
        f"- Statistical Fraud Score: {prediction.get('indicative_score_0_100', 0)}/100 (Tier: {prediction.get('risk_category')})\n"
        f"- Key Factors: {', '.join(base_factors)}\n\n"
        "Requirements:\n"
        "1. Write a concise, professional underwriting summary (2-3 sentences).\n"
        "2. Provide 3-4 actionable review steps for a claims adjuster.\n"
        "3. NEVER assert fraud is proven or accuse the claimant.\n"
        "4. Respond strictly in valid JSON with keys 'underwriting_summary' and 'recommended_review_actions'."
    )

    # 1. Try Gemini if key is provided
    if settings.gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            data = json.loads(response.text)
            return data["underwriting_summary"], data["recommended_review_actions"]
        except Exception as exc:
            logger.warning("Gemini explanation attempt failed, falling back to template: %s", exc)

    # 2. Try OpenAI if key is provided
    if settings.openai_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key, timeout=settings.request_timeout_seconds)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an insurance underwriting risk summarizer. Output valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = completion.choices[0].message.content
            data = json.loads(content)
            return data["underwriting_summary"], data["recommended_review_actions"]
        except Exception as exc:
            logger.warning("OpenAI explanation attempt failed, falling back to template: %s", exc)

    return None


class ExplanationService:
    """Service generating evidence-backed underwriting explanations."""

    def explain_prediction(
        self,
        prediction: Dict[str, Any],
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates explanation components. Tries LLM if configured; otherwise uses deterministic template.
        """
        factors, default_summary, default_actions = generate_deterministic_explanation(
            prediction, features
        )

        source = "template"
        summary = default_summary
        actions = default_actions

        # Attempt optional LLM explanation if configured
        if settings.gemini_api_key or settings.openai_api_key:
            llm_result = try_llm_explanation(prediction, features, factors)
            if llm_result:
                summary, actions = llm_result
                source = "llm"

        return {
            "top_risk_factors": factors,
            "underwriting_summary": summary,
            "recommended_review_actions": actions,
            "explanation_source": source,
        }


def get_explanation_service() -> ExplanationService:
    """Helper to obtain explanation service."""
    return ExplanationService()
