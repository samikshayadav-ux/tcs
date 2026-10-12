"""
Tests for Underwriting Explanation Service.
"""

from unittest.mock import patch
from backend.app.services.explainer import (
    ExplanationService,
    derive_evidence_factors,
    generate_deterministic_explanation,
)


def test_derive_evidence_factors():
    features = {
        "Fault": "Policy Holder",
        "PastNumberOfClaims": "more than 4",
        "DriverRating": 4,
        "VehiclePrice": "more than 69000",
        "Deductible": 700,
        "PoliceReportFiled": "No",
    }
    factors = derive_evidence_factors(features)
    assert any("at-fault" in f.lower() for f in factors)
    assert any("prior claims" in f.lower() for f in factors)
    assert any("rating is 4" in f.lower() for f in factors)
    assert any("more than 69000" in f.lower() for f in factors)
    assert any("700" in f for f in factors)
    assert any("police" in f.lower() for f in factors)


def test_deterministic_explanation_high_risk():
    prediction = {
        "risk_category": "HIGH",
        "fraud_score": 0.825,
        "indicative_score_0_100": 82.5,
    }
    features = {
        "Make": "BMW",
        "PolicyType": "Sport - Collision",
        "Fault": "Policy Holder",
        "PastNumberOfClaims": "2 to 4",
    }
    factors, summary, actions = generate_deterministic_explanation(prediction, features)
    assert len(factors) > 0
    assert "82.5/100" in summary
    assert "BMW" in summary
    assert "Special Investigation Unit" in summary or "SIU" in summary
    assert len(actions) >= 3


def test_explainer_fallback_to_template_without_keys():
    """Verify default explanation source is 'template' when no LLM key is configured."""
    explainer = ExplanationService()
    prediction = {
        "risk_category": "LOW",
        "fraud_score": 0.08,
        "indicative_score_0_100": 8.0,
    }
    features = {"Make": "Honda"}
    result = explainer.explain_prediction(prediction, features)

    assert result["explanation_source"] == "template"
    assert len(result["top_risk_factors"]) > 0
    assert len(result["recommended_review_actions"]) > 0
    assert "Honda" in result["underwriting_summary"]


def test_explainer_llm_mock_success():
    """Verify that successful LLM returns explanation_source='llm'."""
    explainer = ExplanationService()
    prediction = {
        "risk_category": "HIGH",
        "fraud_score": 0.75,
        "indicative_score_0_100": 75.0,
    }
    features = {"Make": "Ford"}

    mock_llm_summary = "AI generated review: High priority audit recommended."
    mock_llm_actions = ["Step 1", "Step 2", "Step 3"]

    with patch(
        "backend.app.services.explainer.try_llm_explanation",
        return_value=(mock_llm_summary, mock_llm_actions)
    ):
        with patch("backend.app.services.explainer.settings.gemini_api_key", "test_key"):
            result = explainer.explain_prediction(prediction, features)

    assert result["explanation_source"] == "llm"
    assert result["underwriting_summary"] == mock_llm_summary
    assert result["recommended_review_actions"] == mock_llm_actions
