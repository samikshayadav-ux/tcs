"""
Tests for Fraud Risk Prediction Endpoint (POST /api/v1/predict).
"""

from fastapi.testclient import TestClient
import pytest
from backend.app.main import app
from backend.app.services.predictor import PredictorService

client = TestClient(app)


@pytest.fixture
def valid_payload():
    return {
        "Make": "Honda",
        "AccidentArea": "Urban",
        "Sex": "Male",
        "MaritalStatus": "Married",
        "Age": 38,
        "Fault": "Policy Holder",
        "PolicyType": "Sedan - Collision",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "20000 to 29000",
        "Deductible": 400,
        "DriverRating": 2,
        "PastNumberOfClaims": "none",
        "AgeOfVehicle": "5 years",
        "AgeOfPolicyHolder": "36 to 40",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
        "AgentType": "External",
        "NumberOfSuppliments": "none",
        "AddressChange_Claim": "no change",
        "NumberOfCars": "1 vehicle",
        "BasePolicy": "Collision",
        "Month": "Jan",
        "DayOfWeek": "Wednesday",
        "WeekOfMonth": 3,
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
    }


def test_predict_valid_payload(valid_payload):
    response = client.post("/api/v1/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert data["predicted_class"] in [0, 1]
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert 0.0 <= data["indicative_score_0_100"] <= 100.0
    assert data["risk_category"] in ["LOW", "MEDIUM", "HIGH"]
    assert len(data["top_risk_factors"]) > 0
    assert len(data["recommended_review_actions"]) > 0
    assert data["explanation_source"] in ["template", "llm"]
    assert "not proof of fraud" in data["disclaimer"].lower()


def test_predict_partial_payload():
    """Test prediction with partial fields safely relying on imputer."""
    partial = {
        "make": "Toyota",
        "age": 45,
        "deductible": 400,
        "accident_area": "Rural",
    }
    response = client.post("/api/v1/predict", json=partial)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [0, 1]
    assert 0.0 <= data["fraud_score"] <= 1.0


def test_reject_annual_mileage(valid_payload):
    bad_payload = {**valid_payload, "annual_mileage": 15000}
    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 422
    assert "annual_mileage" in response.text


def test_reject_driving_experience(valid_payload):
    bad_payload = {**valid_payload, "driving_experience": 10}
    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 422
    assert "driving_experience" in response.text


def test_reject_vehicle_usage(valid_payload):
    bad_payload = {**valid_payload, "vehicle_usage": "Commute"}
    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 422
    assert "vehicle_usage" in response.text


def test_reject_negative_age(valid_payload):
    bad_payload = {**valid_payload, "Age": -10}
    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 422
    assert "cannot be negative" in response.text.lower()


def test_unseen_category_safe(valid_payload):
    """Test that unseen category values are handled safely without crashing."""
    unseen_payload = {
        **valid_payload,
        "Make": "FuturisticAeroCar",
        "AccidentArea": "Underwater",
    }
    response = client.post("/api/v1/predict", json=unseen_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [0, 1]


def test_missing_model_behavior(monkeypatch):
    """Test graceful 503 response if model is unavailable."""
    svc = PredictorService.get_instance()
    # Temporarily set predictor to None
    orig_predictor = svc.predictor
    try:
        svc.predictor = None
        response = client.post("/api/v1/predict", json={"Make": "Ford"})
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"].lower()
    finally:
        svc.predictor = orig_predictor
