"""
Unit and integration tests for FraudPredictor.
"""

import os
import pytest
import pandas as pd
from src.predictor import FraudPredictor, UNSUPPORTED_FIELDS_EXPLANATIONS, DEFAULT_DISCLAIMER


MODEL_DIR = "models"
MODEL_EXISTS = os.path.exists(os.path.join(MODEL_DIR, "fraud_model.joblib"))


@pytest.fixture
def predictor():
    if not MODEL_EXISTS:
        pytest.skip("Trained model artifact not found. Run src/train.py first.")
    return FraudPredictor(model_dir=MODEL_DIR)


@pytest.fixture
def valid_claim():
    return {
        "Make": "Toyota",
        "AccidentArea": "Urban",
        "Sex": "Male",
        "MaritalStatus": "Married",
        "Age": 45,
        "Fault": "Policy Holder",
        "PolicyType": "Sedan - Collision",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "20000 to 29000",
        "Deductible": 400,
        "DriverRating": 2,
        "PastNumberOfClaims": "1",
        "AgeOfVehicle": "5 years",
        "AgeOfPolicyHolder": "41 to 50",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
        "AgentType": "External",
        "NumberOfSuppliments": "none",
        "AddressChange_Claim": "no change",
        "NumberOfCars": "1 vehicle",
        "BasePolicy": "Collision",
        "Month": "Jan",
        "DayOfWeek": "Monday",
        "WeekOfMonth": 3,
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
    }


def test_predictor_initialization(predictor):
    """Test that predictor loads model, schema, and metadata successfully."""
    assert predictor.pipeline is not None
    assert len(predictor.feature_names) == 26
    assert "Make" in predictor.feature_names
    assert "Age" in predictor.feature_names


def test_valid_single_prediction(predictor, valid_claim):
    """Test prediction on a valid dictionary."""
    result = predictor.predict(valid_claim)
    assert isinstance(result, dict)
    assert result["predicted_class"] in [0, 1]
    assert 0.0 <= result["fraud_risk_score"] <= 1.0
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert result["disclaimer"] == DEFAULT_DISCLAIMER
    assert "Toyota" in str(result["input_summary"])


def test_valid_batch_prediction(predictor, valid_claim):
    """Test prediction on a list of claims."""
    claims = [valid_claim, {**valid_claim, "Make": "Honda", "Age": 22}]
    results = predictor.predict(claims)
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert r["predicted_class"] in [0, 1]
        assert 0.0 <= r["fraud_risk_score"] <= 1.0


def test_valid_dataframe_prediction(predictor, valid_claim):
    """Test prediction on a pandas DataFrame."""
    df_claims = pd.DataFrame([valid_claim, {**valid_claim, "Make": "Mazda"}])
    results = predictor.predict(df_claims)
    assert isinstance(results, list)
    assert len(results) == 2


def test_unsupported_fields_rejected(predictor, valid_claim):
    """Test that fields like annual_mileage, driving_experience, and vehicle_usage are rejected."""
    for unsupported_field in UNSUPPORTED_FIELDS_EXPLANATIONS:
        bad_claim = {**valid_claim, unsupported_field: 12000}
        with pytest.raises(ValueError) as excinfo:
            predictor.predict(bad_claim)
        assert f"Unsupported field '{unsupported_field}'" in str(excinfo.value)


def test_unknown_field_rejected(predictor, valid_claim):
    """Test that arbitrary unknown fields are rejected."""
    bad_claim = {**valid_claim, "non_existent_feature_123": "invalid"}
    with pytest.raises(ValueError) as excinfo:
        predictor.predict(bad_claim)
    assert "Unexpected field 'non_existent_feature_123'" in str(excinfo.value)


def test_invalid_numeric_type_rejected(predictor, valid_claim):
    """Test that invalid types for numeric fields raise ValueError."""
    bad_claim = {**valid_claim, "Age": "not_a_number"}
    with pytest.raises(ValueError) as excinfo:
        predictor.predict(bad_claim)
    assert "expects numeric value" in str(excinfo.value)


def test_negative_age_rejected(predictor, valid_claim):
    """Test that negative Age values raise ValueError."""
    bad_claim = {**valid_claim, "Age": -5}
    with pytest.raises(ValueError) as excinfo:
        predictor.predict(bad_claim)
    assert "cannot be negative" in str(excinfo.value)


def test_unseen_categorical_values_safe(predictor, valid_claim):
    """Test that unseen categories do not crash the pipeline (handle_unknown='ignore')."""
    unseen_claim = {
        **valid_claim,
        "Make": "FuturisticEV_Brand",
        "AccidentArea": "Subterranean",
        "VehiclePrice": "infinite",
        "Month": "Undecimber",
    }
    result = predictor.predict(unseen_claim)
    assert isinstance(result, dict)
    assert result["predicted_class"] in [0, 1]
    assert 0.0 <= result["fraud_risk_score"] <= 1.0


def test_omitted_optional_fields_handled(predictor):
    """Test that providing a partial claim relies on imputer safely without crashing."""
    partial_claim = {
        "Make": "Ford",
        "Age": 35,
        "Deductible": 400,
        "VehicleCategory": "Sedan",
    }
    result = predictor.predict(partial_claim)
    assert isinstance(result, dict)
    assert result["predicted_class"] in [0, 1]
    assert 0.0 <= result["fraud_risk_score"] <= 1.0


def test_empty_input_rejected(predictor):
    """Test that empty list or empty DataFrame raises ValueError."""
    with pytest.raises(ValueError):
        predictor.predict([])

    with pytest.raises(ValueError):
        predictor.predict(pd.DataFrame())


def test_invalid_container_type_rejected(predictor):
    """Test that passing invalid types (e.g. string or integer) raises ValueError."""
    with pytest.raises(ValueError):
        predictor.predict("not a valid container")

    with pytest.raises(ValueError):
        predictor.predict(12345)
