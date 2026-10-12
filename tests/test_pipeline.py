"""
Tests for pipeline structure, artifact integrity, and reproducibility.
"""

import json
import os
import pytest
import joblib
import pandas as pd


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "evaluation_metrics.json")
SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_schema.json")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


def test_artifact_files_exist():
    """Verify that all four required model artifacts exist."""
    assert os.path.exists(MODEL_PATH), f"Missing {MODEL_PATH}"
    assert os.path.exists(METRICS_PATH), f"Missing {METRICS_PATH}"
    assert os.path.exists(SCHEMA_PATH), f"Missing {SCHEMA_PATH}"
    assert os.path.exists(METADATA_PATH), f"Missing {METADATA_PATH}"


def test_pipeline_loadable():
    """Verify that the joblib pipeline can be loaded and has expected components."""
    pipeline = joblib.load(MODEL_PATH)
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_schema_integrity():
    """Verify that feature_schema.json contains supported and unsupported fields."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert "supported_features" in schema
    assert "unsupported_fields" in schema
    assert len(schema["supported_features"]) == 26

    # Verify unsupported field explanations
    unsupported = schema["unsupported_fields"]
    assert "annual_mileage" in unsupported
    assert "driving_experience" in unsupported
    assert "vehicle_usage" in unsupported


def test_metadata_integrity():
    """Verify that model_metadata.json contains complete configuration and limitations."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["target_name"] == "FraudFound_P"
    assert "selected_model_name" in meta
    assert "training_timestamp_utc" in meta
    assert "features" in meta
    assert len(meta["limitations"]) > 0

    # Ensure excluded identifiers are logged
    assert "PolicyNumber" in meta["features"]["excluded_features"]
    assert "RepNumber" in meta["features"]["excluded_features"]


def test_metrics_integrity():
    """Verify that evaluation_metrics.json contains actual measured non-null numbers."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "selected_model" in data
    assert "metrics" in data
    selected = data["selected_model"]
    selected_metrics = data["metrics"][selected]

    assert 0.0 <= selected_metrics["precision_fraud"] <= 1.0
    assert 0.0 <= selected_metrics["recall_fraud"] <= 1.0
    assert 0.0 <= selected_metrics["f1_fraud"] <= 1.0
    assert 0.0 <= selected_metrics["roc_auc"] <= 1.0
    assert 0.0 <= selected_metrics["pr_auc"] <= 1.0
    assert 0.0 <= selected_metrics["accuracy"] <= 1.0


def test_prediction_reproducibility():
    """Verify that identical inputs produce identical probability scores."""
    pipeline = joblib.load(MODEL_PATH)

    sample_row = pd.DataFrame([{
        "WeekOfMonth": 3,
        "Age": 45,
        "Deductible": 400,
        "DriverRating": 2,
        "Month": "Jan",
        "DayOfWeek": "Monday",
        "Make": "Honda",
        "AccidentArea": "Urban",
        "Sex": "Male",
        "MaritalStatus": "Married",
        "Fault": "Policy Holder",
        "PolicyType": "Sedan - Collision",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "20000 to 29000",
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "PastNumberOfClaims": "none",
        "AgeOfVehicle": "5 years",
        "AgeOfPolicyHolder": "41 to 50",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
        "AgentType": "External",
        "NumberOfSuppliments": "none",
        "AddressChange_Claim": "no change",
        "NumberOfCars": "1 vehicle",
        "BasePolicy": "Collision",
    }])

    prob_1 = pipeline.predict_proba(sample_row)[0, 1]
    prob_2 = pipeline.predict_proba(sample_row)[0, 1]

    assert prob_1 == pytest.approx(prob_2, rel=1e-6)
