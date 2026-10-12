"""
Tests for Model Info and Dashboard Endpoints.
"""

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_get_model_info():
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["target_name"] == "FraudFound_P"
    assert data["model_type"] == "LogisticRegression"
    assert data["model_name"] == "logistic_regression_balanced"
    assert data["feature_count"] == 26
    assert "PolicyNumber" in data["excluded_features"]
    assert "RepNumber" in data["excluded_features"]

    # Verify actual held-out metrics from Part 1
    metrics = data["evaluation_metrics"]
    assert metrics["recall_fraud"] > 0.8
    assert metrics["roc_auc"] > 0.75
    assert "trade_off_analysis" in data


def test_get_dashboard_stats():
    response = client.get("/api/v1/dashboard-stats")
    assert response.status_code == 200
    data = response.json()

    hist = data["historical_dataset"]
    assert hist["total_historical_claims"] == 15420
    assert hist["fraudulent_claims_count"] == 923
    assert hist["historical_fraud_rate_pct"] == 5.99

    perf = data["model_performance"]
    assert perf["recall_fraud_pct"] > 80.0
    assert perf["roc_auc"] > 0.75

    status = data["service_status"]
    assert status["ml_pipeline_loaded"] is True
    assert "vision_extraction_ready" in status
    assert "notice" in data
