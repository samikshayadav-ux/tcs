"""
Tests for FastAPI Health Check Endpoint.
"""

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert data["model_loaded"] is True
    assert data["model_name"] == "logistic_regression_balanced"
    assert "app_version" in data
    assert "timestamp" in data
