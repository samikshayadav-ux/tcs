"""
Tests for Document Photo Extraction Endpoint (POST /api/v1/extract).
"""

import io
from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest
from backend.app.main import app
from backend.app.services.document_extractor import (
    DocumentExtractor,
    VisionExtractionError,
)

client = TestClient(app)


def test_reject_unsupported_content_type():
    file_content = b"fake pdf content"
    files = {"file": ("document.pdf", io.BytesIO(file_content), "application/pdf")}
    response = client.post("/api/v1/extract", files=files)
    assert response.status_code == 400
    assert "Unsupported content type" in response.json()["detail"]


def test_reject_empty_file():
    empty_content = b""
    files = {"file": ("empty.jpg", io.BytesIO(empty_content), "image/jpeg")}
    response = client.post("/api/v1/extract", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_reject_oversized_file(monkeypatch):
    from backend.app.config import settings
    monkeypatch.setattr(settings, "max_upload_size_bytes", 50)  # limit to 50 bytes

    oversized = b"a" * 100
    files = {"file": ("large.jpg", io.BytesIO(oversized), "image/jpeg")}
    response = client.post("/api/v1/extract", files=files)
    assert response.status_code == 400
    assert "exceeds maximum allowed upload limit" in response.json()["detail"]


def test_unconfigured_vision_provider_returns_honest_error(monkeypatch):
    """When no API key is set, extraction returns honest 503 error without fabricating data."""
    from backend.app.config import settings
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "openai_api_key", "")

    sample_img = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 30
    files = {"file": ("claim.jpg", io.BytesIO(sample_img), "image/jpeg")}
    response = client.post("/api/v1/extract", files=files)

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "GEMINI_API_KEY" in detail or "vision-capable" in detail


def test_successful_mocked_vision_extraction():
    """Test full extraction parsing when vision API returns structured JSON."""
    mock_extracted_json = {
        "document_type": "Insurance Card",
        "policy_number": "POL-987654",
        "make": "Toyota",
        "vehicle_category": "Sedan",
        "age_of_vehicle": "3 years",
        "driver_age": 34,
        "age_of_policy_holder": "31 to 35",
        "sex": "Female",
        "marital_status": "Single",
        "deductible": 400,
        "base_policy": "Collision",
        "policy_type": "Sedan - Collision",
        "accident_date": "2024-05-15",
        "accident_area": "Urban",
        "fault": "Policy Holder",
        "driver_rating": 2,
        "past_number_of_claims": "none",
        "police_report_filed": "Yes",
        "witness_present": "No",
        "agent_type": "External",
        "claim_amount": 3200,
        "raw_extracted_text": "Toyota Corolla sedan policy declaration card."
    }

    sample_img = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 30
    files = {"file": ("card.jpg", io.BytesIO(sample_img), "image/jpeg")}

    with patch.object(DocumentExtractor, "has_configured_provider", return_value=True):
        with patch.object(DocumentExtractor, "_call_vision_api", return_value=mock_extracted_json):
            response = client.post("/api/v1/extract", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["extraction_status"] == "success"
    assert data["document_type"] == "Insurance Card"
    assert data["extracted_fields"]["make"] == "Toyota"
    assert data["extracted_fields"]["policy_number"] == "POL-987654"
    assert "Make" in data["model_compatible_fields"]
    assert data["model_compatible_fields"]["Make"] == "Toyota"
    # Ensure policy_number is in warnings about ML exclusion
    assert any("POL-987654" in w for w in data["warnings"])


def test_vision_api_failure_handled_cleanly():
    """Test that vision provider exceptions return 503 without crashing the server."""
    sample_img = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 30
    files = {"file": ("card.jpg", io.BytesIO(sample_img), "image/jpeg")}

    with patch.object(DocumentExtractor, "has_configured_provider", return_value=True):
        with patch.object(
            DocumentExtractor,
            "_call_vision_api",
            side_effect=VisionExtractionError("Vision API rate limit exceeded")
        ):
            response = client.post("/api/v1/extract", files=files)

    assert response.status_code == 503
    assert "rate limit exceeded" in response.json()["detail"]
