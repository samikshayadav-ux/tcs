"""FastAPI app for AutoRisk AI document review and claim-fraud triage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predictor import DEFAULT_DISCLAIMER, FraudPredictor


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data" / "fraud_oracle.csv"
MAX_UPLOAD_BYTES = int(os.getenv("AUTORISK_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
ACCEPTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}


class PredictRequest(BaseModel):
    """Prediction request using only supported model features."""

    features: Dict[str, Any] = Field(..., description="Supported FraudPredictor feature values.")


class RiskFactor(BaseModel):
    label: str
    value: str
    severity: Literal["low", "medium", "high", "info"]


class PredictResponse(BaseModel):
    predicted_class: int
    fraud_risk_score: Optional[float]
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    score_type: str
    calibrated_probability: bool
    verified_factors: List[RiskFactor]
    underwriting_summary: str
    review_actions: List[str]
    explanation_source: str
    disclaimer: str
    input_summary: Dict[str, Any]


class ExtractedField(BaseModel):
    name: str
    value: Optional[Any] = None
    status: Literal["extracted", "missing", "uncertain", "demo"] = "extracted"
    editable: bool = True


class ExtractResponse(BaseModel):
    status: Literal["completed", "partial", "failed", "demo"]
    fields: List[ExtractedField]
    missing_fields: List[str]
    warnings: List[str]
    supported_features: List[str]
    demo_mode: bool = False


def _load_json(name: str) -> Dict[str, Any]:
    with open(MODEL_DIR / name, "r", encoding="utf-8") as handle:
        return json.load(handle)


predictor = FraudPredictor(model_dir=str(MODEL_DIR))
schema = _load_json("feature_schema.json")
metadata = _load_json("model_metadata.json")
evaluation_metrics = _load_json("evaluation_metrics.json")
supported_features = schema["supported_features"]


app = FastAPI(
    title="AutoRisk AI API",
    version="1.0.0",
    description="Vehicle insurance claim-fraud review API backed by a trained scikit-learn pipeline.",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "AUTORISK_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": True,
        "supported_feature_count": len(supported_features),
    }


@app.get("/api/v1/model-info")
def model_info() -> Dict[str, Any]:
    selected = evaluation_metrics["selected_model"]
    return {
        "model": metadata,
        "schema": schema,
        "selected_model": selected,
        "selected_metrics": evaluation_metrics["metrics"][selected],
        "all_metrics": evaluation_metrics["metrics"],
        "trade_off_analysis": evaluation_metrics.get("trade_off_analysis", {}),
        "score_semantics": {
            "score_type": "fraud_probability",
            "calibrated_probability": False,
            "risk_thresholds": {"LOW": "< 0.35", "MEDIUM": "0.35 to < 0.65", "HIGH": ">= 0.65"},
            "disclaimer": DEFAULT_DISCLAIMER,
        },
    }


@app.get("/api/v1/dashboard-stats")
def dashboard_stats() -> Dict[str, Any]:
    dataset = metadata.get("dataset_split_configuration", {})
    selected_metrics = evaluation_metrics["metrics"][evaluation_metrics["selected_model"]]
    stats: Dict[str, Any] = {
        "source": "historical_training_dataset",
        "usage_stats_available": False,
        "message": "These are historical dataset and holdout evaluation statistics, not live app usage.",
        "dataset": dataset,
        "selected_metrics": selected_metrics,
    }
    if DATA_PATH.exists():
        try:
            frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
            stats["historical_counts"] = {
                "rows": int(len(frame)),
                "fraud_claims": int(frame["FraudFound_P"].sum()),
                "legitimate_claims": int((frame["FraudFound_P"] == 0).sum()),
                "fraud_rate_pct": round(float(frame["FraudFound_P"].mean() * 100), 2),
            }
        except Exception as exc:  # pragma: no cover - defensive observability
            stats["historical_counts_error"] = str(exc)
    return stats


@app.post("/api/v1/extract", response_model=ExtractResponse)
async def extract_document(file: UploadFile = File(...)) -> ExtractResponse:
    if file.content_type not in ACCEPTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type '{file.content_type}'. Accepted: {sorted(ACCEPTED_IMAGE_TYPES)}",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES} byte upload limit.")

    demo_mode = os.getenv("AUTORISK_DEMO_EXTRACTION", "false").lower() == "true"
    if not demo_mode:
        raise HTTPException(
            status_code=501,
            detail=(
                "Document vision extraction is not configured. Set AUTORISK_DEMO_EXTRACTION=true "
                "for an explicitly labeled local demo, or wire a vision provider in this endpoint."
            ),
        )

    demo_values = {
        "Make": "Toyota",
        "AccidentArea": "Urban",
        "Sex": "Male",
        "MaritalStatus": "Married",
        "Age": 42,
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
        "Month": "May",
        "DayOfWeek": "Tuesday",
        "WeekOfMonth": 2,
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
    }
    return ExtractResponse(
        status="demo",
        fields=[ExtractedField(name=name, value=value, status="demo") for name, value in demo_values.items()],
        missing_fields=[],
        warnings=[
            "DEMO MODE: fields are sample values for local presentation only.",
            "Replace with reviewed document values before relying on the prediction workflow.",
        ],
        supported_features=list(supported_features.keys()),
        demo_mode=True,
    )


def _risk_factors(features: Dict[str, Any], score: Optional[float]) -> List[RiskFactor]:
    factors: List[RiskFactor] = []
    if score is not None:
        factors.append(
            RiskFactor(
                label="Model score",
                value=f"{score:.1%} historical claim-fraud review probability",
                severity="high" if score >= 0.65 else "medium" if score >= 0.35 else "low",
            )
        )
    for key in ["Fault", "PastNumberOfClaims", "PoliceReportFiled", "WitnessPresent", "BasePolicy"]:
        if key in features and features[key] not in (None, ""):
            factors.append(RiskFactor(label=key, value=str(features[key]), severity="info"))
    return factors


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict_claim(request: PredictRequest) -> PredictResponse:
    try:
        result = predictor.predict(request.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    score = result["fraud_risk_score"]
    level = result["risk_level"]
    summary = (
        f"The saved {metadata['model_type']} pipeline categorized this claim as {level} "
        "for experimental historical claim-fraud review."
    )
    actions = {
        "LOW": ["Continue standard claim handling.", "Keep document review notes with the claim file."],
        "MEDIUM": ["Perform secondary document verification.", "Review claim history and policy details."],
        "HIGH": ["Route to SIU or senior claim review.", "Do not treat the score as proof of fraud."],
    }[level]
    return PredictResponse(
        predicted_class=result["predicted_class"],
        fraud_risk_score=score,
        risk_level=level,
        score_type="fraud_probability",
        calibrated_probability=False,
        verified_factors=_risk_factors(result["input_summary"], score),
        underwriting_summary=summary,
        review_actions=actions,
        explanation_source="deterministic_backend_summary",
        disclaimer=result["disclaimer"],
        input_summary=result["input_summary"],
    )
