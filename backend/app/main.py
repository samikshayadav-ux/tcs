"""
Main FastAPI Application for AutoRisk AI.

Provides RESTful endpoints for health check, insurance document extraction,
user-confirmed fraud risk prediction, model metadata, and dashboard statistics.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.schemas import (
    DashboardStatsResponse,
    ExtractionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from backend.app.services.dashboard import get_dashboard_service
from backend.app.services.document_extractor import (
    DocumentExtractor,
    VisionExtractionError,
    get_document_extractor,
)
from backend.app.services.explainer import get_explanation_service
from backend.app.services.predictor import get_predictor_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autorisk.api")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AutoRisk AI Backend: Vehicle Insurance Risk Profile Summarizer. "
        "Provides document-photo extraction, user-confirmed fraud prediction, "
        "and underwriting explanation services."
    ),
)

# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Custom Exception Handlers
# ---------------------------------------------------------------------------

from fastapi.encoders import jsonable_encoder

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    # Format readable error message
    error_messages = []
    for err in errors:
        loc = " -> ".join([str(x) for x in err.get("loc", []) if x != "body"])
        msg = str(err.get("msg", "Validation error"))
        error_messages.append(f"{loc}: {msg}" if loc else msg)

    safe_errors = []
    for err in errors:
        clean_err = {k: v for k, v in err.items() if k != "ctx"}
        if "ctx" in err and isinstance(err["ctx"], dict):
            clean_err["ctx"] = {k: str(v) for k, v in err["ctx"].items()}
        safe_errors.append(clean_err)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": "; ".join(error_messages),
            "errors": safe_errors,
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Invalid Input Field",
            "detail": str(exc),
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Verifies service health and Part 1 model load status."""
    predictor_svc = get_predictor_service()
    model_loaded = predictor_svc.is_loaded()
    model_name = (
        predictor_svc.predictor.metadata.get("selected_model_name", "logistic_regression_balanced")
        if model_loaded else None
    )

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_name=model_name,
        app_version=settings.app_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/v1/extract", response_model=ExtractionResponse, tags=["Extraction"])
async def extract_document_photo(
    file: UploadFile = File(..., description="Uploaded insurance or claim document image")
) -> ExtractionResponse:
    """
    Extracts structured policy and claim fields from an insurance document photo.
    Returns proposed fields for user review and confirmation before prediction.
    """
    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {exc}"
        )

    content_type = file.content_type or "image/jpeg"
    extractor = get_document_extractor()

    try:
        result = extractor.extract_document(
            image_bytes=image_bytes,
            content_type=content_type,
            filename=file.filename
        )
        return result
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except VisionExtractionError as vis_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(vis_err)
        )
    except Exception as exc:
        logger.error("Unexpected extraction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal extraction processing failure: {exc}"
        )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_fraud_risk(payload: PredictionRequest) -> PredictionResponse:
    """
    Evaluates confirmed claim fields against the trained fraud risk model.
    Generates an underwriting explanation and suggested review actions.
    """
    predictor_svc = get_predictor_service()
    if not predictor_svc.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "ML prediction model is not loaded. Ensure Part 1 artifacts exist "
                "in 'models/' directory or run src/train.py."
            )
        )

    # 1. Run prediction pipeline
    try:
        raw_input_dict = payload.model_dump(by_alias=False, exclude_none=True)
        # Also include extra fields if provided
        prediction_result = predictor_svc.predict_claim(raw_input_dict)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err)
        )
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction pipeline error: {exc}"
        )

    # 2. Generate underwriting explanation
    explainer_svc = get_explanation_service()
    features_used = prediction_result.get("input_features_used", {})
    explanation = explainer_svc.explain_prediction(prediction_result, features_used)

    return PredictionResponse(
        request_id=prediction_result["request_id"],
        prediction_label=prediction_result["prediction_label"],
        predicted_class=prediction_result["predicted_class"],
        fraud_score=prediction_result["fraud_score"],
        indicative_score_0_100=prediction_result["indicative_score_0_100"],
        score_type=prediction_result["score_type"],
        risk_category=prediction_result["risk_category"],
        top_risk_factors=explanation["top_risk_factors"],
        underwriting_summary=explanation["underwriting_summary"],
        recommended_review_actions=explanation["recommended_review_actions"],
        explanation_source=explanation["explanation_source"],
        disclaimer=prediction_result["disclaimer"],
        input_features_used=features_used,
    )


@app.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
def get_model_information() -> ModelInfoResponse:
    """Returns trained model specifications, feature lists, and actual evaluation metrics."""
    dash_svc = get_dashboard_service()
    return dash_svc.get_model_info()


@app.get("/api/v1/dashboard-stats", response_model=DashboardStatsResponse, tags=["Dashboard"])
def get_dashboard_statistics() -> DashboardStatsResponse:
    """Returns historical dataset statistics and live system service status."""
    dash_svc = get_dashboard_service()
    predictor_svc = get_predictor_service()
    extractor = get_document_extractor()

    return dash_svc.get_dashboard_stats(
        model_loaded=predictor_svc.is_loaded(),
        vision_available=extractor.has_configured_provider(),
        llm_available=bool(settings.gemini_api_key or settings.openai_api_key),
    )
