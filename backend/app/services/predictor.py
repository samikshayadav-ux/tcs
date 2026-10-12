"""
Backend Predictor Service for AutoRisk AI.

Loads the Part 1 trained pipeline once in memory, runs inference on validated
features, scales scores to indicative 0-100 values, and assigns risk tiers.
"""

import os
import sys
import uuid
from typing import Any, Dict, Optional, Tuple

from backend.app.config import settings
from backend.app.services.feature_mapping import map_input_to_model_features

# Ensure project root is in sys.path so src.predictor can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predictor import FraudPredictor, DEFAULT_DISCLAIMER


class PredictorService:
    """Singleton service wrapping the Part 1 model pipeline."""

    _instance: Optional["PredictorService"] = None

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or settings.model_dir
        self.predictor: Optional[FraudPredictor] = None
        self.load_error: Optional[str] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initializes FraudPredictor from disk."""
        try:
            self.predictor = FraudPredictor(model_dir=self.model_dir)
            self.load_error = None
        except Exception as exc:
            self.predictor = None
            self.load_error = str(exc)

    @classmethod
    def get_instance(cls) -> "PredictorService":
        """Returns or creates singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_loaded(self) -> bool:
        """Returns True if the pipeline is successfully loaded."""
        return self.predictor is not None

    def predict_claim(
        self,
        confirmed_fields: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs fraud risk inference on user-confirmed fields.

        Args:
            confirmed_fields: User-verified dictionary of claim features.
            request_id: Optional request ID.

        Returns:
            Dictionary containing prediction labels, probability, and features used.
        """
        if not self.is_loaded():
            raise RuntimeError(
                f"Prediction model is unavailable: {self.load_error or 'Artifacts not loaded.'}"
            )

        req_id = request_id or str(uuid.uuid4())

        # 1. Map and validate inputs to canonical model features
        mapped_features, mapping_warnings = map_input_to_model_features(confirmed_fields)

        # 2. Run inference via Part 1 predictor
        raw_result = self.predictor.predict(mapped_features)

        pred_class = int(raw_result["predicted_class"])
        fraud_prob = raw_result.get("fraud_risk_score")

        # 3. Compute indicative score (0 to 100)
        # Note: Uncalibrated model probability scaled for dashboard review
        indicative_score = round(fraud_prob * 100, 1) if fraud_prob is not None else None

        # 4. Determine risk level
        if fraud_prob is not None:
            if fraud_prob >= 0.65:
                risk_tier = "HIGH"
                prediction_label = "Priority Review Flagged"
            elif fraud_prob >= 0.35:
                risk_tier = "MEDIUM"
                prediction_label = "Secondary Verification Required"
            else:
                risk_tier = "LOW"
                prediction_label = "Standard Underwriting Queue"
        else:
            risk_tier = "HIGH" if pred_class == 1 else "LOW"
            prediction_label = "Flagged for Review" if pred_class == 1 else "Standard Queue"

        return {
            "request_id": req_id,
            "prediction_label": prediction_label,
            "predicted_class": pred_class,
            "fraud_score": fraud_prob,
            "indicative_score_0_100": indicative_score,
            "score_type": "uncalibrated_model_probability",
            "risk_category": risk_tier,
            "input_features_used": mapped_features,
            "mapping_warnings": mapping_warnings,
            "disclaimer": DEFAULT_DISCLAIMER,
        }


def get_predictor_service() -> PredictorService:
    """Helper to access the predictor service singleton."""
    return PredictorService.get_instance()
