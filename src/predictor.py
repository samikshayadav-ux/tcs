"""
Reusable Fraud Prediction Module for AutoRisk AI.

Provides an independent, production-grade interface for scoring insurance claims
against the trained fraud risk model pipeline. Designed for clean importation
by backend services (FastAPI in Part 2) without any web framework coupling.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd


# Explicitly unsupported frontend / OCR fields
UNSUPPORTED_FIELDS_EXPLANATIONS = {
    "annual_mileage": (
        "Field 'annual_mileage' is not present in the historical claim fraud dataset. "
        "The model cannot use annual mileage. Do not map or alias this to another feature."
    ),
    "driving_experience": (
        "Field 'driving_experience' is not supported. The dataset records driver 'Age' and 'DriverRating', "
        "not years of driving experience. Do not treat driving experience as age."
    ),
    "vehicle_usage": (
        "Field 'vehicle_usage' (e.g. Pleasure, Commute, Commercial) is not recorded in the dataset."
    ),
}

DEFAULT_DISCLAIMER = (
    "This risk score is a statistical underwriting risk estimate intended solely for triage "
    "and documentation review. It is NOT proof of fraud or a legal determination."
)


class FraudPredictor:
    """
    Independent predictor class that loads the trained pipeline and schema,
    validates incoming claims, and generates risk estimates.
    """

    def __init__(self, model_dir: str = "models"):
        """
        Initializes the predictor by loading the pipeline, metadata, and schema.

        Args:
            model_dir: Directory containing fraud_model.joblib and JSON metadata.
        """
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "fraud_model.joblib")
        self.schema_path = os.path.join(model_dir, "feature_schema.json")
        self.metadata_path = os.path.join(model_dir, "model_metadata.json")

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads model pipeline and schema files from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{self.model_path}'. "
                f"Please run 'python src/train.py' first."
            )

        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(
                f"Feature schema not found at '{self.schema_path}'. "
                f"Please ensure 'feature_schema.json' is present in '{self.model_dir}'."
            )

        try:
            self.pipeline = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load joblib model from '{self.model_path}': {exc}") from exc

        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failed to load schema from '{self.schema_path}': {exc}") from exc

        self.supported_features = self.schema.get("supported_features", {})
        self.feature_names = list(self.supported_features.keys())

        # Load metadata if available
        self.metadata = {}
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def validate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates a single claim input dictionary against the supported schema.

        Args:
            record: Dictionary of feature name to value.

        Returns:
            Validated and normalized record dictionary.

        Raises:
            ValueError: If unsupported fields, malformed values, or invalid types are detected.
        """
        if not isinstance(record, dict):
            raise ValueError(f"Input record must be a dictionary, received {type(record).__name__}.")

        cleaned_record = {}

        # 1. Check for known unsupported fields and give clear educational errors
        for unsupported_key, explanation in UNSUPPORTED_FIELDS_EXPLANATIONS.items():
            if unsupported_key in record:
                raise ValueError(
                    f"Unsupported field '{unsupported_key}' provided. {explanation}"
                )

        # 2. Check for unknown/unexpected keys
        for key in record:
            if key not in self.supported_features:
                raise ValueError(
                    f"Unexpected field '{key}'. Field is not in the supported feature schema. "
                    f"Supported features are: {self.feature_names}"
                )

        # 3. Validate types and bounds for supported keys
        for feat_name, spec in self.supported_features.items():
            if feat_name in record and record[feat_name] is not None:
                val = record[feat_name]
                expected_type = spec["type"]

                if expected_type == "numeric":
                    try:
                        num_val = float(val)
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"Field '{feat_name}' expects numeric value, received: '{val}'"
                        )
                    if feat_name == "Age" and num_val < 0:
                        raise ValueError(f"Field 'Age' cannot be negative. Received: {num_val}")
                    cleaned_record[feat_name] = num_val

                elif expected_type == "categorical":
                    # Convert to string and strip whitespace
                    str_val = str(val).strip()
                    cleaned_record[feat_name] = str_val
            else:
                # Field omitted or None - will be handled by pipeline SimpleImputer
                cleaned_record[feat_name] = np.nan

        return cleaned_record

    def predict(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generates fraud risk predictions for one or more claims.

        Args:
            data: Single dictionary, list of dictionaries, or pandas DataFrame.

        Returns:
            Dictionary with prediction results, or list of dictionaries for batch input.
        """
        is_single = False
        if isinstance(data, dict):
            is_single = True
            validated_records = [self.validate_record(data)]
        elif isinstance(data, list):
            if not data:
                raise ValueError("Input list cannot be empty.")
            validated_records = [self.validate_record(r) for r in data]
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("Input DataFrame cannot be empty.")
            records = data.to_dict(orient="records")
            validated_records = [self.validate_record(r) for r in records]
        else:
            raise ValueError(
                f"Unsupported data type '{type(data).__name__}'. "
                f"Expected dict, list of dicts, or pandas DataFrame."
            )

        # Convert validated records into DataFrame matching schema column ordering
        df_input = pd.DataFrame(validated_records)[self.feature_names]

        # Model inference
        predicted_classes = self.pipeline.predict(df_input)
        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba(df_input)[:, 1]
        else:
            probabilities = [None] * len(predicted_classes)

        results = []
        for i, pred_cls in enumerate(predicted_classes):
            prob = float(probabilities[i]) if probabilities[i] is not None else None
            cls_int = int(pred_cls)

            # Determine operational risk tier
            if prob is not None:
                if prob >= 0.65:
                    risk_level = "HIGH"
                elif prob >= 0.35:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
            else:
                risk_level = "HIGH" if cls_int == 1 else "LOW"

            result = {
                "predicted_class": cls_int,
                "fraud_risk_score": round(prob, 4) if prob is not None else None,
                "risk_level": risk_level,
                "disclaimer": DEFAULT_DISCLAIMER,
                "input_summary": {
                    k: v for k, v in validated_records[i].items() if pd.notnull(v)
                },
            }
            results.append(result)

        return results[0] if is_single else results
