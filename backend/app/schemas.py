"""
Pydantic Schemas for AutoRisk AI Backend.

Defines strict request/response structures for health, document extraction,
user-confirmed prediction requests, explanation outputs, and dashboard statistics.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


# Common statutory disclaimer
DISCLAIMER_TEXT = (
    "This risk score is an experimental statistical model triage estimate intended solely "
    "to assist manual claim review. It does NOT establish proof of fraud or constitute an adverse action."
)


# ---------------------------------------------------------------------------
# Health Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    model_name: Optional[str] = None
    app_version: str
    timestamp: str


# ---------------------------------------------------------------------------
# Document Extraction Schemas
# ---------------------------------------------------------------------------

class ExtractedDocumentFields(BaseModel):
    document_type: Optional[str] = Field(None, description="Plausible document classification")
    policy_number: Optional[str] = Field(None, description="Identifier for reference only (excluded from ML)")
    make: Optional[str] = Field(None, description="Vehicle make (e.g. Honda, Toyota)")
    vehicle_category: Optional[str] = Field(None, description="Category (Sedan, Sport, Utility)")
    age_of_vehicle: Optional[str] = Field(None, description="Vehicle age bin or years")
    driver_age: Optional[float] = Field(None, description="Numeric age of driver")
    age_of_policy_holder: Optional[str] = Field(None, description="Policyholder age bin")
    sex: Optional[str] = Field(None, description="Sex (Male, Female)")
    marital_status: Optional[str] = Field(None, description="Marital status (Single, Married, Divorced, Widow)")
    deductible: Optional[float] = Field(None, description="Policy deductible amount")
    base_policy: Optional[str] = Field(None, description="Coverage type (Collision, Liability, All Perils)")
    policy_type: Optional[str] = Field(None, description="Full policy type (e.g. Sedan - Collision)")
    accident_date: Optional[str] = Field(None, description="Incident date if visible")
    accident_area: Optional[str] = Field(None, description="Incident location (Urban, Rural)")
    fault: Optional[str] = Field(None, description="Fault determination (Policy Holder, Third Party)")
    driver_rating: Optional[float] = Field(None, description="Driver risk tier (1 to 4)")
    past_number_of_claims: Optional[str] = Field(None, description="Claims history (none, 1, 2 to 4, more than 4)")
    police_report_filed: Optional[str] = Field(None, description="Police report indicator (Yes, No)")
    witness_present: Optional[str] = Field(None, description="Witness indicator (Yes, No)")
    agent_type: Optional[str] = Field(None, description="Agent type (External, Internal)")
    claim_amount: Optional[float] = Field(None, description="Claim monetary amount if indicated")
    raw_extracted_text: Optional[str] = Field(None, description="Raw summary of visible document text")


class ExtractionResponse(BaseModel):
    extraction_status: str = Field(..., description="'success', 'partial', or 'failed'")
    document_type: Optional[str] = None
    extracted_fields: ExtractedDocumentFields
    model_compatible_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Fields mapped to Part 1 feature schema for user confirmation"
    )
    missing_fields: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    unsupported_fields_detected: List[str] = Field(default_factory=list)
    review_instructions: str = (
        "Please review and verify all extracted fields below before submitting for risk scoring. "
        "Missing fields may be manually supplied or left blank to use standard imputation."
    )


# ---------------------------------------------------------------------------
# Prediction Schemas (User Confirmation Contract)
# ---------------------------------------------------------------------------

UNSUPPORTED_FRONTEND_FIELDS = {
    "annual_mileage": (
        "Field 'annual_mileage' is not supported by the historical fraud detection model. "
        "Do not map or alias this to another feature."
    ),
    "driving_experience": (
        "Field 'driving_experience' is not supported. The model uses driver 'Age' and 'DriverRating', "
        "not years of driving experience. Do not treat driving experience as age."
    ),
    "vehicle_usage": (
        "Field 'vehicle_usage' (e.g. commute/pleasure) is not in the model training schema."
    ),
}


class PredictionRequest(BaseModel):
    """
    User-confirmed structured values for fraud risk prediction.
    Accepts both canonical dataset names (e.g. 'Make') and camel/snake case variants.
    Explicitly rejects unsupported fields.
    """
    # Supported Numeric Features
    week_of_month: Optional[float] = Field(None, alias="WeekOfMonth")
    age: Optional[float] = Field(None, alias="Age")
    deductible: Optional[float] = Field(None, alias="Deductible")
    driver_rating: Optional[float] = Field(None, alias="DriverRating")

    # Supported Categorical Features
    month: Optional[str] = Field(None, alias="Month")
    day_of_week: Optional[str] = Field(None, alias="DayOfWeek")
    make: Optional[str] = Field(None, alias="Make")
    accident_area: Optional[str] = Field(None, alias="AccidentArea")
    sex: Optional[str] = Field(None, alias="Sex")
    marital_status: Optional[str] = Field(None, alias="MaritalStatus")
    fault: Optional[str] = Field(None, alias="Fault")
    policy_type: Optional[str] = Field(None, alias="PolicyType")
    vehicle_category: Optional[str] = Field(None, alias="VehicleCategory")
    vehicle_price: Optional[str] = Field(None, alias="VehiclePrice")
    days_policy_accident: Optional[str] = Field(None, alias="Days_Policy_Accident")
    days_policy_claim: Optional[str] = Field(None, alias="Days_Policy_Claim")
    past_number_of_claims: Optional[str] = Field(None, alias="PastNumberOfClaims")
    age_of_vehicle: Optional[str] = Field(None, alias="AgeOfVehicle")
    age_of_policy_holder: Optional[str] = Field(None, alias="AgeOfPolicyHolder")
    police_report_filed: Optional[str] = Field(None, alias="PoliceReportFiled")
    witness_present: Optional[str] = Field(None, alias="WitnessPresent")
    agent_type: Optional[str] = Field(None, alias="AgentType")
    number_of_suppliments: Optional[str] = Field(None, alias="NumberOfSuppliments")
    address_change_claim: Optional[str] = Field(None, alias="AddressChange_Claim")
    number_of_cars: Optional[str] = Field(None, alias="NumberOfCars")
    base_policy: Optional[str] = Field(None, alias="BasePolicy")

    # Optional metadata that is NOT fed into the model
    policy_number: Optional[str] = Field(None, description="Document tracking identifier only (excluded from ML)")

    model_config = {
        "populate_by_name": True,
        "extra": "allow"  # We catch extra keys in model_validator to provide specific error messages
    }

    @model_validator(mode="before")
    @classmethod
    def validate_no_unsupported_fields(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        for field_name, explanation in UNSUPPORTED_FRONTEND_FIELDS.items():
            if field_name in values:
                raise ValueError(f"Unsupported field '{field_name}' provided. {explanation}")

        # Validate non-negative age
        age_val = values.get("age", values.get("Age"))
        if age_val is not None:
            try:
                num_age = float(age_val)
            except (TypeError, ValueError):
                num_age = None
            if num_age is not None and num_age < 0:
                raise ValueError("Field 'Age' cannot be negative.")

        return values


class PredictionResponse(BaseModel):
    request_id: str
    prediction_label: str
    predicted_class: int
    fraud_score: Optional[float]
    indicative_score_0_100: Optional[float]
    score_type: str = "uncalibrated_model_probability"
    risk_category: str
    top_risk_factors: List[str]
    underwriting_summary: str
    recommended_review_actions: List[str]
    explanation_source: str
    disclaimer: str = DISCLAIMER_TEXT
    input_features_used: Dict[str, Any]
    model_version: str = "Part1-LogisticRegression-Balanced"


# ---------------------------------------------------------------------------
# Dashboard and Model Info Schemas
# ---------------------------------------------------------------------------

class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    target_name: str
    training_timestamp: Optional[str]
    feature_count: int
    supported_features: List[str]
    excluded_features: List[str]
    evaluation_metrics: Dict[str, Any]
    trade_off_analysis: Dict[str, Any]
    limitations: List[str]


class DashboardStatsResponse(BaseModel):
    historical_dataset: Dict[str, Any]
    model_performance: Dict[str, Any]
    service_status: Dict[str, Any]
    notice: str = (
        "Dataset statistics reflect historical Kaggle claims baseline. "
        "Production inference runs are tracked in-memory per session."
    )
