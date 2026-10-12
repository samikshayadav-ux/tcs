# AutoRisk AI — Vehicle Insurance Risk Profile Summarizer

AutoRisk AI is an automated insurance claim underwriting risk-scoring and triage platform.
- **Part 1**: Machine Learning & Data Layer (historical claim-fraud classification, scikit-learn pipeline, model comparison, evaluation metrics, and persisted artifacts).
- **Part 2**: FastAPI Backend & AI Document Extraction (vision-based document photo extraction, user-confirmation contract, feature mapping, ML inference integration, explanation service, and dashboard statistics).
- **Part 3**: React Dashboard (to be connected in Part 3).

---

## 1. System Architecture & Flow

```text
[Insurance Document Photo (JPG/PNG)]
                  │
                  ▼
   POST /api/v1/extract (Vision API)
                  │
                  ▼
   [Extracted Fields & Proposed Mappings]
                  │
                  ▼
   [User Review & Confirmation UI]
                  │
                  ▼
   POST /api/v1/predict (Confirmed Fields)
                  │
                  ├─► [Feature Mapping & Validation (Rejects Unsupported)]
                  ├─► [Loaded ML Pipeline: fraud_model.joblib]
                  ├─► [Underwriting Explainer (Template or LLM)]
                  │
                  ▼
   [Structured JSON Response with Indicative Score (0-100) & Actions]
```

> [!IMPORTANT]
> **Semantic Limitation & Disclaimer**:
> The model is trained on historical vehicle insurance claim fraud classification data. It **does not** predict vehicle mechanical breakdown, accident rates, or premium rates. The output is an **experimental claim-review assistance score**, never legal proof or an accusation of fraud.

---

## 2. Project Directory Structure

```text
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py              # Environment configuration & settings
│   │   ├── main.py                # FastAPI endpoints, CORS, exception handlers
│   │   ├── schemas.py             # Pydantic request & response schemas
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── dashboard.py       # Model info & historical dataset statistics
│   │       ├── document_extractor.py # Vision API integration (Gemini/OpenAI)
│   │       ├── explainer.py       # Deterministic template & optional LLM explainer
│   │       ├── feature_mapping.py # Canonical schema mapping & validation
│   │       └── predictor.py       # Singleton ML model inference service
│   ├── tests/
│   │   ├── test_dashboard.py      # Model info & dashboard statistics tests
│   │   ├── test_explainer.py      # Template & LLM fallback tests
│   │   ├── test_extraction.py     # Image upload validation & vision mock tests
│   │   ├── test_health.py         # GET /health test
│   │   └── test_prediction.py      # Feature validation, rejection, and scoring tests
│   ├── requirements.txt           # Backend dependencies
│   └── .env.example               # Environment variables template
├── data/
│   └── fraud_oracle.csv           # Kaggle vehicle insurance dataset (15,420 rows)
├── models/
│   ├── fraud_model.joblib         # Fitted scikit-learn Pipeline
│   ├── model_metadata.json        # Training timestamp, hyperparameters, limitations
│   ├── evaluation_metrics.json    # Confusion matrix, precision, recall, F1, ROC-AUC
│   └── feature_schema.json        # 26 supported canonical features & disallowed fields
├── src/
│   ├── data_inspection.py         # Reproducible dataset EDA CLI
│   ├── train.py                   # Model training and artifact generation CLI
│   ├── evaluate.py                # Standalone evaluation & trade-off analyzer
│   └── predictor.py               # Part 1 FraudPredictor module
├── tests/                         # Part 1 ML pipeline tests (18 tests)
├── .env.example
├── requirements.txt
└── README.md
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Key environment variables:
| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `APP_ENV` | `development` | Runtime environment |
| `HOST` | `0.0.0.0` | Server host binding |
| `PORT` | `8000` | Server port |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173` | Allowed React origins |
| `MODEL_DIR` | `models` | Directory holding Part 1 trained artifacts |
| `MAX_UPLOAD_SIZE_BYTES`| `10485760` (10MB) | Maximum file upload size |
| `GEMINI_API_KEY` | *(empty)* | Optional: Enables document photo extraction via Gemini 1.5 Flash |
| `OPENAI_API_KEY` | *(empty)* | Optional: Enables document photo extraction via GPT-4o-mini |

*Note: If no API key is configured, `/health`, `/api/v1/predict`, `/api/v1/model-info`, and `/api/v1/dashboard-stats` remain fully operational. Calling `/api/v1/extract` will return an honest 503 explaining that a vision API key is required.*

---

## 4. Running the Application

### A. Start the FastAPI Server
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are available at: `http://localhost:8000/docs`

### B. Retrain the Model (if needed)
```bash
python src/train.py --data-path data/fraud_oracle.csv --output-dir models
```

### C. Run the Complete Test Suite
Run all 39 unit and integration tests across ML and Backend:
```bash
python -m pytest tests/ backend/tests/ -v
```

---

## 5. API Reference & Examples

### `GET /health`
Verifies backend service health and model load state.
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "logistic_regression_balanced",
  "app_version": "1.0.0",
  "timestamp": "2026-10-12T14:15:00.000000+00:00"
}
```

---

### `POST /api/v1/extract`
Uploads an insurance document image (JPG/PNG). Returns proposed extracted values for user review.

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@sample_insurance_card.jpg;type=image/jpeg"
```

**Response Example**:
```json
{
  "extraction_status": "success",
  "document_type": "Insurance Card",
  "extracted_fields": {
    "document_type": "Insurance Card",
    "policy_number": "POL-987654",
    "make": "Toyota",
    "vehicle_category": "Sedan",
    "age_of_vehicle": "3 years",
    "driver_age": 34.0,
    "deductible": 400.0,
    "base_policy": "Collision",
    "accident_date": "2024-05-15",
    "accident_area": "Urban",
    "fault": "Policy Holder",
    "driver_rating": 2.0,
    "past_number_of_claims": "none",
    "police_report_filed": "Yes",
    "witness_present": "No",
    "claim_amount": 3200.0
  },
  "model_compatible_fields": {
    "Make": "Toyota",
    "VehicleCategory": "Sedan",
    "AgeOfVehicle": "3 years",
    "Age": 34.0,
    "Deductible": 400.0,
    "BasePolicy": "Collision",
    "AccidentArea": "Urban",
    "Fault": "Policy Holder",
    "DriverRating": 2.0,
    "PastNumberOfClaims": "none",
    "PoliceReportFiled": "Yes",
    "WitnessPresent": "No"
  },
  "missing_fields": ["Days_Policy_Accident", "NumberOfSuppliments", ...],
  "warnings": [
    "Policy number 'POL-987654' was extracted for document tracking, but will be strictly excluded from ML risk scoring to prevent identifier leakage."
  ],
  "unsupported_fields_detected": [],
  "review_instructions": "Please review and verify all extracted fields below before submitting for risk scoring."
}
```

---

### `POST /api/v1/predict`
Accepts **confirmed structured values** from the user. Runs ML inference and produces underwriting summary.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Make": "Honda",
    "VehicleCategory": "Sedan",
    "Age": 38,
    "Deductible": 400,
    "DriverRating": 3,
    "AccidentArea": "Urban",
    "Fault": "Policy Holder",
    "PastNumberOfClaims": "2 to 4",
    "AgeOfVehicle": "5 years",
    "BasePolicy": "Collision"
  }'
```

**Response Example**:
```json
{
  "request_id": "9a38f322-d784-4d87-9bc0-8809a80e1cf2",
  "prediction_label": "Priority Review Flagged",
  "predicted_class": 1,
  "fraud_score": 0.8124,
  "indicative_score_0_100": 81.2,
  "score_type": "uncalibrated_model_probability",
  "risk_category": "HIGH",
  "top_risk_factors": [
    "Policyholder is designated at-fault in the incident description.",
    "Elevated claim frequency: policyholder has '\''2 to 4'\'' prior claims on record.",
    "High risk driver tier: policy rating is 3 (scale 1-4)."
  ],
  "underwriting_summary": "The statistical model assigned an elevated review priority score of 81.2/100 (uncalibrated fraud-class score: 0.812) for this Honda claim under Collision. Key risk indicators include policyholder is designated at-fault in the incident description. This claim warrants secondary verification by the Special Investigation Unit (SIU) before settlement.",
  "recommended_review_actions": [
    "Request itemized repair estimate and shop invoices for physical damage validation.",
    "Verify proof of loss and confirm incident timeline against police / tow records.",
    "Review policyholder historical claims records across industry databases.",
    "Assign to Special Investigation Unit (SIU) desk for documentation audit."
  ],
  "explanation_source": "template",
  "disclaimer": "This risk score is an experimental statistical model triage estimate intended solely to assist manual claim review. It does NOT establish proof of fraud or constitute an adverse action.",
  "input_features_used": {
    "Make": "Honda",
    "VehicleCategory": "Sedan",
    "Age": 38.0,
    "Deductible": 400.0,
    "DriverRating": 3.0,
    "AccidentArea": "Urban",
    "Fault": "Policy Holder",
    "PastNumberOfClaims": "2 to 4",
    "AgeOfVehicle": "5 years",
    "BasePolicy": "Collision"
  },
  "model_version": "Part1-LogisticRegression-Balanced"
}
```

---

### `GET /api/v1/model-info`
Returns exact model metadata, features, and actual measured metrics on holdout test set (3,084 claims):
- **Recall (Fraud Class)**: 88.65%
- **Precision (Fraud Class)**: 13.05%
- **ROC-AUC**: 0.8067
- **PR-AUC**: 0.1534
- **Accuracy**: 63.88% (Secondary metric)

---

### `GET /api/v1/dashboard-stats`
Returns historical dataset baseline distributions and real-time backend service status.

---

## 6. Supported Feature Schema & Strict Validations

### 26 Supported Canonical Model Features
- **Numeric (4)**: `WeekOfMonth`, `Age`, `Deductible`, `DriverRating`
- **Categorical (22)**: `Month`, `DayOfWeek`, `Make`, `AccidentArea`, `Sex`, `MaritalStatus`, `Fault`, `PolicyType`, `VehicleCategory`, `VehiclePrice`, `Days_Policy_Accident`, `Days_Policy_Claim`, `PastNumberOfClaims`, `AgeOfVehicle`, `AgeOfPolicyHolder`, `PoliceReportFiled`, `WitnessPresent`, `AgentType`, `NumberOfSuppliments`, `AddressChange_Claim`, `NumberOfCars`, `BasePolicy`.

### Unsupported Fields Policy
The following fields are **explicitly rejected** with an informative `HTTP 422`:
1. `annual_mileage`: Not in historical claim dataset.
2. `driving_experience`: Not in dataset; distinct from Age and DriverRating.
3. `vehicle_usage`: Commute vs pleasure tiers are not recorded.

### Identifiers Policy
`PolicyNumber` and `RepNumber` are stripped from ML features to prevent identifier leakage and memorization.
