# AutoRisk AI — Vehicle Insurance Risk Profile Summarizer (Part 1: ML & Data Layer)

AutoRisk AI is an underwriting risk-scoring and claim-triage engine. This repository houses **Part 1**, implementing the data inspection, preprocessing pipeline, model training and selection, evaluation, persisted artifacts, and an independent, reusable prediction module (`FraudPredictor`) designed for seamless integration into Part 2 (API and document-extraction flow).

---

## 1. Project Overview & Scope

- **Objective**: Train and persist a machine learning pipeline on historical vehicle insurance claim fraud records to score incoming claims for underwriting and investigative triage.
- **Dataset**: Kaggle Vehicle Insurance Claim Fraud Detection dataset by shivamb (`fraud_oracle.csv`).
- **Target**: `FraudFound_P` (`0` = Legitimate / Non-Fraudulent, `1` = Fraudulent).
- **Core Scope**: Pure ML and data layer. Does **not** include React UI, OCR vision API, or FastAPI endpoints (reserved for Part 2).

> [!IMPORTANT]
> **Insurance Domain Purpose & Disclaimer**:
> This dataset concerns historical claim fraud classification from the 1990s. It **does not** predict vehicle mechanical reliability, accident probability, or insurance premiums. Model output probabilities represent statistical risk estimates for human triage, **never definitive proof or legal accusations of fraud**.

---

## 2. Directory Structure

```text
├── data/
│   └── fraud_oracle.csv           # Kaggle vehicle insurance dataset (15,420 rows)
├── models/
│   ├── fraud_model.joblib         # Fitted scikit-learn Pipeline (preprocessing + model)
│   ├── model_metadata.json        # Training timestamp, feature lists, hyperparameters, limitations
│   ├── evaluation_metrics.json    # Measured confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC
│   └── feature_schema.json        # Supported feature names, types, categories, and unsupported fields
├── src/
│   ├── __init__.py                # Package exports (FraudPredictor)
│   ├── data_inspection.py         # Reproducible dataset EDA, schema, and leakage inspection CLI
│   ├── train.py                   # Stratified split, ColumnTransformer, model comparison & artifact export
│   ├── evaluate.py                # Standalone evaluation & underwriting trade-off analyzer CLI
│   └── predictor.py               # Reusable FraudPredictor class for Part 2 API consumption
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py           # Pipeline integrity, artifact existence, reproducibility tests
│   └── test_predictor.py          # Input validation, unknown handling, unsupported field rejection tests
├── requirements.txt               # Locked dependencies
└── README.md                      # Documentation
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Virtual environment recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset Location
Place `fraud_oracle.csv` inside the `data/` directory:
```bash
mkdir -p data
# Place fraud_oracle.csv at data/fraud_oracle.csv
```
The dataset contains 15,420 rows and 33 attributes. The data loader handles UTF-8 BOM encoding automatically.

---

## 4. Reproducible Commands

### A. Data Inspection
Run the data inspection CLI to analyze shape, data types, class balance, zero values, and potential leakage:
```bash
python src/data_inspection.py --data-path data/fraud_oracle.csv
```

### B. Model Training & Artifact Generation
Run model training to execute stratified train/test split, train candidate models, evaluate metrics, and export artifacts to `models/`:
```bash
python src/train.py --data-path data/fraud_oracle.csv --output-dir models --random-seed 42
```

### C. Standalone Model Evaluation
Evaluate the persisted model on test claims and generate an insurance trade-off report:
```bash
python src/evaluate.py --data-path data/fraud_oracle.csv --model-path models/fraud_model.joblib
```

### D. Automated Test Suite
Run the 18 automated unit and integration tests:
```bash
python -m pytest tests/ -v
```

---

## 5. Model Evaluation & Comparison Results

Evaluated on a stratified 20% holdout test set (**3,084 claims**, containing **185 fraudulent claims** and **2,899 legitimate claims**):

| Model Candidate | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | 94.00% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.0600 |
| **Dummy (Stratified)** | 88.26% | 6.40% | 7.03% | 0.0670 | 0.5024 | 0.0603 |
| **Logistic Regression (Balanced)** *(Selected)* | **63.88%** | **13.05%** | **88.65%** | **0.2275** | **0.8067** | **0.1534** |
| **Random Forest (Balanced)** | 66.54% | 13.08% | 81.08% | 0.2252 | 0.8077 | 0.1760 |
| **Random Forest (Default)** | 94.03% | 100.00% | 0.54% | 0.0108 | 0.8186 | 0.2360 |

### Confusion Matrix for Selected Model (`Logistic Regression Balanced`)
- **True Negatives (TN)**: 1,806 (Legitimate claims cleared without audit)
- **False Positives (FP)**: 1,093 (Legitimate claims routed to secondary verification)
- **False Negatives (FN)**: 21 (Fraudulent claims missed)
- **True Positives (TP)**: 164 (Fraudulent claims caught out of 185 total)

### Insurance Trade-off Rationale
1. **The Accuracy Fallacy**: A naive baseline predicting "no fraud" achieves 94% accuracy while catching zero fraudulent claims. Similarly, unweighted Random Forest achieves 94.03% accuracy but catches only 1 out of 185 frauds.
2. **False Negatives (Severe Financial Risk)**: Fraudulent payouts result in permanent indemnity capital loss that cannot easily be recovered.
3. **False Positives (Manageable Operational Cost)**: Routing a flagged claim to the Special Investigation Unit (SIU) costs ~1 hour of investigator audit time. Flagging 164/185 frauds (88.65% recall) substantially mitigates leakage.

---

## 6. Schema Compatibility & Future Document Extraction Contract

The future Part 2 pipeline will extract fields from insurance documents (e.g. policy declarations, registration, claim reports).

### Field Feasibility & Mapping

| Field | Source / Origin | Status | Handling Strategy |
| :--- | :--- | :--- | :--- |
| `Make` | Insurance card / Dec page | Supported | Categorical (e.g. Honda, Toyota) |
| `VehicleCategory` | Dec page / registration | Supported | Categorical ('Sedan', 'Sport', 'Utility') |
| `AgeOfVehicle` | Registration / Dec page | Supported | Binned string ('new', '1 year', ..., '7 years', 'more than 7') |
| `Deductible` | Policy Dec page | Supported | Numeric (300, 400, 500, 700) |
| `BasePolicy` | Coverage section | Supported | Categorical ('Collision', 'Liability', 'All Perils') |
| `PolicyType` | Coverage section | Supported | Categorical (e.g. 'Sedan - Collision') |
| `Age` / `AgeOfPolicyHolder` | Driver ID / Dec page | Supported | Numeric (Age) / Binned string |
| `PastNumberOfClaims` | Claims history record | Supported | Categorical ('none', '1', '2 to 4', 'more than 4') |
| `DriverRating` | Policy underwriting tier | Supported | Numeric (1, 2, 3, 4) |
| `AccidentArea` | Incident description | Supported | Categorical ('Urban', 'Rural') |
| `Fault` | Police / Incident report | Supported | Categorical ('Policy Holder', 'Third Party') |
| **`annual_mileage`** | **Frontend concept** | **UNSUPPORTED** | **Explicitly rejected**; not in historical data |
| **`driving_experience`** | **Frontend concept** | **UNSUPPORTED** | **Explicitly rejected**; distinct from Age / Rating |
| **`vehicle_usage`** | **Frontend concept** | **UNSUPPORTED** | **Explicitly rejected**; no usage tier recorded |

> [!CAUTION]
> **No Silent Mapping**:
> `FraudPredictor` explicitly raises a descriptive `ValueError` if consumers attempt to pass `annual_mileage`, `driving_experience`, or `vehicle_usage`. It does not alias or invent substitute mappings.

---

## 7. Reusable Predictor Usage (Part 2 API Integration)

```python
from src.predictor import FraudPredictor

# 1. Initialize predictor (loads pipeline, metadata, and schema)
predictor = FraudPredictor(model_dir="models")

# 2. Input claim payload (dictionary, list of dicts, or pandas DataFrame)
claim_payload = {
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
    "Month": "May",
    "DayOfWeek": "Tuesday",
    "WeekOfMonth": 2,
    "Days_Policy_Accident": "more than 30",
    "Days_Policy_Claim": "more than 30",
}

# 3. Predict
result = predictor.predict(claim_payload)

print(result)
# Output:
# {
#   "predicted_class": 0,
#   "fraud_risk_score": 0.0842,
#   "risk_level": "LOW",
#   "disclaimer": "This risk score is a statistical underwriting risk estimate intended solely for triage and documentation review. It is NOT proof of fraud or a legal determination.",
#   "input_summary": { ... }
# }
```

### Safe Handling of Unseen Categories
If an incoming claim contains an unseen category (e.g. `Make: "Genesis"`), the fitted `OneHotEncoder(handle_unknown="ignore")` encodes the category as all zeros without raising runtime exceptions.

---

## 8. Limitations & Boundary Conditions

1. **Historical Domain Data**: The dataset dates from 1994 to 1996. Newer vehicle technologies (EVs, telematics, autonomous features) are not represented.
2. **Review vs Decisioning**: High fraud risk scores require human investigation by qualified insurance auditors. The model is an automated triage filter, not an autonomous decider.
3. **Threshold Calibration**: The default decision threshold (0.5) can be calibrated in Part 2 based on insurer risk tolerance and investigator staffing capacity.
