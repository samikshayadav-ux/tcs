# AutoRisk AI - Vehicle Insurance Risk Profile Summarizer

AutoRisk AI is a hackathon-ready vehicle insurance claim review demo. It combines:

- A trained scikit-learn pipeline for historical insurance claim-fraud triage.
- A FastAPI backend that exposes model metadata, dashboard statistics, document extraction, and prediction endpoints.
- A React dashboard that uploads document photos, reviews extracted fields, sends confirmed model inputs, and displays the saved model's prediction.

Important limitation: the Kaggle dataset predicts historical insurance claim fraud labels. It does not predict accident probability, driver safety, vehicle reliability, or insurance premiums. Results are experimental review indicators, not proof of fraud.

## Project Structure

```text
data/fraud_oracle.csv                 # Historical Kaggle dataset
models/fraud_model.joblib             # Saved scikit-learn pipeline
models/feature_schema.json            # Supported model fields and categories
models/model_metadata.json            # Training metadata and limitations
models/evaluation_metrics.json        # Holdout evaluation metrics
src/predictor.py                      # Reusable FraudPredictor
src/api/main.py                       # FastAPI backend
frontend/                             # React + Vite dashboard
tests/                                # Existing ML tests
```

## Prerequisites

- Python 3.10+
- Node.js 20+ recommended for the React frontend
- The model artifacts in `models/`
- The dataset at `data/fraud_oracle.csv` for full dashboard statistics

## Backend Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Backend Environment Variables

```bash
AUTORISK_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
AUTORISK_MAX_UPLOAD_BYTES=8388608
AUTORISK_DEMO_EXTRACTION=false
```

`AUTORISK_DEMO_EXTRACTION=true` enables an explicitly labeled local demo extraction response. When it is false, `POST /api/v1/extract` returns a clear configuration error instead of fabricating document OCR results.

## Vision API Configuration

Live document extraction is intentionally isolated in `src/api/main.py` at `POST /api/v1/extract`. Add your paid or internal vision provider there, keep the multipart field name as `file`, and map only fields that exist in `models/feature_schema.json`. Do not log full document contents or API secrets.

## Optional Explanation API

The current backend returns `explanation_source: deterministic_backend_summary`. If you add an LLM explanation service, it must only explain the saved model output. It must not alter `predicted_class`, `fraud_risk_score`, or `risk_level`.

## Frontend Setup

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

Set the backend URL in `frontend/.env`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Build:

```bash
npm run build
```

## API Endpoints

- `GET /health`
- `GET /api/v1/model-info`
- `GET /api/v1/dashboard-stats`
- `POST /api/v1/extract` with multipart field `file`
- `POST /api/v1/predict` with `{ "features": { ...supported model fields } }`

Unsupported frontend concepts such as `annual_mileage`, `driving_experience`, and `vehicle_usage` are rejected by the predictor because they are not in the historical dataset.

## Complete Workflow Test

1. Start the backend.
2. Start the frontend.
3. Confirm model metrics load from `/api/v1/model-info` and `/api/v1/dashboard-stats`.
4. Upload a supported image.
5. If no live vision provider is configured, verify the UI shows the extraction configuration error and does not show fake extracted values.
6. For a local demo, restart the backend with `AUTORISK_DEMO_EXTRACTION=true`, upload an image, review the clearly marked demo values, edit any fields, and analyze the claim.
7. Confirm samples call the real `/api/v1/predict` endpoint and do not hardcode scores.
8. Confirm the result is labeled as an experimental historical claim-fraud review indicator.

## Tests

```bash
python -m pytest tests/ -v
cd frontend
npm run build
```

External vision or LLM APIs should be mocked in automated tests. Paid credentials are not required for the default test suite.

## Known Limitations

- Historical dataset covers 1994-1996 claim records and is not representative of all modern vehicles or insurance products.
- The model probability is not calibrated, as reported by the backend score semantics.
- Dashboard counts are historical dataset/evaluation statistics, not live application usage.
- Live OCR/vision extraction is not bundled; demo extraction must be explicitly enabled and is visibly labeled.
