import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { AlertCircle, BarChart3, CheckCircle2, FileImage, RefreshCw, ShieldCheck, UploadCloud, X } from "lucide-react";
import { api, DashboardStats, ExtractResponse, FeatureSpec, ModelInfo, PredictionResponse } from "./services/api";
import "./styles.css";

type Step = "upload" | "review" | "analyze" | "results";
type FeatureValues = Record<string, string | number | null>;

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"];
const MAX_FILE_SIZE = 8 * 1024 * 1024;

const samples: Array<{ name: string; subtitle: string; features: FeatureValues }> = [
  {
    name: "Standard sedan claim",
    subtitle: "Urban collision with routine documentation",
    features: {
      Make: "Honda", AccidentArea: "Urban", Sex: "Male", MaritalStatus: "Married", Age: 45, Fault: "Policy Holder",
      PolicyType: "Sedan - Collision", VehicleCategory: "Sedan", VehiclePrice: "20000 to 29000", Deductible: 400,
      DriverRating: 2, PastNumberOfClaims: "none", AgeOfVehicle: "5 years", AgeOfPolicyHolder: "41 to 50",
      PoliceReportFiled: "No", WitnessPresent: "No", AgentType: "External", NumberOfSuppliments: "none",
      AddressChange_Claim: "no change", NumberOfCars: "1 vehicle", BasePolicy: "Collision", Month: "Jan",
      DayOfWeek: "Monday", WeekOfMonth: 3, Days_Policy_Accident: "more than 30", Days_Policy_Claim: "more than 30",
    },
  },
  {
    name: "Complex all-perils claim",
    subtitle: "Higher-value vehicle and prior claim history",
    features: {
      Make: "BMW", AccidentArea: "Urban", Sex: "Female", MaritalStatus: "Single", Age: 29, Fault: "Policy Holder",
      PolicyType: "Sedan - All Perils", VehicleCategory: "Sedan", VehiclePrice: "more than 69000", Deductible: 500,
      DriverRating: 4, PastNumberOfClaims: "2 to 4", AgeOfVehicle: "2 years", AgeOfPolicyHolder: "26 to 30",
      PoliceReportFiled: "No", WitnessPresent: "No", AgentType: "External", NumberOfSuppliments: "more than 5",
      AddressChange_Claim: "under 6 months", NumberOfCars: "1 vehicle", BasePolicy: "All Perils", Month: "Mar",
      DayOfWeek: "Friday", WeekOfMonth: 4, Days_Policy_Accident: "8 to 15", Days_Policy_Claim: "15 to 30",
    },
  },
  {
    name: "Rural third-party claim",
    subtitle: "Lower review pressure, complete claim timing",
    features: {
      Make: "Ford", AccidentArea: "Rural", Sex: "Male", MaritalStatus: "Married", Age: 58, Fault: "Third Party",
      PolicyType: "Utility - Liability", VehicleCategory: "Utility", VehiclePrice: "30000 to 39000", Deductible: 400,
      DriverRating: 1, PastNumberOfClaims: "none", AgeOfVehicle: "more than 7", AgeOfPolicyHolder: "51 to 65",
      PoliceReportFiled: "Yes", WitnessPresent: "Yes", AgentType: "Internal", NumberOfSuppliments: "none",
      AddressChange_Claim: "no change", NumberOfCars: "2 vehicles", BasePolicy: "Liability", Month: "Oct",
      DayOfWeek: "Wednesday", WeekOfMonth: 2, Days_Policy_Accident: "more than 30", Days_Policy_Claim: "more than 30",
    },
  },
];

function pct(value: unknown) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "n/a";
}

function App() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loadingMeta, setLoadingMeta] = useState(true);
  const [metaError, setMetaError] = useState("");
  const [step, setStep] = useState<Step>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [extracting, setExtracting] = useState(false);
  const [extractError, setExtractError] = useState("");
  const [extraction, setExtraction] = useState<ExtractResponse | null>(null);
  const [formValues, setFormValues] = useState<FeatureValues>({});
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState("");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    Promise.all([api.modelInfo(), api.dashboardStats()])
      .then(([info, dashboard]) => {
        setModelInfo(info);
        setStats(dashboard);
      })
      .catch((error: Error) => setMetaError(error.message))
      .finally(() => setLoadingMeta(false));
  }, []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return;
    }
    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  const featureSpecs = modelInfo?.schema.supported_features || {};
  const featureNames = useMemo(() => Object.keys(featureSpecs), [featureSpecs]);

  function chooseFile(nextFile: File | null) {
    setExtractError("");
    setExtraction(null);
    setPrediction(null);
    if (!nextFile) {
      setFile(null);
      return;
    }
    if (!ACCEPTED_TYPES.includes(nextFile.type)) {
      setExtractError("Choose a JPEG, PNG, WEBP, HEIC, or HEIF document photo.");
      return;
    }
    if (nextFile.size > MAX_FILE_SIZE) {
      setExtractError("Choose an image under 8 MB.");
      return;
    }
    setFile(nextFile);
  }

  async function runExtraction() {
    if (!file || extracting) return;
    setExtracting(true);
    setExtractError("");
    try {
      const response = await api.extract(file);
      const nextValues: FeatureValues = {};
      response.fields.forEach((field) => {
        if (field.name in featureSpecs) nextValues[field.name] = field.value;
      });
      setExtraction(response);
      setFormValues(nextValues);
      setStep("review");
    } catch (error) {
      setExtractError((error as Error).message);
    } finally {
      setExtracting(false);
    }
  }

  async function runPrediction(values = formValues) {
    if (predicting) return;
    setPredicting(true);
    setPredictError("");
    setStep("analyze");
    try {
      const filtered = Object.fromEntries(Object.entries(values).filter(([key, value]) => key in featureSpecs && value !== "")) as FeatureValues;
      const result = await api.predict(filtered);
      setPrediction(result);
      setStep("results");
    } catch (error) {
      setPredictError((error as Error).message);
      setStep("review");
    } finally {
      setPredicting(false);
    }
  }

  function resetWorkflow() {
    setStep("upload");
    setFile(null);
    setExtraction(null);
    setFormValues({});
    setPrediction(null);
    setExtractError("");
    setPredictError("");
  }

  function FieldInput({ name, spec }: { name: string; spec: FeatureSpec }) {
    const value = formValues[name] ?? "";
    const extractedStatus = extraction?.fields.find((field) => field.name === name)?.status;
    const isMissing = extraction?.missing_fields.includes(name) || value === "";
    return (
      <label className={`field ${isMissing ? "missing" : ""}`}>
        <span>
          {name}
          {extractedStatus ? <em>{extractedStatus}</em> : null}
        </span>
        {spec.type === "categorical" && spec.allowed_categories ? (
          <select value={String(value)} onChange={(event) => setFormValues({ ...formValues, [name]: event.target.value })}>
            <option value="">Select value</option>
            {spec.allowed_categories.map((option) => <option key={option} value={option}>{option}</option>)}
          </select>
        ) : (
          <input
            type="number"
            min={spec.min}
            max={spec.max}
            value={String(value)}
            onChange={(event) => setFormValues({ ...formValues, [name]: event.target.value === "" ? "" : Number(event.target.value) })}
          />
        )}
      </label>
    );
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand"><ShieldCheck size={28} /> AutoRisk AI</div>
        <nav>
          <a className="active"><UploadCloud size={18} /> Document workflow</a>
          <a><BarChart3 size={18} /> Model metrics</a>
        </nav>
        <p className="sidebar-note">Experimental historical claim-fraud review indicator. Not proof of fraud, driver safety, accident risk, or premium value.</p>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <p className="eyebrow">Vehicle insurance risk profile summarizer</p>
            <h1>Claim document review and fraud-triage analysis</h1>
          </div>
          <button className="ghost" onClick={resetWorkflow}><RefreshCw size={16} /> Reset</button>
        </header>

        <section className="progress">
          {["Upload document", "Review extracted fields", "Analyze claim", "View results"].map((label, index) => (
            <div key={label} className={index <= ["upload", "review", "analyze", "results"].indexOf(step) ? "done" : ""}>
              <span>{index + 1}</span>{label}
            </div>
          ))}
        </section>

        <section className="grid">
          <div className="panel workflow">
            {step === "upload" && (
              <>
                <div className="section-title"><FileImage /> Upload document photo</div>
                <label className="dropzone">
                  <input type="file" accept={ACCEPTED_TYPES.join(",")} onChange={(event) => chooseFile(event.target.files?.[0] || null)} />
                  {previewUrl ? <img src={previewUrl} alt="Selected claim document preview" /> : <UploadCloud size={44} />}
                  <strong>{file ? file.name : "Choose an insurance or claim image"}</strong>
                  <span>Accepted: JPEG, PNG, WEBP, HEIC, HEIF. Limit: 8 MB.</span>
                </label>
                {file && <div className="row-actions"><button onClick={runExtraction} disabled={extracting}>{extracting ? "Extracting..." : "Extract fields"}</button><button className="secondary" onClick={() => chooseFile(null)}><X size={16} /> Remove</button></div>}
                {extractError && <ErrorBox message={extractError} action={file ? runExtraction : undefined} />}
              </>
            )}

            {(step === "review" || step === "analyze") && (
              <>
                <div className="section-title"><CheckCircle2 /> Review extracted values</div>
                {extraction?.demo_mode && <div className="warning">DEMO MODE extraction is active. These are sample values and must be reviewed.</div>}
                {extraction?.warnings.map((warning) => <div className="warning" key={warning}>{warning}</div>)}
                {predictError && <ErrorBox message={predictError} action={() => runPrediction()} />}
                <div className="form-grid">
                  {featureNames.map((name) => <FieldInput key={name} name={name} spec={featureSpecs[name]} />)}
                </div>
                <div className="row-actions"><button onClick={() => runPrediction()} disabled={predicting}>{predicting ? "Analyzing..." : "Confirm and analyze claim"}</button><button className="secondary" onClick={() => setStep("upload")}>Back</button></div>
              </>
            )}

            {step === "results" && prediction && (
              <>
                <div className="section-title"><ShieldCheck /> Review result</div>
                <div className={`score ${prediction.risk_level.toLowerCase()}`}>
                  <span>{prediction.risk_level}</span>
                  <strong>{prediction.fraud_risk_score == null ? "n/a" : `${(prediction.fraud_risk_score * 100).toFixed(1)}%`}</strong>
                  <p>{prediction.score_type.replace("_", " ")} Â· {prediction.calibrated_probability ? "calibrated" : "not calibrated"}</p>
                </div>
                <p className="summary">{prediction.underwriting_summary}</p>
                <div className="columns">
                  <div><h3>Verified factors</h3>{prediction.verified_factors.map((factor) => <p className="pill" key={`${factor.label}-${factor.value}`}>{factor.label}: {factor.value}</p>)}</div>
                  <div><h3>Review actions</h3>{prediction.review_actions.map((action) => <p className="action" key={action}>{action}</p>)}</div>
                </div>
                <p className="disclaimer">{prediction.disclaimer}</p>
                <div className="row-actions"><button onClick={resetWorkflow}>Analyze another document</button><button className="secondary" onClick={() => setStep("review")}>Edit inputs</button></div>
              </>
            )}
          </div>

          <aside className="panel side">
            <h2>Sample profiles</h2>
            {samples.map((sample) => (
              <button className="sample" key={sample.name} onClick={() => { setFormValues(sample.features); setExtraction(null); void runPrediction(sample.features); }}>
                <strong>{sample.name}</strong><span>{sample.subtitle}</span>
              </button>
            ))}
            <h2>Backend</h2>
            <p className="muted">API base: {api.baseUrl}</p>
            {loadingMeta && <p className="muted">Loading model metadata...</p>}
            {metaError && <ErrorBox message={metaError} />}
          </aside>
        </section>

        <section className="metrics">
          <Metric title="Historical rows" value={stats?.historical_counts?.rows.toLocaleString() || "n/a"} note={stats?.message || "Historical dataset statistics"} />
          <Metric title="Holdout recall" value={pct(stats?.selected_metrics.recall_fraud)} note="Fraud recall on saved evaluation split" />
          <Metric title="Holdout precision" value={pct(stats?.selected_metrics.precision_fraud)} note="Fraud precision on saved evaluation split" />
          <Metric title="Model" value={modelInfo?.model.selected_model_name || "n/a"} note={modelInfo?.model.model_type || "Loaded from backend"} />
        </section>
      </main>
    </div>
  );
}

function ErrorBox({ message, action }: { message: string; action?: () => void }) {
  return <div className="error"><AlertCircle size={18} /><span>{message}</span>{action && <button className="secondary" onClick={action}>Retry</button>}</div>;
}

function Metric({ title, value, note }: { title: string; value: string; note: string }) {
  return <article className="metric"><span>{title}</span><strong>{value}</strong><p>{note}</p></article>;
}

createRoot(document.getElementById("root")!).render(<App />);

