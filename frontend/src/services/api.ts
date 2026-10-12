const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const REQUEST_TIMEOUT_MS = 20000;

export type FeatureSpec = {
  type: "numeric" | "categorical";
  allowed_categories?: string[];
  min?: number;
  max?: number;
  description?: string;
};

export type ExtractedField = {
  name: string;
  value: string | number | null;
  status: "extracted" | "missing" | "uncertain" | "demo";
  editable: boolean;
};

export type ExtractResponse = {
  status: "completed" | "partial" | "failed" | "demo";
  fields: ExtractedField[];
  missing_fields: string[];
  warnings: string[];
  supported_features: string[];
  demo_mode: boolean;
};

export type PredictionResponse = {
  predicted_class: number;
  fraud_risk_score: number | null;
  risk_level: "LOW" | "MEDIUM" | "HIGH";
  score_type: string;
  calibrated_probability: boolean;
  verified_factors: Array<{ label: string; value: string; severity: "low" | "medium" | "high" | "info" }>;
  underwriting_summary: string;
  review_actions: string[];
  explanation_source: string;
  disclaimer: string;
  input_summary: Record<string, string | number>;
};

export type ModelInfo = {
  model: {
    model_type: string;
    selected_model_name: string;
    limitations: string[];
    dataset_split_configuration: Record<string, unknown>;
  };
  schema: {
    supported_features: Record<string, FeatureSpec>;
    unsupported_fields: Record<string, string>;
  };
  selected_metrics: Record<string, number | Record<string, number>>;
  score_semantics: {
    score_type: string;
    calibrated_probability: boolean;
    risk_thresholds: Record<string, string>;
    disclaimer: string;
  };
};

export type DashboardStats = {
  source: string;
  usage_stats_available: boolean;
  message: string;
  dataset: Record<string, unknown>;
  historical_counts?: {
    rows: number;
    fraud_claims: number;
    legitimate_claims: number;
    fraud_rate_pct: number;
  };
  selected_metrics: Record<string, number | Record<string, number>>;
};

class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.headers || {}),
      },
      signal: controller.signal,
    });

    const text = await response.text();
    let body: unknown = null;
    if (text) {
      try {
        body = JSON.parse(text);
      } catch {
        throw new ApiError("The server returned malformed JSON.", response.status);
      }
    }

    if (!response.ok) {
      const detail = typeof body === "object" && body && "detail" in body ? String((body as { detail: unknown }).detail) : response.statusText;
      throw new ApiError(detail || "Request failed.", response.status);
    }

    return body as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("The request timed out. Check that the backend is running.", 0);
    }
    throw new ApiError("Unable to reach the AutoRisk API. Check the backend URL and CORS settings.", 0);
  } finally {
    window.clearTimeout(timeout);
  }
}

export const api = {
  baseUrl: API_BASE_URL,
  health: () => request<{ status: string; model_loaded: boolean }>("/health"),
  modelInfo: () => request<ModelInfo>("/api/v1/model-info"),
  dashboardStats: () => request<DashboardStats>("/api/v1/dashboard-stats"),
  extract: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return request<ExtractResponse>("/api/v1/extract", {
      method: "POST",
      body: formData,
    });
  },
  predict: (features: Record<string, string | number | null>) =>
    request<PredictionResponse>("/api/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    }),
};
