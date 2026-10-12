"""
Configuration Module for AutoRisk AI Backend.

Loads environment variables, handles defaults for development and production,
and ensures secrets are never hardcoded or printed.
"""

import os
from typing import List
from pydantic import BaseModel, Field


def _get_project_root() -> str:
    """Calculates absolute project root path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # backend/app -> backend -> project_root
    return os.path.abspath(os.path.join(current_dir, "..", ".."))


PROJECT_ROOT = _get_project_root()


class Settings(BaseModel):
    """Application settings schema and environment variable loader."""

    app_name: str = "AutoRisk AI Backend"
    app_version: str = "1.0.0"
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))

    # CORS origins
    cors_origins: List[str] = Field(default_factory=lambda: [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if origin.strip()
    ])

    # Paths to Part 1 artifacts
    model_dir: str = Field(default_factory=lambda: os.getenv(
        "MODEL_DIR",
        os.path.join(PROJECT_ROOT, "models")
    ))
    data_path: str = Field(default_factory=lambda: os.getenv(
        "DATA_PATH",
        os.path.join(PROJECT_ROOT, "data", "fraud_oracle.csv")
    ))

    # File upload limits
    max_upload_size_bytes: int = Field(default_factory=lambda: int(
        os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024))  # 10 MB default
    ))
    allowed_image_types: List[str] = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp"
    ]

    # External Vision & LLM API Configuration
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    vision_provider: str = Field(default_factory=lambda: os.getenv("VISION_PROVIDER", "auto"))
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto"))
    request_timeout_seconds: float = Field(default_factory=lambda: float(
        os.getenv("REQUEST_TIMEOUT_SECONDS", "30.0")
    ))


settings = Settings()
