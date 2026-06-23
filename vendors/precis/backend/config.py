# =============================================================================
# © JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT
# =============================================================================
# Centralized configuration loaded from environment variables and .env file.
# All settings are typed and validated by Pydantic at startup.
#
# Usage:
#   from backend.config import settings
#   api_key = settings.OPENAI_API_KEY  # Auto-loaded from .env
#
# Related:
#   .env.example  — Template with all available settings
#   docker-compose.yml — Passes env vars to containers
# =============================================================================

from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Precis application settings.

    All fields are automatically loaded from environment variables
    or the .env file. Pydantic validates types at startup — if a required
    field is missing or has the wrong type, the app fails fast with a clear error.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore unknown env vars (don't crash)
    )

    # --- Application ---
    APP_NAME: str = "Precis"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"  # development | staging | production
    LOG_LEVEL: str = "INFO"           # DEBUG | INFO | WARNING | ERROR

    # --- Database ---
    DATABASE_URL: str = "sqlite:///data/app.db"

    # --- Redis (Optional) ---
    REDIS_URL: Optional[str] = None

    # --- External LLM Providers ---
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None

    # --- Local LLM (Ollama) ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL: str = "llama3"

    # --- Default LLM Provider ---
    DEFAULT_LLM_PROVIDER: str = "deepseek"  # deepseek | openai | anthropic | google | ollama

    # --- RBF Predictor Hyperparameters ---
    # See: backend/agents/radial_interpol.py for usage
    # See: Alt_DNN.pdf Theorem 2.1 for mathematical foundation
    RBF_TAU: float = 500.0    # Sharpness — higher τ → more exact interpolation at training points
    RBF_GAMMA: float = 1.0    # Kernel width — controls influence radius of each training node

    # --- NoGAN Synthesizer Defaults ---
    # See: backend/agents/dist_free_synth.py for usage
    NOGAN_DEFAULT_BINS: int = 50
    NOGAN_MODE: str = "random_counts"  # random_counts | fixed_counts

    # --- Anomaly Detection Thresholds ---
    # See: backend/agents/stat_anomaly.py for usage
    ANOMALY_MULTI_ENTITY_THRESHOLD: int = 50
    ANOMALY_SPIKE_SIGMA: float = 3.0

    # --- Evaluation Thresholds ---
    # See: backend/agents/veri_score.py for usage
    EVAL_MIN_RELEVANCY: float = 0.6
    EVAL_MIN_TRUSTWORTHINESS: float = 0.5

    # --- Security ---
    CORS_ORIGINS: str = "http://localhost:3000"
    MAX_UPLOAD_SIZE_MB: int = 50


# Global settings instance — import this everywhere
settings = Settings()
