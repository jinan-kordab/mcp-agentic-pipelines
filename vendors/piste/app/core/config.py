# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Piste — Centralized Settings
================================
All configuration via environment variables + .env file.
Uses pydantic-settings for validation.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # --- App ---
    APP_NAME: str = "Piste"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- CORS ---
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # --- Database (PostgreSQL 16) ---
    DATABASE_URL: str = "postgresql+asyncpg://piste:piste@localhost:5432/piste"
    DATABASE_URL_SYNC: str = "postgresql+psycopg2://piste:piste@localhost:5432/piste"

    # --- Redis 7.2 ---
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- FAISS ---
    FAISS_INDEX_PATH: str = "./data/faiss_evidence.index"
    FAISS_DIMENSION: int = 1536  # OpenAI embedding dimension

    # --- LiteLLM ---
    LITELLM_MODEL: str = "deepseek/deepseek-chat"
    LITELLM_FALLBACK_MODELS: List[str] = ["deepseek/deepseek-chat", "claude-3-haiku-20240307"]
    LITELLM_REQUEST_TIMEOUT: int = 600
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""

    # --- DSPy ---
    DSPY_OPTIMIZER: str = "BootstrapFewShot"  # or "MIPROv2"
    DSPY_MAX_LABELED_EXAMPLES: int = 100

    # --- Pipeline ---
    VOTING_COMPLETIONS: int = 3
    VOTING_THRESHOLD: float = 0.67  # 2/3 majority
    MAX_RETRY_LOOPS: int = 3        # Loop 1 max retries
    FAISS_CACHE_THRESHOLD: float = 0.92  # Cosine similarity for cache hit

    # --- Idempotency ---
    VERDICT_CACHE_TTL_SECONDS: int = 86400  # 24 hours
    IDEMPOTENCY_LOCK_TTL_SECONDS: int = 3600  # 1 hour

    # --- Rate Limiting ---
    RATE_LIMIT_PER_USER: int = 10    # claims per minute
    RATE_LIMIT_GLOBAL: int = 100     # claims per minute

    # --- Auth ---
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"

    # --- Observability ---
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "piste"
    PROMETHEUS_PORT: int = 9090

    # --- Search Providers ---
    TAVILY_API_KEY: str = ""
    SERPER_API_KEY: str = ""
    GOOGLE_CSE_API_KEY: str = ""
    GOOGLE_CSE_ID: str = ""


settings = Settings()
