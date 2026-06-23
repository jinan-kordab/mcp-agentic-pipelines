"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

from typing import Dict, Optional, Type
from backend.llm.base import LLMProvider
from backend.config import settings


class LLMFactory:
    """Central registry for LLM providers. Hot-swap by changing config."""

    _providers: Dict[str, Type[LLMProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[LLMProvider]) -> None:
        """Register a provider class. Called at startup for all 5 providers."""
        cls._providers[name] = provider_class

    @classmethod
    def create(cls, name: str, **kwargs) -> LLMProvider:
        """Create a provider instance by name. Raises ValueError if unknown."""
        provider_class = cls._providers.get(name)
        if provider_class is None:
            available = list(cls._providers.keys())
            raise ValueError(f"Unknown provider '{name}'. Available: {available}")
        api_keys = {
            "openai": settings.OPENAI_API_KEY, "anthropic": settings.ANTHROPIC_API_KEY,
            "google": settings.GOOGLE_API_KEY, "deepseek": settings.DEEPSEEK_API_KEY,
        }
        api_key = api_keys.get(name)
        if api_key and name != "ollama" and "api_key" not in kwargs:
            return provider_class(api_key=api_key, **kwargs)
        return provider_class(**kwargs)

    @classmethod
    def create_default(cls) -> LLMProvider:
        """Create the default provider from config. Falls back to first available."""
        name = settings.DEFAULT_LLM_PROVIDER
        if name in cls._providers:
            try:
                return cls.create(name)
            except Exception:
                pass
        for fallback in cls._providers:
            try:
                return cls.create(fallback)
            except Exception:
                continue
        raise RuntimeError(f"No LLM provider available. Registered: {list(cls._providers.keys())}")

    @classmethod
    def create_with_fallback(cls, primary: str = "openai", fallback: str = "ollama") -> LLMProvider:
        """Create primary provider, with automatic fallback on failure."""
        try:
            return cls.create(primary)
        except Exception:
            return cls.create(fallback)

    @classmethod
    def list_available(cls) -> list:
        return list(cls._providers.keys())
