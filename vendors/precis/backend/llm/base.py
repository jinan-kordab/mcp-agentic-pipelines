# =============================================================================
# © JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT
# =============================================================================
# Model-agnostic interface for all external LLM providers.
# Every provider implements this interface → orchestrator doesn't care which LLM.
#
# Supported providers:
#   - OpenAIProvider      → GPT-4o, GPT-4, GPT-3.5 (backend/llm/openai_provider.py)
#   - AnthropicProvider   → Claude 3.5 Sonnet, Claude 3 Opus
#   - GoogleProvider      → Gemini 1.5 Pro, Gemini 1.5 Flash
#   - DeepSeekProvider    → DeepSeek V4 Pro, DeepSeek Reasoner
#   - OllamaProvider      → Llama 3, Mistral, Phi-3 (local, no API key)
#
# Usage:
#   provider = LLMFactory.create("openai")  # or "ollama" for local
#   response = await provider.generate("Hello, world!")
#
# Related:
#   backend/llm/factory.py     — Factory for creating providers from config
#   backend/config.py          — Settings (API keys, default provider)
# =============================================================================

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator


class LLMProvider(ABC):
    """
    Abstract base for all LLM providers.

    Every provider must implement:
      - generate(): Single-turn text generation
      - generate_stream(): Streaming text generation (for real-time UI)
      - Embed (optional): Generate embeddings (for semantic fallback)

    The orchestrator uses this interface exclusively — it never
    imports provider-specific classes directly.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """
        Generate a text completion for the given prompt.

        Args:
            prompt: The user/agent prompt
            system_prompt: Optional system-level instruction
            temperature: Creativity level (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens in the response
            **kwargs: Provider-specific parameters

        Returns:
            Generated text response
        """
        ...
        # TODO: Each provider implements this differently
        #   - OpenAI: client.chat.completions.create()
        #   - Anthropic: client.messages.create()
        #   - Google: model.generate_content()
        #   - Ollama: client.chat()
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a text completion token by token.

        Used by the WebSocket endpoint to stream agent outputs
        to the frontend in real-time (AgentTimeline component).

        Args:
            prompt: The user/agent prompt
            system_prompt: Optional system-level instruction
            temperature: Creativity level
            max_tokens: Maximum tokens

        Yields:
            Text tokens as they are generated
        """
        ...
        # TODO: Each provider implements streaming differently
        pass

    @abstractmethod
    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Used ONLY for semantic fallback search — not the primary retrieval path.
        Primary retrieval uses ExactHash (96% accurate, no embeddings needed).

        Args:
            texts: List of texts to embed
            model: Optional model override

        Returns:
            List of embedding vectors
        """
        ...
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the currently active model name.

        Used for logging, cost tracking, and UI display.
        """
        ...
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for a text string.

        Used for:
          - Budget tracking (cost estimation before API call)
          - Context window management (don't exceed model limits)
          - Prompt optimization (trim if needed)

        Args:
            text: The text to count tokens for

        Returns:
            Estimated token count
        """
        ...
        pass
